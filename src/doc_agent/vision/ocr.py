"""Stage 3 — OCR/HTR (BASELINE = pretrained foundation, fine-tuned)

Single-pass VLM reading (Step 0 decision D2): one Qwen2.5-VL call per region, no
GraDeT-HTR stage and no confidence-routed fallback. A two-stage reader needs a calibrated
confidence signal to route on, and there is no OCR accuracy measurement yet, so the
fallback branch would be tuned against nothing. Revisit once eval/metrics.py:ocr_f1 scores
the held-out pages.

The GraDeT-HTR + DocLayout-YOLO path is DEFERRED, NOT REJECTED. It was always intended as
a Bangla-handwriting-specific fallback for when the single VLM underperforms (DocLayout-YOLO
only because GraDeT-HTR does no layout detection of its own). Deferring it is a sequencing
call -- measure the single pass first, then decide -- not a judgement that it is the wrong
fallback.

CHECKPOINT: Qwen2.5-VL-3B-Instruct, run LOCALLY (config.yaml ocr.model). Local inference is
a deliberate design constraint, not a convenience: a production land-deed system handles
sensitive records, and hosted LLM APIs may retain what is sent to them. Note precisely that
THIS corpus is public (Mendeley Dolil, CC BY 4.0) and carries no such risk itself -- the
constraint is about the deployment this project models. 3B over 7B because 7B fp16 does not
fit the available T4, and 3B avoids a quantization toolchain whose build can break; if
ocr_f1 says 3B is not enough, escalating to 7B-AWQ is then an evidence-backed change.

One Chunk per PAGE, not per region: Stage 4's legal-semantic chunker cuts on deed and
paragraph boundaries, and it cannot find a boundary that spans two regions if this stage
has already cut the text up. Regions are concatenated in the reading order layout emitted.

A Region carries only page_id and bbox, but a Chunk needs doc_id and the reader needs the
image. Both come from the page index below, which prefers the ingest sidecar and otherwise
reads the deed grouping directly. It refuses to invent a doc_id unless explicitly allowed:
doc_id is what keeps the train/val/test split document-level, so a silent page-as-document
fallback would manufacture exactly the leak the corpus policy exists to prevent.
"""

from __future__ import annotations

import csv
import json
import unicodedata
from pathlib import Path

from PIL import Image, ImageOps

from ..contracts import *  # noqa
from ..llm.prompts import TRANSCRIBE
from ..logging_conf import get_logger

logger = get_logger(__name__)

# A1 commits to mapping Bangla numerals to ASCII so that "২১৬৩" and "2163" compare equal
# under Exact Match -- a plot or deed number must match to the character.
_BANGLA_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def normalize(text: str, ascii_digits: bool = True) -> str:
    """Unicode NFC + optional Bangla->ASCII digits, per the A1 normalization policy."""
    out = unicodedata.normalize("NFC", text).strip()
    return out.translate(_BANGLA_DIGITS) if ascii_digits else out


def load_page_index(cfg: dict) -> dict[str, dict[str, str]]:
    """page_id -> {image_path, doc_id}, from the ingest sidecar or the raw dir + grouping."""
    params = cfg["ocr"]
    sidecar = Path(params["sidecar"])

    if sidecar.is_file():
        index: dict[str, dict[str, str]] = {}
        for line in sidecar.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            # Prefer the deskewed greyscale: it is what Stage 1 hands downstream, and the
            # binarization can drop faint seals and matras the reader still needs.
            image = record.get("greyscale_path") or record["original_path"]
            index[record["page_id"]] = {"image_path": image, "doc_id": record["doc_id"]}
        return index

    pages_dir = Path(params["pages_dir"])
    if not pages_dir.is_dir():
        raise FileNotFoundError(
            f"no ingest sidecar at {sidecar} and no page directory at {pages_dir}"
        )

    groups = _load_deed_groups(params)
    index = {}
    for path in sorted(pages_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        page_id = path.stem
        doc_id = groups[page_id] if groups is not None else page_id
        index[page_id] = {"image_path": str(path), "doc_id": doc_id}
    return index


def _load_deed_groups(params: dict) -> dict[str, str] | None:
    """The page->deed map, or None when running standalone with page-as-document allowed."""
    groups_path = Path(params["deed_groups"])
    if not groups_path.is_file():
        if params.get("allow_page_as_doc"):
            logger.warning(
                "no deed grouping at %s -- treating every page as its own document. This "
                "makes the split leak by construction; standalone development only.",
                groups_path,
            )
            return None
        raise FileNotFoundError(
            f"no ingest sidecar and no deed_groups file at {groups_path}. Stage 3 cannot "
            "invent doc_id: it is what keeps the train/val/test split document-level. Run "
            "ingest first, or set ocr.allow_page_as_doc for standalone development."
        )

    with groups_path.open(encoding="utf-8", newline="") as fh:
        return {row["page_id"]: row["doc_id"] for row in csv.DictReader(fh)}


class Reader:
    """Model set by cfg['ocr']. Baseline: pretrained Qwen2.5-VL, one call per region."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg["ocr"]
        self.pages = load_page_index(cfg)
        self._model = None
        self._processor = None

    def crop(self, region: Region) -> Image.Image:
        """The region's pixels, cut from its page image, in display orientation.

        exif_transpose is required, not cosmetic: some of this corpus is phone photos
        carrying an EXIF orientation tag. cv2.imread (used by Stage 2 layout) applies that
        tag, so region bboxes are in display coordinates, while PIL.Image.open does not.
        Without this the reader would silently be handed the wrong part of a rotated page.
        """
        page = self.pages.get(region.page_id)
        if page is None:
            raise ValueError(f"no image known for page {region.page_id!r}")
        with Image.open(page["image_path"]) as image:
            upright = ImageOps.exif_transpose(image)
            return upright.convert("RGB").crop(region.bbox)

    def _load(self) -> None:
        """Load the VLM on first use, so importing this module stays cheap and CPU-safe."""
        if self._model is not None:
            return
        from transformers import AutoModelForVision2Seq, AutoProcessor  # heavy, load late

        name = self.cfg["model"]
        logger.info("loading reader %s", name)
        self._processor = AutoProcessor.from_pretrained(name)
        self._model = AutoModelForVision2Seq.from_pretrained(name, device_map="auto")

    def transcribe_region(self, region: Region) -> str:
        """Read one region with a single VLM call. Transcription only -- see prompts.TRANSCRIBE."""
        self._load()
        image = self.crop(region)
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": TRANSCRIBE}],
            }
        ]
        prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(text=[prompt], images=[image], return_tensors="pt").to(
            self._model.device
        )
        generated = self._model.generate(
            **inputs, max_new_tokens=int(self.cfg["max_new_tokens"]), do_sample=False
        )
        trimmed = generated[0][inputs["input_ids"].shape[1] :]
        return self._processor.decode(trimmed, skip_special_tokens=True)


def transcribe(regions: list[Region], cfg: dict, reader: object | None = None) -> list[Chunk]:
    """Regions -> one Chunk per page, in the order layout emitted them.

    `reader` is injectable so the assembly around the model can be tested without loading
    a 7B VLM; production callers (pipeline.py) leave it None.
    """
    if not regions:
        return []

    params = cfg["ocr"]
    ascii_digits = bool(params["ascii_digits"])
    if reader is None:
        reader = Reader(cfg)
        pages = reader.pages  # already built; don't read the sidecar twice
    else:
        pages = load_page_index(cfg)

    ordered_pages: list[str] = []
    by_page: dict[str, list[Region]] = {}
    for region in regions:
        if region.page_id not in by_page:
            by_page[region.page_id] = []
            ordered_pages.append(region.page_id)
        by_page[region.page_id].append(region)

    unknown = [page_id for page_id in ordered_pages if page_id not in pages]
    if unknown:
        raise ValueError(
            f"{len(unknown)} region page(s) missing from the page index, e.g. {unknown[:5]}"
        )

    chunks: list[Chunk] = []
    for page_id in ordered_pages:
        pieces = []
        for region in by_page[page_id]:
            text = normalize(reader.transcribe_region(region), ascii_digits)
            if text:
                pieces.append(text)

        if not pieces:
            # An entirely unreadable page yields no chunk rather than an empty one: an
            # empty chunk would be indexed and could be retrieved as evidence for nothing.
            logger.warning("page %s produced no readable text", page_id)
            continue

        chunks.append(
            Chunk(
                id=f"{page_id}#p",
                doc_id=pages[page_id]["doc_id"],
                text="\n".join(pieces),
                page_ids=[page_id],
            )
        )

    logger.info("read %d page(s) into %d chunk(s)", len(ordered_pages), len(chunks))
    return chunks
