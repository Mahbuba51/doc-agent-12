"""Stage 3 — OCR/HTR (BASELINE = pretrained foundation, fine-tuned)

Single-pass VLM reading (Step 0 decision D2): one Qwen2.5-VL call per region, no
GraDeT-HTR stage and no confidence-routed fallback. A two-stage reader needs a calibrated
confidence signal to route on, and there was no OCR accuracy measurement when this was
written, so the fallback branch would have been tuned against nothing. That measurement now
exists and it OVERTURNS the decision -- see D2 MEASURED below. What follows in this file is
the measured baseline, not the intended production reader.

D2 MEASURED 2026-08-13 -- SINGLE-PASS QWEN2.5-VL FAILS ON THIS CORPUS.
scripts/score_heldout.py over the 7 human-typed gold pages, on a Kaggle T4 (fp16, greedy,
max_pixels 1280*28*28 ~= 1.0M px, max_new_tokens 768), recorded in reports/ocr_baseline.json:

    mean ocr_f1 0.022   median 0.000   range 0.000-0.076   22 of 22 regions hit the cap

against a headline target of F1 >= 0.92. The reader is not misreading the handwriting -- it
is not reading it. It emits fluent, invented Bangla prose about a document-shaped image:
dolil_66 came back as "established on 20 May 1657 ... one of the oldest national memorials
of an artist", dolil_605 as "established by the founder of the university on 10 May 1972".
Neither page says anything of the kind. It then degenerates into repetition until the token
cap. The 100% cap-hit rate is the symptom; the confabulation is the defect.

NOT A TUNING PROBLEM. Ablation on dolil_66 (43 gold words, 1 region), one variable at a
time, same prompt and crop as production:

    control (1.0M px, no penalty)     f1 0.000   113 words   capped
    max_pixels x4 (~3.2M px)          f1 0.014   148 words   capped
    repetition_penalty 1.1            f1 0.000   108 words   capped
    both                              f1 0.108    14 words   capped

Read the predicted text, not the F1 column: the highest-scoring cell is the one that read
LEAST. Its 14 words are mostly numerals, one of them a 90-digit run, so the score is
numeric collision, not reading. Four cells, four cap-hits, no knob.

CONSEQUENCES.
1. 3B -> 7B-AWQ is NOT supported by this evidence, contrary to what this docstring claimed
   before (corrected below). The gap is categorical, not marginal: a larger model with the
   same missing capability confabulates more fluently, not more accurately.
2. GraDeT-HTR is promoted from deferred fallback to candidate reader.
3. That re-opens D1. GraDeT-HTR is TrOCR-style and reads LINE crops, so layout becomes
   load-bearing again -- and vision/layout.py records that 4 of the 8 held-out scans yield
   a single page-sized region because the writing runs continuously down the sheet. Line
   segmentation on those pages is unsolved work, not a formality.
4. OPEN, UNMEASURED: whether this same checkpoint reads PRINTED Bangla (dolil_20 carries
   printed gold). That bounds whether any VLM survives anywhere in this pipeline.
5. This is evidence about the READER ONLY. The ~78,000-word floor in data/provenance.md
   stands as an estimate and must NOT be revised from this run -- the reader under-extracts,
   which is exactly the error that estimate already warns about.

The GraDeT-HTR + DocLayout-YOLO path is PROMOTED (was: deferred, not rejected). It was
always intended as a Bangla-handwriting-specific fallback for when the single VLM
underperforms (DocLayout-YOLO only because GraDeT-HTR does no layout detection of its own),
and the single VLM has now underperformed by an order of magnitude. Promotion is blocked on
one check nobody has run: that obtainable weights exist under a usable licence. If they do
not, the choice is between a different Bangla HTR checkpoint, fine-tuning one (which needs
line-level labels this project does not have), and re-opening the local-inference constraint
below -- all three are team decisions, not lane decisions.

CHECKPOINT: Qwen2.5-VL-3B-Instruct, run LOCALLY (config.yaml ocr.model). Local inference is
a deliberate design constraint, not a convenience: a production land-deed system handles
sensitive records, and hosted LLM APIs may retain what is sent to them. Note precisely that
THIS corpus is public (Mendeley Dolil, CC BY 4.0) and carries no such risk itself -- the
constraint is about the deployment this project models. 3B over 7B because 7B fp16 does not
fit the available T4, and 3B avoids a quantization toolchain whose build can break. This
paragraph used to end "if ocr_f1 says 3B is not enough, escalating to 7B-AWQ is then an
evidence-backed change" -- ocr_f1 has now said 3B is not enough AND that 7B is the wrong
escalation, because what is missing is Bangla handwriting capability, not parameters.

The local-inference constraint is what rules out the hosted OCR services that do read Bangla
handwriting well. It is a real constraint, deliberately chosen -- but D2 MEASURED is the
point at which its cost became known rather than hypothetical, so the team should re-affirm
it explicitly instead of inheriting it.

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
    excluded: list[str] = []
    for path in sorted(pages_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        page_id = path.stem
        if groups is None:
            doc_id = page_id
        else:
            # The grouping defines the corpus: 105 duplicate re-scans and 2 non-content pages
            # carry no row (data/provenance.md). Such a page is EXCLUDED, not an error -- and
            # skipping it is not the doc_id invention this module refuses to do, because an
            # excluded page has a correct answer: leave it out. Failing hard here meant one
            # dedup pass took down every caller, including scripts/score_heldout.py.
            grouped = groups.get(page_id)
            if grouped is None:
                excluded.append(page_id)
                continue
            doc_id = grouped
        index[page_id] = {"image_path": str(path), "doc_id": doc_id}

    if excluded:
        logger.warning(
            "%d page(s) in %s are absent from the deed grouping and were excluded, e.g. %s. "
            "If one of these is a page you meant to read, it needs a row in %s.",
            len(excluded),
            pages_dir,
            excluded[:5],
            params["deed_groups"],
        )
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
        # Per-region generation bookkeeping, read back by scripts/score_heldout.py. A short
        # chunk looks the same downstream whether the model stopped or we cut it off at the
        # cap, and those call for opposite fixes -- a bigger model vs a bigger budget.
        self.generation_stats: list[dict[str, object]] = []

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
        max_new_tokens = int(self.cfg["max_new_tokens"])
        generated = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = generated[0][inputs["input_ids"].shape[1] :]
        # Greedy decoding stops on EOS, so reaching the cap means the page was cut off
        # mid-transcription rather than finished.
        n_generated = int(trimmed.shape[0])
        self.generation_stats.append(
            {
                "page_id": region.page_id,
                "generated_tokens": n_generated,
                "truncated": n_generated >= max_new_tokens,
            }
        )
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
