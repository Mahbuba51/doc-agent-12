"""Stage 3 OCR/HTR — reader plumbing and region->Chunk assembly. CI runs these.

The VLM call itself is not exercised here: a fake reader is injected so these tests stay
CPU-only and deterministic. What is tested is everything around it -- page lookup, reading
order, normalization, and the doc_id sourcing that keeps the document-level split honest.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from doc_agent.contracts import Region
from doc_agent.vision import ocr

# The shipped layout parameters (configs/config.yaml), so the harness tests exercise the
# same segmentation the baseline run will.
_LAYOUT = {
    "model": "projection-heuristic",
    "score_thr": 0.5,
    "min_ink_frac": 0.01,
    "line_gap_px": 18,
    "min_height_px": 10,
    "min_width_px": 30,
    "pad_px": 4,
}


def _write_page(path, size=(200, 300)):
    rng = np.random.default_rng(0)
    arr = rng.integers(200, 255, size=(size[1], size[0]), dtype=np.uint8)
    Image.fromarray(arr).save(path, format="JPEG", quality=95)
    return path


def _cfg(tmp_path, **overrides):
    params = {
        "model": "fake/model",
        "finetune": False,
        "sidecar": str(tmp_path / "preprocess.jsonl"),
        "pages_dir": str(tmp_path / "raw"),
        "deed_groups": str(tmp_path / "deed_groups.csv"),
        "allow_page_as_doc": False,
        "ascii_digits": True,
        "max_new_tokens": 768,
    }
    params.update(overrides)
    return {"ocr": params}


@pytest.fixture()
def corpus(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    records = []
    for n, doc in ((1, "deed_001"), (2, "deed_001"), (3, "deed_007")):
        path = _write_page(raw / f"dolil_{n}.jpg")
        records.append({"page_id": f"dolil_{n}", "doc_id": doc, "greyscale_path": str(path)})
    (tmp_path / "preprocess.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )
    return tmp_path


class FakeReader:
    """Returns one canned string per region, and records what it was asked to read."""

    def __init__(self, texts=None):
        self.texts = list(texts or [])
        self.seen: list[tuple[str, tuple[int, int, int, int]]] = []

    def transcribe_region(self, region):
        self.seen.append((region.page_id, region.bbox))
        return self.texts.pop(0) if self.texts else f"text-{len(self.seen)}"


def _regions(page_id, *boxes):
    return [Region(page_id=page_id, bbox=b, kind="text") for b in boxes]


def test_one_chunk_per_page_not_per_region(corpus):
    regions = _regions("dolil_1", (0, 0, 100, 50), (0, 60, 100, 110), (0, 120, 100, 170))

    chunks = ocr.transcribe(regions, _cfg(corpus), reader=FakeReader())

    assert len(chunks) == 1


def test_regions_are_concatenated_in_reading_order(corpus):
    regions = _regions("dolil_1", (0, 0, 100, 50), (0, 60, 100, 110))

    chunk = ocr.transcribe(regions, _cfg(corpus), reader=FakeReader(["first", "second"]))[0]

    assert chunk.text == "first\nsecond"


def test_each_page_becomes_its_own_chunk(corpus):
    regions = _regions("dolil_1", (0, 0, 100, 50)) + _regions("dolil_2", (0, 0, 100, 50))

    chunks = ocr.transcribe(regions, _cfg(corpus), reader=FakeReader())

    assert [c.page_ids for c in chunks] == [["dolil_1"], ["dolil_2"]]


def test_chunk_ids_are_unique(corpus):
    regions = _regions("dolil_1", (0, 0, 100, 50)) + _regions("dolil_2", (0, 0, 100, 50))

    chunks = ocr.transcribe(regions, _cfg(corpus), reader=FakeReader())

    assert len({c.id for c in chunks}) == 2


def test_doc_id_comes_from_the_ingest_sidecar(corpus):
    regions = _regions("dolil_1", (0, 0, 100, 50)) + _regions("dolil_3", (0, 0, 100, 50))

    chunks = ocr.transcribe(regions, _cfg(corpus), reader=FakeReader())

    assert [c.doc_id for c in chunks] == ["deed_001", "deed_007"]


def test_an_unreadable_region_does_not_leave_a_blank_line(corpus):
    regions = _regions("dolil_1", (0, 0, 100, 50), (0, 60, 100, 110), (0, 120, 100, 170))

    chunk = ocr.transcribe(regions, _cfg(corpus), reader=FakeReader(["a", "   ", "b"]))[0]

    assert chunk.text == "a\nb"


def test_a_page_whose_regions_are_all_empty_produces_no_chunk(corpus):
    regions = _regions("dolil_1", (0, 0, 100, 50))

    assert ocr.transcribe(regions, _cfg(corpus), reader=FakeReader([""])) == []


def test_text_is_nfc_normalized(corpus):
    decomposed = unicodedata.normalize("NFD", "কাঁচা")
    regions = _regions("dolil_1", (0, 0, 100, 50))

    chunk = ocr.transcribe(regions, _cfg(corpus), reader=FakeReader([decomposed]))[0]

    assert chunk.text == unicodedata.normalize("NFC", decomposed)


def test_bangla_digits_map_to_ascii_when_enabled(corpus):
    regions = _regions("dolil_1", (0, 0, 100, 50))

    chunk = ocr.transcribe(regions, _cfg(corpus), reader=FakeReader(["দাগ ২১৬৩"]))[0]

    assert "2163" in chunk.text


def test_bangla_digits_are_left_alone_when_disabled(corpus):
    regions = _regions("dolil_1", (0, 0, 100, 50))

    chunk = ocr.transcribe(
        regions, _cfg(corpus, ascii_digits=False), reader=FakeReader(["দাগ ২১৬৩"])
    )[0]

    assert "২১৬৩" in chunk.text


def test_a_region_for_an_unknown_page_is_an_error(corpus):
    regions = _regions("dolil_999", (0, 0, 100, 50))

    with pytest.raises(ValueError, match="dolil_999"):
        ocr.transcribe(regions, _cfg(corpus), reader=FakeReader())


def test_without_a_sidecar_or_deed_groups_it_refuses_to_invent_doc_ids(corpus):
    (corpus / "preprocess.jsonl").unlink()
    regions = _regions("dolil_1", (0, 0, 100, 50))

    with pytest.raises(FileNotFoundError, match="deed_groups"):
        ocr.transcribe(regions, _cfg(corpus), reader=FakeReader())


def _write_groups(tmp_path, mapping):
    path = tmp_path / "deed_groups.csv"
    rows = ["page_id,doc_id,confidence,notes,duplicate_of"]
    rows += [f"{page_id},{doc_id},manual,," for page_id, doc_id in mapping.items()]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_a_page_missing_from_the_deed_grouping_is_skipped(corpus):
    """The grouping DEFINES the corpus, so a page with no row is excluded, not an error.

    105 duplicate re-scans and 2 non-content pages have no row (data/provenance.md). Skipping
    them is not the same as inventing a doc_id -- the guard this module defends is against
    fabricating one, and an excluded page has a correct answer: leave it out. Failing hard
    instead means one dedup pass takes down the whole reader.
    """
    (corpus / "preprocess.jsonl").unlink()
    _write_groups(corpus, {"dolil_1": "deed_001", "dolil_3": "deed_007"})  # dolil_2 excluded

    index = ocr.load_page_index(_cfg(corpus))

    assert sorted(index) == ["dolil_1", "dolil_3"]


def test_pages_that_are_in_the_grouping_keep_their_doc_id(corpus):
    (corpus / "preprocess.jsonl").unlink()
    _write_groups(corpus, {"dolil_1": "deed_001", "dolil_3": "deed_007"})

    index = ocr.load_page_index(_cfg(corpus))

    assert [index[p]["doc_id"] for p in ("dolil_1", "dolil_3")] == ["deed_001", "deed_007"]


def test_page_as_doc_fallback_only_applies_when_explicitly_allowed(corpus):
    (corpus / "preprocess.jsonl").unlink()
    regions = _regions("dolil_1", (0, 0, 100, 50))

    chunk = ocr.transcribe(regions, _cfg(corpus, allow_page_as_doc=True), reader=FakeReader())[0]

    assert chunk.doc_id == "dolil_1"


def test_reader_receives_each_region_once(corpus):
    reader = FakeReader()
    regions = _regions("dolil_1", (0, 0, 100, 50), (0, 60, 100, 110))

    ocr.transcribe(regions, _cfg(corpus), reader=reader)

    assert [box for _, box in reader.seen] == [(0, 0, 100, 50), (0, 60, 100, 110)]


def test_reader_crops_the_region_from_the_page_image(corpus):
    reader = ocr.Reader(_cfg(corpus))
    crop = reader.crop(Region(page_id="dolil_1", bbox=(10, 20, 110, 140), kind="text"))

    assert crop.size == (100, 120)


def _exif_rotated_page(tmp_path, page_id="dolil_212"):
    """Stored sideways with EXIF orientation=6, as two of the held-out scans are.

    cv2.imread (layout) honours the tag, so region bboxes are in DISPLAY coordinates.
    PIL.Image.open does not, so a naive crop reads the raw raster and returns the wrong
    pixels -- the reader would be handed the wrong part of the page, silently.
    """
    display = np.full((600, 400), 240, dtype=np.uint8)
    display[0:120, 0:100] = 20  # a dark marker at the display-space top-left corner
    stored = np.rot90(display)
    image = Image.fromarray(stored)
    exif = image.getexif()
    exif[274] = 6
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    path = raw / f"{page_id}.jpg"
    image.save(path, format="JPEG", quality=95, exif=exif)
    (tmp_path / "preprocess.jsonl").write_text(
        json.dumps({"page_id": page_id, "doc_id": "deed_009", "greyscale_path": str(path)}) + "\n",
        encoding="utf-8",
    )
    return path


def test_crop_honours_exif_orientation(tmp_path):
    _exif_rotated_page(tmp_path)
    reader = ocr.Reader(_cfg(tmp_path))

    crop = reader.crop(Region(page_id="dolil_212", bbox=(0, 0, 100, 120), kind="text"))

    assert np.asarray(crop.convert("L")).mean() < 80  # the dark marker, not blank page


def test_crop_size_matches_the_bbox_on_an_exif_rotated_page(tmp_path):
    _exif_rotated_page(tmp_path)
    reader = ocr.Reader(_cfg(tmp_path))

    crop = reader.crop(Region(page_id="dolil_212", bbox=(0, 0, 100, 120), kind="text"))

    assert crop.size == (100, 120)


# --- generation stats -------------------------------------------------------------------
# The pilot read (977803f) confirmed two defects on dense pages: output truncated at the
# token cap, and repetition loops. Both look identical downstream -- a short, wrong chunk --
# so the reader records how many tokens each call actually generated and whether it hit the
# cap. Without it, a low ocr_f1 cannot distinguish "the model cannot read this" from "we cut
# it off mid-page", and the 3B-vs-7B decision would be made on the wrong evidence.
#
# The VLM is faked at the processor/model seam: the real one is a 3B checkpoint that needs a
# GPU, and what is under test here is the bookkeeping around generate(), not generate().


class _FakeBatch(dict):
    """What a processor returns: a mapping whose .to(device) yields itself."""

    def to(self, device):
        return self


class _FakeProcessor:
    """The three processor calls transcribe_region makes, and nothing else."""

    def __init__(self, prompt_tokens=5, decoded="পড়া"):
        self.prompt_tokens = prompt_tokens
        self.decoded = decoded

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "<prompt>"

    def __call__(self, text=None, images=None, return_tensors=None):
        import torch

        self.images = images
        return _FakeBatch(input_ids=torch.zeros((1, self.prompt_tokens), dtype=torch.long))

    def decode(self, ids, skip_special_tokens=True):
        return self.decoded


class _FakeModel:
    """Emits exactly `generated` new tokens after the prompt."""

    device = "cpu"

    def __init__(self, generated):
        self.generated = generated

    def generate(self, **kwargs):
        import torch

        prompt_len = kwargs["input_ids"].shape[1]
        return torch.zeros((1, prompt_len + self.generated), dtype=torch.long)


def _fake_reader(corpus, generated, **overrides):
    reader = ocr.Reader(_cfg(corpus, **overrides))
    reader._processor = _FakeProcessor()
    reader._model = _FakeModel(generated)  # non-None, so _load() stays a no-op
    return reader


def test_transcribe_region_records_a_stat_per_region(corpus):
    reader = _fake_reader(corpus, generated=40)

    reader.transcribe_region(Region(page_id="dolil_1", bbox=(0, 0, 100, 50), kind="text"))
    reader.transcribe_region(Region(page_id="dolil_2", bbox=(0, 0, 100, 50), kind="text"))

    assert [s["page_id"] for s in reader.generation_stats] == ["dolil_1", "dolil_2"]
    assert [s["generated_tokens"] for s in reader.generation_stats] == [40, 40]


def test_a_read_that_hits_the_token_cap_is_flagged_truncated(corpus):
    reader = _fake_reader(corpus, generated=64, max_new_tokens=64)

    reader.transcribe_region(Region(page_id="dolil_1", bbox=(0, 0, 100, 50), kind="text"))

    assert reader.generation_stats[-1]["truncated"] is True


def test_a_read_that_stops_before_the_cap_is_not_flagged(corpus):
    reader = _fake_reader(corpus, generated=63, max_new_tokens=64)

    reader.transcribe_region(Region(page_id="dolil_1", bbox=(0, 0, 100, 50), kind="text"))

    assert reader.generation_stats[-1]["truncated"] is False


def test_stats_survive_a_full_transcribe_run(corpus):
    """transcribe() owns the region loop, so the stats have to be readable after it.

    This is how the scoring harness attributes truncation to a page.
    """
    reader = _fake_reader(corpus, generated=64, max_new_tokens=64)
    regions = _regions("dolil_1", (0, 0, 100, 50)) + _regions("dolil_2", (0, 0, 100, 50))

    ocr.transcribe(regions, _cfg(corpus), reader=reader)

    assert [s["page_id"] for s in reader.generation_stats] == ["dolil_1", "dolil_2"]
    assert all(s["truncated"] for s in reader.generation_stats)


# --- the held-out scoring harness -------------------------------------------------------
# scripts/score_heldout.py is the baseline measurement itself, so its assembly is tested
# here with a fake reader. Loaded by path because scripts/ is not an importable package.


def _harness():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "score_heldout.py"
    spec = importlib.util.spec_from_file_location("score_heldout", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _labels(tmp_path, rows):
    path = tmp_path / "labels.jsonl"
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    return path


def test_only_human_completed_labels_are_scored(tmp_path):
    """A TODO row has no transcription -- scoring against it would score against "".

    Silently treating an unfilled label as empty gold would report a confident 0.0 for a
    page nobody has read yet, which is worse than reporting nothing.
    """
    harness = _harness()
    path = _labels(
        tmp_path,
        [
            {"page_id": "dolil_13", "status": "done", "text": "ক খ"},
            {"page_id": "dolil_11", "status": "TODO", "text": ""},
        ],
    )

    assert harness.load_gold(path) == {"dolil_13": "ক খ"}


def test_a_perfect_read_scores_one_per_page(corpus):
    harness = _harness()
    cfg = _cfg(corpus, allow_page_as_doc=True)
    cfg["layout"] = _LAYOUT

    rows = harness.score_pages({"dolil_1": "ক খ গ"}, cfg, reader=FakeReader(["ক খ গ"]))

    assert [r["page_id"] for r in rows] == ["dolil_1"]
    assert rows[0]["f1"] == 1.0


def test_a_missed_page_scores_zero_rather_than_disappearing(corpus):
    """An unreadable page produces no chunk (by design). It still has to appear at 0.0."""
    harness = _harness()
    cfg = _cfg(corpus, allow_page_as_doc=True)
    cfg["layout"] = _LAYOUT

    rows = harness.score_pages({"dolil_1": "ক খ গ"}, cfg, reader=FakeReader([""]))

    assert [r["page_id"] for r in rows] == ["dolil_1"]
    assert rows[0]["f1"] == 0.0


def test_truncation_is_attributed_to_the_page_it_happened_on(corpus):
    harness = _harness()
    cfg = _cfg(corpus, allow_page_as_doc=True, max_new_tokens=64)
    cfg["layout"] = _LAYOUT
    reader = _fake_reader(corpus, generated=64, max_new_tokens=64, allow_page_as_doc=True)

    rows = harness.score_pages({"dolil_1": "ক খ"}, cfg, reader=reader)

    assert rows[0]["truncated"] is True
    assert rows[0]["generated_tokens"] == 64


def test_summary_reports_mean_f1_and_the_truncation_count():
    harness = _harness()
    rows = [
        {"page_id": "a", "f1": 1.0, "truncated": True},
        {"page_id": "b", "f1": 0.5, "truncated": False},
    ]

    summary = harness.summarise(rows)

    assert summary["pages"] == 2
    assert summary["mean_f1"] == pytest.approx(0.75)
    assert summary["truncated_pages"] == 1
