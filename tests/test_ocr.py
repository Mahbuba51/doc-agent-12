"""Stage 3 OCR/HTR — reader plumbing and region->Chunk assembly. CI runs these.

The VLM call itself is not exercised here: a fake reader is injected so these tests stay
CPU-only and deterministic. What is tested is everything around it -- page lookup, reading
order, normalization, and the doc_id sourcing that keeps the document-level split honest.
"""

from __future__ import annotations

import json
import unicodedata

import numpy as np
import pytest
from PIL import Image

from doc_agent.contracts import Region
from doc_agent.vision import ocr


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
        records.append(
            {"page_id": f"dolil_{n}", "doc_id": doc, "greyscale_path": str(path)}
        )
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


def test_page_as_doc_fallback_only_applies_when_explicitly_allowed(corpus):
    (corpus / "preprocess.jsonl").unlink()
    regions = _regions("dolil_1", (0, 0, 100, 50))

    chunk = ocr.transcribe(
        regions, _cfg(corpus, allow_page_as_doc=True), reader=FakeReader()
    )[0]

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
