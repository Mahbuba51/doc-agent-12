"""Unit tests for ingest (Stage 1 preprocess). CPU-only, synthetic images -- no dependency
on the real corpus, so CI never needs data/raw."""

from __future__ import annotations

import json
import math

import numpy as np
from PIL import Image, ImageFilter

from doc_agent.contracts import Page
from doc_agent.ingest import preprocess


def _synthetic_page(h: int = 200, w: int = 300, n_lines: int = 6) -> np.ndarray:
    """A white page with a few horizontal black "text lines" -- enough structure for the
    projection-profile skew search to have a real preference."""
    img = np.full((h, w), 255, dtype=np.uint8)
    line_h = 4
    gap = h // (n_lines + 1)
    for i in range(1, n_lines + 1):
        y = i * gap
        img[y : y + line_h, 20 : w - 20] = 0
    return img


def _rotate_pil(arr: np.ndarray, angle_deg: float) -> np.ndarray:
    """Ground-truth rotation via PIL, independent of preprocess.rotate, for the skew test."""
    im = Image.fromarray(arr)
    rotated = im.rotate(angle_deg, resample=Image.BICUBIC, fillcolor=255, expand=False)
    return np.array(rotated)


# ---- skew estimation / rotation ----------------------------------------------------


def test_rotate_is_identity_at_zero_degrees():
    grey = _synthetic_page()
    out = preprocess.rotate(grey, 0.0)
    assert np.array_equal(out, grey)


def test_rotate_preserves_shape_and_fills_white():
    grey = _synthetic_page()
    out = preprocess.rotate(grey, 3.0)
    assert out.shape == grey.shape
    # corners are exposed by any nonzero rotation and must be filled white, not black
    assert out[0, 0] > 200


def test_estimate_skew_recovers_a_known_angle():
    grey = _synthetic_page()
    skewed = _rotate_pil(grey, 3.0)
    detected = preprocess.estimate_skew_deg(skewed, search_range=4.0, step=0.5)
    # correcting the skew should rotate back roughly -3 degrees
    assert math.isclose(detected, -3.0, abs_tol=1.0)


def test_estimate_skew_is_near_zero_for_unskewed_page():
    grey = _synthetic_page()
    detected = preprocess.estimate_skew_deg(grey, search_range=4.0, step=0.5)
    assert abs(detected) <= 0.5


# ---- denoise / binarize --------------------------------------------------------------


def test_denoise_preserves_shape_and_dtype():
    grey = _synthetic_page()
    out = preprocess.denoise(grey)
    assert out.shape == grey.shape
    assert out.dtype == grey.dtype


def test_sauvola_binarize_output_is_strictly_binary():
    grey = _synthetic_page()
    binary = preprocess.sauvola_binarize(grey)
    assert set(np.unique(binary)).issubset({0, 255})
    assert binary.shape == grey.shape


def test_sauvola_binarize_keeps_dark_lines_dark():
    h, w = 200, 300
    grey = _synthetic_page(h, w)
    binary = preprocess.sauvola_binarize(grey)
    # the synthetic text lines were pure black (0), first one at row h // 7; Sauvola
    # should mark that ink as 0
    first_line_y = h // 7
    assert (binary[first_line_y : first_line_y + 2, 50:60] == 0).all()


# ---- quality proxy ----------------------------------------------------------------


def test_quality_score_ranks_sharp_above_blurred():
    sharp = _synthetic_page()
    blurred = np.array(Image.fromarray(sharp).filter(ImageFilter.GaussianBlur(4)))
    assert preprocess.quality_score(sharp) > preprocess.quality_score(blurred)


def test_quality_score_of_blank_page_is_near_zero():
    blank = np.full((200, 300), 255, dtype=np.uint8)
    assert preprocess.quality_score(blank) < 0.05


# ---- run() end-to-end ---------------------------------------------------------------


def _write_page_image(tmp_path, page_id: str, arr: np.ndarray):
    path = tmp_path / f"{page_id}.png"
    Image.fromarray(arr).save(path)
    return Page(id=page_id, image_path=str(path), doc_id=f"deed_{page_id}")


def _cfg(tmp_path):
    out_dir = tmp_path / "preprocessed"
    return {
        "preprocess": {"out_dir": str(out_dir)},
        "ocr": {"sidecar": str(out_dir / "preprocess.jsonl")},
    }


def test_run_writes_sidecar_with_expected_schema(tmp_path):
    page = _write_page_image(tmp_path, "dolil_1", _synthetic_page())
    cfg = _cfg(tmp_path)

    preprocess.run([page], cfg)

    sidecar = json.loads(open(cfg["ocr"]["sidecar"], encoding="utf-8").readline())
    for key in (
        "page_id",
        "doc_id",
        "original_path",
        "greyscale_path",
        "binary_path",
        "skew_deg",
        "quality_score",
        "low_legibility",
    ):
        assert key in sidecar
    assert sidecar["page_id"] == "dolil_1"
    assert sidecar["original_path"] == page.image_path


def test_run_returns_pages_pointing_at_greyscale_output(tmp_path):
    page = _write_page_image(tmp_path, "dolil_2", _synthetic_page())
    cfg = _cfg(tmp_path)

    out_pages = preprocess.run([page], cfg)

    assert len(out_pages) == 1
    out = out_pages[0]
    assert out.id == page.id
    assert out.doc_id == page.doc_id
    assert out.image_path != page.image_path  # points at the new greyscale, not the raw file
    assert out.image_path.endswith("_grey.png")
    assert Image.open(out.image_path) is not None  # file actually exists and opens


def test_run_never_modifies_the_original_file(tmp_path):
    page = _write_page_image(tmp_path, "dolil_3", _synthetic_page())
    original_bytes = open(page.image_path, "rb").read()
    cfg = _cfg(tmp_path)

    preprocess.run([page], cfg)

    assert open(page.image_path, "rb").read() == original_bytes


def test_run_flags_low_legibility_pages(tmp_path):
    sharp = _write_page_image(tmp_path, "dolil_sharp", _synthetic_page())
    blurred_arr = np.array(Image.fromarray(_synthetic_page()).filter(ImageFilter.GaussianBlur(6)))
    blurred = _write_page_image(tmp_path, "dolil_blurred", blurred_arr)
    cfg = _cfg(tmp_path)

    preprocess.run([sharp, blurred], cfg)

    records = {}
    for line in open(cfg["ocr"]["sidecar"], encoding="utf-8"):
        rec = json.loads(line)
        records[rec["page_id"]] = rec

    assert records["dolil_blurred"]["low_legibility"] is True
    assert records["dolil_sharp"]["low_legibility"] is False
