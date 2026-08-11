"""Stage 2 layout — projection-heuristic region detection. CI runs these."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from PIL import Image

from doc_agent.contracts import Page
from doc_agent.vision import layout

CFG_LAYOUT = {
    "model": "projection-heuristic",
    "score_thr": 0.5,
    "min_ink_frac": 0.01,
    "line_gap_px": 18,
    "min_height_px": 10,
    "min_width_px": 30,
    "pad_px": 4,
}


def _blank(height=600, width=400):
    return np.full((height, width), 235, dtype=np.uint8)


def _add_block(img, top, n_lines=3, left=40, right=360, line_h=8, leading=14):
    """Draw a paragraph: n_lines dark bars separated by `leading` blank rows."""
    y = top
    for _ in range(n_lines):
        img[y : y + line_h, left:right] = 60
        y += line_h + leading
    return img


def _page(tmp_path, img, page_id="dolil_1"):
    path = tmp_path / f"{page_id}.jpg"
    Image.fromarray(img).save(path, format="JPEG", quality=95)
    return Page(id=page_id, image_path=str(path), doc_id="deed_001")


@pytest.fixture()
def three_block_page(tmp_path):
    img = _blank()
    for top in (40, 240, 440):
        _add_block(img, top)
    return _page(tmp_path, img)


def _cfg():
    return {"layout": dict(CFG_LAYOUT)}


def test_paragraphs_separated_by_wide_gaps_become_separate_regions(three_block_page):
    regions = layout.detect([three_block_page], _cfg())

    assert len(regions) == 3


def test_lines_within_a_paragraph_stay_in_one_region(tmp_path):
    page = _page(tmp_path, _add_block(_blank(), 40, n_lines=5))

    assert len(layout.detect([page], _cfg())) == 1


def test_regions_are_ordered_top_to_bottom(three_block_page):
    tops = [r.bbox[1] for r in layout.detect([three_block_page], _cfg())]

    assert tops == sorted(tops)


def test_region_bbox_covers_the_ink_and_excludes_the_margins(tmp_path):
    img = _blank()
    _add_block(img, 100, n_lines=2, left=60, right=300)
    page = _page(tmp_path, img)

    x0, y0, x1, y1 = layout.detect([page], _cfg())[0].bbox

    assert 50 <= x0 <= 60 and 300 <= x1 <= 310
    assert 90 <= y0 <= 100 and y1 >= 118


def test_regions_carry_their_page_id_and_are_typed_text(three_block_page):
    regions = layout.detect([three_block_page], _cfg())

    assert {r.page_id for r in regions} == {"dolil_1"}
    assert {r.kind for r in regions} == {"text"}


def test_a_blank_page_yields_no_regions(tmp_path):
    assert layout.detect([_page(tmp_path, _blank())], _cfg()) == []


def test_speckle_noise_below_the_size_floor_is_not_a_region(tmp_path):
    img = _blank()
    img[300:303, 200:204] = 40  # a dust speck, not a text line

    assert layout.detect([_page(tmp_path, img)], _cfg()) == []


def test_regions_from_several_pages_are_all_returned(tmp_path, three_block_page):
    other = _page(tmp_path, _add_block(_blank(), 60), page_id="dolil_2")

    regions = layout.detect([three_block_page, other], _cfg())

    assert {r.page_id for r in regions} == {"dolil_1", "dolil_2"}
    assert len(regions) == 4


def test_bbox_stays_inside_the_page_bounds(tmp_path):
    img = _blank(height=200, width=200)
    img[0:12, 0:200] = 50  # ink flush against the top and side edges

    x0, y0, x1, y1 = layout.detect([_page(tmp_path, img)], _cfg())[0].bbox

    assert 0 <= x0 < x1 <= 200
    assert 0 <= y0 < y1 <= 200


def test_detection_survives_a_dark_uneven_scan_ground(tmp_path):
    """Otsu, not a fixed threshold: aged scans are grey, not white."""
    img = np.full((600, 400), 150, dtype=np.uint8)
    gradient = np.linspace(0, 60, 400, dtype=np.uint8)
    img = np.clip(img.astype(np.int16) - gradient[None, :], 0, 255).astype(np.uint8)
    _add_block(img, 100, n_lines=3)
    img = cv2.GaussianBlur(img, (3, 3), 0)

    assert len(layout.detect([_page(tmp_path, img)], _cfg())) == 1


def _exif_rotated_page(tmp_path, page_id="dolil_212"):
    """A page stored sideways with EXIF orientation=6, i.e. 'rotate 90 CW to display'.

    Two of the eight held-out scans are stored exactly like this. Viewers honour the tag,
    so the page looks upright to a human; cv2.imread ignores it, so the pipeline would see
    vertical text lines and find no blank rows at all.
    """
    upright = _blank(height=600, width=400)
    for top in (40, 240, 440):
        _add_block(upright, top)
    sideways = np.rot90(upright)  # what is actually stored in the file
    image = Image.fromarray(sideways)
    exif = image.getexif()
    exif[274] = 6
    path = tmp_path / f"{page_id}.jpg"
    image.save(path, format="JPEG", quality=95, exif=exif)
    return Page(id=page_id, image_path=str(path), doc_id="deed_001")


def test_exif_orientation_is_honoured_when_reading_a_page(tmp_path):
    page = _exif_rotated_page(tmp_path)

    assert len(layout.detect([page], _cfg())) == 3


def test_exif_rotated_bboxes_are_in_display_coordinates(tmp_path):
    page = _exif_rotated_page(tmp_path)

    for region in layout.detect([page], _cfg()):
        assert region.bbox[2] <= 400  # page is 400 wide once displayed upright
        assert region.bbox[3] <= 600
