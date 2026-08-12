"""Stage 2 — layout detection / segmentation

Training-free projection heuristic (Step 0 decision D1). Most pages are dense handwritten
deed body text, where ink projected onto the vertical axis is enough: handwritten lines
produce clear ink bands, and the blank runs between paragraphs are wider than the leading
within one. The heuristic costs no checkpoint, no GPU pass per page and no extra pinned
dependency.

D1 RATIONALE CORRECTED 2026-08-12. This docstring previously claimed the corpus has "no
tables and no multi-column pages". That is FALSE and the decision must not be defended on
it. Measured over all 653 raw pages (crop to the paper first, then threshold ink -- an
Otsu pass over a whole phone photo separates paper from the surface it was shot on, not
ink from paper, and gives nonsense):

    widest interior ink-free gutter / inked span
    p50 0.000   p90 0.026   p95 0.048   p99 0.109
    > 0.06 of span:  23 pages (3.5%)     > 0.15 of span: 4 pages (0.6%)

and dolil_74 is a তফসিল (schedule) laid out as a FOUR-COLUMN TABLE. Treat 3.5% as a LOWER
BOUND on tabular pages, not a rate: a wide-gutter test finds two-column splits but misses
dense tables whose inter-column gaps are narrow. Horizontal projection will merge such
columns into single bands and scramble reading order on those pages.

The decision still stands, on a different and narrower argument: reading is a single VLM
call over the whole page (D2), so the VLM sees the table itself and layout governs only
cropping. Revisit if ocr_f1 turns out to be materially worse on tabular pages -- that is
the measurement that should decide it, not this docstring.

ALSO UNHANDLED: some pages are rotated 90 degrees in-content (e.g. dolil_429), which EXIF
correction does not fix and which defeats horizontal projection outright. Prevalence not
yet measured.

Every region is emitted as kind="text". Heading/table/figure are left unused rather than
guessed at -- an unconsumed class nobody evaluates is untested surface, and the reader
handles structure inside a region anyway.

Self-contained by design: this does its own Otsu binarization from whatever image the Page
points at, so it works on a raw scan or on the ingest stage's preprocessed greyscale.

KNOWN LIMITATION, measured on the 8 held-out scans (correctness check, nothing tuned):
this corpus largely does not contain the blank horizontal bands the heuristic looks for.
The 4 flatbed-style pages split into 3-9 regions; the 4 camera-photo pages yield a single
page-sized region, and no amount of deskewing or adaptive thresholding produced blank rows
on them (both were tried and measured -- the writing genuinely runs continuously down the
sheet).

DECIDED: region = page. A page-sized region is the expected output here, not a failure to
segment -- Qwen2.5-VL (Step 0 decision D2) reads a whole page in one call, so finer
segmentation would buy nothing and would risk splitting a plot number or deed amount
across two crops, which the precision-first NFR cannot afford. The banding below is kept
because it still yields tighter crops on flatbed-style pages, but nothing downstream may
assume more than one region per page.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..contracts import *  # noqa
from ..logging_conf import get_logger

logger = get_logger(__name__)


def _read_grey(image_path: str) -> np.ndarray:
    grey = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if grey is None:
        raise ValueError(f"unreadable page image: {image_path}")
    return grey


def _ink_mask(grey: np.ndarray) -> np.ndarray:
    """Binary ink mask (True = ink), via Otsu so an aged grey ground still separates."""
    # Otsu picks the threshold per page: scan ground ranges from near-white to heavily
    # aged grey across this corpus, and a fixed cutoff would erase the darker scans.
    _, binary = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary > 0


def _bands(ink_rows: np.ndarray, gap: int) -> list[tuple[int, int]]:
    """Row indices where ink is present -> [start, end) bands, merging gaps below `gap`."""
    if not ink_rows.any():
        return []

    rows = np.flatnonzero(ink_rows)
    bands: list[list[int]] = [[int(rows[0]), int(rows[0]) + 1]]
    for row in rows[1:]:
        row = int(row)
        # A blank run shorter than line_gap_px is the leading between lines of one
        # paragraph; anything wider is a real block boundary.
        if row - bands[-1][1] <= gap:
            bands[-1][1] = row + 1
        else:
            bands.append([row, row + 1])
    return [(start, end) for start, end in bands]


def detect(pages: list[Page], cfg: dict) -> list[Region]:
    """Detect text regions on each page. Returns them in page order, top to bottom."""
    params = cfg["layout"]
    min_ink_frac = float(params["min_ink_frac"])
    gap = int(params["line_gap_px"])
    min_height = int(params["min_height_px"])
    min_width = int(params["min_width_px"])
    pad = int(params["pad_px"])

    regions: list[Region] = []
    for page in pages:
        mask = _ink_mask(_read_grey(page.image_path))
        height, width = mask.shape

        # A row counts as text only if enough of it is ink -- this is what keeps dust
        # specks and scan-edge artefacts from opening a band.
        row_ink = mask.sum(axis=1) >= max(1, int(min_ink_frac * width))

        for top, bottom in _bands(row_ink, gap):
            columns = np.flatnonzero(mask[top:bottom].any(axis=0))
            if columns.size == 0:
                continue
            left, right = int(columns[0]), int(columns[-1]) + 1

            if (bottom - top) < min_height or (right - left) < min_width:
                continue

            regions.append(
                Region(
                    page_id=page.id,
                    bbox=(
                        max(0, left - pad),
                        max(0, top - pad),
                        min(width, right + pad),
                        min(height, bottom + pad),
                    ),
                    kind="text",
                )
            )

    logger.info("layout: %d region(s) across %d page(s)", len(regions), len(pages))
    return regions
