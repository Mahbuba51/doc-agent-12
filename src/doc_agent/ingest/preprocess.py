"""Stage 1 — deskew / denoise / binarize / augment

Classical pipeline only (design_choices.md Row 1): no VAE/diffusion here -- a generative
model could reconstruct, reshape, or remove a handwritten stroke and change a legally
significant plot number, name, date, or amount. Deterministic, rule-based, no training
loss, conservative parameters chosen to preserve thin strokes over aggressive cleanup.

Per page: bake in EXIF orientation (so cv2.imread in Stage 2 and PIL+exif_transpose in
Stage 3 read the SAME pixel grid off the greyscale this stage writes, regardless of which
file variant they end up reading) -> estimate skew by projection-profile search in
[-4,+4] degrees -> rotate (white fill) -> light median + non-local-means denoise for
show-through -> Sauvola adaptive threshold for a binary copy -> a normalized high-pass
blur-variance quality proxy, flagging low-legibility pages.

The untouched original is never modified or copied -- only referenced by path -- so the
raw evidence stays recoverable if a preprocessing setting turns out to hurt a downstream
field read. Outputs (greyscale + binary) and per-page stats go to a JSONL sidecar that
Stage 3's `load_page_index()` already knows how to read (it prefers `greyscale_path`,
falling back to `original_path`); the Page objects this function returns point at the
same greyscale files, so Stage 2 layout and the sidecar can never disagree about which
image a page means.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from ..contracts import *  # noqa
from ..logging_conf import get_logger

logger = get_logger(__name__)


def _load_upright_grey(image_path: str) -> np.ndarray:
    """Read a page as greyscale with EXIF orientation baked into the pixels."""
    with Image.open(image_path) as im:
        upright = ImageOps.exif_transpose(im.convert("L"))
        return np.array(upright)


def estimate_skew_deg(grey: np.ndarray, search_range: float = 4.0, step: float = 0.5) -> float:
    """Projection-profile skew search: the rotation angle whose row-ink-sum profile has
    the highest variance is the one where text lines are most horizontal -- a skewed line
    smears ink across more rows, flattening the profile."""
    _, ink = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = ink.shape
    center = (w / 2.0, h / 2.0)
    best_angle, best_score = 0.0, -1.0
    angle = -search_range
    while angle <= search_range + 1e-9:
        if abs(angle) < 1e-9:
            rotated = ink
        else:
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(ink, matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        score = float(rotated.sum(axis=1).astype(np.float64).var())
        if score > best_score:
            best_score, best_angle = score, angle
        angle += step
    return best_angle


def rotate(grey: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate by angle_deg, filling the exposed border white (255) -- a scan's ground is
    light, so a white fill does not manufacture a false dark margin for the ink mask."""
    if abs(angle_deg) < 1e-6:
        return grey
    h, w = grey.shape
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(grey, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)


def denoise(grey: np.ndarray) -> np.ndarray:
    """Light 3x3 median (scanner speckle) + non-local-means (reverse-side show-through).
    Conservative by design: aggressive denoising risks thinning or erasing faint strokes
    the precision-first NFR needs intact."""
    median = cv2.medianBlur(grey, 3)
    return cv2.fastNlMeansDenoising(median, h=7, templateWindowSize=7, searchWindowSize=21)


def sauvola_binarize(
    grey: np.ndarray, window: int = 15, k: float = 0.2, r: float = 128.0
) -> np.ndarray:
    """Sauvola adaptive threshold: local (mean, std) over `window`, threshold =
    mean * (1 + k * (std/r - 1)). Local, not global (Otsu), because uneven ballpoint ink
    under uneven illumination needs a threshold that adapts across the page."""
    grey_f = grey.astype(np.float64)
    mean = cv2.boxFilter(grey_f, ddepth=-1, ksize=(window, window))
    sqmean = cv2.boxFilter(grey_f * grey_f, ddepth=-1, ksize=(window, window))
    std = np.sqrt(np.maximum(sqmean - mean * mean, 0.0))
    threshold = mean * (1.0 + k * (std / r - 1.0))
    return np.where(grey_f > threshold, 255, 0).astype(np.uint8)


def quality_score(grey: np.ndarray, radius: int = 3, norm: float = 40.0) -> float:
    """Normalized high-pass blur-variance proxy for scan legibility: subtract a Gaussian
    blur from the image and take the variance of what's left. Crisp ink strokes leave a
    lot of high-frequency energy behind; faint/blurred ink does not."""
    blurred = cv2.GaussianBlur(grey, (0, 0), radius)
    high_pass = grey.astype(np.float32) - blurred.astype(np.float32)
    return float(high_pass.var() / norm)


def run(pages: list[Page], cfg: dict) -> list[Page]:
    """Deskew, denoise, and binarize every page; write a greyscale + binary variant and a
    JSONL sidecar per page. Returns Pages whose image_path points at the deskewed
    greyscale (Stage 2/3's preferred variant)."""
    params = cfg.get("preprocess", {})
    skew_range = float(params.get("skew_range_deg", 4.0))
    skew_step = float(params.get("skew_step_deg", 0.5))
    sauvola_window = int(params.get("sauvola_window", 15))
    sauvola_k = float(params.get("sauvola_k", 0.2))
    sauvola_r = float(params.get("sauvola_r", 128.0))
    quality_radius = int(params.get("quality_blur_radius", 3))
    quality_norm = float(params.get("quality_norm", 40.0))
    quality_threshold = float(params.get("quality_threshold", 0.35))

    out_dir = Path(params.get("out_dir", "data/interim/preprocessed"))
    out_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = Path(cfg.get("ocr", {}).get("sidecar", str(out_dir / "preprocess.jsonl")))
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)

    out_pages: list[Page] = []
    records: list[dict] = []
    n_low = 0
    for page in pages:
        grey = _load_upright_grey(page.image_path)

        skew_deg = estimate_skew_deg(grey, skew_range, skew_step)
        deskewed = rotate(grey, skew_deg)
        cleaned = denoise(deskewed)
        binary = sauvola_binarize(cleaned, sauvola_window, sauvola_k, sauvola_r)
        quality = quality_score(cleaned, quality_radius, quality_norm)
        low_legibility = quality < quality_threshold
        n_low += int(low_legibility)

        grey_path = out_dir / f"{page.id}_grey.png"
        binary_path = out_dir / f"{page.id}_binary.png"
        cv2.imwrite(str(grey_path), cleaned)
        cv2.imwrite(str(binary_path), binary)

        records.append(
            {
                "page_id": page.id,
                "doc_id": page.doc_id,
                "original_path": page.image_path,
                "greyscale_path": str(grey_path),
                "binary_path": str(binary_path),
                "skew_deg": skew_deg,
                "quality_score": quality,
                "low_legibility": low_legibility,
            }
        )
        out_pages.append(Page(id=page.id, image_path=str(grey_path), doc_id=page.doc_id))

    with sidecar_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(
        "preprocess: %d page(s) processed, %d flagged low-legibility (score<%.2f), sidecar %s",
        len(records),
        n_low,
        quality_threshold,
        sidecar_path,
    )
    return out_pages
