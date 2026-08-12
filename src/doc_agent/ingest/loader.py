"""Stage 1 — load scanned page-images"""

from __future__ import annotations

import csv
from pathlib import Path

from ..contracts import Page
from ..data.validate import validate
from ..data.versioning import snapshot
from ..logging_conf import get_logger

logger = get_logger(__name__)

_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def _load_deed_groups(path: Path) -> dict[str, dict[str, str]]:
    """page_id -> its data/deed_groups.csv row (doc_id, confidence, notes, duplicate_of).

    Doubles as the corpus's inclusion list: a page absent from this file (today, only
    dolil_30 -- a confirmed non-content Instagram screenshot, see data/provenance.md) is
    not loaded at all, the same fail-closed contract vision/ocr.py's standalone mode uses
    for this file. Most rows are `confidence=unreviewed` singletons rather than confirmed
    deed groups -- real registry-serial grouping is still a manual, in-progress pass; a
    singleton is a safe default (it never wrongly merges two different deeds) even though
    it doesn't yet give the leakage guarantee a correct multi-page group would.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"no deed grouping at {path}. Stage 1 cannot invent doc_id: it is what keeps "
            "the train/val/test split document-level. Build data/deed_groups.csv first."
        )
    with path.open(encoding="utf-8", newline="") as fh:
        return {row["page_id"]: row for row in csv.DictReader(fh)}


def load_pages(cfg: dict) -> list[Page]:
    """Read data/raw/ -> list[Page], scoped and grouped by data/deed_groups.csv.

    Pages marked duplicate_of another page are excluded here, not downstream: they're a
    re-scan of content already covered by their canonical page's doc_id, so indexing both
    would duplicate content in the vector store for no benefit, and Page (contracts.py)
    has no field to carry "duplicate, skip me" through the pipeline for a later stage to
    act on instead.
    """
    params = cfg["ocr"]  # pages_dir/deed_groups are shared with Stage 3's standalone mode
    pages_dir = Path(params["pages_dir"])
    if not pages_dir.is_dir():
        raise FileNotFoundError(f"no page directory at {pages_dir}")

    groups = _load_deed_groups(Path(params["deed_groups"]))

    pages: list[Page] = []
    skipped_duplicates = 0
    for path in pages_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        row = groups.get(path.stem)
        if row is None:
            continue  # not in deed_groups.csv -- confirmed non-content, e.g. dolil_30
        if row.get("duplicate_of"):
            skipped_duplicates += 1
            continue
        pages.append(Page(id=path.stem, image_path=str(path), doc_id=row["doc_id"]))

    pages.sort(key=lambda p: int(p.id.rsplit("_", 1)[-1]))

    logger.info(
        "loaded %d page(s), skipped %d confirmed duplicate(s), from %s",
        len(pages),
        skipped_duplicates,
        pages_dir,
    )
    validate(pages)
    snapshot(str(pages_dir))
    return pages
