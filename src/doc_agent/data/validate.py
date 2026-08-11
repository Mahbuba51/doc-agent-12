"""Data — data schema/quality validation at ingest"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .. import config as config_module
from ..contracts import Page
from ..logging_conf import get_logger

logger = get_logger(__name__)


def validate(pages: list[Page]) -> None:
    """Assert min pages/words, format, no leakage across splits.

    `Page` (contracts.py) only carries {id, image_path, doc_id} -- no extracted text and
    no split label -- because this runs at ingest, before OCR (Stage 3) and before any
    train/val/test split is assigned. So it checks everything that IS knowable here:

      - the page-count floor from configs/task.yaml (corpus.min_pages)
      - every page id is unique (no duplicate ingestion)
      - every image_path actually exists on disk (catches a broken/partial data/raw/)
      - every doc_id is non-empty -- document-level splitting (the leakage rule: never
        split the same document across train/val/test) is keyed on doc_id, so a page
        with no doc_id can't be kept safely on one side of a split

    What this does NOT check, and why it's not here:
      - the word-count floor (configs/task.yaml corpus.min_words) needs extracted text,
        which doesn't exist until Stage 3 OCR runs
      - the near-duplicate scan (pHash across page images) needs to open and hash every
        image, which is too heavy for an ingest-time gate -- it stays in
        notebooks/eda.ipynb, run once as exploration, not on every ingest
      - whether a *document* actually ends up split-safe, i.e. all its pages landing in
        one split -- non-empty doc_id only proves a page COULD be grouped correctly, not
        that the split step actually did it. That real leakage check needs the split
        assignment itself, so it lives in validate_splits() below, run after the split
        step, not at ingest.
    """
    task = config_module.load_task()
    min_pages = task.get("corpus", {}).get("min_pages")

    errors: list[str] = []

    if min_pages is None:
        errors.append("configs/task.yaml has no corpus.min_pages")
    elif len(pages) < min_pages:
        errors.append(
            f"only {len(pages)} pages, need >= {min_pages} (configs/task.yaml corpus.min_pages)"
        )

    id_counts = Counter(p.id for p in pages)
    dupes = [pid for pid, n in id_counts.items() if n > 1]
    if dupes:
        errors.append(f"{len(dupes)} duplicate page id(s), e.g. {dupes[:5]}")

    missing_images = [p.id for p in pages if not Path(p.image_path).is_file()]
    if missing_images:
        errors.append(
            f"{len(missing_images)} page(s) with missing image_path, e.g. {missing_images[:5]}"
        )

    missing_doc_id = [p.id for p in pages if not p.doc_id]
    if missing_doc_id:
        errors.append(
            f"{len(missing_doc_id)} page(s) with empty doc_id -- document-level split "
            f"leakage can't be prevented without it, e.g. {missing_doc_id[:5]}"
        )

    if errors:
        message = "data validation failed:\n  - " + "\n  - ".join(errors)
        logger.error(message)
        raise ValueError(message)

    n_docs = len(set(p.doc_id for p in pages))
    logger.info("data validation passed: %d pages across %d documents", len(pages), n_docs)


def validate_splits(
    splits_path: str | Path = "data/splits.json",
    pages: list[Page] | None = None,
) -> None:
    """Assert the split is honest. Run after the split step, not at ingest.

    `splits_path` is a JSON file of {split_name: [page_id, ...]}, written by whatever
    produces the train/val/test assignment (today: notebooks/eda.ipynb Section 5). A
    "skipped" key is allowed and excluded from the leakage check below, not just ignored.

    Structural checks (always run, no `pages` needed):
      - train/val/test keys are all present
      - no page id appears in more than one split

    Document-leakage check (only runs if `pages` is given, since doc_id lives on Page,
    not in the split file): no doc_id has its pages spread across more than one split --
    this is the actual bug non-empty-doc_id in validate() above cannot catch on its own
    (609 pages with 607 distinct doc_ids passes that check while still leaking).
    """
    path = Path(splits_path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found -- write it from notebooks/eda.ipynb")

    splits: dict[str, list[str]] = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []

    missing = {"train", "val", "test"} - set(splits)
    if missing:
        errors.append(f"missing split(s): {sorted(missing)}")

    where: dict[str, str] = {}
    for name, page_ids in splits.items():
        for pid in page_ids:
            if pid in where:
                errors.append(f"{pid} appears in both {where[pid]} and {name}")
            where[pid] = name

    if pages is not None:
        uncovered = {p.id for p in pages} - set(where)
        if uncovered:
            errors.append(f"{len(uncovered)} page(s) in no split, e.g. {sorted(uncovered)[:5]}")

        doc_splits: dict[str, set[str]] = {}
        for p in pages:
            if p.id in where and where[p.id] != "skipped":
                doc_splits.setdefault(p.doc_id, set()).add(where[p.id])
        torn = [d for d, s in doc_splits.items() if len(s) > 1]
        if torn:
            errors.append(f"{len(torn)} doc_id(s) span more than one split, e.g. {torn[:5]}")

    if errors:
        message = "split validation failed:\n  - " + "\n  - ".join(errors)
        logger.error(message)
        raise ValueError(message)

    logger.info("split validation passed: %d pages across %d splits", len(where), len(splits))
