"""Data — corpus versioning (which corpus version -> which result)"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from ..logging_conf import get_logger

logger = get_logger(__name__)

VERSIONS_LOG = Path("data/interim/corpus_versions.jsonl")


def snapshot(corpus_dir: str) -> str:
    """Hash the corpus directory's contents and record a version id.

    The id is a short sha256 digest over every file's (relative path, size, mtime),
    sorted for determinism -- not the file bytes themselves. Hashing the full byte
    content of a ~2GB scan corpus on every run would be far too slow for what's really
    just a change check; path+size+mtime already catches the additions, removals, and
    replacements that define "a different corpus snapshot" for our purposes. Swap in a
    full content hash, or wire DVC (`dvc add` + `dvc push` against .dvc/config), if a
    stronger guarantee is ever needed.

    Returns the version id and appends one record to `data/interim/corpus_versions.jsonl`
    (gitignored, like the rest of data/interim/) so a later result can be traced back to
    the exact corpus state that produced it.
    """
    root = Path(corpus_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"corpus_dir does not exist: {corpus_dir}")

    files = sorted(p for p in root.rglob("*") if p.is_file())
    if not files:
        raise ValueError(f"corpus_dir has no files to version: {corpus_dir}")

    hasher = hashlib.sha256()
    for f in files:
        stat = f.stat()
        record = f"{f.relative_to(root).as_posix()}:{stat.st_size}:{int(stat.st_mtime)}"
        hasher.update(record.encode("utf-8"))
    version_id = hasher.hexdigest()[:16]

    VERSIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "version_id": version_id,
        "corpus_dir": str(root),
        "n_files": len(files),
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with VERSIONS_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")

    logger.info(
        "corpus snapshot %s (%d files) recorded to %s", version_id, len(files), VERSIONS_LOG
    )
    return version_id
