"""HITL — persistent review queue (survives restarts)

JSON, not sqlite. The queue is small (a handful of escalations per run), it is written once
per decision, and a human reviewer has to be able to open it and see what the agent got
stuck on. An inspectable file is worth more here than concurrent-write safety this project
has no way to exercise -- one agent process, one reviewer, never at the same instant.

PRIVACY NOTE for whoever wires the reviewer UI: `context` is written verbatim, and on this
corpus it will contain deed text -- party names, addresses. That is intentional (a reviewer
who cannot see the passage cannot rule on it) but it means this file carries the same
handling obligations as the corpus, and it must not be committed. See governance/pii.py for
the redaction that applies to answers and logs, which deliberately does NOT apply here.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from ..contracts import *  # noqa
from ..logging_conf import get_logger

logger = get_logger(__name__)

STORE_PATH = Path("data/interim/hitl_queue.json")

PENDING = "pending"


def _load() -> list[dict]:
    if not STORE_PATH.is_file():
        return []
    text = STORE_PATH.read_text(encoding="utf-8").strip()
    return json.loads(text) if text else []


def _save(items: list[dict]) -> None:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    # ensure_ascii=False so Bangla deed text stays readable to the human reviewer rather
    # than turning into a wall of \uXXXX escapes.
    STORE_PATH.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def enqueue(item: dict) -> str:
    """Persist a pending review item; return its id."""
    items = _load()
    record = dict(item)
    record["id"] = uuid.uuid4().hex[:12]
    record["status"] = PENDING
    record["decision"] = None
    record["created_at"] = datetime.now(UTC).isoformat()
    items.append(record)
    _save(items)
    logger.info("queued review item %s: %s", record["id"], record.get("reason", ""))
    return record["id"]


def pending() -> list[dict]:
    """Items still awaiting a human decision, oldest first."""
    return [i for i in _load() if i.get("status") == PENDING]


def resolve(item_id: str, decision: str) -> None:
    """Record a human decision. Raises KeyError if the id is unknown."""
    items = _load()
    for item in items:
        if item.get("id") == item_id:
            item["status"] = decision
            item["decision"] = decision
            item["resolved_at"] = datetime.now(UTC).isoformat()
            _save(items)
            logger.info("review item %s resolved: %s", item_id, decision)
            return
    raise KeyError(f"no review item with id {item_id!r}")


def find(item_id: str) -> dict | None:
    """One item by id, or None. Used by hitl.escalate to read a prior decision."""
    for item in _load():
        if item.get("id") == item_id:
            return item
    return None


def find_by_key(key: str) -> dict | None:
    """The most recent item for an escalation key, or None.

    Most recent, not first: if a reviewer rejects an escalation and a later run raises the
    same one again after the underlying page was re-read, the newer decision is the live one.
    """
    matches = [i for i in _load() if i.get("key") == key]
    return matches[-1] if matches else None
