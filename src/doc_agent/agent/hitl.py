"""HITL — human-in-the-loop review queue

FAIL CLOSED. escalate() returns ToolResult.ok=False for anything not explicitly approved by
a human -- pending included. Queuing is not consent. This matters because the agent reads
`ok` to decide whether to proceed: an escalation that returned ok=True on the strength of
having been *filed* would hand the agent permission it was specifically asking a human for,
which is worse than never escalating at all.

HOW "BLOCK UNTIL APPROVED" WORKS WITHOUT BLOCKING. A batch eval cannot sit on a socket
waiting for a reviewer, so approval is asynchronous: the first call files the item and
refuses, a human resolves it in the queue, and a LATER run of the same escalation reads that
decision and proceeds. The escalation key is what connects the two runs, which is also why
the store has to survive a restart -- the approving reviewer and the run that benefits from
it are, by construction, different processes.
"""

from __future__ import annotations

import hashlib
import json

from ..contracts import *  # noqa
from ..logging_conf import get_logger
from . import hitl_store

logger = get_logger(__name__)

APPROVED = "approved"


def _key(reason: str, context: dict) -> str:
    """A stable id for 'this same escalation', so a decision carries across runs.

    sort_keys because dict ordering must not change the key -- an approval granted on one
    run has to be found by the next one, which may have built the context in another order.
    """
    payload = json.dumps({"reason": reason, "context": context}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def escalate(reason: str, context: dict) -> ToolResult:
    """Queue for human review; block the action unless a human has approved it."""
    key = _key(reason, context)
    existing = hitl_store.find_by_key(key)

    if existing is None:
        item_id = hitl_store.enqueue({"key": key, "reason": reason, "context": context})
        logger.warning("escalated to human review (%s): %s", item_id, reason)
        return ToolResult(
            ok=False, payload={"id": item_id, "status": hitl_store.PENDING, "reason": reason}
        )

    status = existing.get("status", hitl_store.PENDING)
    # Only an explicit approval unblocks. Pending, rejected, or any decision string a
    # reviewer UI invents later all mean "not approved" and keep the action blocked.
    return ToolResult(
        ok=status == APPROVED,
        payload={"id": existing["id"], "status": status, "reason": reason},
    )


def review_queue() -> list[dict]:
    """Return pending items for the reviewer UI."""
    return hitl_store.pending()
