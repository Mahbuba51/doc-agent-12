"""HITL — the review queue must persist, and an unapproved escalation must fail closed."""

import pytest

from doc_agent.agent import hitl, hitl_store


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(hitl_store, "STORE_PATH", tmp_path / "hitl_queue.json")
    return tmp_path / "hitl_queue.json"


def test_an_enqueued_item_shows_up_as_pending():
    item_id = hitl_store.enqueue({"reason": "illegible plot number", "page_id": "dolil_66"})

    assert [i["id"] for i in hitl_store.pending()] == [item_id]


def test_the_queue_survives_a_restart(store):
    """'Persistent' has to mean on disk, not in a module-level list."""
    item_id = hitl_store.enqueue({"reason": "illegible plot number"})

    assert store.is_file()
    assert [i["id"] for i in hitl_store.pending()] == [item_id]


def test_two_items_get_different_ids():
    first = hitl_store.enqueue({"reason": "a"})
    second = hitl_store.enqueue({"reason": "b"})

    assert first != second


def test_a_resolved_item_leaves_the_pending_queue():
    item_id = hitl_store.enqueue({"reason": "illegible plot number"})

    hitl_store.resolve(item_id, "approved")

    assert hitl_store.pending() == []


def test_resolving_records_the_decision():
    item_id = hitl_store.enqueue({"reason": "illegible plot number"})

    hitl_store.resolve(item_id, "rejected")

    assert hitl_store.find(item_id)["decision"] == "rejected"


def test_resolving_an_unknown_id_is_an_error():
    with pytest.raises(KeyError, match="nope"):
        hitl_store.resolve("nope", "approved")


def test_an_escalation_is_not_approved_just_because_it_was_queued():
    """Fail closed: queuing is not consent. The agent must not proceed on a pending item."""
    result = hitl.escalate("plot number illegible", {"page_id": "dolil_66"})

    assert result.ok is False
    assert result.payload["status"] == "pending"


def test_escalating_the_same_thing_twice_does_not_queue_it_twice():
    hitl.escalate("plot number illegible", {"page_id": "dolil_66"})
    hitl.escalate("plot number illegible", {"page_id": "dolil_66"})

    assert len(hitl_store.pending()) == 1


def test_a_human_approval_lets_the_next_run_proceed():
    hitl.escalate("plot number illegible", {"page_id": "dolil_66"})
    (item,) = hitl_store.pending()

    hitl_store.resolve(item["id"], "approved")

    assert hitl.escalate("plot number illegible", {"page_id": "dolil_66"}).ok is True


def test_a_human_rejection_keeps_the_action_blocked():
    hitl.escalate("plot number illegible", {"page_id": "dolil_66"})
    (item,) = hitl_store.pending()

    hitl_store.resolve(item["id"], "rejected")

    result = hitl.escalate("plot number illegible", {"page_id": "dolil_66"})
    assert result.ok is False
    assert result.payload["status"] == "rejected"


def test_a_different_page_is_a_separate_escalation():
    hitl.escalate("plot number illegible", {"page_id": "dolil_66"})
    (item,) = hitl_store.pending()
    hitl_store.resolve(item["id"], "approved")

    assert hitl.escalate("plot number illegible", {"page_id": "dolil_20"}).ok is False


def test_the_review_queue_shows_the_reviewer_what_is_waiting():
    hitl.escalate("plot number illegible", {"page_id": "dolil_66"})

    (waiting,) = hitl.review_queue()
    assert waiting["reason"] == "plot number illegible"
    assert waiting["context"]["page_id"] == "dolil_66"
