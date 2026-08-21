"""Tracing — every seam the FIXED agent loop calls must land a TraceStep in traces/run.jsonl.

Driven through hooks.run() directly, with the same ctx shapes agent.Agent.run passes, so the
audit trail is testable before the agent policy exists.
"""

import json

import pytest

from doc_agent import hooks, logging_conf
from doc_agent.contracts import Answer, TraceStep


@pytest.fixture
def trace(tmp_path, monkeypatch):
    path = tmp_path / "run.jsonl"
    monkeypatch.setattr(logging_conf, "TRACE_PATH", path)
    hooks.clear()
    logging_conf.register(hooks)
    yield path
    hooks.clear()


def _lines(path):
    return [TraceStep(**json.loads(ln)) for ln in path.read_text().splitlines() if ln.strip()]


def test_on_step_records_the_query_and_what_decide_saw(trace):
    hooks.run(
        hooks.ON_STEP,
        {"state": {"query": "who owns plot 2163?", "obs": [{"top_score": 0.31, "k": 10}]}},
    )

    (step,) = _lines(trace)
    assert step.tool == "decide"
    assert step.args["query"] == "who owns plot 2163?"
    assert step.obs == {"top_score": 0.31, "k": 10}


def test_on_tool_call_records_the_tool_and_its_arguments(trace):
    hooks.run(hooks.ON_TOOL_CALL, {"action": {"tool": "retrieve", "query": "plot 2163", "k": 20}})

    (step,) = _lines(trace)
    assert step.tool == "retrieve"
    assert step.args == {"query": "plot 2163", "k": 20}


def test_after_answer_records_whether_the_answer_was_grounded(trace):
    answer = Answer(text="রহিম উদ্দিন", citations=[], grounded=False, confidence=0.2)

    hooks.run(hooks.AFTER_ANSWER, {"answer": answer})

    (step,) = _lines(trace)
    assert step.tool == "answer"
    assert step.obs["grounded"] is False


def test_steps_are_numbered_in_the_order_they_happened(trace):
    hooks.run(hooks.ON_STEP, {"state": {"query": "q", "obs": []}})
    hooks.run(hooks.ON_TOOL_CALL, {"action": {"tool": "retrieve", "k": 10}})
    hooks.run(hooks.ON_STEP, {"state": {"query": "q", "obs": [{"top_score": 0.9, "k": 20}]}})

    assert [s.step for s in _lines(trace)] == [1, 2, 3]
    assert [s.tool for s in _lines(trace)] == ["decide", "retrieve", "decide"]


def test_rewiring_starts_a_fresh_trace_rather_than_interleaving_runs(trace):
    hooks.run(hooks.ON_STEP, {"state": {"query": "first run", "obs": []}})

    hooks.clear()
    logging_conf.register(hooks)
    hooks.run(hooks.ON_STEP, {"state": {"query": "second run", "obs": []}})

    steps = _lines(trace)
    assert [s.args["query"] for s in steps] == ["second run"]


def test_a_widening_re_search_is_readable_from_the_trace(trace):
    """The A3 agentic gate: the trace must show k widening after weak evidence."""
    hooks.run(hooks.ON_STEP, {"state": {"query": "q", "obs": [{"top_score": 0.31, "k": 10}]}})
    hooks.run(
        hooks.ON_TOOL_CALL, {"action": {"tool": "retrieve", "query": "q reformulated", "k": 20}}
    )

    decide, retrieve = _lines(trace)
    assert decide.obs["top_score"] < 0.35 and decide.obs["k"] == 10
    assert retrieve.args["k"] == 20
