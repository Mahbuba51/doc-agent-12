"""LLM client — the fake backend must answer without a model, so CI never needs a GPU."""

import pytest

from doc_agent.llm.client import LLM


def _cfg(**agent):
    return {"agent": {"max_steps": 8, "autonomy": "act-with-approval", "budget_usd": 0.05, **agent}}


def test_fake_backend_completes_without_loading_a_model():
    llm = LLM(_cfg(backend="fake"))
    assert isinstance(llm.complete("who owns plot 2163?"), str)


def test_fake_backend_replays_scripted_responses_in_order():
    llm = LLM(_cfg(backend="fake", fake_responses=["first", "second"]))
    assert [llm.complete("a"), llm.complete("b")] == ["first", "second"]


def test_fake_backend_raises_when_the_script_runs_out():
    llm = LLM(_cfg(backend="fake", fake_responses=["only"]))
    llm.complete("a")
    with pytest.raises(RuntimeError, match="fake_responses"):
        llm.complete("b")


def test_fake_backend_records_every_prompt_it_was_sent():
    llm = LLM(_cfg(backend="fake"))
    llm.complete("who owns plot 2163?")
    assert llm.calls == ["who owns plot 2163?"]


def test_local_backend_sends_the_prompt_to_the_injected_generator():
    seen = []

    def generate(prompt, **kw):
        seen.append(prompt)
        return "মালিক: রহিম উদ্দিন"

    llm = LLM(_cfg(backend="local", model="fake/checkpoint"), generate=generate)
    assert llm.complete("who owns plot 2163?") == "মালিক: রহিম উদ্দিন"
    assert seen == ["who owns plot 2163?"]


def test_unknown_backend_names_itself_in_the_error():
    llm = LLM(_cfg(backend="wishful"))
    with pytest.raises(ValueError, match="wishful"):
        llm.complete("a")
