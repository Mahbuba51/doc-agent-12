"""Grounding gate — citations are verified against the evidence, not taken on trust.

Citation.span indexes into Answer.text, because eval/metrics.groundedness(answer) and
citation_accuracy(answer) are handed the Answer alone and must be scorable from it.
"""

from doc_agent import hooks
from doc_agent.contracts import Chunk
from doc_agent.llm import postprocess
from doc_agent.llm.postprocess import format_answer
from doc_agent.llm.prompts import ABSTAIN


def _chunk(cid, text="দাগ নং ২১৬৩ এর মালিক রহিম উদ্দিন।"):
    return Chunk(id=cid, doc_id="deed_p0038", text=text, page_ids=["dolil_38"], score=0.9)


def test_the_abstain_sentinel_produces_an_ungrounded_answer():
    answer = format_answer(ABSTAIN, [_chunk("dolil_38#p")])

    assert answer.grounded is False
    assert answer.citations == []
    assert answer.confidence == 0.0


def test_a_cited_claim_is_grounded_and_the_marker_is_stripped():
    answer = format_answer("মালিক রহিম উদ্দিন [dolil_38#p]", [_chunk("dolil_38#p")])

    assert answer.grounded is True
    assert "[dolil_38#p]" not in answer.text
    assert [c.chunk_id for c in answer.citations] == ["dolil_38#p"]


def test_a_citation_span_points_at_the_claim_it_supports():
    answer = format_answer("মালিক রহিম উদ্দিন [dolil_38#p]", [_chunk("dolil_38#p")])

    start, end = answer.citations[0].span
    assert answer.text[start:end].strip() == "মালিক রহিম উদ্দিন"


def test_an_answer_with_no_citation_is_not_grounded():
    answer = format_answer("মালিক রহিম উদ্দিন", [_chunk("dolil_38#p")])

    assert answer.grounded is False
    assert answer.citations == []


def test_a_citation_to_a_chunk_that_was_never_retrieved_is_dropped():
    """The hallucinated-citation case: a plausible id the agent never actually read."""
    answer = format_answer("মালিক রহিম উদ্দিন [dolil_999#p]", [_chunk("dolil_38#p")])

    assert answer.citations == []
    assert answer.grounded is False


def test_the_readers_illegible_marker_is_not_mistaken_for_a_citation():
    """TRANSCRIBE writes [?] for unreadable characters, so brackets occur in chunk text."""
    answer = format_answer("দাগ নং ২১[?]৩ [dolil_38#p]", [_chunk("dolil_38#p")])

    assert [c.chunk_id for c in answer.citations] == ["dolil_38#p"]
    assert "[?]" in answer.text


def test_every_sentence_must_be_cited_for_the_answer_to_be_grounded():
    """Precision-first: one cited sentence does not launder an uncited one beside it."""
    raw = "মালিক রহিম উদ্দিন [dolil_38#p]। জমির দাম ৫০০০ টাকা।"

    answer = format_answer(raw, [_chunk("dolil_38#p")])

    assert answer.grounded is False


def test_confidence_reflects_how_much_of_the_answer_is_cited():
    one = format_answer("মালিক রহিম উদ্দিন [dolil_38#p]", [_chunk("dolil_38#p")])
    half = format_answer(
        "মালিক রহিম উদ্দিন [dolil_38#p]। জমির দাম ৫০০০ টাকা।", [_chunk("dolil_38#p")]
    )

    assert one.confidence > half.confidence


def test_the_gate_marks_a_state_with_no_evidence_for_abstention():
    hooks.clear()
    postprocess.register(hooks)
    state = {"query": "who owns plot 2163?", "obs": []}

    hooks.run(hooks.BEFORE_ANSWER, {"state": state})

    assert state["abstain"] is True
    hooks.clear()


def test_the_gate_leaves_a_state_with_real_evidence_alone():
    hooks.clear()
    postprocess.register(hooks)
    state = {"query": "who owns plot 2163?", "obs": [{"top_score": 0.81, "k": 10}]}

    hooks.run(hooks.BEFORE_ANSWER, {"state": state})

    assert state.get("abstain") is False
    hooks.clear()
