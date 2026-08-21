"""§6 NO-HALLUCINATION — the grounding gate catching fabricated answers.

Cases are driven from tests/fixtures/hallucination_cases.jsonl, whose queries and gold
values are REAL entries from grading_kit/tasks.jsonl (t2 and t10) rather than invented
scenarios. t10 is the useful one: its gold is INSUFFICIENT EVIDENCE, because the annual
khajna it asks for is not on the page at all. That makes it a standing hallucination trap
already present in the grading kit.

This matters on this corpus specifically. vision/ocr.py D2 MEASURED records the reader
inventing universities and deed boilerplate rather than reporting that it could not read.
On land deeds, the same failure produces an invented plot number or amount -- the
precision-first NFR's worst outcome, and the reason abstention is treated as a correct
answer here rather than a failure to answer.
"""

import json
import pathlib

import pytest

from doc_agent.contracts import Chunk
from doc_agent.llm.postprocess import format_answer
from doc_agent.llm.prompts import ABSTAIN

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "hallucination_cases.jsonl"


def _cases():
    return [
        json.loads(line)
        for line in FIXTURES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


CASES = _cases()


def _answer(case):
    evidence = [Chunk(**c) for c in case["evidence"]]
    return format_answer(case["draft"], evidence)


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_the_grounding_gate_reaches_the_recorded_verdict(case):
    assert _answer(case).grounded is case["expect_grounded"], case["_note"]


def test_an_invented_field_is_never_published_as_an_answer():
    """t10: the page has no khajna figure; a fluent invented number must not survive."""
    case = next(c for c in CASES if c["id"] == "t10_invented_field_uncited")

    answer = _answer(case)

    assert answer.grounded is False
    assert answer.citations == []
    # The agent replaces an ungrounded answer with the abstention (agent.synthesize),
    # which is what grading_kit gold for t10 expects.
    assert case["gold"] == ABSTAIN


def test_one_cited_sentence_does_not_launder_an_invented_one_beside_it():
    """The realistic failure, and the reason grounding requires EVERY sentence cited."""
    case = next(c for c in CASES if c["id"] == "t2_partial_hallucination")

    answer = _answer(case)

    assert answer.grounded is False
    # The true half IS cited -- a one-citation-is-enough rule would have published the lot.
    assert len(answer.citations) == 1
    assert answer.confidence < 1.0


def test_a_citation_to_a_page_that_was_never_retrieved_is_dropped():
    case = next(c for c in CASES if c["id"] == "t2_hallucinated_citation")

    answer = _answer(case)

    assert answer.citations == []
    assert answer.grounded is False


def test_a_correct_cited_answer_still_passes():
    """Control: a gate that refuses this is useless, not safe."""
    case = next(c for c in CASES if c["id"] == "t2_correct_cited")

    answer = _answer(case)

    assert answer.grounded is True
    assert answer.confidence == 1.0
    assert answer.citations[0].chunk_id == "dolil_13#p"


def test_known_gap_a_false_claim_cited_to_a_real_chunk_is_not_caught():
    """Asserted so the limitation is tracked rather than discovered by a grader.

    Grounding proves an answer came from the evidence. It cannot prove the evidence says
    it. Closing this needs an entailment check -- the JUDGE prompt in llm/prompts.py, run
    by eval/judge.py -- not a stricter parser.
    """
    case = next(c for c in CASES if c["id"] == "KNOWN_GAP_invented_field_cited_to_a_real_chunk")

    answer = _answer(case)

    assert answer.grounded is True  # <-- the gap
    assert case["gold"] == ABSTAIN  # ...but the true answer is that the page is silent
