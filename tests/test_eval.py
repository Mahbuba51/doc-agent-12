"""Stage 9 metrics — OCR scoring. CI runs these.

ocr_f1 is the number every deferred Stage 2/3 decision is waiting on (3B vs 7B, single-pass
vs the GraDeT-HTR fallback, page-level layout on tabular pages), so its edge cases are
pinned here rather than discovered later against a held-out score nobody can reproduce.
"""

from __future__ import annotations

import unicodedata

import pytest

from doc_agent.contracts import Answer, Chunk, Citation
from doc_agent.eval import metrics
from doc_agent.eval.metrics import (
    answer_f1,
    citation_accuracy,
    critical_field_exact_match,
    ece,
    groundedness,
    ocr_f1,
    recall_at_k,
    subgroup_gap,
)


def test_answer_f1_uses_same_matching_policy_as_ocr_f1():
    pred = "deed 2163 party Rahim"
    gold = "deed \u09e8\u09e7\u09ec\u09e9 party Karim"
    assert answer_f1(pred, gold) == pytest.approx(ocr_f1(pred, gold))


def test_critical_field_exact_match_scores_scalars_and_lists():
    pred = {
        "deed_date": "30/1/88",
        "land_amount": "10 1/2",
        "party_names": ["Karim", "Rahim"],
    }
    gold = {
        "deed_date": "30/1/88",
        "land_amount": "10 1/2",
        "party_names": ["Rahim", "Karim"],
    }
    assert critical_field_exact_match(pred, gold) == 1.0


def test_critical_field_exact_match_counts_normalized_misses():
    pred = {"deed_date": "30/1/88", "land_amount": "11", "party_names": ["Rahim"]}
    gold = {"deed_date": "30/1/88", "land_amount": "10", "party_names": ["Karim"]}
    assert critical_field_exact_match(pred, gold) == pytest.approx(1 / 3)


def test_critical_field_exact_match_skips_unscorable_gold_values():
    pred = {
        "deed_serial": "anything",
        "deed_date": "30/1/88",
        "land_amount": "10",
        "party_names": ["Rahim"],
    }
    gold = {
        "deed_serial": "",
        "deed_date": "30/1/88",
        "land_amount": "[illegible]",
        "party_names": ["Rahim", "[?]"],
    }
    assert critical_field_exact_match(pred, gold) == 1.0


def test_critical_field_exact_match_accepts_selected_fields():
    pred = {"deed_date": "30/1/88", "land_amount": "wrong"}
    gold = {"deed_date": "30/1/88", "land_amount": "10"}
    assert critical_field_exact_match(pred, gold, critical_fields=["land_amount"]) == 0.0


def test_identical_text_scores_one():
    assert ocr_f1("দলিল নং ২১৬৩", "দলিল নং ২১৬৩") == 1.0


def test_disjoint_text_scores_zero():
    assert ocr_f1("ক খ গ", "ঘ ঙ চ") == 0.0


def test_partial_overlap_is_token_level_f1():
    # 2 of 3 predicted tokens are right, 2 of 3 gold tokens are found: P = R = F1 = 2/3.
    assert ocr_f1("ক খ গ", "ক খ ঘ") == pytest.approx(2 / 3)


def test_bangla_and_ascii_digits_compare_equal():
    """The A1 normalization policy: "২১৬৩" and "2163" are the same deed number."""
    assert ocr_f1("দলিল ২১৬৩", "দলিল 2163") == 1.0


def test_decomposed_and_composed_forms_compare_equal():
    """NFC, so a decomposed matra from the reader is not scored as a different word.

    U+09CB (ো) decomposes to U+09C7 + U+09BE, so the two spellings are byte-different but
    the same word -- exactly the mismatch NFC exists to fold.
    """
    composed = "কো"
    decomposed = unicodedata.normalize("NFD", composed)
    assert decomposed != composed
    assert ocr_f1(decomposed, composed) == 1.0


def test_repetition_loop_is_penalised():
    """The pilot's failure mode: the reader loops one token. Counts must cap the match.

    Bag-of-tokens without multiplicity would score this 1.0 and hide the defect.
    """
    # overlap = min(4, 1) = 1 -> P = 1/4, R = 1/2, F1 = 1/3.
    assert ocr_f1("ক ক ক ক", "ক খ") == pytest.approx(1 / 3)


def test_truncated_prediction_loses_recall_not_precision():
    """Cut-off text is fully correct as far as it goes: P = 1, R = 1/2, F1 = 2/3."""
    assert ocr_f1("ক খ", "ক খ গ ঘ") == pytest.approx(2 / 3)


def test_illegible_markers_are_excluded_from_both_sides():
    """[?] (reader) and [illegible] (labeller) mean "unread", not a token to match.

    Scoring them as tokens would let a reader inflate F1 by emitting [?] wherever the gold
    says [illegible] -- agreeing about failure would read as agreeing about content.
    """
    assert ocr_f1("ক [?] গ", "ক [illegible] গ") == pytest.approx(1.0)
    # And an all-[?] read is an empty read, not a perfect one.
    assert ocr_f1("[?] [?]", "ক খ") == 0.0


def test_both_empty_scores_one_and_one_empty_scores_zero():
    assert ocr_f1("", "") == 1.0
    assert ocr_f1("   ", "\n") == 1.0
    assert ocr_f1("", "ক খ") == 0.0
    assert ocr_f1("ক খ", "") == 0.0


def test_recall_at_k_scores_unique_gold_hits_in_top_k():
    assert recall_at_k(["c1", "c2", "c3"], ["c2", "c4"], 2) == pytest.approx(0.5)
    assert recall_at_k(["c1", "c1", "c2"], ["c1", "c2"], 2) == pytest.approx(0.5)


def test_recall_at_k_accepts_chunk_objects():
    retrieved = [Chunk(id="c1", doc_id="d1", text="x", page_ids=["p1"])]
    gold = [
        Chunk(id="c1", doc_id="d1", text="x", page_ids=["p1"]),
        Chunk(id="c2", doc_id="d1", text="y", page_ids=["p1"]),
    ]
    assert recall_at_k(retrieved, gold, 10) == pytest.approx(0.5)


def test_recall_at_k_edge_cases():
    assert recall_at_k(["c1"], [], 5) == 1.0
    assert recall_at_k([], ["c1"], 5) == 0.0
    assert recall_at_k(["c1"], ["c1"], 0) == 0.0


def test_groundedness_scores_supported_answers_and_fabrications():
    supported = Answer(text="supported answer", citations=[], grounded=True, confidence=0.9)
    unsupported = Answer(text="unsupported answer", citations=[], grounded=False, confidence=0.2)
    assert groundedness(supported) == 1.0
    assert groundedness(unsupported) == 0.0


def test_groundedness_treats_empty_abstention_as_non_hallucination():
    abstained = Answer(text="", citations=[], grounded=False, confidence=0.1)
    inconsistent = Answer(text="   ", citations=[], grounded=True, confidence=0.1)
    assert groundedness(abstained) == 1.0
    assert groundedness(inconsistent) == 0.0


def test_citation_accuracy_requires_cited_span_to_appear_in_answer(monkeypatch):
    monkeypatch.setattr(metrics, "_CHUNK_TEXTS", {"c1": "alpha beta gamma"})
    answer = Answer(
        text="The cited value is beta.",
        citations=[Citation(chunk_id="c1", span=(6, 10))],
        grounded=True,
        confidence=0.9,
    )
    assert citation_accuracy(answer) == 1.0


def test_citation_accuracy_rejects_wrong_or_unresolved_spans(monkeypatch):
    monkeypatch.setattr(metrics, "_CHUNK_TEXTS", {"c1": "alpha beta gamma"})
    missing_text = Answer(
        text="The cited value is delta.",
        citations=[Citation(chunk_id="c1", span=(6, 10))],
        grounded=True,
        confidence=0.9,
    )
    missing_chunk = Answer(
        text="The cited value is beta.",
        citations=[Citation(chunk_id="missing", span=(0, 4))],
        grounded=True,
        confidence=0.9,
    )
    bad_span = Answer(
        text="The cited value is beta.",
        citations=[Citation(chunk_id="c1", span=(6, 99))],
        grounded=True,
        confidence=0.9,
    )
    assert citation_accuracy(missing_text) == 0.0
    assert citation_accuracy(missing_chunk) == 0.0
    assert citation_accuracy(bad_span) == 0.0


def test_citation_accuracy_averages_over_citations(monkeypatch):
    monkeypatch.setattr(metrics, "_CHUNK_TEXTS", {"c1": "alpha beta gamma", "c2": "delta"})
    answer = Answer(
        text="The cited value is beta.",
        citations=[Citation(chunk_id="c1", span=(6, 10)), Citation(chunk_id="c2", span=(0, 5))],
        grounded=True,
        confidence=0.9,
    )
    assert citation_accuracy(answer) == pytest.approx(0.5)


def test_citation_accuracy_abstention_and_missing_citation_cases(monkeypatch):
    monkeypatch.setattr(metrics, "_CHUNK_TEXTS", {"c1": "alpha beta gamma"})
    assert citation_accuracy(Answer(text="", citations=[], grounded=False, confidence=0.1)) == 1.0
    assert (
        citation_accuracy(
            Answer(
                text="",
                citations=[Citation(chunk_id="c1", span=(6, 10))],
                grounded=False,
                confidence=0.1,
            )
        )
        == 0.0
    )
    assert (
        citation_accuracy(
            Answer(text="The cited value is beta.", citations=[], grounded=True, confidence=0.9)
        )
        == 0.0
    )


def test_ece_returns_zero_for_empty_or_perfectly_calibrated_inputs():
    assert ece([], []) == 0.0
    assert ece([1.0, 0.0], [True, False]) == 0.0


def test_ece_weights_bin_confidence_accuracy_gaps():
    # Bin 9 has two of three examples: avg confidence 0.95, accuracy 1.0.
    # Bin 2 has one of three examples: avg confidence 0.25, accuracy 0.0.
    assert ece([0.9, 1.0, 0.25], [True, True, False]) == pytest.approx(7 / 60)


def test_ece_rejects_mismatched_lengths_and_invalid_confidences():
    with pytest.raises(ValueError, match="same length"):
        ece([0.5], [True, False])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ece([1.2], [True])


def test_subgroup_gap_scores_max_minus_min_group_mean():
    scores = {
        "flatbed": [0.9, 0.8, 1.0],
        "phone_photo": [0.6, 0.7],
        "printed": 0.85,
    }
    assert subgroup_gap(scores) == pytest.approx(0.9 - 0.65)


def test_subgroup_gap_ignores_empty_groups_and_singletons_have_no_gap():
    assert subgroup_gap({}) == 0.0
    assert subgroup_gap({"flatbed": [0.9], "empty": []}) == 0.0
    assert subgroup_gap({"empty": [], "none": None}) == 0.0


def test_punctuation_does_not_split_a_matching_token():
    """A danda glued to the last word must not make it a miss."""
    assert ocr_f1("ক খ।", "ক খ") == 1.0
