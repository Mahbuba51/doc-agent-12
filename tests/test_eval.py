"""Stage 9 metrics — OCR scoring. CI runs these.

ocr_f1 is the number every deferred Stage 2/3 decision is waiting on (3B vs 7B, single-pass
vs the GraDeT-HTR fallback, page-level layout on tabular pages), so its edge cases are
pinned here rather than discovered later against a held-out score nobody can reproduce.
"""

from __future__ import annotations

import unicodedata

import pytest

from doc_agent.eval.metrics import ocr_f1


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


def test_punctuation_does_not_split_a_matching_token():
    """A danda glued to the last word must not make it a miss."""
    assert ocr_f1("ক খ।", "ক খ") == 1.0
