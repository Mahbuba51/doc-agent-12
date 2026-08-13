"""Stage 9 — metrics"""

from __future__ import annotations

from collections import Counter

from ..contracts import *  # noqa
from ..vision.ocr import normalize

# Stripped from token edges before matching, so a danda glued to the last word of a line is
# not scored as a different word. Bangla danda/double-danda first, then ASCII punctuation.
_PUNCT = "।॥.,;:!?\"'()[]{}<>-–—/\\|*_"

# "[?]" is the reader's illegible marker (llm/prompts.TRANSCRIBE), "[illegible]" the human
# labeller's (grading_kit/README.md). Both mean "not read", so neither is a token to match.
_ILLEGIBLE = {"[?]", "[illegible]"}


def _tokens(text: str) -> Counter[str]:
    """Whitespace tokens under the A1 normalization policy, minus illegible markers.

    Normalization is imported from the reader rather than reimplemented: the metric has to
    fold NFC and Bangla->ASCII digits exactly the way Stage 3 emits them, or "২১৬৩" scores
    as a miss against a gold "2163" that the pipeline itself would have matched.
    """
    counts: Counter[str] = Counter()
    for raw in normalize(text).split():
        if raw.lower() in _ILLEGIBLE:
            continue
        token = raw.strip(_PUNCT)
        if token:
            counts[token] += 1
    return counts


def ocr_f1(pred: str, gold: str) -> float:
    """Token-level F1 of a transcription against its gold label, in [0, 1].

    Multiplicity is kept (Counter, not set) so a repetition loop -- the failure the pilot
    read confirmed on dense pages -- costs precision instead of scoring as a clean match.
    Two empty reads agree; one empty read does not.
    """
    pred_tokens = _tokens(pred)
    gold_tokens = _tokens(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    overlap = sum((pred_tokens & gold_tokens).values())
    if not overlap:
        return 0.0

    precision = overlap / sum(pred_tokens.values())
    recall = overlap / sum(gold_tokens.values())
    return 2 * precision * recall / (precision + recall)


def recall_at_k(retrieved: list, gold: list, k: int) -> float:
    raise NotImplementedError


def groundedness(answer: Answer) -> float:
    raise NotImplementedError  # no-hallucination


def citation_accuracy(answer: Answer) -> float:
    raise NotImplementedError


def ece(confidences, correct) -> float:
    raise NotImplementedError  # calibration


def subgroup_gap(scores_by_group: dict) -> float:
    raise NotImplementedError  # fairness
