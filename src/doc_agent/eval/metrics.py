"""Stage 9 — metrics"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from ..contracts import *  # noqa
from ..contracts import Answer
from ..vision.ocr import normalize

# Stripped from token edges before matching, so a danda glued to the last word of a line is
# not scored as a different word. Bangla danda/double-danda first, then ASCII punctuation.
_PUNCT = "।॥.,;:!?\"'()[]{}<>-–—/\\|*_"

# "[?]" is the reader's illegible marker (llm/prompts.TRANSCRIBE), "[illegible]" the human
# labeller's (grading_kit/README.md). Both mean "not read", so neither is a token to match.
_ILLEGIBLE = {"[?]", "[illegible]"}

_CHUNKS_PATH = Path(__file__).resolve().parents[3] / "data" / "interim" / "index" / "chunks.jsonl"
_CHUNK_ID_KEY = "chunk_id"
_CHUNK_TEXT_KEY = "chunk_text"
_CHUNK_TEXTS: dict[str, str] | None = None


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


def _match_text(text: str) -> str:
    return " ".join(normalize(text).split())


def _load_chunk_texts() -> dict[str, str]:
    """Load the sidecar schema emitted by index.chunk._make_chunk via index.store.build."""
    global _CHUNK_TEXTS
    if _CHUNK_TEXTS is not None:
        return _CHUNK_TEXTS

    chunk_texts: dict[str, str] = {}
    with _CHUNKS_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            chunk_id = record.get(_CHUNK_ID_KEY)
            chunk_text = record.get(_CHUNK_TEXT_KEY)
            if isinstance(chunk_id, str) and isinstance(chunk_text, str):
                chunk_texts[chunk_id] = chunk_text
    _CHUNK_TEXTS = chunk_texts
    return chunk_texts


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


def answer_f1(pred: str, gold: str) -> float:
    """Token-level F1 for answer text, using the same normalization policy as OCR.

    The A1 target names answer-F1 separately from OCR quality, but for verifiable
    short-answer text the matching policy should stay identical: NFC/digit normalization,
    punctuation stripped at token edges, and illegible markers excluded.
    """
    return ocr_f1(pred, gold)


def _field_scalar(value: object) -> str:
    return _match_text(str(value))


def _is_unscorable_field_value(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        normalized = _field_scalar(value).lower()
        return not normalized or any(marker in normalized for marker in _ILLEGIBLE)
    if isinstance(value, Iterable) and not isinstance(value, str | bytes):
        return not any(not _is_unscorable_field_value(item) for item in value)
    return False


def _field_key(value: object) -> tuple[str, ...]:
    """Normalize a scalar/list field into an order-insensitive exact-match key."""
    if isinstance(value, Iterable) and not isinstance(value, str | bytes):
        return tuple(
            sorted(
                _field_scalar(item)
                for item in value
                if not _is_unscorable_field_value(item)
            )
        )
    if _is_unscorable_field_value(value):
        return ()
    return (_field_scalar(value),)


def critical_field_exact_match(
    pred_fields: dict,
    gold_fields: dict,
    critical_fields: Iterable[str] | None = None,
) -> float:
    """Exact Match over scorable critical fields.

    Empty or explicitly illegible gold values are skipped, because they are not reliable
    oracle values. List-valued fields such as party names are compared as normalized
    unordered sets; scalar fields such as dates and amounts require a normalized exact
    string match.
    """
    fields = list(critical_fields) if critical_fields is not None else list(gold_fields)
    total = 0
    correct = 0
    for field in fields:
        gold_value = gold_fields.get(field)
        if _is_unscorable_field_value(gold_value):
            continue

        total += 1
        if _field_key(pred_fields.get(field)) == _field_key(gold_value):
            correct += 1

    return correct / total if total else 1.0


def recall_at_k(retrieved: list, gold: list, k: int) -> float:
    if not gold:
        return 1.0
    if k <= 0 or not retrieved:
        return 0.0

    def key(item):
        return getattr(item, "id", item)

    gold_ids = {key(item) for item in gold}
    retrieved_ids = {key(item) for item in retrieved[:k]}
    return len(gold_ids & retrieved_ids) / len(gold_ids)


def groundedness(answer: Answer) -> float:
    if not answer.text.strip():
        return 1.0 if not answer.grounded else 0.0
    return 1.0 if answer.grounded else 0.0


def citation_accuracy(answer: Answer) -> float:
    # chunks.jsonl has to be loaded before this metric can validate cited spans.
    if not answer.text.strip():
        return 1.0 if not answer.citations else 0.0
    if not answer.citations:
        return 0.0

    chunk_texts = _load_chunk_texts()
    answer_text = _match_text(answer.text)
    valid = 0
    for citation in answer.citations:
        chunk_text = chunk_texts.get(citation.chunk_id)
        if chunk_text is None:
            continue

        start, end = citation.span
        if not (0 <= start < end <= len(chunk_text)):
            continue

        cited_text = _match_text(chunk_text[start:end])
        if cited_text and cited_text in answer_text:
            valid += 1

    return valid / len(answer.citations)


def ece(confidences, correct) -> float:
    confidences = list(confidences)
    correct = list(correct)
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must have the same length")
    if not confidences:
        return 0.0

    n_bins = 10
    bins: list[list[tuple[float, float]]] = [[] for _ in range(n_bins)]
    for confidence, is_correct in zip(confidences, correct, strict=True):
        confidence = float(confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidences must be in [0, 1]")
        accuracy = 1.0 if bool(is_correct) else 0.0
        bin_index = min(n_bins - 1, int(confidence * n_bins))
        bins[bin_index].append((confidence, accuracy))

    total = len(confidences)
    error = 0.0
    for bucket in bins:
        if not bucket:
            continue
        avg_confidence = sum(item[0] for item in bucket) / len(bucket)
        avg_accuracy = sum(item[1] for item in bucket) / len(bucket)
        error += (len(bucket) / total) * abs(avg_confidence - avg_accuracy)
    return error


def subgroup_gap(scores_by_group: dict) -> float:
    means: list[float] = []
    for scores in scores_by_group.values():
        if isinstance(scores, Iterable) and not isinstance(scores, str | bytes):
            values = [float(score) for score in scores]
            if not values:
                continue
            means.append(sum(values) / len(values))
        elif scores is not None:
            means.append(float(scores))

    if len(means) < 2:
        return 0.0
    return max(means) - min(means)
