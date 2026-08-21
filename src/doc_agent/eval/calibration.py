"""Stage 9 — confidence calibration (calibrated-confidence NFR)"""
from __future__ import annotations

from ..contracts import *  # noqa


def temperature_scale(logits, labels):
    """Fit temperature on val; return scaler. IMPLEMENT."""
    raise NotImplementedError("Calibration: temperature scaling")
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

