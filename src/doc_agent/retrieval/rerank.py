"""Stage 5 - cross-encoder reranking."""

from __future__ import annotations

import math
import os
from typing import Any

from ..contracts import Chunk
from ..logging_conf import get_logger

logger = get_logger(__name__)
_MODELS: dict[tuple[str, str | None], Any] = {}


def _retrieve_cfg(cfg: dict) -> dict:
    return cfg.get("retrieve", {})


def _device(cfg: dict) -> str | None:
    return cfg.get("embed", {}).get("device", cfg.get("device"))


def _cache_dir(cfg: dict) -> str | None:
    return cfg.get("embed", {}).get("cache_dir")


def _model(cfg: dict) -> Any:
    retrieve_cfg = _retrieve_cfg(cfg)
    name = str(retrieve_cfg.get("reranker", "BAAI/bge-reranker-v2-m3"))
    device = _device(cfg)
    key = (name, device)
    if key not in _MODELS:
        from sentence_transformers import CrossEncoder

        kwargs = {}
        cache_dir = _cache_dir(cfg)
        if cache_dir:
            os.environ.setdefault("HF_HOME", str(cache_dir))
            os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir))
        if device:
            kwargs["device"] = device
        if retrieve_cfg.get("rerank_max_length") is not None:
            kwargs["max_length"] = int(retrieve_cfg["rerank_max_length"])
        if retrieve_cfg.get("reranker_trust_remote_code") is not None:
            kwargs["trust_remote_code"] = bool(retrieve_cfg["reranker_trust_remote_code"])
        _MODELS[key] = CrossEncoder(name, **kwargs)
        logger.info("reranker loaded: %s", name)
    return _MODELS[key]


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _copy_with_score(chunk: Chunk, score: float) -> Chunk:
    return chunk.model_copy(update={"score": float(score)})


def rerank(query: str, candidates: list[Chunk], cfg: dict) -> list[Chunk]:
    """Cross-encoder rerank if cfg['retrieve']['rerank']."""
    retrieve_cfg = _retrieve_cfg(cfg)
    if not candidates or not bool(retrieve_cfg.get("rerank", False)):
        return candidates

    top_n = min(int(retrieve_cfg.get("rerank_top_n", len(candidates))), len(candidates))
    head = candidates[:top_n]
    tail = candidates[top_n:]
    pairs = [(query, chunk.text) for chunk in head]
    raw_scores = [float(score) for score in _model(cfg).predict(pairs)]
    reranked = [
        _copy_with_score(chunk, _sigmoid(score))
        for chunk, score in zip(head, raw_scores, strict=False)
    ]
    reranked.sort(key=lambda chunk: chunk.score, reverse=True)
    logger.info("rerank: scored=%d returned=%d", len(reranked), len(candidates))
    return reranked + tail
