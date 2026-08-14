"""Stage 4 — embed chunks"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from ..contracts import *  # noqa
from ..contracts import Chunk


def _embedding_cfg(cfg: dict) -> dict:
    return cfg.get("embed", {})


def _configure_cache(emb_cfg: dict) -> None:
    cache_dir = emb_cfg.get("cache_dir")
    if cache_dir:
        os.environ.setdefault("HF_HOME", str(cache_dir))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir))


def _model(cfg: dict) -> Any:
    emb_cfg = _embedding_cfg(cfg)
    _configure_cache(emb_cfg)
    from sentence_transformers import SentenceTransformer

    name = emb_cfg.get("model", "intfloat/multilingual-e5-small")
    device = emb_cfg.get("device", cfg.get("device", "cpu"))
    return SentenceTransformer(name, device=device, cache_folder=emb_cfg.get("cache_dir"))


def _as_float32(vectors: Any) -> np.ndarray:
    return np.asarray(vectors, dtype=np.float32)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vectors / norms


def _encode_texts(texts: list[str], cfg: dict, prefix_key: str) -> np.ndarray:
    emb_cfg = _embedding_cfg(cfg)
    dim = int(emb_cfg.get("dimension", emb_cfg.get("dim", 384)))
    if not texts:
        return np.zeros((0, dim), dtype=np.float32)

    prefix = emb_cfg.get(prefix_key, "passage: " if prefix_key == "document_prefix" else "query: ")
    model = _model(cfg)
    vectors = _as_float32(
        model.encode(
            [prefix + text for text in texts],
            batch_size=int(emb_cfg.get("initial_batch_size", 8)),
            convert_to_numpy=True,
            normalize_embeddings=bool(emb_cfg.get("normalize_embeddings", True)),
            show_progress_bar=False,
        )
    )
    if vectors.shape[1] != dim:
        raise ValueError(f"embedding dimension {vectors.shape[1]} does not match configured {dim}")
    if bool(emb_cfg.get("normalize_embeddings", True)):
        vectors = _normalize(vectors).astype(np.float32, copy=False)
    return vectors


def encode_query(query: str, cfg: dict) -> np.ndarray:
    return _encode_texts([query], cfg, "query_prefix")


def encode(chunks: list[Chunk], cfg: dict):  # type: ignore[no-untyped-def]
    """Embed with cfg['embed']['model']. IMPLEMENT."""
    return _encode_texts([chunk.text for chunk in chunks], cfg, "document_prefix")
    raise NotImplementedError("Stage 4: embed")
