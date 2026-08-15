"""Stage 4 — vector store"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts import *  # noqa
from ..contracts import Chunk


def _index_cfg(cfg: dict) -> dict:
    return cfg.get("index", {})


def _embedding_cfg(cfg: dict) -> dict:
    return cfg.get("embed", {})


def _index_dir(cfg: dict) -> Path:
    return Path(_index_cfg(cfg).get("path", "data/interim/index"))


def _index_path(cfg: dict) -> Path:
    return _index_dir(cfg) / "chunks.faiss"


def _metadata_path(cfg: dict) -> Path:
    return _index_dir(cfg) / "chunks.jsonl"


def _as_matrix(vectors: Any, dim: int) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("vectors must be a 2D matrix")
    if matrix.shape[1] != dim:
        raise ValueError(f"vector dimension {matrix.shape[1]} does not match configured {dim}")
    return matrix


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype(np.float32, copy=False)


def _fallback_metadata(chunk: Chunk, chunk_index: int) -> dict[str, Any]:
    page_id = chunk.page_ids[0] if chunk.page_ids else ""
    return {
        "chunk_id": chunk.id,
        "page_id": page_id,
        "page_ids": list(chunk.page_ids),
        "doc_id": chunk.doc_id,
        "chunk_index": chunk_index,
        "paragraph_indices": [],
        "block_order_indices": [],
        "chunk_text": chunk.text,
        "token_count": 0,
        "boundary_type": "unknown",
    }


def _metadata(chunks: list[Chunk]) -> list[dict[str, Any]]:
    from . import chunk as chunking

    records = []
    for chunk_index, item in enumerate(chunks):
        record = chunking.metadata_for(item.id) or _fallback_metadata(item, chunk_index)
        record["chunk_index"] = chunk_index
        records.append(record)
    return records


def build(chunks, vectors, cfg: dict) -> None:  # type: ignore[no-untyped-def]
    """Persist a vector index (cfg['index']['type']). IMPLEMENT."""
    import faiss

    index_cfg = _index_cfg(cfg)
    if index_cfg.get("type", "faiss:flat") != "faiss:flat":
        raise ValueError("only faiss:flat is supported")
    dim = int(_embedding_cfg(cfg).get("dimension", _embedding_cfg(cfg).get("dim", 384)))
    matrix = _as_matrix(vectors, dim)
    if bool(_embedding_cfg(cfg).get("normalize_embeddings", True)):
        matrix = _normalize(matrix)

    out_dir = _index_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)
    faiss.write_index(index, str(_index_path(cfg)))

    with _metadata_path(cfg).open("w", encoding="utf-8") as fh:
        for record in _metadata(chunks):
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return None
    raise NotImplementedError("Stage 4: build index")


def load(cfg: dict):  # type: ignore[no-untyped-def]
    import faiss

    index = faiss.read_index(str(_index_path(cfg)))
    metadata = []
    with _metadata_path(cfg).open(encoding="utf-8") as fh:
        metadata = [json.loads(line) for line in fh if line.strip()]
    chunks = [
        Chunk(
            id=record["chunk_id"],
            doc_id=record.get("doc_id", ""),
            text=record["chunk_text"],
            page_ids=record.get("page_ids", [record["page_id"]]),
        )
        for record in metadata
    ]
    return {"index": index, "metadata": metadata, "chunks": chunks}
