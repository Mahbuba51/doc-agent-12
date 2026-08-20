"""Stage 5 - hybrid dense/BM25 retrieval."""

from __future__ import annotations

import hashlib
import json
import pickle
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts import Chunk
from ..index import embed, store
from ..logging_conf import get_logger

logger = get_logger(__name__)
_BANGLA_DIGITS = str.maketrans(
    "\u09e6\u09e7\u09e8\u09e9\u09ea\u09eb\u09ec\u09ed\u09ee\u09ef",
    "0123456789",
)
_TOKEN_RE = re.compile(r"[0-9]+(?:[/-][0-9]+)*|[A-Za-z]+(?:[.'-][A-Za-z]+)*|[\u0980-\u09ff]+")
_SPARSE_SCHEMA_VERSION = 1


def _retrieve_cfg(cfg: dict) -> dict:
    return cfg.get("retrieve", {})


def _index_cfg(cfg: dict) -> dict:
    return cfg.get("index", {})


def _metadata_path(cfg: dict) -> Path:
    return Path(_index_cfg(cfg).get("path", "data/interim/index")) / "chunks.jsonl"


def _bm25_cache_paths(cfg: dict) -> tuple[Path, Path]:
    index_dir = Path(_index_cfg(cfg).get("path", "data/interim/index"))
    name = str(_retrieve_cfg(cfg).get("bm25_cache_name", "bm25"))
    return index_dir / f"{name}.pkl", index_dir / f"{name}.meta.json"


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _as_tokens(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return []


def _fallback_tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFC", text).translate(_BANGLA_DIGITS).lower()
    return _TOKEN_RE.findall(re.sub(r"\s+", " ", normalized).strip())


def _query_tokens(query: str) -> list[str]:
    """Mirror index.chunk's sparse schema for BM25 query-time tokens."""
    from ..index import chunk as chunking

    expanded = chunking._expanded_sparse_tokens(query)  # type: ignore[attr-defined]
    grams = chunking._char_ngrams(expanded)  # type: ignore[attr-defined]
    return expanded + [f"char:{gram}" for gram in grams]


def _copy_with_score(chunk: Chunk, score: float) -> Chunk:
    return chunk.model_copy(update={"score": float(score)})


class Retriever:
    def __init__(self, cfg: dict) -> None:
        self.full_cfg = cfg
        self.cfg = cfg["retrieve"]
        self._loaded: dict[str, Any] | None = None
        self._bm25: Any | None = None
        self._bm25_cache_id: str | None = None

    def _load_index(self) -> dict[str, Any]:
        if self._loaded is None:
            self._loaded = store.load(self.full_cfg)
            logger.info("retrieval index loaded: %d chunk(s)", len(self._loaded["chunks"]))
        return self._loaded

    def _bm25_corpus(self, metadata: list[dict[str, Any]]) -> list[list[str]]:
        token_field = str(self.cfg.get("bm25_token_field", "bm25_tokens"))
        corpus = []
        for record in metadata:
            tokens = _as_tokens(record.get(token_field))
            if not tokens:
                tokens = _fallback_tokens(str(record.get("chunk_text", "")))
            corpus.append(tokens)
        return corpus

    def _bm25_meta(self, metadata_hash: str, record_count: int) -> dict[str, Any]:
        return {
            "metadata_sha256": metadata_hash,
            "record_count": record_count,
            "token_field": str(self.cfg.get("bm25_token_field", "bm25_tokens")),
            "bm25_class": "BM25Okapi",
            "k1": float(self.cfg.get("bm25_k1", 1.5)),
            "b": float(self.cfg.get("bm25_b", 0.75)),
            "epsilon": float(self.cfg.get("bm25_epsilon", 0.25)),
            "sparse_schema_version": int(
                _index_cfg(self.full_cfg).get("sparse_schema_version", _SPARSE_SCHEMA_VERSION)
            ),
        }

    def _load_or_build_bm25(self) -> Any:
        if not bool(self.cfg.get("bm25_enabled", True)):
            return None

        loaded = self._load_index()
        metadata = loaded["metadata"]
        metadata_hash = _sha256(_metadata_path(self.full_cfg))
        expected_meta = self._bm25_meta(metadata_hash, len(metadata))
        cache_id = json.dumps(expected_meta, sort_keys=True)
        if self._bm25 is not None and self._bm25_cache_id == cache_id:
            return self._bm25

        cache_path, meta_path = _bm25_cache_paths(self.full_cfg)
        if bool(self.cfg.get("bm25_cache", True)) and cache_path.is_file() and meta_path.is_file():
            cached_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if cached_meta == expected_meta:
                with cache_path.open("rb") as fh:
                    self._bm25 = pickle.load(fh)
                self._bm25_cache_id = cache_id
                logger.info("bm25 cache hit: %s", cache_path)
                return self._bm25

        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RuntimeError(
                "BM25 retrieval is enabled but rank-bm25 is not installed. "
                "Install project dependencies after the pyproject.toml update."
            ) from exc

        corpus = self._bm25_corpus(metadata)
        self._bm25 = BM25Okapi(
            corpus,
            k1=float(self.cfg.get("bm25_k1", 1.5)),
            b=float(self.cfg.get("bm25_b", 0.75)),
            epsilon=float(self.cfg.get("bm25_epsilon", 0.25)),
        )
        self._bm25_cache_id = cache_id
        if bool(self.cfg.get("bm25_cache", True)):
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("wb") as fh:
                pickle.dump(self._bm25, fh)
            meta_path.write_text(
                json.dumps(expected_meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info("bm25 cache rebuilt: %s", cache_path)
        else:
            logger.info("bm25 built in memory: %d document(s)", len(corpus))
        return self._bm25

    def _effective_k(self, k: int | None, n_chunks: int) -> int:
        requested = int(k if k is not None else self.cfg.get("k", 10))
        return max(0, min(requested, n_chunks))

    def _candidate_pool(self, k: int, n_chunks: int) -> int:
        multiplier = int(self.cfg.get("candidate_multiplier", 4))
        configured = int(self.cfg.get("candidate_pool", k * multiplier))
        return max(k, min(max(configured, k * multiplier), n_chunks))

    def _dense_search(self, query: str, search_k: int) -> list[tuple[int, float]]:
        if search_k <= 0:
            return []
        loaded = self._load_index()
        qvec = np.asarray(embed.encode_query(query, self.full_cfg), dtype=np.float32)
        scores, indices = loaded["index"].search(qvec, search_k)
        hits = []
        for idx, score in zip(indices[0].tolist(), scores[0].tolist(), strict=False):
            if idx >= 0:
                hits.append((int(idx), float(score)))
        return hits

    def _bm25_search(self, query: str, search_k: int) -> list[tuple[int, float]]:
        bm25 = self._load_or_build_bm25()
        if bm25 is None or search_k <= 0:
            return []
        scores = np.asarray(bm25.get_scores(_query_tokens(query)), dtype=np.float32)
        if scores.size == 0:
            return []
        top = np.argsort(scores)[::-1]
        hits = [(int(idx), float(scores[idx])) for idx in top if scores[idx] > 0.0]
        return hits[:search_k]

    def _rrf_merge(
        self,
        dense_hits: list[tuple[int, float]],
        bm25_hits: list[tuple[int, float]],
        limit: int,
    ) -> list[tuple[int, float]]:
        rrf_k = float(self.cfg.get("rrf_k", 60))
        dense_weight = float(self.cfg.get("dense_weight", 0.65))
        bm25_weight = float(self.cfg.get("bm25_weight", 0.35))
        fused: dict[int, float] = {}

        for rank, (idx, _) in enumerate(dense_hits, start=1):
            fused[idx] = fused.get(idx, 0.0) + dense_weight / (rrf_k + rank)
        for rank, (idx, _) in enumerate(bm25_hits, start=1):
            fused[idx] = fused.get(idx, 0.0) + bm25_weight / (rrf_k + rank)

        return sorted(fused.items(), key=lambda item: item[1], reverse=True)[:limit]

    def _evidence_scores(
        self,
        ranked: list[tuple[int, float]],
        dense_hits: list[tuple[int, float]],
        bm25_hits: list[tuple[int, float]],
    ) -> dict[int, float]:
        dense = {idx: max(0.0, score) for idx, score in dense_hits}
        bm25 = {idx: max(0.0, score) for idx, score in bm25_hits}
        max_bm25 = max(bm25.values(), default=0.0) or 1.0
        dense_weight = float(self.cfg.get("dense_weight", 0.65))
        bm25_weight = float(self.cfg.get("bm25_weight", 0.35))

        out = {}
        for idx, _ in ranked:
            bm25_norm = bm25.get(idx, 0.0) / max_bm25
            out[idx] = min(1.0, dense_weight * dense.get(idx, 0.0) + bm25_weight * bm25_norm)
        return out

    def retrieve(self, query: str, k: int | None = None) -> list[Chunk]:
        """Top-k dense retrieval. Set chunk.score (relevance) on every result so decide() can judge
        whether the evidence is weak."""
        from .rerank import rerank

        loaded = self._load_index()
        n_chunks = len(loaded["chunks"])
        top_k = self._effective_k(k, n_chunks)
        if top_k == 0:
            return []

        search_k = self._candidate_pool(top_k, n_chunks)
        dense_hits = self._dense_search(query, search_k)
        bm25_hits = self._bm25_search(query, search_k)
        ranked = self._rrf_merge(dense_hits, bm25_hits, search_k)
        if not ranked:
            ranked = dense_hits[:search_k] or bm25_hits[:search_k]

        evidence = self._evidence_scores(ranked, dense_hits, bm25_hits)
        candidates = [
            _copy_with_score(loaded["chunks"][idx], evidence.get(idx, score))
            for idx, score in ranked
        ]
        if bool(self.cfg.get("rerank", False)):
            candidates = rerank(query, candidates, self.full_cfg)

        results = candidates[:top_k]
        logger.info(
            "retrieve: k=%d pool=%d dense=%d bm25=%d returned=%d top_score=%.4f",
            top_k,
            search_k,
            len(dense_hits),
            len(bm25_hits),
            len(results),
            top_score(results),
        )
        return results


# --- evidence-strength policy: read by agent.decide() for evidence-gated re-search ---
def top_score(chunks: list[Chunk]) -> float:
    """Strength of the current evidence = best chunk score (0.0 if empty)."""
    return max((c.score for c in chunks), default=0.0)


def is_weak(chunks: list[Chunk], cfg: dict) -> bool:
    """Weak evidence = best score below cfg.retrieve.weak_threshold."""
    return top_score(chunks) < cfg["retrieve"]["weak_threshold"]


def next_k(k: int, cfg: dict) -> int | None:
    """Widen the net: k + k_step, or None once it would exceed k_max (signal to ABSTAIN)."""
    nk = k + cfg["retrieve"]["k_step"]
    return nk if nk <= cfg["retrieve"]["k_max"] else None
