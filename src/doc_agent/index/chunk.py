"""Stage 4 — chunk text"""

from __future__ import annotations

import os
import re
from typing import Any

from ..contracts import *  # noqa
from ..contracts import Chunk

_CHUNK_METADATA: dict[str, dict[str, Any]] = {}
_ADMIN_MARKERS = (
    "deed",
    "registry",
    "registrar",
    "office",
    "book",
    "volume",
    "serial",
    "page",
    "date",
    "no.",
    "number",
)


def _index_cfg(cfg: dict) -> dict:
    return cfg.get("index", {})


def _embedding_cfg(cfg: dict) -> dict:
    return cfg.get("embed", {})


def _configure_cache(emb_cfg: dict) -> None:
    cache_dir = emb_cfg.get("cache_dir")
    if cache_dir:
        os.environ.setdefault("HF_HOME", str(cache_dir))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir))


def _load_tokenizer(cfg: dict) -> Any:
    emb_cfg = _embedding_cfg(cfg)
    _configure_cache(emb_cfg)
    from transformers import AutoTokenizer

    model = emb_cfg.get("model", "intfloat/multilingual-e5-small")
    cache_dir = emb_cfg.get("cache_dir")
    return AutoTokenizer.from_pretrained(model, use_fast=True, cache_dir=cache_dir)


def _token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _paragraph_texts(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"\n\s*\n+", text)
    if len(parts) == 1:
        parts = text.splitlines()
    return [part.strip() for part in parts if part.strip()]


def _admin_like(text: str) -> bool:
    lowered = text.lower()
    if any(marker in lowered for marker in _ADMIN_MARKERS):
        return True
    digits = sum(char.isdigit() for char in text)
    separators = text.count(":") + text.count("/") + text.count("-")
    return len(text) <= 120 and separators > 0 and digits > 0


def _same_boundary_kind(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return _admin_like(str(left["text"])) == _admin_like(str(right["text"]))


def _offsets(tokenizer: Any, text: str) -> list[tuple[int, int]]:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return [(int(start), int(end)) for start, end in encoded["offset_mapping"]]


def _safe_window_tokens(tokenizer: Any, cfg: dict) -> int:
    index_cfg = _index_cfg(cfg)
    emb_cfg = _embedding_cfg(cfg)
    prefix = emb_cfg.get("document_prefix", "passage: ")
    prefix_tokens = _token_count(tokenizer, prefix)
    max_model_tokens = int(emb_cfg.get("max_model_tokens", 512))
    hard_max = int(index_cfg.get("hard_max_tokens", index_cfg.get("chunk_tokens", 450)))
    return max(1, min(hard_max, max_model_tokens - prefix_tokens - 2))


def _make_chunk(
    source: Chunk,
    chunk_index: int,
    text: str,
    paragraph_indices: list[int],
    block_order_indices: list[int],
    token_count: int,
    boundary_type: str,
    extra: dict[str, Any] | None = None,
) -> Chunk:
    page_id = source.page_ids[0] if source.page_ids else source.id
    chunk_id = f"{page_id}#c{chunk_index:04d}"
    meta = {
        "chunk_id": chunk_id,
        "page_id": page_id,
        "page_ids": list(source.page_ids),
        "doc_id": source.doc_id,
        "chunk_index": chunk_index,
        "paragraph_indices": paragraph_indices,
        "block_order_indices": block_order_indices,
        "chunk_text": text,
        "token_count": token_count,
        "boundary_type": boundary_type,
    }
    if extra:
        meta.update(extra)
    _CHUNK_METADATA[chunk_id] = meta
    return Chunk(id=chunk_id, doc_id=source.doc_id, text=text, page_ids=list(source.page_ids))


def _split_oversized(
    source: Chunk,
    paragraph: dict[str, Any],
    start_chunk_index: int,
    tokenizer: Any,
    cfg: dict,
) -> list[Chunk]:
    index_cfg = _index_cfg(cfg)
    chunk_size = min(
        int(index_cfg.get("chunk_tokens", index_cfg.get("hard_max_tokens", 450))),
        _safe_window_tokens(tokenizer, cfg),
    )
    overlap = int(index_cfg.get("overlap", index_cfg.get("split_overlap_tokens", 40)))
    overlap = max(0, min(overlap, chunk_size - 1))
    stride = chunk_size - overlap
    paragraph_text = str(paragraph["text"])
    paragraph_idx = int(paragraph["paragraph_idx"])
    block_order_idx = int(paragraph["block_order_idx"])
    offsets = _offsets(tokenizer, paragraph_text)
    chunks: list[Chunk] = []
    start = 0
    split_part_idx = 0
    while start < len(offsets):
        end = min(start + chunk_size, len(offsets))
        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]
        text = paragraph_text[char_start:char_end].strip()
        if text:
            chunks.append(
                _make_chunk(
                    source,
                    start_chunk_index + len(chunks),
                    text,
                    [paragraph_idx],
                    [block_order_idx],
                    end - start,
                    "token_window",
                    {
                        "split_part_idx": split_part_idx,
                        "token_start": start,
                        "token_end": end,
                    },
                )
            )
        if end >= len(offsets):
            break
        start += stride
        split_part_idx += 1
    return chunks


def metadata_for(chunk_id: str) -> dict[str, Any] | None:
    return _CHUNK_METADATA.get(chunk_id)


def _flush_pending(owner: Chunk, pending: list[dict[str, Any]], out: list[Chunk]) -> None:
    if not pending:
        return
    text = "\n\n".join(item["text"] for item in pending)
    out.append(
        _make_chunk(
            owner,
            len(out),
            text,
            [int(item["paragraph_idx"]) for item in pending],
            [int(item["block_order_idx"]) for item in pending],
            sum(int(item["token_count"]) for item in pending),
            "merged_paragraphs" if len(pending) > 1 else "layout_paragraph",
        )
    )
    pending.clear()


def split(chunks: list[Chunk], cfg: dict) -> list[Chunk]:
    """Re-chunk to cfg['index'] size/overlap. IMPLEMENT."""
    tokenizer = _load_tokenizer(cfg)
    index_cfg = _index_cfg(cfg)
    merge_below = int(index_cfg.get("merge_below_tokens", 80))
    target_max = int(index_cfg.get("target_max_tokens", 300))
    preferred_max = min(
        int(index_cfg.get("preferred_max_tokens", 350)),
        _safe_window_tokens(tokenizer, cfg),
    )
    hard_max = min(int(index_cfg.get("hard_max_tokens", 450)), _safe_window_tokens(tokenizer, cfg))
    out: list[Chunk] = []
    _CHUNK_METADATA.clear()

    for source in chunks:
        page_id = source.page_ids[0] if source.page_ids else source.id
        paragraphs = [
            {
                "page_id": page_id,
                "paragraph_idx": idx,
                "block_order_idx": idx,
                "text": text,
                "token_count": _token_count(tokenizer, text),
            }
            for idx, text in enumerate(_paragraph_texts(source.text))
        ]
        pending: list[dict[str, Any]] = []

        for paragraph in paragraphs:
            token_count = int(str(paragraph["token_count"]))
            paragraph_text = str(paragraph["text"])
            paragraph_idx = int(str(paragraph["paragraph_idx"]))
            block_order_idx = int(str(paragraph["block_order_idx"]))
            if token_count > hard_max:
                _flush_pending(source, pending, out)
                out.extend(_split_oversized(source, paragraph, len(out), tokenizer, cfg))
                continue
            if token_count >= merge_below or token_count > preferred_max:
                _flush_pending(source, pending, out)
                out.append(
                    _make_chunk(
                        source,
                        len(out),
                        paragraph_text,
                        [paragraph_idx],
                        [block_order_idx],
                        token_count,
                        "layout_paragraph",
                    )
                )
                continue
            pending_total = sum(int(item["token_count"]) for item in pending)
            if pending and not _same_boundary_kind(pending[-1], paragraph):
                _flush_pending(source, pending, out)
                pending_total = 0
            if pending and pending_total + token_count > target_max:
                _flush_pending(source, pending, out)
            pending.append(paragraph)
        _flush_pending(source, pending, out)

    return out
