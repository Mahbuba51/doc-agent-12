"""Stage 4 — chunk text"""

from __future__ import annotations

import os
import re
import unicodedata
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
    "registered",
    "registration",
    "sub-registrar",
    "dist.",
    "section",
    "rule",
    "act",
    "stamp",
    "stamped",
    "duty",
    "sd/-",
    "tk",
    "taka",
    "s. r.",
    "\u09a6\u09b2\u09bf\u09b2",  # dolil/deed
    "\u09a8\u0982",  # no.
    "\u09a8\u0995\u09b2",  # certified copy/copy
    "\u09b0\u09c7\u099c\u09bf\u09b8\u09cd\u099f\u09cd\u09b0\u09bf",  # registry
    "\u09b0\u09c7\u099c\u09bf\u09b7\u09cd\u099f\u09cd\u09b0\u09bf",  # registry spelling variant
    "\u09b0\u09c7\u099c\u09bf\u09b8\u09cd\u099f\u09cd\u09b0\u09be\u09b0",  # registrar
    "\u09b0\u09c7\u099c\u09bf\u09b7\u09cd\u099f\u09cd\u09b0\u09be\u09b0",  # registrar spelling variant
    "\u09b0\u09c7\u099c\u09bf\u0983",  # abbreviated registry
    "\u09b8\u09be\u09ac",  # sub
    "\u09b8\u09a6\u09b0",  # sadar/head office
    "\u09ac\u09b9\u09bf",  # book
    "\u09ad\u09b2\u09bf\u09af\u09bc\u09be\u09ae",  # volume
    "\u09ad\u09b2\u09bf\u09df\u09be\u09ae",  # volume, precomposed ya variant
    "\u09aa\u09c3\u09b7\u09cd\u09a0\u09be",  # page
    "\u09aa\u09c3\u09b7\u09cd\u09a0\u09be\u09b0",  # page, inflected
    "\u09a4\u09be\u09b0\u09bf\u0996",  # date
    "\u09b8\u09a8",  # year
    "\u09b8\u09a8\u09c7\u09b0",  # year's
    "\u09ae\u09c2\u09b2\u09cd\u09af",  # value/price
    "\u099f\u09be\u0995\u09be",  # taka
    "\u09b8\u09cd\u099f\u09cd\u09af\u09be\u09ae\u09cd\u09aa",  # stamp
    "\u09ae\u09cc\u099c\u09be",  # mouza
    "\u09a5\u09be\u09a8\u09be",  # thana
    "\u099c\u09c7\u09b2\u09be",  # district
    "\u099c\u09bf\u09b2\u09be",  # district spelling variant
    "\u09a6\u09be\u0997",  # plot/dag
    "\u0996\u09a4\u09bf\u09af\u09bc\u09be\u09a8",  # khatian
    "\u0996\u09a4\u09bf\u09df\u09be\u09a8",  # khatian, precomposed ya variant
    "\u09a4\u09ab\u09b8\u09bf\u09b2",  # property schedule
    "\u09a6\u09be\u09a4\u09be",  # grantor
    "\u09a6\u09be\u09a4\u09cd\u09b0\u09c0",  # female grantor
    "\u0997\u09cd\u09b0\u09b9\u09bf\u09a4\u09be",  # grantee
    "\u0997\u09cd\u09b0\u09b9\u09c0\u09a4\u09be",  # grantee spelling variant
    "\u09b6\u09aa\u09a5",  # affidavit/oath
    "\u09b9\u09c7\u09ac\u09be",  # heba deed type
)
_BANGLA_DIGITS = str.maketrans("\u09e6\u09e7\u09e8\u09e9\u09ea\u09eb\u09ec\u09ed\u09ee\u09ef", "0123456789")
_BANGLA_BLOCK = "\u0980-\u09ff"
_TOKEN_RE = re.compile(
    rf"[0-9]+(?:[/-][0-9]+)*|[A-Za-z]+(?:[.'-][A-Za-z]+)*|[{_BANGLA_BLOCK}]+"
)
_DATE_RE = re.compile(r"\b[0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4}\b")
_DIGIT_RE = re.compile(r"\d+")
_AMOUNT_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?(?:/\d+)?\b"
    r"(?=\s*(?:/-|tk|taka|rs|bdt|\u099f\u09be\u0995\u09be))",
    re.IGNORECASE,
)
_VARIANT_MAP: dict[str, tuple[str, ...]] = {
    "dist": ("district",),
    "no": ("number",),
    "no.": ("number",),
    "s.r": ("sub-registrar",),
    "s.": ("sub",),
    "r.": ("registrar",),
    "sd": ("signature",),
    "\u09a8\u0982": ("\u09a8\u09ae\u09cd\u09ac\u09b0",),
    "\u09a8\u0982\u0983": ("\u09a8\u09ae\u09cd\u09ac\u09b0",),
    "\u09a8\u0982-": ("\u09a8\u09ae\u09cd\u09ac\u09b0",),
    "\u09b0\u09c7\u099c\u09bf\u09b7\u09cd\u099f\u09cd\u09b0\u09bf": (
        "\u09b0\u09c7\u099c\u09bf\u09b8\u09cd\u099f\u09cd\u09b0\u09bf",
    ),
    "\u09b0\u09c7\u099c\u09bf\u09b7\u09cd\u099f\u09cd\u09b0\u09be\u09b0": (
        "\u09b0\u09c7\u099c\u09bf\u09b8\u09cd\u099f\u09cd\u09b0\u09be\u09b0",
    ),
    "\u09b0\u09c7\u099c\u09bf\u09b8\u09cd\u099f\u09cd\u09b0\u09be\u09b0\u09c0": (
        "\u09b0\u09c7\u099c\u09bf\u09b8\u09cd\u099f\u09cd\u09b0\u09bf",
        "\u09b0\u09c7\u099c\u09bf\u09b8\u09cd\u099f\u09cd\u09b0\u09be\u09b0",
    ),
    "\u09b0\u09c7\u099c\u09bf\u09b7\u09cd\u099f\u09cd\u09b0\u09be\u09b0\u09c0": (
        "\u09b0\u09c7\u099c\u09bf\u09b8\u09cd\u099f\u09cd\u09b0\u09bf",
        "\u09b0\u09c7\u099c\u09bf\u09b8\u09cd\u099f\u09cd\u09b0\u09be\u09b0",
    ),
    "\u09b0\u09c7\u099c\u09bf\u0983": (
        "\u09b0\u09c7\u099c\u09bf\u09b8\u09cd\u099f\u09cd\u09b0\u09bf",
    ),
    "\u09ad\u09b2\u09bf\u09df\u09be\u09ae": ("\u09ad\u09b2\u09bf\u09af\u09bc\u09be\u09ae",),
    "\u0996\u09a4\u09bf\u09df\u09be\u09a8": ("\u0996\u09a4\u09bf\u09af\u09bc\u09be\u09a8",),
    "\u0997\u09cd\u09b0\u09b9\u09c0\u09a4\u09be": ("\u0997\u09cd\u09b0\u09b9\u09bf\u09a4\u09be",),
    "\u099c\u09bf\u09b2\u09be": ("\u099c\u09c7\u09b2\u09be",),
}
_COMPOUND_MAP: dict[str, tuple[str, ...]] = {
    "sub-registrar": ("sub", "registrar"),
    "\u09b8\u09be\u09ac\u09b0\u09c7\u099c\u09bf\u09b8\u09cd\u099f\u09cd\u09b0\u09be\u09b0": (
        "\u09b8\u09be\u09ac",
        "\u09b0\u09c7\u099c\u09bf\u09b8\u09cd\u099f\u09cd\u09b0\u09be\u09b0",
    ),
    "\u09b8\u09be\u09ac\u09b0\u09c7\u099c\u09bf\u09b7\u09cd\u099f\u09cd\u09b0\u09be\u09b0": (
        "\u09b8\u09be\u09ac",
        "\u09b0\u09c7\u099c\u09bf\u09b8\u09cd\u099f\u09cd\u09b0\u09be\u09b0",
    ),
    "\u09b8\u09be\u09ac-\u09b0\u09c7\u099c\u09bf\u09b8\u09cd\u099f\u09cd\u09b0\u09be\u09b0\u09c0": (
        "\u09b8\u09be\u09ac",
        "\u09b0\u09c7\u099c\u09bf\u09b8\u09cd\u099f\u09cd\u09b0\u09bf",
    ),
    "\u09b8\u09cd\u099f\u09cd\u09af\u09be\u09ae\u09cd\u09aa\u09a1\u09bf\u0989\u099f\u09bf": (
        "\u09b8\u09cd\u099f\u09cd\u09af\u09be\u09ae\u09cd\u09aa",
        "duty",
    ),
    "\u09a6\u09b2\u09bf\u09b2\u09a6\u09be\u09a4\u09be": (
        "\u09a6\u09b2\u09bf\u09b2",
        "\u09a6\u09be\u09a4\u09be",
    ),
    "\u09a6\u09b2\u09bf\u09b2\u0997\u09cd\u09b0\u09b9\u09bf\u09a4\u09be": (
        "\u09a6\u09b2\u09bf\u09b2",
        "\u0997\u09cd\u09b0\u09b9\u09bf\u09a4\u09be",
    ),
}
_LEGAL_ROOTS = {
    "\u09a6\u09b2\u09bf\u09b2",
    "\u09a8\u09ae\u09cd\u09ac\u09b0",
    "\u09a8\u0982",
    "\u09a8\u0995\u09b2",
    "\u09b0\u09c7\u099c\u09bf\u09b8\u09cd\u099f\u09cd\u09b0\u09bf",
    "\u09b0\u09c7\u099c\u09bf\u09b8\u09cd\u099f\u09cd\u09b0\u09be\u09b0",
    "\u09b8\u09be\u09ac",
    "\u09b8\u09a6\u09b0",
    "\u09ac\u09b9\u09bf",
    "\u09ad\u09b2\u09bf\u09af\u09bc\u09be\u09ae",
    "\u09aa\u09c3\u09b7\u09cd\u09a0\u09be",
    "\u09a4\u09be\u09b0\u09bf\u0996",
    "\u09b8\u09a8",
    "\u09ae\u09c2\u09b2\u09cd\u09af",
    "\u099f\u09be\u0995\u09be",
    "\u09b8\u09cd\u099f\u09cd\u09af\u09be\u09ae\u09cd\u09aa",
    "\u09ae\u09cc\u099c\u09be",
    "\u09a5\u09be\u09a8\u09be",
    "\u099c\u09c7\u09b2\u09be",
    "\u09a6\u09be\u0997",
    "\u0996\u09a4\u09bf\u09af\u09bc\u09be\u09a8",
    "\u09a4\u09ab\u09b8\u09bf\u09b2",
    "\u09a6\u09be\u09a4\u09be",
    "\u09a6\u09be\u09a4\u09cd\u09b0\u09c0",
    "\u0997\u09cd\u09b0\u09b9\u09bf\u09a4\u09be",
    "\u09b6\u09aa\u09a5",
    "\u09b9\u09c7\u09ac\u09be",
}
_BANGLA_SUFFIXES = (
    "\u0997\u09c1\u09b2\u09cb",
    "\u0997\u09c1\u09b2\u09bf",
    "\u09a6\u09c7\u09b0",
    "\u09af\u09bc\u09c7\u09b0",
    "\u09df\u09c7\u09b0",
    "\u098f\u09b0",
    "\u09c7\u09b0",
    "\u09a4\u09c7",
    "\u09c7",
    "\u09b0",
)


def _index_cfg(cfg: dict) -> dict:
    return cfg.get("index", {})


def _embedding_cfg(cfg: dict) -> dict:
    return cfg.get("embed", {})


def _normalize_for_sparse(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_BANGLA_DIGITS)
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()


def _sparse_tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(_normalize_for_sparse(text))


def _stem_expansions(token: str) -> list[str]:
    stems: list[str] = []
    for suffix in _BANGLA_SUFFIXES:
        if not token.endswith(suffix) or len(token) <= len(suffix) + 2:
            continue
        stem = token[: -len(suffix)]
        if stem in _LEGAL_ROOTS:
            stems.append(stem)
    return stems


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _expanded_sparse_tokens(text: str) -> list[str]:
    expanded: list[str] = []
    pending = list(_sparse_tokens(text))
    while pending:
        token = pending.pop(0)
        before = len(expanded)
        expanded.append(token)
        expanded.extend(_VARIANT_MAP.get(token, ()))
        expanded.extend(_COMPOUND_MAP.get(token, ()))
        expanded.extend(_stem_expansions(token))

        for new_token in expanded[before + 1 :]:
            if new_token not in pending and new_token not in expanded[:before]:
                pending.append(new_token)

    return _dedupe_preserve_order(expanded)


def _char_ngrams(tokens: list[str], n: int = 3) -> list[str]:
    grams: list[str] = []
    for token in tokens:
        if token.isdigit() or len(token) < n + 1:
            continue
        grams.extend(token[index : index + n] for index in range(len(token) - n + 1))
    return _dedupe_preserve_order(grams)


def _digit_runs(text: str) -> list[str]:
    return _dedupe_preserve_order(_DIGIT_RE.findall(_normalize_for_sparse(text)))


def _date_candidates(text: str) -> list[str]:
    return _dedupe_preserve_order(_DATE_RE.findall(_normalize_for_sparse(text)))


def _amount_candidates(text: str) -> list[str]:
    return _dedupe_preserve_order(_AMOUNT_RE.findall(_normalize_for_sparse(text)))


def _identifier_candidates(text: str) -> list[str]:
    normalized = _normalize_for_sparse(text)
    identifiers = _DIGIT_RE.findall(normalized)
    identifiers.extend(_DATE_RE.findall(normalized))
    return _dedupe_preserve_order(identifiers)


def _sparse_metadata(text: str) -> dict[str, Any]:
    normalized = _normalize_for_sparse(text)
    tokens = _sparse_tokens(text)
    expanded_tokens = _expanded_sparse_tokens(text)
    char_3grams = _char_ngrams(expanded_tokens)
    return {
        "normalized_text": normalized,
        "sparse_tokens": tokens,
        "sparse_token_count": len(tokens),
        "expanded_sparse_tokens": expanded_tokens,
        "expanded_sparse_token_count": len(expanded_tokens),
        "sparse_char_3grams": char_3grams,
        "bm25_tokens": expanded_tokens + [f"char:{gram}" for gram in char_3grams],
        "bm25_token_count": len(expanded_tokens) + len(char_3grams),
        "digit_runs": _digit_runs(text),
        "date_candidates": _date_candidates(text),
        "amount_candidates": _amount_candidates(text),
        "identifier_candidates": _identifier_candidates(text),
        "admin_like": _admin_like(text),
        "has_digits": any(char.isdigit() for char in normalized),
        "has_bangla": bool(re.search(rf"[{_BANGLA_BLOCK}]", normalized)),
        "has_english": bool(re.search(r"[a-z]", normalized)),
    }


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
    if len(text.strip()) <= 40 and digits >= 4:
        return True
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
        **_sparse_metadata(text),
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
