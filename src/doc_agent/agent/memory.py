"""Stage 6 — working/episodic memory"""
from __future__ import annotations
from ..contracts import *  # noqa

class Memory:
    def __init__(self) -> None:
        self.items: list = []
    def add(self, item) -> None:
        self.items.append(item)
    def recall(self, query: str) -> list:
        import re

        query_tokens = set(re.findall(r"\w+", query.casefold()))
        if not query_tokens:
            return list(self.items)

        ranked = []
        for position, item in enumerate(self.items):
            payload = item.payload if isinstance(item, ToolResult) else item
            text_parts = []
            if isinstance(payload, dict):
                text_parts.append(str(payload.get("query", "")))
                text_parts.extend(str(chunk.text) for chunk in payload.get("chunks", []))
            else:
                text_parts.append(str(payload))

            item_tokens = set(re.findall(r"\w+", " ".join(text_parts).casefold()))
            overlap = len(query_tokens & item_tokens)
            if overlap:
                ranked.append((overlap, position, item))

        ranked.sort(key=lambda entry: (-entry[0], entry[1]))
        return [item for _, _, item in ranked]

