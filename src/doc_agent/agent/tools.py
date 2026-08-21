"""Stage 6 — FIXED tool interface — the agent's tools"""
from __future__ import annotations
from ..contracts import *  # noqa

from abc import ABC, abstractmethod

class Tool(ABC):
    name: str
    @abstractmethod
    def __call__(self, **kwargs) -> ToolResult: ...

# FIXED tool set — names & signatures locked (test_tools.py checks these).
class Retrieve(Tool):
    name = "retrieve"
    def __call__(self, query: str, k: int = 10) -> ToolResult:
        """Retrieve evidence and expose the values used by the agent's re-search policy."""
        chunks = self.retriever.retrieve(query, k)
        best_score = max((chunk.score for chunk in chunks), default=0.0)

        return ToolResult(
            ok=bool(chunks),
            payload={
                "chunks": chunks,
                "chunk_ids": [chunk.id for chunk in chunks],
                "top_score": best_score,
                "k": k,
                "query": query,
            },
        )

class Rerank(Tool):
    name = "rerank"
    def __call__(self, query: str, candidates: list) -> ToolResult:
        from ..retrieval.rerank import rerank

        ranked = rerank(query, candidates, self.cfg)
        return ToolResult(
            ok=bool(ranked),
            payload={
                "chunks": ranked,
                "chunk_ids": [chunk.id for chunk in ranked],
                "top_score": max((chunk.score for chunk in ranked), default=0.0),
                "query": query,
            },
        )

class ReadPage(Tool):
    name = "read_page"
    def __call__(self, page_id: str) -> ToolResult:
        from PIL import Image, ImageOps

        from ..vision.ocr import Reader, normalize

        reader = Reader(self.cfg)
        page = reader.pages.get(page_id)
        if page is None:
            return ToolResult(
                ok=False,
                payload={"page_id": page_id, "error": "page not found"},
            )

        with Image.open(page["image_path"]) as image:
            width, height = ImageOps.exif_transpose(image).size
        region = Region(page_id=page_id, bbox=(0, 0, width, height), kind="text")
        text = normalize(
            reader.transcribe_region(region),
            bool(self.cfg.get("ocr", {}).get("ascii_digits", True)),
        )
        return ToolResult(
            ok=bool(text),
            payload={
                "page_id": page_id,
                "doc_id": page["doc_id"],
                "image_path": page["image_path"],
                "text": text,
            },
        )

class EnhancePage(Tool):
    name = "enhance_page"
    def __call__(self, page_id: str) -> ToolResult:
        from ..ingest.enhance import run
        from ..vision.ocr import load_page_index

        page_data = load_page_index(self.cfg).get(page_id)
        if page_data is None:
            return ToolResult(
                ok=False,
                payload={"page_id": page_id, "error": "page not found"},
            )

        page = Page(
            id=page_id,
            image_path=page_data["image_path"],
            doc_id=page_data["doc_id"],
        )
        enhanced = run([page], self.cfg)
        if not enhanced:
            return ToolResult(
                ok=False,
                payload={"page_id": page_id, "error": "enhancement produced no page"},
            )
        output = enhanced[0]
        return ToolResult(
            ok=True,
            payload={
                "page_id": output.id,
                "doc_id": output.doc_id,
                "image_path": output.image_path,
                "enhanced": bool(self.cfg.get("enhance", {}).get("enabled", False)),
            },
        )

class Extract(Tool):
    name = "extract"
    def __call__(self, field: str, chunk_id: str) -> ToolResult:
        chunk = next((item for item in self.chunks if item.id == chunk_id), None)
        if chunk is None:
            return ToolResult(
                ok=False,
                payload={"field": field, "chunk_id": chunk_id, "error": "chunk not found"},
            )

        field_key = field.casefold().strip()
        matches = [line.strip() for line in chunk.text.splitlines() if field_key in line.casefold()]
        if not matches:
            return ToolResult(
                ok=False,
                payload={
                    "field": field,
                    "chunk_id": chunk_id,
                    "value": None,
                    "error": "field not found in chunk",
                },
            )
        return ToolResult(
            ok=True,
            payload={"field": field, "chunk_id": chunk_id, "value": "\n".join(matches)},
        )

class Aggregate(Tool):
    name = "aggregate"
    def __call__(self, op: str, items: list) -> ToolResult:
        operation = op.casefold().strip()
        if operation == "count":
            value = len(items)
        elif operation == "unique":
            value = list(dict.fromkeys(items))
        elif operation in {"sum", "min", "max", "average", "mean"}:
            try:
                numbers = [float(item) for item in items]
            except (TypeError, ValueError):
                return ToolResult(
                    ok=False,
                    payload={"op": op, "error": "operation requires numeric items"},
                )
            if not numbers and operation != "sum":
                return ToolResult(ok=False, payload={"op": op, "error": "items are empty"})
            if operation == "sum":
                value = sum(numbers)
            elif operation == "min":
                value = min(numbers)
            elif operation == "max":
                value = max(numbers)
            else:
                value = sum(numbers) / len(numbers)
        else:
            return ToolResult(
                ok=False,
                payload={"op": op, "error": "unsupported aggregation"},
            )
        return ToolResult(ok=True, payload={"op": op, "value": value})

class Cite(Tool):
    name = "cite"
    def __call__(self, chunk_id: str, span: tuple) -> ToolResult:
        chunk = next((item for item in self.chunks if item.id == chunk_id), None)
        if chunk is None:
            return ToolResult(
                ok=False,
                payload={"chunk_id": chunk_id, "error": "chunk not found"},
            )
        if len(span) != 2:
            return ToolResult(
                ok=False,
                payload={"chunk_id": chunk_id, "error": "span must contain start and end"},
            )
        start, end = span
        if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(chunk.text):
            return ToolResult(
                ok=False,
                payload={"chunk_id": chunk_id, "error": "span is outside the chunk text"},
            )
        citation = Citation(chunk_id=chunk_id, span=(start, end))
        return ToolResult(
            ok=True,
            payload={"citation": citation, "quote": chunk.text[start:end]},
        )

class Calculator(Tool):
    name = "calculator"
    def __call__(self, expr: str) -> ToolResult:
        import ast
        import operator

        if len(expr) > 200:
            return ToolResult(ok=False, payload={"expr": expr, "error": "expression is too long"})
        try:
            root = ast.parse(expr, mode="eval").body
        except SyntaxError:
            return ToolResult(ok=False, payload={"expr": expr, "error": "invalid expression"})

        binary = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
        }
        unary = {ast.UAdd: operator.pos, ast.USub: operator.neg}
        values = {}
        pending = [(root, False)]
        try:
            while pending:
                node, visited = pending.pop()
                if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
                    if abs(node.value) > 1_000_000_000_000:
                        return ToolResult(
                            ok=False,
                            payload={"expr": expr, "error": "numeric literal is too large"},
                        )
                    values[id(node)] = node.value
                elif isinstance(node, ast.BinOp) and type(node.op) in binary:
                    if not visited:
                        pending.extend([(node, True), (node.right, False), (node.left, False)])
                    else:
                        if isinstance(node.op, ast.Pow) and abs(values[id(node.right)]) > 10:
                            return ToolResult(
                                ok=False,
                                payload={"expr": expr, "error": "exponent is too large"},
                            )
                        values[id(node)] = binary[type(node.op)](
                            values[id(node.left)], values[id(node.right)]
                        )
                elif isinstance(node, ast.UnaryOp) and type(node.op) in unary:
                    if not visited:
                        pending.extend([(node, True), (node.operand, False)])
                    else:
                        values[id(node)] = unary[type(node.op)](values[id(node.operand)])
                else:
                    return ToolResult(
                        ok=False,
                        payload={"expr": expr, "error": "expression contains a disallowed operation"},
                    )
            value = values[id(root)]
        except (ArithmeticError, OverflowError, KeyError):
            return ToolResult(ok=False, payload={"expr": expr, "error": "calculation failed"})
        return ToolResult(ok=True, payload={"expr": expr, "value": value})

class EscalateToHuman(Tool):     # HITL entry
    name = "escalate_to_human"
    def __call__(self, reason: str, context: dict) -> ToolResult:
        from .hitl import escalate

        return escalate(reason, context)

REGISTRY = [Retrieve, Rerank, ReadPage, EnhancePage, Extract,
            Aggregate, Cite, Calculator, EscalateToHuman]

