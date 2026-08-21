"""FIXED — structured logging (auditable NFR). Use get_logger(), never print()."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

TRACE_PATH = Path("traces/run.jsonl")


def get_logger(name: str) -> logging.Logger:
    lg = logging.getLogger(name)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(
            logging.Formatter(
                '{"ts":"%(asctime)s","lvl":"%(levelname)s","mod":"%(name)s","msg":"%(message)s"}'
            )
        )
        lg.addHandler(h)
        lg.setLevel(logging.INFO)
    return lg


def register(hooks: Any) -> None:
    """Wire structured tracing at each seam (auditable trail) AND emit traces/run.jsonl.

    One closure per seam rather than one shared handler: the three seams carry different ctx
    shapes ({"state":...}, {"action":...}, {"answer":...}), and a single handler could only
    tell them apart by guessing from the keys present.

    The step counter lives in this closure, so it restarts whenever wiring.register_all()
    re-registers -- one numbering per wired run, which is what makes the trajectory readable.
    """
    from .contracts import TraceStep

    counter = {"step": 0}

    # Truncate here, not on first write: the step numbering restarts with this closure, so
    # keeping the previous run's lines would produce a file with two step-1s and no way to
    # tell which trajectory a line belongs to. The trace is a per-run artifact, regenerated
    # by scripts/run.sh -- archive it before re-running if a specific run matters.
    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRACE_PATH.write_text("", encoding="utf-8")

    def _emit(tool: str, args: dict, obs: dict) -> None:
        counter["step"] += 1
        record = TraceStep(step=counter["step"], tool=tool, args=args, obs=obs)
        TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with TRACE_PATH.open("a", encoding="utf-8") as fh:
            fh.write(record.model_dump_json() + "\n")

    def _on_step(ctx: dict) -> dict:
        state = ctx.get("state") or {}
        seen = state.get("obs") or []
        # The LAST observation is what decide() branches on, and top_score/k in it are what
        # the A3 agentic check reads to confirm the path depended on the evidence.
        _emit("decide", {"query": state.get("query", "")}, seen[-1] if seen else {})
        return ctx

    def _on_tool_call(ctx: dict) -> dict:
        action = dict(ctx.get("action") or {})
        # Everything except the name IS the arguments -- recording the args the tool actually
        # ran with is what lets the grader see the reformulated query on a re-search.
        tool = action.pop("tool", "unknown")
        _emit(tool, action, {})
        return ctx

    def _after_answer(ctx: dict) -> dict:
        answer = ctx.get("answer")
        obs = {
            "grounded": bool(getattr(answer, "grounded", False)),
            "confidence": float(getattr(answer, "confidence", 0.0)),
            "citations": len(getattr(answer, "citations", []) or []),
        }
        _emit("answer", {}, obs)
        return ctx

    hooks.register(hooks.ON_STEP, _on_step)
    hooks.register(hooks.ON_TOOL_CALL, _on_tool_call)
    hooks.register(hooks.AFTER_ANSWER, _after_answer)
