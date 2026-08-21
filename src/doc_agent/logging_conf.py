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


# Defined after get_logger, not before: this module supplies the logger every other module
# uses, so it cannot borrow one from anywhere else.
logger = get_logger(__name__)


def register(hooks: Any) -> None:
    """Wire structured tracing at each seam (auditable trail) AND emit traces/run.jsonl.

    One closure per seam rather than one shared handler: the three seams carry different ctx
    shapes ({"state":...}, {"action":...}, {"answer":...}), and a single handler could only
    tell them apart by guessing from the keys present.

    The step counter lives in this closure, so it restarts whenever wiring.register_all()
    re-registers -- one numbering per wired run, which is what makes the trajectory readable.
    """
    from .contracts import ToolResult, TraceStep

    counter = {"step": 0}

    # Truncate here, not on first write: the step numbering restarts with this closure, so
    # keeping the previous run's lines would produce a file with two step-1s and no way to
    # tell which trajectory a line belongs to. The trace is a per-run artifact, regenerated
    # by scripts/run.sh -- archive it before re-running if a specific run matters.
    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRACE_PATH.write_text("", encoding="utf-8")

    def _compact(payload: dict) -> dict:
        """Payload with bulk collections collapsed to a count.

        traces/ is COMMITTED -- traces/example_trace.md is tracked, because the worked trace
        is an A3 deliverable. A payload's "chunks" key holds whole Chunk objects, so copying
        it verbatim would write party names and addresses from the deeds into git history.
        The scalars the audit actually needs (top_score, k, ok) and the compact "chunk_ids"
        reference survive; the bodies they point at do not.

        This is deliberately structural rather than a PII pattern match: it cannot be
        defeated by a name the redactor does not recognise, because it never copies the
        field at all. Residual risk stays in scalar strings such as a reformulated query --
        see governance/pii.py for the policy that has to cover those.
        """
        out: dict = {}
        for key, value in payload.items():
            if isinstance(value, (str, int, float, bool, type(None))):
                out[key] = value
            elif isinstance(value, (list, tuple)) and all(
                isinstance(v, (str, int, float, bool)) for v in value
            ):
                out[key] = list(value)
            else:
                out[key] = (
                    f"<{len(value)} item(s) omitted>" if hasattr(value, "__len__") else "<omitted>"
                )
        return out

    def _emit(tool: str, args: dict, obs: dict) -> None:
        counter["step"] += 1
        record = TraceStep(step=counter["step"], tool=tool, args=args, obs=obs)
        TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with TRACE_PATH.open("a", encoding="utf-8") as fh:
            fh.write(record.model_dump_json() + "\n")

    def _as_obs(observation: object) -> dict:
        """Whatever the loop appended to state['obs'], as a dict TraceStep will accept.

        Agent.run appends ToolResult, not dicts, so the payload has to be unwrapped -- and
        `ok` carried across with it, because a trace that shows a failed tool call as an
        ordinary one is an audit trail that hides the thing worth auditing. Payload keys win
        a collision: they are the tool's own report.
        """
        if isinstance(observation, ToolResult):
            return {"ok": observation.ok, **_compact(observation.payload)}
        if isinstance(observation, dict):
            return _compact(observation)
        # Never raise. Tracing that takes down the run it is auditing loses the whole
        # trajectory to protect one field; record the surprise and let the run finish.
        logger.warning("unexpected observation type %s in trace", type(observation).__name__)
        return {"raw": repr(observation)}

    def _on_step(ctx: dict) -> dict:
        state = ctx.get("state") or {}
        seen = state.get("obs") or []
        # The LAST observation is what decide() branches on, and top_score/k in it are what
        # the A3 agentic check reads to confirm the path depended on the evidence.
        _emit("decide", {"query": state.get("query", "")}, _as_obs(seen[-1]) if seen else {})
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
