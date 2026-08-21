"""Stage 6 - FIXED loop - perceive -> decide -> act -> observe, with cross-cutting seams.
Implement decide() and synthesize() only. Security, grounding, PII, and tracing run via hooks at the
marked seams - do NOT inline them here."""
from __future__ import annotations
from ..contracts import *  # noqa
from .. import hooks
from .memory import Memory

class Agent:
    """FIXED loop. Implement decide() (the policy) and synthesize() only."""
    def __init__(self, cfg: dict, retriever) -> None:
        self.cfg = cfg["agent"]; self.retriever = retriever; self.mem = Memory()

    def run(self, query_text: str) -> Answer:
        state = {"query": query_text, "obs": []}
        for _ in range(self.cfg["max_steps"]):
            hooks.run(hooks.ON_STEP, {"state": state})
            action = self.decide(state)                      # IMPLEMENT (policy)
            if action["tool"] == "stop":
                break
            hooks.run(hooks.ON_TOOL_CALL, {"action": action})   # guardrails/injection/trace
            result = self.act(action)                        # runs the tool via REGISTRY
            state["obs"].append(result); self.mem.add(result)
        hooks.run(hooks.BEFORE_ANSWER, {"state": state})     # grounding gate / PII redact
        ans = self.synthesize(state)                         # IMPLEMENT (grounded answer)
        hooks.run(hooks.AFTER_ANSWER, {"answer": ans})       # trace / metrics
        return ans

    def decide(self, state: dict) -> dict:
        """Evidence-gated re-search — the MANDATORY agentic behaviour (A3 gate, fail-closed).
        Read the last observation (top_score, k) and branch on the NUMBER, using retrieval.retriever:
          1. retrieve at k = cfg.retrieve.k
          2. if is_weak(chunks, cfg):  k2 = next_k(k, cfg)
               - k2 is not None -> retrieve AGAIN at the wider k2 (widen the net), then re-check
               - k2 is None (hit k_max) and still weak -> ABSTAIN ("insufficient evidence")
          3. else -> synthesize a grounded, cited answer
        Emit obs {"top_score": ..., "k": ...} on each step. A fixed retrieve->answer path is NOT agentic
        and caps the grade. Rule-based (baseline) or RL policy (Stage 7)."""
        retrieve_cfg = getattr(self.retriever, "cfg", {})
        initial_k = int(retrieve_cfg.get("k", 10))

        if state.get("abstain"):
            return {"tool": "stop"}

        observations = state.get("obs", [])
        if not observations:
            return {"tool": "retrieve", "query": state["query"], "k": initial_k}

        last = observations[-1]
        payload = last.payload if isinstance(last, ToolResult) else last
        if not isinstance(payload, dict) or "top_score" not in payload:
            return {"tool": "stop"}

        score = float(payload.get("top_score", 0.0))
        threshold = float(retrieve_cfg.get("weak_threshold", 0.35))
        if score >= threshold:
            return {"tool": "stop"}

        current_k = int(payload.get("k", initial_k))
        next_value = current_k + int(retrieve_cfg.get("k_step", 10))
        maximum_k = int(retrieve_cfg.get("k_max", 40))
        if next_value > maximum_k:
            state["abstain"] = True
            return {"tool": "stop"}

        return {"tool": "retrieve", "query": state["query"], "k": next_value}

    def act(self, action: dict) -> ToolResult:
        from . import tools

        tool_name = action.get("tool")
        tool_class = next((item for item in tools.REGISTRY if item.name == tool_name), None)
        if tool_class is None:
            return ToolResult(ok=False, payload={"error": f"unknown tool: {tool_name}"})

        tool = tool_class()
        if tool_name == "retrieve":
            tool.retriever = self.retriever
        if tool_name in {"rerank", "read_page", "enhance_page"}:
            tool.cfg = self.retriever.full_cfg
        if tool_name in {"extract", "cite"}:
            tool.chunks = []
            for observation in self.mem.items:
                payload = observation.payload if isinstance(observation, ToolResult) else observation
                if isinstance(payload, dict):
                    tool.chunks.extend(payload.get("chunks", []))

        arguments = {key: value for key, value in action.items() if key != "tool"}
        return tool(**arguments)

    def synthesize(self, state: dict) -> Answer:
        """Grounded, cited answer; abstain if unsupported (no-hallucination)."""
        from ..llm.client import LLM
        from ..llm.postprocess import format_answer
        from ..llm.prompts import ABSTAIN, SYNTHESIZE

        chunks = []
        seen = set()
        last_score = None
        for observation in state.get("obs", []):
            payload = observation.payload if isinstance(observation, ToolResult) else observation
            if not isinstance(payload, dict):
                continue
            if "top_score" in payload:
                last_score = float(payload["top_score"])
            for chunk in payload.get("chunks", []):
                if chunk.id not in seen:
                    chunks.append(chunk)
                    seen.add(chunk.id)

        threshold = float(getattr(self.retriever, "cfg", {}).get("weak_threshold", 0.35))
        if state.get("abstain") or not chunks or last_score is None or last_score < threshold:
            return format_answer(ABSTAIN, [])

        evidence = "\n\n".join(f"[{chunk.id}]\n{chunk.text}" for chunk in chunks)
        prompt = SYNTHESIZE.format(query=state["query"], evidence=evidence)
        raw = LLM({"agent": self.cfg}).complete(prompt)
        answer = format_answer(raw, chunks)

        if not answer.grounded:
            return format_answer(ABSTAIN, [])
        return answer
