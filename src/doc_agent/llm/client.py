"""LLM — the single LLM call wrapper (all model calls go through here)

Every model call in the agent goes through LLM.complete(), so this is also the one place a
backend can be swapped without touching a caller. Two exist:

  fake   scripted, deterministic, no weights. CI runs on it -- the grading kit and the
         cross-cutting tests (grounding, injection, PII, trace) all assert on agent
         BEHAVIOUR given a model output, and behaviour is only assertable when the output
         is fixed. It is a stub with a script, not a small model.
  local  a real checkpoint via transformers, loaded lazily on first use.

`local`, not a hosted API, is the default backend on purpose: vision/ocr.py records
local inference as a deliberate design constraint for this project, on the grounds that a
production land-deed system handles sensitive records and hosted APIs may retain what is
sent to them. settings.llm_api_key stays available for a hosted backend if the team ever
re-affirms that constraint the other way, but no backend here reads it today.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..contracts import *  # noqa
from ..logging_conf import get_logger

logger = get_logger(__name__)

# What the fake says when a test did not care what the model said. Deliberately not
# answer-shaped: a test that accidentally depends on this string should look wrong.
FAKE_DEFAULT = "fake completion"


class LLM:
    """Model set by cfg['agent']. Key from settings. Backend set by cfg['agent']['backend']."""

    def __init__(self, cfg: dict, generate: Callable[..., str] | None = None) -> None:
        self.cfg = cfg["agent"]
        self.backend = self.cfg.get("backend", "local")
        # Scripted fake replies, consumed in order by the fake backend.
        self._script = list(self.cfg.get("fake_responses") or [])
        # Every prompt this client was sent, in order. The injection test needs to assert on
        # what actually reached the model, not just on what came back.
        self.calls: list[str] = []
        # Injectable so the wiring around a checkpoint is testable without downloading one;
        # production callers leave it None (same contract as ocr.transcribe's `reader`).
        self._generate = generate
        # Any: lazily set to whatever _load() instantiates from the chosen checkpoint.
        self._model: Any = None
        self._tokenizer: Any = None

    def complete(self, prompt: str, **kw: Any) -> str:
        self.calls.append(prompt)
        if self.backend == "fake":
            return self._complete_fake(prompt)
        if self.backend == "local":
            return self._complete_local(prompt, **kw)
        raise ValueError(
            f"unknown llm backend {self.backend!r} in cfg['agent']['backend']; "
            "expected 'fake' or 'local'"
        )

    def _complete_fake(self, prompt: str) -> str:
        if not self.cfg.get("fake_responses"):
            return FAKE_DEFAULT
        if not self._script:
            # Loud, not wrap-around: a loop asking for more completions than the test
            # scripted has taken a path the test did not describe, and silently repeating
            # the last reply would hide exactly that.
            raise RuntimeError(
                f"fake backend ran out of fake_responses after {len(self.calls) - 1} call(s); "
                "the agent asked for more completions than the script provides"
            )
        return self._script.pop(0)

    def _complete_local(self, prompt: str, **kw: Any) -> str:
        if self._generate is None:
            self._generate = self._load()
        return self._generate(prompt, **kw)

    def _load(self) -> Callable[..., str]:
        """Build the generate callable on first use, so importing this module stays CPU-safe."""
        name = self.cfg.get("model") or ""
        if not name:
            # Checked before the import so the failure names the decision, not a missing
            # weight file. No answering checkpoint has been agreed for this project yet.
            raise ValueError(
                "backend is 'local' but cfg['agent']['model'] is empty -- set agent.model in "
                "configs/config.yaml to a checkpoint, or set agent.backend: 'fake' to run "
                "without weights"
            )

        from transformers import AutoModelForCausalLM, AutoTokenizer  # heavy, load late

        logger.info("loading llm %s", name)
        self._tokenizer = AutoTokenizer.from_pretrained(name)
        self._model = AutoModelForCausalLM.from_pretrained(name, device_map="auto")

        def generate(prompt: str, **kw: Any) -> str:
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            # Greedy by default: the precision-first NFR wants a reproducible answer for a
            # given page far more than it wants a varied one.
            out = self._model.generate(
                **inputs,
                max_new_tokens=int(kw.get("max_new_tokens", self.cfg.get("max_new_tokens", 512))),
                do_sample=False,
            )
            trimmed = out[0][inputs["input_ids"].shape[1] :]
            return self._tokenizer.decode(trimmed, skip_special_tokens=True)

        return generate
