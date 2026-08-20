"""Stage 6 — SECURITY — autonomy, budgets, prompt-injection defense

INSTRUCTION/CONTENT ISOLATION, and why it is enforced on the ACTION.
The seam available here is ON_TOOL_CALL, so this module sees the actions the agent chose,
not the chunks it read. That is the right place regardless: a dolil page that says "ignore
your instructions" is inert while it stays DATA, and only becomes an attack when its words
are copied into something the system executes -- a tool argument. Retrieved text reaching a
tool argument is the observable moment of an injection succeeding, and it is exactly what
_INJECTION catches.

So the defense is not "detect evil documents". It is: document text may be read, quoted, and
cited, but may never become an instruction the agent acts on. A page that is merely ABOUT
instructions still retrieves and still gets answered from; what is refused is the call whose
arguments have been rewritten by the page.
"""

from __future__ import annotations

import re
from typing import Any

from ..contracts import *  # noqa
from ..logging_conf import get_logger

logger = get_logger(__name__)

# The nine locked tool names (tests/test_structure.py::test_tool_names_locked). An action
# naming anything else did not come from the registry, so it is refused rather than dispatched.
ALLOWED_TOOLS = frozenset(
    {
        "retrieve",
        "rerank",
        "read_page",
        "enhance_page",
        "extract",
        "aggregate",
        "cite",
        "calculator",
        "escalate_to_human",
    }
)

# Tools that only look things up. Under read-only autonomy nothing else may run.
READ_ONLY_TOOLS = frozenset(
    {"retrieve", "rerank", "read_page", "extract", "aggregate", "cite", "calculator"}
)

AUTONOMY_LEVELS = ("read-only", "act-with-approval", "autonomous")

# Imperatives aimed at the system rather than at the deed. Deliberately a small, explicit
# list: this is a demonstrable defense for the A3 security section, not a claim to catch
# every phrasing. A determined paraphrase gets through, and the honest mitigation is that
# the agent's own prompt keeps retrieved text in a quoted evidence block (llm/prompts.py).
_INJECTION = re.compile(
    r"ignore\s+(your|all|the|previous|prior|above)"
    r"|disregard\s+(your|all|the|previous|prior|above)"
    r"|forget\s+(your|all|the|previous|prior)\s+instruction"
    r"|you\s+are\s+now\s+"
    r"|new\s+instructions?\s*:"
    r"|system\s+prompt"
    r"|reveal\s+(the\s+)?(key|secret|password|api)",
    re.IGNORECASE,
)


class GuardrailError(Exception):
    """A refused tool call. Raised, not returned: a blocked action must not reach the tool."""


class Guardrails:
    """Enforce autonomy level, step/cost budget, and instruction/content isolation."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg["agent"]
        self.reset()

    def reset(self) -> None:
        self.spent = 0.0
        self.steps = 0

    def check(self, action: dict) -> None:
        """Raise GuardrailError if over budget / disallowed autonomy / injection detected."""
        tool = action.get("tool", "")

        if tool not in ALLOWED_TOOLS:
            raise GuardrailError(
                f"tool {tool!r} is not one of the nine registry tools; refusing to dispatch it"
            )

        autonomy = self.cfg.get("autonomy", "act-with-approval")
        if autonomy not in AUTONOMY_LEVELS:
            raise GuardrailError(
                f"unknown autonomy level {autonomy!r}; expected one of {AUTONOMY_LEVELS}"
            )
        if autonomy == "read-only" and tool not in READ_ONLY_TOOLS:
            raise GuardrailError(f"tool {tool!r} acts, and autonomy is read-only; refusing")

        # Injection before budgets: a hijacked call should be reported as a hijacked call,
        # not masked by whichever limit it happened to trip on the way through.
        for name, value in action.items():
            if name == "tool" or not isinstance(value, str):
                continue
            found = _INJECTION.search(value)
            if found:
                logger.warning("refused %s: injected text in argument %r", tool, name)
                raise GuardrailError(
                    f"possible prompt injection in {tool} argument {name!r}: "
                    f"{found.group(0)!r} -- document text must stay evidence, not instruction"
                )

        cost = float(action.get("cost_usd", 0.0))
        budget = float(self.cfg["budget_usd"])
        if self.spent + cost > budget:
            raise GuardrailError(
                f"budget exhausted: {self.spent + cost:.4f} USD would exceed budget_usd {budget}"
            )

        max_steps = int(self.cfg["max_steps"])
        if self.steps >= max_steps:
            raise GuardrailError(f"step budget exhausted: max_steps is {max_steps}")

        # Only counted once the call is cleared, so a refused action does not burn budget.
        self.steps += 1
        self.spent += cost


def register(hooks: Any, cfg: dict) -> None:
    """Wire guardrails into every tool call."""
    g = Guardrails(cfg)
    g.reset()

    def _check(ctx: dict) -> dict:
        g.check(ctx["action"])  # budgets / autonomy / injection
        return ctx

    hooks.register(hooks.ON_TOOL_CALL, _check)
