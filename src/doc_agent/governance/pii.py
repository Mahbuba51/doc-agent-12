"""Governance — PII detection + redaction (mandatory)"""
from __future__ import annotations
from ..contracts import *  # noqa

def detect(text: str) -> list[tuple[int,int,str]]:
    """Return (start,end,type) PII spans. IMPLEMENT."""
    return []  # TODO: replace passthrough with real PII span detection later.
def redact(text: str) -> str:
    return text  # TODO: replace passthrough with real PII redaction later.



def register(hooks) -> None:
    """Wire PII redaction into the pipeline. IMPLEMENT the handler (call redact())."""
    def _scrub(ctx: dict) -> dict:
        return ctx  # TODO: call redact() on text/answer/log fields when PII is implemented.
    hooks.register(hooks.AFTER_OCR, _scrub)       # scrub extracted text before indexing
    hooks.register(hooks.BEFORE_ANSWER, _scrub)   # scrub the outgoing answer
    hooks.register(hooks.ON_LOG, _scrub)          # scrub logs
