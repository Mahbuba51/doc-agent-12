"""LLM — answer post-process / format / abstention

THE GROUNDING GATE, and why it is two separate things.

prompts.SYNTHESIZE asks the model to cite every claim and to abstain when the deeds are
silent. That is a request. A model under-supported by evidence still produces fluent,
confident, uncited prose -- vision/ocr.py records exactly this failure on this corpus, where
the reader invented universities and deed boilerplate rather than admitting it could not
read. On land deeds the same failure mode means an invented plot number, which is the
precision-first NFR's worst outcome. So format_answer() VERIFIES what the prompt requested:
every parsed citation is checked against the evidence actually retrieved, and an answer whose
sentences are not all cited is marked ungrounded no matter how well it reads.

WHERE THE TWO HALVES RUN. The FIXED loop fires BEFORE_ANSWER with {"state": ...} *before*
synthesize() builds the answer, so the hook cannot inspect an answer that does not exist
yet. It does the half it can: reads the evidence the agent gathered and marks the state to
abstain when there is none. format_answer() does the other half afterwards, on the text.

CITATION SPANS INDEX INTO Answer.text. eval/metrics.groundedness(answer) and
citation_accuracy(answer) are handed the Answer alone, with no chunks, so everything they
score has to be locatable inside it.
"""

from __future__ import annotations

import re
from typing import Any

from ..contracts import *  # noqa
from ..logging_conf import get_logger
from .prompts import ABSTAIN

logger = get_logger(__name__)

# A bracketed run with no whitespace or nesting. This deliberately over-matches: "[?]" (the
# reader's illegible marker) and "[illegible]" (the labeller's) both hit it. Validating the
# captured id against the retrieved chunk ids is what separates a citation from a bracket
# that merely looks like one -- the syntax alone cannot, because chunk text contains both.
_MARKER = re.compile(r"\[([^\[\]\s]+)\]")

# Sentence terminators: Bangla danda and double-danda first, then the ASCII stops. A Bangla
# answer ends its sentences with "।", so splitting on "." alone would see one long sentence
# and let a single trailing citation vouch for the entire answer.
_SENTENCE_END = re.compile(r"[।॥.!?]+")


def format_answer(raw: str, citations: list) -> Answer:
    """Attach citations, set grounded/confidence, enforce abstention.

    `citations` is the evidence the answer is allowed to cite -- the Chunks retrieved for
    this query (ids alone are accepted too). The Citation objects on the returned Answer are
    built here, from the [chunk_id] markers the model emitted, after checking each one
    against that evidence.
    """
    allowed = _allowed_ids(citations)
    text = raw.strip()

    if text == ABSTAIN:
        # The honest answer when the deeds are silent. Not a failure -- but not grounded
        # either, because there is nothing under it to be grounded in.
        return Answer(text=text, citations=[], grounded=False, confidence=0.0)

    clean, found = _strip_markers(text, allowed)
    cited_spans = [span for span, _ in found]
    parsed = [Citation(chunk_id=cid, span=span) for span, cid in found]

    covered = _covered_sentences(clean, cited_spans)
    total = len(_sentences(clean))
    # Grounded only if EVERY sentence carries a citation. One cited sentence must not
    # launder an uncited one sitting beside it -- that is how a real finding and an
    # invented one end up in the same paragraph wearing the same confidence.
    grounded = bool(parsed) and total > 0 and covered == total
    confidence = (covered / total) if total else 0.0

    if parsed and not grounded:
        logger.warning(
            "answer cites %d chunk(s) but only %d/%d sentences are cited",
            len(parsed),
            covered,
            total,
        )

    return Answer(text=clean, citations=parsed, grounded=grounded, confidence=confidence)


def _allowed_ids(citations: list) -> set[str]:
    """Chunk ids the answer may legitimately cite, from Chunks or bare id strings."""
    ids: set[str] = set()
    for item in citations or []:
        cid = item if isinstance(item, str) else getattr(item, "id", None)
        if cid:
            ids.add(cid)
    return ids


def _strip_markers(text: str, allowed: set[str]) -> tuple[str, list[tuple[tuple[int, int], str]]]:
    """Remove valid [chunk_id] markers, returning the display text and (span, id) pairs.

    The span runs from the end of the previous citation (or the start of the sentence the
    marker closes) up to the marker, which is the stretch of answer that citation vouches
    for. Offsets are into the RETURNED text, not the raw string.
    """
    out: list[str] = []
    found: list[tuple[tuple[int, int], str]] = []
    cursor = 0  # read position in `text`
    claim_start = 0  # write position in `out` where the current uncited claim began

    for match in _MARKER.finditer(text):
        chunk_id = match.group(1)
        if chunk_id not in allowed:
            # Either a hallucinated id or an ordinary bracket like [?]. Both stay in the
            # text verbatim: dropping "[?]" would silently repair an illegible reading.
            if chunk_id not in {"?", "illegible"}:
                logger.warning("dropping citation to %r, which was not retrieved", chunk_id)
            continue
        out.append(text[cursor : match.start()])
        cursor = match.end()
        end = len("".join(out))
        found.append(((claim_start, end), chunk_id))
        claim_start = end

    out.append(text[cursor:])
    clean = "".join(out)
    # Trailing whitespace left by a stripped marker would push spans past their claim.
    return clean.rstrip(), [((s, min(e, len(clean.rstrip()))), cid) for (s, e), cid in found]


def _sentences(text: str) -> list[tuple[int, int]]:
    """(start, end) spans of each non-empty sentence in `text`."""
    spans = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        if text[start : match.end()].strip():
            spans.append((start, match.end()))
        start = match.end()
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans


def _covered_sentences(text: str, cited: list[tuple[int, int]]) -> int:
    """How many sentences have at least one citation span overlapping them."""
    return sum(
        1
        for s_start, s_end in _sentences(text)
        if any(c_start < s_end and c_end > s_start for c_start, c_end in cited)
    )


def register(hooks: Any) -> None:
    """Wire the grounding / abstention gate (abstain if answer unsupported by evidence)."""

    def _ground(ctx: dict) -> dict:
        state = ctx.get("state")
        if state is None:
            return ctx
        # BEFORE_ANSWER runs ahead of synthesize(), so the only thing to judge here is the
        # evidence. No observations at all means there is nothing to ground an answer in,
        # and synthesize() should abstain rather than answer from the model's priors.
        state["abstain"] = not (state.get("obs") or [])
        if state["abstain"]:
            logger.warning("no evidence gathered for %r; answer must abstain", state.get("query"))
        return ctx

    hooks.register(hooks.BEFORE_ANSWER, _ground)
