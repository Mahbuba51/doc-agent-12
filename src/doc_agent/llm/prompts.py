"""LLM — FIXED prompt template registry (all prompts live here)"""

from __future__ import annotations

from ..contracts import *  # noqa

# Fill the template bodies; do NOT scatter prompt strings elsewhere.
DECIDE = "IMPLEMENT: tool-selection prompt"
SYNTHESIZE = "IMPLEMENT: grounded-answer prompt (must force citations + abstention)"
JUDGE = "IMPLEMENT: LLM-as-judge prompt for open-ended inference"

# Stage 3 reading prompt. Transcription only -- the reader must never summarise, translate,
# or "correct" a legal field, and must mark what it cannot read rather than guessing at it,
# because a confidently invented plot number is the precision-first NFR's worst failure.
TRANSCRIBE = (
    "Transcribe every character of handwritten and printed text in this image of a Bangla "
    "land deed (dolil), exactly as written.\n"
    "Rules:\n"
    "- Preserve the original script: Bangla stays Bangla, English stays English.\n"
    "- Preserve numerals exactly as written, including Bangla digits.\n"
    "- Keep the reading order and line breaks of the page.\n"
    "- Do NOT translate, summarise, correct spelling, or expand abbreviations.\n"
    "- If a word or digit is illegible, write [?] in its place. Never guess.\n"
    "Output only the transcription."
)
