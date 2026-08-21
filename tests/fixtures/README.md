# tests/fixtures/ — synthetic adversarial pages

**Nothing in this folder is corpus data.** These pages are written by hand as attack fixtures
for the A3 security section. They are NOT scans, NOT transcriptions, and must never be
copied into `grading_kit/labels.jsonl`, `grading_kit/tasks.jsonl`, or the FAISS index built
from `data/`.

`grading_kit/README.md` requires every `labels.jsonl` transcription to be typed by a human
reading a real page image, because a fabricated "ground truth" silently corrupts every
downstream OCR and Exact-Match score. These files are fabricated on purpose, which is
exactly why they live here and not there.

## injection_pages.jsonl

One JSON object per line, each a `Chunk` as `retrieval/retriever.py` would emit, plus a
`_note` field describing what the page is testing. `_note` is stripped before the Chunk is
built — it is documentation, not payload.

| id | what it is |
|---|---|
| `synthetic_inj_bn#p` | Bangla injection — the realistic attack on THIS corpus |
| `synthetic_inj_en#p` | English injection — the case the pattern layer catches |
| `synthetic_benign#p` | a page that *discusses* instructions without attacking — false-positive control |

## hallucination_cases.jsonl

`(query, evidence, draft answer, expected verdict)` cases for the grounding gate, driven by
`tests/test_hallucination.py` and captured in `reports/hallucination_demo.md`.

Unlike the injection pages, these are NOT fabricated end to end: `query` and `gold` are real
entries from `grading_kit/tasks.jsonl` (t2, t10) and `evidence` is a trimmed excerpt of the
human transcription of `dolil_13` in `grading_kit/labels.jsonl`. Only `draft` is written by
hand — it is the model output being tested, not ground truth.

The excerpt is trimmed to the lines t2 and t10 turn on. The full page carries party names,
and duplicating them into a second committed file is gratuitous.
