# grading_kit/ — the one folder that makes this project reproducible and gradable.

- **manifest.yaml** — the single entry point: the three axes (domain, data speciality, primary NFR) + pointers to the corpus,
  the held-out slice, and the build/run/eval commands.
- **heldout_pages/** — page-images set aside, never OCR-trained on.
- **labels.jsonl** — ground-truth transcriptions for the held-out pages (the oracle:
  OCR is scored against them, and fresh grading questions are authored from them).

A grader (or you) opens ONLY `manifest.yaml`; it points to everything else. The build/run
scripts and the eval tasks are named there, not copied here, so they never go stale.

## `labels.jsonl` must be typed by a human

`text` and every value under `fields` in `labels.jsonl` are the OCR scoring oracle — they must be
typed by a person reading the actual page image, **never generated or guessed by a model**
(including an AI assistant helping with this repo). A model cannot read this corpus reliably, and
a wrong "ground truth" would silently corrupt every downstream OCR/Exact-Match score.

If a page — or a specific field on a page — genuinely can't be read (faded ink, seal covering the
text, etc.), type `[illegible]` in that field instead of a best guess. `status` stays `"TODO"`
until a human has filled in `text` and `fields` for that page, then set it to `"done"` (or
`"illegible"` if the whole page is unreadable).
