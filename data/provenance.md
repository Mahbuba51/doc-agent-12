# Corpus provenance (A1 — carried into A2)

- **Source (URL):** Bangla Handwritten Dolil Dataset (653 Images) — Mendeley Data.
  https://data.mendeley.com/datasets/yk3c3xy9vm/1 (DOI: `10.17632/yk3c3xy9vm.1`)
  Citation: Anik, Saifullah; Hossain, Md. Obydul; Arifen, Radoanul; Ahammed, Md. Shakib (2025),
  "Bangla Handwritten Dolil Dataset (653 Images)", Mendeley Data, V1.
- **Licence / usage rights:** CC BY 4.0 — copying, redistribution, modification, and commercial
  reuse permitted with attribution to the dataset creators above. We may re-share the corpus.
- **Pages:** 653 scanned JPEGs on disk, **all 653 usable**. An earlier pass of this document (and
  the A1 form) flagged 44 files under 0.2MB (`dolil_313`–`dolil_324`, `dolil_431`–`dolil_462`) as
  non-content thumbnail/corrupt and excluded them — **that was wrong**. Direct visual inspection
  of over half of those 44 files, plus an automated integrity check (every file opens as a valid,
  reasonably-sized image) on all of them, confirms every one is a real, legible scanned deed page;
  they're just more compressed or lower-resolution than the rest of the batch. File size alone
  was never a valid corrupt/blank detector. (Corrected in `notebooks/eda.ipynb` Section 1.)
- **Words:** not yet counted — extraction requires the Stage 3 OCR/HTR reader (the corpus is
  handwritten, so no text layer exists to count directly). A1 estimate: ~95,000 words (originally
  estimated over the wrongly-reduced 609-page count; now spread over the full 653 pages), to be
  finalized once A2's VLM/HTR reading stage runs.
- **Size on disk:** 2.33 GB across all 653 JPEGs.
- **Meets the profile:** ≥300 usable pages AND ≥60,000 words (pending final word count from A2).

## Scan/script difficulty notes
Three difficulties stack on every page:
1. **Script** — cursive handwritten Bangla with dense conjuncts (যুক্তাক্ষর) and stacked matras
   that touch/overlap, plus older legal orthography.
2. **Code-switching** — English and Bangla-digit numerals appear inline for the fields that
   matter most (deed no., khatian/dag numbers, dates, money), and English legal boilerplate is
   handwritten in the margin (e.g. "Admissible under Rule 2(1); under Section 26(J)-33(L) of the
   Bengal Tenancy Act 1933..." on `dolil_100`).
3. **Physical degradation** — faint/uneven ballpoint ink, skew from flatbed capture, show-through
   from the reverse side, and government seals/round stamps overprinted on the text.

Standard Bangla OCR (Tesseract) fails on all three, which is why the reading stage (Stage 3) uses
a VLM-based reader (Qwen2.5-VL + GraDeT-HTR) instead of a conventional OCR engine.

## Split policy (by document)
**Target: 70% train / 15% validation / 15% test, split by DEED, never by page.** A deed spans a
variable number of consecutive sheets sharing an incrementing registry serial number (e.g.
`dolil_13/14/15` carry serials `2162843/2162844/2162845`), and every sheet also prints an `x/N`
"page-in-deed" marker top-left. The real grouping needs those serials/markers read off the page,
which needs the Stage 3 VLM/HTR reader that doesn't exist yet — so until A2, `notebooks/eda.ipynb`
uses a placeholder: sort pages by page number (which tracks scan/deed order closely) and cut
70/15/15 in that order, dropping the 6 pages immediately before each cut into a `skipped` bucket
excluded from every split, as a cheap buffer against a deed's sheets landing on both sides of a
boundary. `data/splits.json` (written by that notebook) is the current split; `deed_id`/
`page_in_deed_marker` grouping (hand-read, see the deed-group CSV workflow) will replace it.

**Leak check:** the most likely leak is a multi-sheet deed (same header boilerplate, parties,
mouza, serial run) scattered across train/test, letting the model "recognise" a test sheet from a
near-twin seen in training. The placeholder split above only approximates document-level safety;
`notebooks/eda.ipynb` verifies it two ways: (a) a perceptual-hash (pHash) near-duplicate scan
flagging pages that look alike regardless of split, and (b) checking how many of those near-dup
pairs land in different splits (down to 4 of 90 under the buffered page-order split, from 52 under
an earlier deed-group-hash attempt — both numbers are in the notebook). `src/doc_agent/data/
validate.py`'s `validate_splits()` re-runs the structural + per-`doc_id` version of this check
once real `doc_id`s exist. Residual risk: the 4 remaining cross-split pairs, and any deed the
page-order placeholder split without a human catching it, are re-checked once A2 reads the real
serials and the deed-group CSV replaces the placeholder.
