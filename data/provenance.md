# Corpus provenance (A1 — carried into A2)

- **Source (URL):** Bangla Handwritten Dolil Dataset (653 Images) — Mendeley Data.
  https://data.mendeley.com/datasets/yk3c3xy9vm/1 (DOI: `10.17632/yk3c3xy9vm.1`)
  Citation: Anik, Saifullah; Hossain, Md. Obydul; Arifen, Radoanul; Ahammed, Md. Shakib (2025),
  "Bangla Handwritten Dolil Dataset (653 Images)", Mendeley Data, V1.
- **Licence / usage rights:** CC BY 4.0 — copying, redistribution, modification, and commercial
  reuse permitted with attribution to the dataset creators above. We may re-share the corpus.
- **Pages:** 653 scanned JPEGs on disk, **652 usable**. An earlier pass of this document (and
  the A1 form) flagged 44 files under 0.2MB (`dolil_313`–`dolil_324`, `dolil_431`–`dolil_462`) as
  non-content thumbnail/corrupt and excluded them — **that was wrong**. Direct visual inspection
  of over half of those 44 files, plus an automated integrity check (every file opens as a valid,
  reasonably-sized image) on all of them, confirms every one is a real, legible scanned deed page;
  they're just more compressed or lower-resolution than the rest of the batch. File size alone
  was never a valid corrupt/blank detector. (Corrected in `notebooks/eda.ipynb` Section 1.) One
  page, `dolil_30`, is confirmed genuine non-content — an Instagram screenshot, found by direct
  inspection during manual deed-grouping, not a scan of anything — and is excluded, giving 652.
- **Words:** ~78,000+ words (conservative estimate) — 652 usable pages (653 raw minus the
  confirmed non-content `dolil_30`, an Instagram screenshot, not a scan) x 120 words/page. That
  120 is itself a floor, not a typical value: 5 pages hand-counted directly against the scans
  came in at 259–447 words each. This supersedes two earlier, less reliable numbers: the A1
  form's rough ~95,000 guess, and a Qwen2.5-VL pilot-read of 60 sampled pages (median ~60
  words/page after cleaning model repetition-loop artifacts and `[?]`-marker inflation) that
  significantly undercounts real content on dense pages — the reader under-extracts, the corpus
  itself does not fall short. A full recount once the Stage 3 reader is properly built and tuned
  will replace this estimate; it should only move upward.
- **Size on disk:** 2.33 GB across all 653 JPEGs.
- **Meets the profile:** >=300 usable pages (652, confirmed) AND >=60,000 words (~78,000+
  estimated, see above) — **both floors cleared.**

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
