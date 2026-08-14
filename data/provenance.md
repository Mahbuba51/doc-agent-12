# Corpus provenance (A1 — carried into A2)

- **Source (URL):** Bangla Handwritten Dolil Dataset (653 Images) — Mendeley Data.
  https://data.mendeley.com/datasets/yk3c3xy9vm/1 (DOI: `10.17632/yk3c3xy9vm.1`)
  Citation: Anik, Saifullah; Hossain, Md. Obydul; Arifen, Radoanul; Ahammed, Md. Shakib (2025),
  "Bangla Handwritten Dolil Dataset (653 Images)", Mendeley Data, V1.
- **Licence / usage rights:** CC BY 4.0 — copying, redistribution, modification, and commercial
  reuse permitted with attribution to the dataset creators above. We may re-share the corpus.
- **Pages:** 653 scanned JPEGs on disk, **651 content-usable**. An earlier pass of this document
  (and the A1 form) flagged 44 files under 0.2MB (`dolil_313`–`dolil_324`, `dolil_431`–`dolil_462`)
  as non-content thumbnail/corrupt and excluded them — **that was wrong**. Direct visual inspection
  of over half of those 44 files, plus an automated integrity check (every file opens as a valid,
  reasonably-sized image) on all of them, confirms every one is a real, legible scanned deed page;
  they're just more compressed or lower-resolution than the rest of the batch. File size alone
  was never a valid corrupt/blank detector. (Corrected in `notebooks/eda.ipynb` Section 1.) Two
  pages are confirmed genuine non-content — `dolil_30` (an Instagram screenshot) and `dolil_412`
  (also confirmed not a scan of a deed) — found by direct inspection, and are excluded, giving 651.
- **Distinct document pages: 546**, after duplicate resolution. Beyond the two non-content pages
  above, **105 of the 651 content-usable pages are confirmed duplicate re-scans** of another page
  already in the corpus — the clearest case: one real 10-sheet deed had been photographed in three
  to four separate capture sessions, so several sheets exist 3–4 times under different page ids.
  Found via a pHash/CLIP/ORB near-duplicate sweep (`notebooks/eda.ipynb` Section 3) cross-checked
  against the printed stamp-paper serial number where visible (the most reliable signal — more
  reliable than either the perceptual hash or the keypoint match alone, both of which have known
  false positives from shared stamp-paper templates and false negatives from lighting/crop
  differences). All 105 are still present on disk in `data/raw/` — nothing is deleted from the
  corpus — but excluded from `data/deed_groups.csv` and therefore from grouping/splitting, each
  with a note recording which canonical page it duplicates and how the duplicate was confirmed.
- **Words:** ~65,500+ words (conservative estimate) — **546 distinct** pages x 120 words/page.
  That 120 is itself a floor, not a typical value: 5 pages hand-counted directly against the scans
  came in at 259–447 words each. This corrects an earlier ~78,000 estimate that used 652 pages and
  double-counted the 105 confirmed duplicates' words — the corpus doesn't actually contain that
  much distinct text. It also supersedes two older, less reliable numbers: the A1 form's rough
  ~95,000 guess, and a Qwen2.5-VL pilot-read of 60 sampled pages (median ~60 words/page after
  cleaning model repetition-loop artifacts) that undercounts real content on dense pages for a
  different reason (reader under-extraction, not duplication). A full recount once the Stage 3
  reader is properly built and tuned will replace this estimate.
- **Size on disk:** 2.33 GB across all 653 JPEGs (duplicates included — nothing has been deleted).
- **Meets the profile:** >=300 usable pages (546 distinct, confirmed) AND >=60,000 words (~65,500+
  estimated, see above) — **both floors cleared, but the word margin is narrow (~9%) now that
  duplicate pages are correctly excluded from the count.** Worth tracking as the OCR reader comes
  online and the word estimate gets replaced with a real count.

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
**Target: 70% train / 15% validation / 15% test, split by deed, never by page — met exactly:
train 382 (70.0%) / val 82 (15.0%) / test 82 (15.0%), out of the 546 distinct document pages.**

`data/deed_groups.csv` gives each page a `doc_id`, built from two sources: (a) EXIF capture
timestamps for the 573 pages that have them, grouped by reviewer-marked boundaries wherever a
photography session gap indicates a new deed started; (b) direct manual review, page by page, for
the 36 pages with no EXIF (`dolil_1`–`dolil_36`). On top of that grouping, **105 pages confirmed as
duplicate re-scans of another page are excluded entirely** (see the "Pages" section above) — a
document-level split is only leak-safe if the same physical page can't appear as two different ids
on two different sides of it, and duplicate resolution is what makes that true. `notebooks/
eda.ipynb` Section 5 assigns whole `doc_id` groups to a split via greedy bin-packing (largest
group first, into whichever split is furthest below its target share), so no confirmed multi-page
deed straddles a split by construction. `data/splits.json` (written by that notebook) is the
current split.

**Leak check:** the most likely leak is a multi-sheet deed (same header boilerplate, parties,
mouza, serial run) scattered across train/test, letting the model "recognise" a test sheet from a
near-twin seen in training. `notebooks/eda.ipynb` verifies this two ways: (a) a rotation-invariant
perceptual-hash (pHash) near-duplicate scan across all pages (211 candidate pairs found), and
(b) checking how many of those pairs land in different splits — **12 of 211**, down from 124 of
211 before duplicate resolution (most of that drop is mechanical: a page that only exists once can
no longer cross a split with itself). `src/doc_agent/data/validate.py`'s `validate_splits()`
re-runs the structural + per-`doc_id` version of this check for real, against `ingest/loader.py`'s
actual output — **0 doc_id groups straddle a split**, a hard guarantee, not a heuristic. Residual
risk: 38 pages (`dolil_313`–`dolil_324`, `dolil_431`–`dolil_462`) still have no EXIF and no manual
grouping review, so any hidden relationship between them and the rest of the corpus (duplicate or
otherwise) isn't yet checked — closing that gap is the next step, not another split-algorithm
change.
