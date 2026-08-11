# Corpus provenance (A1 — carried into A2)

- **Source (URL):** Bangla Handwritten Dolil Dataset (653 Images) — Mendeley Data.
  https://data.mendeley.com/datasets/yk3c3xy9vm/1 (DOI: `10.17632/yk3c3xy9vm.1`)
  Citation: Anik, Saifullah; Hossain, Md. Obydul; Arifen, Radoanul; Ahammed, Md. Shakib (2025),
  "Bangla Handwritten Dolil Dataset (653 Images)", Mendeley Data, V1.
- **Licence / usage rights:** CC BY 4.0 — copying, redistribution, modification, and commercial
  reuse permitted with attribution to the dataset creators above. We may re-share the corpus.
- **Pages:** 653 scanned JPEGs on disk. 44 are non-content thumbnail/corrupt files (<0.2MB —
  `dolil_313`–`dolil_324`, `dolil_431`–`dolil_462`), leaving **609 usable pages**. (Exact counts
  and the legibility breakdown are computed in `notebooks/eda.ipynb`.)
- **Words:** not yet counted — extraction requires the Stage 3 OCR/HTR reader (the corpus is
  handwritten, so no text layer exists to count directly). A1 estimate: ~95,000 words, to be
  finalized once A2's VLM/HTR reading stage runs.
- **Size on disk:** 2.2 GB across all 653 JPEGs.
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
**70% train / 15% validation / 15% test, split by DEED, never by page.** A deed spans a variable
number of consecutive sheets sharing an incrementing registry serial number (e.g. `dolil_13/14/15`
carry serials `2162843/2162844/2162845`), and every sheet also prints an `x/N` "page-in-deed"
marker top-left. Sheets are grouped into deeds by clustering consecutive serials and confirming
each boundary with the `x/N` marker; whole deed-groups are then assigned to splits by hashing the
deed key (deterministic, reproducible). Serial numbers are only read reliably once the A2 VLM
stage runs, so at this stage the grouping is seeded from a hand-labelled sample of deed
boundaries and finalized once A2's reader is in place.

**Leak check:** the most likely leak is a multi-sheet deed (same header boilerplate, parties,
mouza, serial run) scattered across train/test, letting the model "recognise" a test sheet from a
near-twin seen in training. Prevented by clustering on the registry serial number *before*
splitting, so a whole serial-run stays in one split. Verified two ways (both in `eda.ipynb`):
(a) assert no deed group appears in more than one split, (b) a perceptual-hash (pHash)
near-duplicate scan across splits to catch re-scans/consecutive sheets. Residual risk: any deed
mis-grouped before A2 reads the real serials is re-checked and corrected once that stage runs.
