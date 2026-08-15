# Knowledge-base pipeline (A2)

The fixed stage order lives in `src/doc_agent/pipeline.py:build_knowledge_base()` and may not be
reordered. Everything below is what actually runs today, including the stage that is measured and
rejected — this is a build report, not a plan.

```mermaid
flowchart TD
    RAW["data/raw/<br/>653 scanned JPEGs<br/>546 distinct after dedup"]

    subgraph S1["Stage 1 — ingest"]
        LOAD["loader.load_pages()<br/>deed_groups.csv → doc_id"]
        PREP["preprocess.run()<br/>deskew ±4°, denoise,<br/>Sauvola 15/0.2/128"]
        ENH["enhance.run()<br/><b>OFF</b> — generative repair rejected:<br/>it can reshape a stroke"]
    end

    subgraph S2["Stage 2 — layout"]
        LAY["layout.detect()<br/>Otsu + row projection<br/>min_ink_frac .01, gap 18px"]
    end

    subgraph S3["Stage 3 — OCR/HTR"]
        OCR["ocr.transcribe()<br/>Qwen2.5-VL-3B, greedy, 768 tok<br/><b>MEASURED: macro F1 0.022 → REJECTED</b>"]
    end

    subgraph S4["Stage 4 — index"]
        CH["chunk.split()<br/>paragraph-aware, 200-300 tok,<br/>hard max 450, overlap 40"]
        EMB["embed.encode()<br/>multilingual-e5-small, 384-d<br/>'passage: ' prefix, normalised"]
        ST["store.build()<br/>FAISS IndexFlatIP<br/>+ chunks.jsonl sidecar"]
    end

    GOLD["grading_kit/labels.jsonl<br/>17 human-typed pages<br/>2,398 words"]
    IDX[("data/interim/index/<br/>chunks.faiss + chunks.jsonl")]
    Q(["query → embed.encode_query()<br/>'query: ' prefix → FAISS search"])

    RAW --> LOAD --> PREP --> ENH --> LAY --> OCR
    OCR -.->|"output unusable<br/>(confabulated)"| CH
    GOLD ==>|"what the demo KB<br/>actually indexes"| CH
    CH --> EMB --> ST --> IDX --> Q

    classDef dead fill:#fdd,stroke:#c00,stroke-width:2px
    classDef live fill:#dfd,stroke:#080
    class OCR,ENH dead
    class GOLD,CH,EMB,ST live
```

## Contracts on each edge

The types are frozen in `src/doc_agent/contracts.py`; a stage may only change what flows *inside*
them.

| Edge | Type | Note |
|---|---|---|
| loader → preprocess → enhance | `list[Page]` | `id`, `image_path`, `doc_id` |
| layout → OCR | `list[Region]` | `page_id`, `bbox`, `kind` — always `"text"` today |
| OCR → chunk | `list[Chunk]` | **one Chunk per PAGE**, not per region, so the chunker can still cut on deed/paragraph boundaries that span regions |
| chunk → embed → store | `list[Chunk]` + `ndarray` | 384-d, L2-normalised, so inner product = cosine |
| retrieval → agent | `list[Chunk]` with `.score` | Stage 5, not built yet |

## Cross-cutting seams

Horizontal features attach at fixed hook points (`hooks.py`, wired in `wiring.py`) rather than
being inlined into stage code:

```
after_ingest ──► (fairness metadata)
after_ocr    ──► governance/pii.redact      ← PII must not reach the index
before_index ──► logging/trace
```

## Status, honestly

| Stage | State |
|---|---|
| 1 Ingest | implemented; enhancement deliberately OFF |
| 2 Layout | implemented, 12 tests. Region = page on camera-photo scans, which is expected, not a failure |
| **3 OCR** | implemented and **measured at macro F1 0.022 against a 0.92 target** — the reader confabulates Bangla prose rather than reading handwriting, on both a 3B and a 9B model. It *does* read printed text correctly. GraDeT-HTR is the promoted candidate; weights are request-only |
| 4 Index | implemented; no dedicated test file yet |
| 5 Retrieval | `Retriever.retrieve()` still raises — A3 |

**Why the diagram has a dotted line into chunking.** Stage 3's output is not fit to index: filling
the store with invented text is worse than an empty store, because retrieval would then return
confident citations to sentences no deed contains. So the demo knowledge base is built from the 17
human-transcribed gold pages (~3.1% of the corpus) — real Bangla legal text, honestly reported as a
subset. `chunk`/`embed`/`store` operate on whatever text a `Chunk` carries, so the identical code
indexes the full corpus the moment Stage 3 has a working reader.

Full decision records: `src/doc_agent/vision/ocr.py` (D2 MEASURED), `configs/design_choices.md`.
