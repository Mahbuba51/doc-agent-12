# Knowledge-base pipeline diagram (A2)

```mermaid
flowchart LR
    L[Load pages] --> P[Preprocess]
    P --> E[Optional enhancement]
    E --> D[Layout detection]
    D --> O[OCR / HTR]
    O --> K[Chunk]
    K --> M[Embed]
    M --> S[Store]
```

## Stage descriptions

**Stage 1 - Load pages**

`ingest/loader.py` reads scanned `.jpg`, `.jpeg`, and `.png` pages from the configured raw page directory. It uses `data/deed_groups.csv` as both the inclusion list and the page-to-document map, which protects document-level train/validation/test splitting. Pages marked as duplicates are skipped before indexing. The loader validates the resulting `Page` objects, snapshots the input directory for provenance, sorts pages by numeric suffix, and returns page IDs, image paths, and `doc_id`s.

**Stage 1 - Preprocess**

`ingest/preprocess.py` performs deterministic cleanup while preserving the original evidence. It applies EXIF orientation, estimates skew with projection profiles, rotates with a white border, denoises lightly, creates a Sauvola binary copy, and computes a blur-based quality score. It writes greyscale and binary outputs plus a JSONL sidecar containing paths, skew, quality, and legibility flags. Downstream layout and OCR use the greyscale path so they operate on the same corrected pixels.

**Stage 1 - Optional enhancement**

`ingest/enhance.py` is a configurable generative enhancement hook for VAE or diffusion-based denoise/super-resolution. In the current code, it only runs when `cfg["enhance"]["enabled"]` is true, and the `Enhancer.train()` and `Enhancer.apply()` methods remain unimplemented. If disabled, pages pass through unchanged. This avoids overprocessing legal handwriting unless a tested enhancement model is explicitly added and enabled.

**Stage 2 - Layout detection**

`vision/layout.py` detects text regions using a training-free projection heuristic. It thresholds each page into an ink mask, finds horizontal ink bands, merges close gaps, filters tiny artefacts, pads bounding boxes, and emits `Region` objects in page order. Every emitted region is labeled `text`; table, figure, and heading classes are deliberately not guessed. The code acknowledges that dense or tabular pages may become page-sized regions, which is acceptable for whole-page readers.

**Stage 3 - OCR / HTR**

`vision/ocr.py` crops each detected region and reads it with a configured local vision-language model baseline. It normalizes Unicode and can translate Bangla digits to ASCII for exact-match evaluation. Regions are reassembled into one page-level `Chunk`, preserving document IDs from the ingest sidecar or deed grouping file. The current docstring records that Qwen2.5-VL performed poorly on held-out handwritten pages, so a Bangla HTR path is the planned replacement.

**Stage 4 - Chunking**

`index/chunk.py` converts page-level OCR output into retrieval-sized chunks. It splits text into paragraphs or lines, counts tokens with the embedding model tokenizer, preserves paragraph and block order metadata, merges small compatible paragraphs, and splits oversized paragraphs with token-window overlap. Chunk IDs encode the source page and order. Metadata is kept for later storage so retrieved evidence can be traced back to page IDs, document IDs, paragraph positions, and boundary types.

**Stage 4 - Embedding**

`index/embed.py` encodes chunk text with SentenceTransformers, defaulting to `intfloat/multilingual-e5-small`. It applies configurable document and query prefixes, batch size, device, cache directory, embedding dimension, and normalization. Empty inputs produce an empty matrix of the configured dimension. The code verifies that model output matches the configured dimension and normalizes vectors to `float32`, making them suitable for inner-product FAISS search.

**Stage 4 - Storage**

`index/store.py` persists the knowledge base as a FAISS flat inner-product index plus a JSONL metadata file. It validates vector shape and dimension, optionally normalizes embeddings, builds `IndexFlatIP`, writes `chunks.faiss`, and serializes chunk metadata to `chunks.jsonl`. The loader reconstructs `Chunk` objects from metadata, so retrieval can connect vector hits back to text, document IDs, page IDs, chunk IDs, and source-order information.
