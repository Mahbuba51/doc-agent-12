#!/usr/bin/env python3
"""Score the Stage 3 reader against the human-typed held-out labels.

This is the baseline measurement four parked decisions are waiting on:
  * Qwen2.5-VL-3B vs escalating to 7B-AWQ (vision/ocr.py docstring),
  * single-pass reading vs the deferred GraDeT-HTR fallback (Step 0 decision D2),
  * whether page-level layout hurts on tabular pages (D1, vision/layout.py),
  * the corpus word count, currently a 120-words/page floor (data/provenance.md).

It reads through the production path -- layout.detect then ocr.transcribe -- rather than
calling the model directly, so the number describes what the pipeline actually produces,
including the region cropping and the normalization applied to every chunk.

Per page it reports F1 against gold AND how the generation ended. A short chunk scores the
same whether the model gave up or we cut it off at max_new_tokens, and the pilot read
(977803f) confirmed the cap is being hit on dense pages -- so the fix that a low score
calls for cannot be chosen from the score alone.

Usage:
    python scripts/score_heldout.py                       # score every "done" label
    python scripts/score_heldout.py --page dolil_20       # one page
    python scripts/score_heldout.py --out reports/ocr_baseline.json

Needs a GPU for the real reader: the 3B checkpoint is loaded on first region.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doc_agent import config  # noqa: E402
from doc_agent.contracts import Page  # noqa: E402
from doc_agent.eval.metrics import ocr_f1  # noqa: E402
from doc_agent.vision import layout, ocr  # noqa: E402

LABELS = ROOT / "grading_kit" / "labels.jsonl"
HELDOUT_PAGES = ROOT / "grading_kit" / "heldout_pages"


def load_gold(path: Path = LABELS) -> dict[str, str]:
    """page_id -> human transcription, for labels a human has actually finished.

    A "TODO" row carries an empty `text`. Scoring against it would report a confident 0.0
    for a page nobody has read yet -- a fabricated measurement, not a missing one.
    """
    gold: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("status") != "done":
            continue
        text = (record.get("text") or "").strip()
        if text:
            gold[record["page_id"]] = text
    return gold


def score_pages(gold: dict[str, str], cfg: dict, reader: object | None = None) -> list[dict]:
    """Read each gold page through layout + OCR and score the result against its label."""
    index = ocr.load_page_index(cfg)
    missing = sorted(set(gold) - set(index))
    if missing:
        raise ValueError(f"labelled page(s) not found in the page index: {missing}")

    pages = [
        Page(id=page_id, image_path=index[page_id]["image_path"], doc_id=index[page_id]["doc_id"])
        for page_id in sorted(gold)
    ]
    regions = layout.detect(pages, cfg)

    if reader is None:
        reader = ocr.Reader(cfg)
    chunks = ocr.transcribe(regions, cfg, reader=reader)
    predicted = {chunk.page_ids[0]: chunk.text for chunk in chunks}

    stats: dict[str, list[dict]] = {}
    for stat in getattr(reader, "generation_stats", []):
        stats.setdefault(str(stat["page_id"]), []).append(stat)

    rows: list[dict[str, Any]] = []
    for page in pages:
        # A page whose regions all came back empty yields no chunk (ocr.transcribe drops
        # it). That is a total miss, not an absent measurement -- score it 0.0 and keep it
        # in the table, or the mean silently improves by losing its worst pages.
        pred = predicted.get(page.id, "")
        page_stats = stats.get(page.id, [])
        rows.append(
            {
                "page_id": page.id,
                "f1": ocr_f1(pred, gold[page.id]),
                "regions": sum(1 for r in regions if r.page_id == page.id),
                "pred_words": len(pred.split()),
                "gold_words": len(gold[page.id].split()),
                "generated_tokens": sum(int(s["generated_tokens"]) for s in page_stats),
                "truncated": any(bool(s["truncated"]) for s in page_stats),
                "pred": pred,
            }
        )
    return rows


def summarise(rows: list[dict]) -> dict:
    scores = [float(row["f1"]) for row in rows]
    return {
        "pages": len(rows),
        "mean_f1": statistics.fmean(scores) if scores else 0.0,
        "median_f1": statistics.median(scores) if scores else 0.0,
        "min_f1": min(scores) if scores else 0.0,
        "max_f1": max(scores) if scores else 0.0,
        "truncated_pages": sum(1 for row in rows if row.get("truncated")),
    }


def _print_table(rows: list[dict], summary: dict) -> None:
    print(f"\n{'page':<12}{'f1':>7}{'regions':>9}{'pred_w':>8}{'gold_w':>8}{'tokens':>8}  cut")
    for row in sorted(rows, key=lambda r: r["f1"]):
        print(
            f"{row['page_id']:<12}{row['f1']:>7.3f}{row['regions']:>9}"
            f"{row['pred_words']:>8}{row['gold_words']:>8}"
            f"{row['generated_tokens']:>8}  {'YES' if row['truncated'] else ''}"
        )
    print(
        f"\n{summary['pages']} page(s)  mean F1 {summary['mean_f1']:.3f}  "
        f"median {summary['median_f1']:.3f}  "
        f"range {summary['min_f1']:.3f}-{summary['max_f1']:.3f}  "
        f"truncated {summary['truncated_pages']}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page", action="append", help="score only this page (repeatable)")
    parser.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    parser.add_argument("--out", default=str(ROOT / "reports" / "ocr_baseline.json"))
    args = parser.parse_args(argv)

    cfg = config.load(args.config)
    # The held-out scans live in their own folder and are deliberately not in the ingest
    # sidecar; doc_id still comes from the real deed grouping, never from the page id.
    # Paths are anchored to the repo root so the script works from any working directory.
    cfg["ocr"] = {
        **cfg["ocr"],
        "sidecar": "",
        "pages_dir": str(HELDOUT_PAGES),
        "deed_groups": str(ROOT / cfg["ocr"]["deed_groups"]),
    }

    gold = load_gold()
    if args.page:
        gold = {page_id: gold[page_id] for page_id in args.page if page_id in gold}
        unknown = [page_id for page_id in args.page if page_id not in gold]
        if unknown:
            print(f"no finished label for: {', '.join(unknown)}", file=sys.stderr)
    # A labelled page is only scorable if its image is here AND it survived the dedup. Ten of
    # the 17 gold pages live in gitignored data/raw/, and dolil_66 was dropped from the deed
    # grouping entirely -- so report what is being left out and score the rest, rather than
    # letting one unresolved page block all measurement. score_pages itself stays strict.
    index = ocr.load_page_index(cfg)
    unscorable = sorted(set(gold) - set(index))
    if unscorable:
        print(
            f"skipping {len(unscorable)} labelled page(s) with no image in {HELDOUT_PAGES} "
            f"or no row in the deed grouping: {', '.join(unscorable)}",
            file=sys.stderr,
        )
        gold = {page_id: text for page_id, text in gold.items() if page_id in index}

    if not gold:
        print("nothing to score -- no label has status 'done' with text", file=sys.stderr)
        return 1

    rows = score_pages(gold, cfg)
    summary = summarise(rows)
    _print_table(rows, summary)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"model": cfg["ocr"]["model"], "config": cfg["ocr"], "summary": summary, "pages": rows},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
