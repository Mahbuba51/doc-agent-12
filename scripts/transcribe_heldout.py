#!/usr/bin/env python3
"""Fill grading_kit/labels.jsonl by hand, one held-out page at a time.

These are GOLD labels: eval/metrics.py:ocr_f1 scores the Stage 3 reader against them, so
they have to be written by someone who reads Bangla. A model-written transcription would
make the metric measure agreement-with-a-model instead of correctness, and would do it
invisibly -- if the reader and the labeller share a failure mode, the score comes out high.

Usage:
    python scripts/transcribe_heldout.py            # work through unfilled pages
    python scripts/transcribe_heldout.py --page dolil_20   # just one page
    python scripts/transcribe_heldout.py --show     # print progress and exit

Resumable: pages that already have text are skipped unless you name them with --page.
Each page is saved as soon as you finish it, so stopping half way loses nothing.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

LABELS = Path("grading_kit/labels.jsonl")
PAGES_DIR = Path("grading_kit/heldout_pages")

FIELD_PROMPTS = {
    "deed_serial": "Registry serial (as printed, or blank if illegible)",
    "deed_date": "Deed date as written",
    "land_amount": "Land amount / consideration",
    "party_names": "Party names, comma-separated",
}


def load_records() -> list[dict]:
    return [
        json.loads(line)
        for line in LABELS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def save_records(records: list[dict]) -> None:
    LABELS.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )


def open_image(page_id: str) -> None:
    """Best-effort: open the scan in the desktop viewer, else just print the path."""
    path = PAGES_DIR / f"{page_id}.jpg"
    print(f"\n  image: {path.resolve()}")
    for opener in ("xdg-open", "open"):
        try:
            subprocess.Popen(
                [opener, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            return
        except FileNotFoundError:
            continue
    print("  (no image viewer found -- open the path above manually)")


def read_multiline(prompt: str) -> str:
    """Read until a line containing only '.' -- Bangla text is pasted, not typed blind."""
    print(f"\n{prompt}")
    print("  Paste or type the transcription. End with a single '.' on its own line.")
    print("  Use [?] for anything illegible. Leave empty and enter '.' to skip this page.")
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == ".":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def transcribe(record: dict) -> bool:
    """Fill one record in place. Returns True if anything was entered."""
    page_id = record["page_id"]
    print("\n" + "=" * 72)
    print(f"PAGE {page_id}")
    if record.get("notes"):
        print(f"\n  existing notes: {record['notes'][:300]}...")
    open_image(page_id)

    text = read_multiline("Full page transcription (reading order, line breaks preserved):")
    if not text:
        print("  skipped.")
        return False
    record["text"] = text

    print("\nNow the critical fields (Enter to leave blank):")
    for key, prompt in FIELD_PROMPTS.items():
        value = input(f"  {prompt}: ").strip()
        if not value:
            continue
        record.setdefault("fields", {})[key] = (
            [v.strip() for v in value.split(",") if v.strip()]
            if key == "party_names"
            else value
        )

    record["notes"] = (record.get("notes", "") + " | HUMAN-TRANSCRIBED").lstrip(" |")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page", help="transcribe only this page id, even if already filled")
    parser.add_argument("--show", action="store_true", help="print progress and exit")
    args = parser.parse_args()

    if not LABELS.is_file():
        print(f"{LABELS} not found -- run from the repo root.", file=sys.stderr)
        return 1

    records = load_records()

    if args.show:
        done = sum(1 for r in records if r.get("text"))
        print(f"{done}/{len(records)} pages transcribed")
        for r in records:
            mark = "x" if r.get("text") else " "
            filled = sum(1 for v in r.get("fields", {}).values() if v)
            print(f"  [{mark}] {r['page_id']:10} fields {filled}/{len(FIELD_PROMPTS)}")
        return 0

    if args.page:
        targets = [r for r in records if r["page_id"] == args.page]
        if not targets:
            print(f"no held-out page called {args.page!r}", file=sys.stderr)
            return 1
    else:
        targets = [r for r in records if not r.get("text")]
    if not targets:
        print("Nothing to do -- every page already has a transcription.")
        return 0

    print(f"{len(targets)} page(s) to transcribe. Ctrl-C to stop; progress is saved as you go.")
    for record in targets:
        try:
            if transcribe(record):
                save_records(records)
                print(f"  saved {record['page_id']}")
        except KeyboardInterrupt:
            print("\nStopped. Progress saved.")
            break

    done = sum(1 for r in records if r.get("text"))
    print(f"\n{done}/{len(records)} pages now transcribed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
