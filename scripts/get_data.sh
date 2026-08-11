#!/usr/bin/env bash
# A1 — fetch or recreate the corpus into data/raw/
#
# Corpus: Bangla Handwritten Dolil Dataset (653 Images), Mendeley Data,
#         DOI 10.17632/yk3c3xy9vm.1, CC BY 4.0.
# https://data.mendeley.com/datasets/yk3c3xy9vm/1
#
# Mendeley serves downloads via a JS "Download All" button with no stable,
# scriptable file URL, so this cannot be a plain curl one-liner. Instead:
#   1. If SRC_DIR points at an already-downloaded local copy, copy it in.
#   2. Otherwise, print the manual steps and exit non-zero.
set -euo pipefail

RAW_DIR="data/raw"
EXPECTED_COUNT=653

mkdir -p "$RAW_DIR"

if [ -n "${SRC_DIR:-}" ]; then
  if [ ! -d "$SRC_DIR" ]; then
    echo "SRC_DIR '$SRC_DIR' does not exist." >&2
    exit 1
  fi
  echo "Copying corpus from $SRC_DIR into $RAW_DIR ..."
  cp -n "$SRC_DIR"/*.jpg "$RAW_DIR"/
else
  count=$(find "$RAW_DIR" -maxdepth 1 -iname '*.jpg' | wc -l | tr -d ' ')
  if [ "$count" -ge "$EXPECTED_COUNT" ]; then
    echo "$RAW_DIR already has $count/$EXPECTED_COUNT images — nothing to do."
    exit 0
  fi
  cat >&2 <<EOF
No corpus found in $RAW_DIR (have $count/$EXPECTED_COUNT images) and SRC_DIR not set.

Manual download (Mendeley has no stable direct-download URL):
  1. Open https://data.mendeley.com/datasets/yk3c3xy9vm/1
  2. Click "Download All" and unzip it somewhere, e.g. ~/Downloads/dolil/
  3. Re-run: SRC_DIR=~/Downloads/dolil/"Dolil Dataset" scripts/get_data.sh
EOF
  exit 1
fi

count=$(find "$RAW_DIR" -maxdepth 1 -iname '*.jpg' | wc -l | tr -d ' ')
echo "$RAW_DIR now has $count images (expected $EXPECTED_COUNT)."
if [ "$count" -ne "$EXPECTED_COUNT" ]; then
  echo "WARNING: count mismatch — verify the source folder is complete." >&2
fi
