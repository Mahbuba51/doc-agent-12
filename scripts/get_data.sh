#!/usr/bin/env bash
# A1 — fetch or recreate the corpus into data/raw/
#
# Corpus: Bangla Handwritten Dolil Dataset (653 Images), Mendeley Data,
#         DOI 10.17632/yk3c3xy9vm.1, CC BY 4.0.
# https://data.mendeley.com/datasets/yk3c3xy9vm/1
#
# The dataset page itself is a JS "Download All" button with no stable URL behind it, but
# Mendeley's public zip API is real and curl-able (verified: returns a signed S3 redirect,
# Content-Type: application/zip, ~2.21GB) — that's what this uses by default.
#   1. If data/raw/ already has the expected count, do nothing.
#   2. Else if SRC_DIR points at an already-downloaded local copy, copy it in (fastest if a
#      teammate already has it — no need to re-download 2.2GB per machine).
#   3. Else download+unzip via the Mendeley zip API.
set -euo pipefail

RAW_DIR="data/raw"
EXPECTED_COUNT=653
ZIP_URL="https://data.mendeley.com/public-api/zip/yk3c3xy9vm/download/1"

mkdir -p "$RAW_DIR"

count() { find "$RAW_DIR" -maxdepth 1 -iname '*.jpg' | wc -l | tr -d ' '; }

if [ "$(count)" -ge "$EXPECTED_COUNT" ]; then
  echo "$RAW_DIR already has $(count)/$EXPECTED_COUNT images — nothing to do."
  exit 0
fi

if [ -n "${SRC_DIR:-}" ]; then
  if [ ! -d "$SRC_DIR" ]; then
    echo "SRC_DIR '$SRC_DIR' does not exist." >&2
    exit 1
  fi
  echo "Copying corpus from $SRC_DIR into $RAW_DIR ..."
  cp -n "$SRC_DIR"/*.jpg "$RAW_DIR"/
else
  echo "Downloading corpus from Mendeley (~2.2GB, may take a while) ..."
  tmp_zip="$(mktemp -t dolil_XXXXXX).zip"
  tmp_extract="$(mktemp -d -t dolil_extract_XXXXXX)"
  trap 'rm -f "$tmp_zip"; rm -rf "$tmp_extract"' EXIT

  if ! curl -fL --retry 3 --retry-delay 5 -o "$tmp_zip" "$ZIP_URL"; then
    cat >&2 <<EOF
Download failed (network issue, or Mendeley changed its API since this was written).

Manual fallback:
  1. Open https://data.mendeley.com/datasets/yk3c3xy9vm/1
  2. Click "Download All" and unzip it somewhere, e.g. ~/Downloads/dolil/
  3. Re-run: SRC_DIR=~/Downloads/dolil/"Dolil Dataset" scripts/get_data.sh
EOF
    exit 1
  fi

  echo "Unzipping ..."
  unzip -q "$tmp_zip" -d "$tmp_extract"
  # The archive nests the images a few folders deep; find them wherever they landed
  # rather than assuming an exact path, so a Mendeley folder-naming change doesn't break this.
  find "$tmp_extract" -iname '*.jpg' -exec cp -n {} "$RAW_DIR"/ \;
fi

n="$(count)"
echo "$RAW_DIR now has $n images (expected $EXPECTED_COUNT)."
if [ "$n" -ne "$EXPECTED_COUNT" ]; then
  echo "WARNING: count mismatch — verify the source folder/zip is complete." >&2
fi
