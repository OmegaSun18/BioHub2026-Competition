#!/usr/bin/env bash
# HOST: puts the features computed on Kaggle (fout.tar.gz from the Kaggle cell) next to the matching detections: <model folder>/features/ + .complete.
# A model is only marked complete when it has one features file per detections file AND the same number of rows in each. Already-complete models are left untouched.
#   TAR=/path/fout.tar.gz bash scripts/unpack_features.sh        ROOTS="..."  to change where detections are searched
set -uo pipefail
VOL=${VOL:-/mnt/data/biohub_working/split_dataset/biohub_runs}; TAR=${TAR:?set TAR=.../fout.tar.gz}
ROOTS=${ROOTS:-"$VOL/test_four_0.2 $VOL/test_spots"}
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT; tar xzf "$TAR" -C "$tmp"
mapfile -t DIRS < <(for r in $ROOTS; do [[ -d "$r" ]] && find "$r" -type d -name detections | sort; done)
for src in "$tmp"/fout/*/; do
  m=$(basename "$src"); det=""
  for d in "${DIRS[@]}"; do [[ "$(basename "$(dirname "$d")")" == "$m" ]] && det="$d"; done
  [[ -n "$det" ]] || { echo "[no detections found for] $m"; continue; }
  dst="$(dirname "$det")/features"
  [[ -f "$dst/.complete" ]] && { echo "[skip] $m (already complete)"; continue; }
  mkdir -p "$dst"; cp "$src"/features/*.parquet "$dst"/
  python3 - "$det" "$dst" <<'PY' && touch "$dst/.complete" && echo "[ok] $m" || echo "[NOT complete] $m (rows or file count do not match the detections)"
import sys
from pathlib import Path
import pyarrow.parquet as pq
det, dst = Path(sys.argv[1]), Path(sys.argv[2])
a = sorted(p.name for p in det.glob("*.parquet")); b = sorted(p.name for p in dst.glob("*.parquet"))
assert a == b, f"{len(a)} detection files vs {len(b)} feature files"
for n in a:
    assert pq.read_metadata(det / n).num_rows == pq.read_metadata(dst / n).num_rows, n
PY
done
