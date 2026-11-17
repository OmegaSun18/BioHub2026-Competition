#!/usr/bin/env bash
# CPU/SSH side: track the detections that ALREADY exist on this machine, variants v0-v4, score with the official metric, time tracking, and write the organisers' submission CSV.
# Finds every <root>/**/detections folder.   v1-v4 need <detections folder>/../features (run scripts/features_existing.sh first); a model without complete features only gets v0.
#   ROOTS="dir1 dir2"   ONLY="model names"   VARIANTS="v0 v1 v2 v3 v4"   GT=<your split_dataset/test>   DRY=1 lists what it found   WAPP=1  THR_DEFAULT=0.1  CDIV=9.5  CBIRTH=10  SUBM=$VOL/test_submissions
# Per model and variant: <model folder>/tracks_<variant>/{*.geff,timing_track.json,score.json}  and  $SUBM/<model>/<variant>/submission.csv (the organisers' geffs_to_csv.py, unchanged). A variant that already has score.json is skipped.
# thr: the dev-chosen value for the 4 small models, -1e9 (no cutoff) for dog_detector/log_detector, THR_DEFAULT for the rest. w_app is not tuned. Nothing is tuned on test.
# Detections written before the dup_frame column existed are tracked with every frame treated as a normal frame.
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
VOL=${VOL:-/mnt/data/biohub_working/split_dataset/biohub_runs}; GT=${GT:?set GT=.../split_dataset/test}
ROOTS=${ROOTS:-"$VOL/test_four_0.2 $VOL/test_spots"}; VARIANTS=${VARIANTS:-"v0 v1 v2 v3 v4"}
SUBM=${SUBM:-$VOL/test_submissions}; THR_DEFAULT=${THR_DEFAULT:-0.1}; CDIV=${CDIV:-9.5}; CBIRTH=${CBIRTH:-10}; WAPP=${WAPP:-1}
export PYTHONPATH="$HERE:${PYTHONPATH:-}"
thr_of() { case "$1" in temporal_diff+dog@small|temporal_diff+erode@small) echo 0.05;; temporal_stack+erode@small|raw_only+dog+edge+erode@small) echo 0.1;; dog_detector|log_detector) echo -1e9;; *) echo "$THR_DEFAULT";; esac; }
mapfile -t DIRS < <(for r in $ROOTS; do [[ -d "$r" ]] && find "$r" -type d -name detections | sort; done)
if [[ -n "${ONLY:-}" ]]; then keep=(); for det in "${DIRS[@]}"; do m=$(basename "$(dirname "$det")"); [[ " $ONLY " == *" $m "* ]] && keep+=("$det"); done; DIRS=("${keep[@]}"); fi
echo "found ${#DIRS[@]} detection folders"
for det in "${DIRS[@]}"; do
  m=$(basename "$(dirname "$det")"); n=$(ls "$det"/*.parquet 2>/dev/null | wc -l); ft=no; [[ -f "$(dirname "$det")/features/.complete" ]] && ft=yes
  echo "  $m  videos=$n  features=$ft  thr=$(thr_of "$m")  <- $det"
done
[[ "${DRY:-0}" == "1" ]] && exit 0
echo "load average: $(cut -d' ' -f1-3 /proc/loadavg) on $(nproc) cpus"
for det in "${DIRS[@]}"; do
  m=$(basename "$(dirname "$det")"); base="$(dirname "$det")"
  for v in $VARIANTS; do
    d="$base/tracks_$v"; fa=()
    [[ -f "$d/score.json" ]] && { echo "[skip] $m $v (already scored)"; continue; }
    if [[ "$v" != v0 ]]; then [[ -f "$base/features/.complete" ]] || { echo "[skip] $m $v: features not complete (run scripts/features_existing.sh)"; continue; }; fa=(--feat-dir "$base/features"); fi
    echo "[$(date +%H:%M:%S)] track $m $v  thr=$(thr_of "$m")"
    (cd "$HERE" && python3 -m tracker.cli run --variant "$v" "${fa[@]}" --det-dir "$det" --gt-dir "$GT" --out-dir "$d" --thr="$(thr_of "$m")" --c-div "$CDIV" --c-birth "$CBIRTH" --w-app "$WAPP" --score) > "$base/track_$v.log" 2>&1 \
      && grep -E "^tracking" "$base/track_$v.log" || { echo "  FAILED $m $v (see $base/track_$v.log)"; continue; }
    mkdir -p "$SUBM/$m/$v"
    (cd "$HERE" && python3 scripts/geffs_to_csv.py --in-dir "$d" --csv "$SUBM/$m/$v/submission.csv") > "$SUBM/$m/$v/csv.log" 2>&1 && echo "  submission: $SUBM/$m/$v/submission.csv" || echo "  FAILED csv $m $v (see $SUBM/$m/$v/csv.log)"
  done
done
python3 - "$VARIANTS" "${DIRS[@]}" <<'PY'
import json, sys
from pathlib import Path
import pandas as pd
vs, rows = sys.argv[1].split(), []
for det in sys.argv[2:]:
    b = Path(det).parent
    for v in vs:
        try:
            s = json.loads((b / f"tracks_{v}" / "score.json").read_text()); t = json.loads((b / f"tracks_{v}" / "timing_track.json").read_text())
        except Exception:
            continue
        rows.append(dict(model=b.name, variant=v, thr=s["thr"], videos=s["n_videos"], edge_j=round(s["edge_jaccard"], 4), adj_edge_j=round(s["adj_edge_jaccard"], 4),
                         div_j=round(s["division_jaccard"], 4), score=round(s["score"], 4), track_s_per_video=round(t["mean_track_s_per_video"], 2)))
if rows:
    df = pd.DataFrame(rows).sort_values("score", ascending=False); print("\n" + df.to_string(index=False))
PY
