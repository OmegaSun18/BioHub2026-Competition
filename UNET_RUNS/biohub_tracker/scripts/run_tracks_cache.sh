#!/usr/bin/env bash
# SSH/CPU side: run tracker variants v0-v4 on the Kaggle cache (detections + features of the 10 models, test half of the 40 videos), score with the official metric, time tracking per video.
# No GPU, no models, no biohub needed - only the cache and the ground truth.
#   CACHE   folder holding <model>/test_0.2/{detections,features}   (the extracted cache.tar.gz -> .../cache)
#   GT      folder with the test-half <video>.geff (+ .zarr for the voxel scale): your split_dataset/test
#   OUT     where tracks/scores go (default $CACHE/../tracks)
#   VARIANTS="v0 v1 v2 v3 v4"   MODELS="m1 m2" (default: all in CACHE)   THR_DEFAULT=0.1   CDIV=9.5 CBIRTH=10   WAPP=1   JOBS=1   DRY=1
# dog_detector / log_detector (converted spot caches, scripts/convert_spots.py --sub test_0.2) have no features -> only v0 runs for them, with the score cutoff off (their own detector threshold already applied).
# Settings: thr per model for the 4 small ones = what the DEV sweep chose; every other model gets THR_DEFAULT (no dev detections exist for them). w_app is NOT tuned. Nothing here is tuned on test.
# JOBS>1 runs several at once and inflates the per-video tracking times; keep JOBS=1 for timings you will report.
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
CACHE=${CACHE:?set CACHE=.../cache}; GT=${GT:?set GT=.../split_dataset/test}
OUT=${OUT:-$(dirname "$CACHE")/tracks}
VARIANTS=${VARIANTS:-"v0 v1 v2 v3 v4"}; THR_DEFAULT=${THR_DEFAULT:-0.1}; CDIV=${CDIV:-9.5}; CBIRTH=${CBIRTH:-10}; WAPP=${WAPP:-1}; JOBS=${JOBS:-1}
if [[ -n "${MODELS:-}" ]]; then read -r -a MS <<< "$MODELS"; else mapfile -t MS < <(ls "$CACHE"); fi
export HERE CACHE GT OUT THR_DEFAULT CDIV CBIRTH WAPP PYTHONPATH="$HERE:${PYTHONPATH:-}"
thr_of() { case "$1" in temporal_diff+dog@small|temporal_diff+erode@small) echo 0.05;; temporal_stack+erode@small|raw_only+dog+edge+erode@small) echo 0.1;; dog_detector|log_detector) echo -1e9;; *) echo "$THR_DEFAULT";; esac; }
export -f thr_of
one() {
  m="$1"; v="$2"; td="$CACHE/$m/test_0.2"; d="$OUT/$m/$v"
  [[ -d "$td/detections" ]] || { echo "[missing] $m"; return; }
  [[ -f "$d/score.json" ]] && { echo "[skip] $m $v"; return; }
  fa=(); if [[ "$v" != v0 ]]; then [[ -f "$td/features/.complete" ]] || { echo "[skip] $m $v: no appearance features for this model"; return; }; fa=(--feat-dir "$td/features"); fi
  echo "[$(date +%H:%M:%S)] $m $v thr=$(thr_of "$m")"
  (cd "$HERE" && python3 -m tracker.cli run --variant "$v" "${fa[@]}" --det-dir "$td/detections" --gt-dir "$GT" --out-dir "$d" --thr="$(thr_of "$m")" --c-div "$CDIV" --c-birth "$CBIRTH" --w-app "$WAPP" --score) > "$d.log" 2>&1 \
    || { mkdir -p "$(dirname "$d")"; echo "  FAILED $m $v (see $d.log)"; }
}
export -f one
if [[ "${DRY:-0}" == "1" ]]; then for m in "${MS[@]}"; do echo "$m thr=$(thr_of "$m")"; done; echo "variants: $VARIANTS  c_div=$CDIV c_birth=$CBIRTH w_app=$WAPP jobs=$JOBS out=$OUT"; exit 0; fi
mkdir -p "$OUT"; for m in "${MS[@]}"; do mkdir -p "$OUT/$m"; done
echo "load average: $(cut -d' ' -f1-3 /proc/loadavg) on $(nproc) cpus"
for m in "${MS[@]}"; do for v in $VARIANTS; do echo "$m $v"; done; done | xargs -P "$JOBS" -L1 bash -c 'one "$0" "$1"'
python3 - "$OUT" "$VARIANTS" "${MS[@]}" <<'PY'
import json, sys
from pathlib import Path
import pandas as pd
out, vs, rows = Path(sys.argv[1]), sys.argv[2].split(), []
for m in sys.argv[3:]:
    for v in vs:
        try:
            s = json.loads((out / m / v / "score.json").read_text()); t = json.loads((out / m / v / "timing_track.json").read_text())
        except Exception:
            continue
        rows.append(dict(model=m, variant=v, thr=s["thr"], videos=s["n_videos"], edge_j=round(s["edge_jaccard"], 4), adj_edge_j=round(s["adj_edge_jaccard"], 4), div_j=round(s["division_jaccard"], 4),
                         score=round(s["score"], 4), track_s_per_video=round(t["mean_track_s_per_video"], 2)))
if rows:
    df = pd.DataFrame(rows); df.to_csv(out / "summary.csv", index=False); print("\n" + df.to_string(index=False))
PY
