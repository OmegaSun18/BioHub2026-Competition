#!/usr/bin/env bash
# Tracker v0 on the 4 top small models, DEV only (the held-out 'val' videos of dev; nothing from test is touched).
#   1) detections of the val videos with each small model   (GPU, small; copies ckpt+config, the grid run folders are not touched)
#   2) tracker sweep over thr / c_div / c_birth, scored with the organisers' metric   (CPU only)
# Writes ONLY to $OUT (default $VOL/track_dev). Does not modify biohub_unet or any run folder, so it can run next to the medium/large jobs.
#   bash scripts/run_track_dev.sh            DRY=1 to print the plan     STAGES="detect sweep"
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
BIOHUB=${BIOHUB:-/workspace/UNET_RUNS/biohub_unet}
VOL=${VOL:-/kaggle/working/split_dataset/biohub_runs}; ROOT=${ROOT:-/kaggle/working/split_dataset}
OUT=${OUT:-$VOL/track_dev}; STAGES=${STAGES:-"detect sweep"}
THR=${THR:-0.2,0.35,0.5}; CDIV=${CDIV:-8,10.5,11.5}; CBIRTH=${CBIRTH:-10,14}
MODELS=("temporal_diff+dog@small" "temporal_stack+erode@small" "temporal_diff+erode@small" "raw_only+dog+edge+erode@small")
[[ "${DRY:-0}" == "1" ]] && { printf '%s\n' "${MODELS[@]}"; echo "out: $OUT  data: $ROOT/dev  thr=$THR c_div=$CDIV c_birth=$CBIRTH"; exit 0; }
mkdir -p "$OUT"
{
for n in "${MODELS[@]}"; do
  d="$OUT/$n"; src="$VOL/grid/$n"
  if [[ " $STAGES " == *" detect "* ]]; then
    [[ -f "$src/ckpt/last.pt" ]] || { echo "[missing] $n"; continue; }
    mkdir -p "$d/ckpt"; cp -n "$src/ckpt/last.pt" "$d/ckpt/"; cp -n "$src/config.yaml" "$d/"
    echo "[$(date +%H:%M:%S)] detect (dev val videos) $n"
    (cd "$BIOHUB" && python3 -m biohub.cli run --run-dir "$d" --stages detect --split val --set "data.root=$ROOT" --set "data.subdir=dev" --set "eval.dup_um=3.0") >> "$d/track_dev.log" 2>&1 || { echo "  FAILED detect $n (see $d/track_dev.log)"; continue; }
  fi
  if [[ " $STAGES " == *" sweep "* ]]; then
    echo "[$(date +%H:%M:%S)] tracker sweep $n"
    (cd "$HERE" && python3 -m tracker.cli sweep --det-dir "$d/detections" --gt-dir "$ROOT/dev" --out-dir "$d/sweep" --thr "$THR" --c-div "$CDIV" --c-birth "$CBIRTH") >> "$d/sweep.log" 2>&1 \
      && { echo "  best of $n:"; head -3 "$d/sweep/sweep.csv"; } || echo "  FAILED sweep $n (see $d/sweep.log)"
  fi
done
} 2>&1 | tee -a "$OUT.log"
