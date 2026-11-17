#!/usr/bin/env bash
# CPU/SSH side: appearance features (raw, P, dog, erode sampled at every detection) for detections that ALREADY exist here, so that tracker variants v1-v4 can run on them. No detection is redone.
# Writes <detections folder>/../features/<video>.parquet (same rows as the detections) and a marker file .complete when all videos are done. Resumable: finished videos are skipped.
#   ROOTS="dir1 dir2"  ONLY="model names"  DATA=<your split_dataset/test>  DEVICE=cpu  DRY=1
# The filter channels come from a model's own config (the channels it saw). For U-Net models: that model's run folder ($VOL/grid/<m> or $VOL/long/<m>).
# For dog_detector / log_detector (no model): the config of REF (default grid/temporal_diff+erode@small) - the filter parameters (DoG 2.5 um, gain 4, erosion 2 um) are the defaults.
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
BIOHUB=${BIOHUB:-/mnt/data/UNET_RUNS/biohub_unet}
VOL=${VOL:-/mnt/data/biohub_working/split_dataset/biohub_runs}; DATA=${DATA:?set DATA=.../split_dataset/test}; DEVICE=${DEVICE:-cpu}
ROOTS=${ROOTS:-"$VOL/test_four_0.2 $VOL/test_spots"}; REF=${REF:-grid/temporal_diff+erode@small}
export PYTHONPATH="$BIOHUB/src:$BIOHUB:$HERE:${PYTHONPATH:-}"
mapfile -t DIRS < <(for r in $ROOTS; do [[ -d "$r" ]] && find "$r" -type d -name detections | sort; done)
if [[ -n "${ONLY:-}" ]]; then keep=(); for det in "${DIRS[@]}"; do m=$(basename "$(dirname "$det")"); [[ " $ONLY " == *" $m "* ]] && keep+=("$det"); done; DIRS=("${keep[@]}"); fi
run_of() {  # model name -> its run folder (checkpoint + config)
  case "$1" in dog_detector|log_detector) echo "$VOL/$REF";; *) for k in grid long; do [[ -f "$VOL/$k/$1/ckpt/last.pt" ]] && { echo "$VOL/$k/$1"; return; }; done;; esac; }
echo "found ${#DIRS[@]} detection folders"
for det in "${DIRS[@]}"; do m=$(basename "$(dirname "$det")"); r=$(run_of "$m"); echo "  $m  videos=$(ls "$det"/*.parquet | wc -l)  config from: ${r:-NOT FOUND}"; done
[[ "${DRY:-0}" == "1" ]] && exit 0
for det in "${DIRS[@]}"; do
  m=$(basename "$(dirname "$det")"); r=$(run_of "$m"); f="$(dirname "$det")/features"
  [[ -n "$r" && -f "$r/ckpt/last.pt" ]] || { echo "[missing run folder] $m"; continue; }
  [[ -f "$f/.complete" ]] && { echo "[skip] $m (features complete)"; continue; }
  echo "[$(date +%H:%M:%S)] features $m  ($(ls "$det"/*.parquet | wc -l) videos, device $DEVICE)"
  (cd "$HERE" && python3 -m tracker.features --run-dir "$r" --data-dir "$DATA" --det-dir "$det" --out-dir "$f" --device "$DEVICE") > "$(dirname "$det")/features.log" 2>&1 \
    && [[ $(ls "$f"/*.parquet | wc -l) -eq $(ls "$det"/*.parquet | wc -l) ]] && touch "$f/.complete" && echo "  done" || echo "  FAILED/INCOMPLETE $m (see $(dirname "$det")/features.log)"
done
