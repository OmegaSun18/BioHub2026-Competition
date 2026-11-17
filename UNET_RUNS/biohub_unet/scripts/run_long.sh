#!/usr/bin/env bash
# 20 medium/large runs of the 108-run grid, retrained at their OWN size for longer. Chosen like this: among the medium and large runs
# with recall@1.0x >= 0.93 (23 runs, all within a few hundredths of the best, 0.962) the 20 with the LOWEST peaks/est (least over-prediction).
#   medium runs for 4000 steps, large runs for 6000 steps (the grid used 2000 for everything). 20 runs, same seed, data and evaluation (dup 3).
#   bash scripts/run_long.sh                 # run (inside tmux); resumable: a run with an eval_val.json is skipped
#   DRY=1 bash scripts/run_long.sh           # print the 20 runs, run nothing
# Results: runs/long/<base>+<extras>@<size>-<steps>/   ranking:  biohub leaderboard --root runs/long --top 60
# Safety: stops (does not train on) if no GPU is visible or less than MIN_FREE_GB of disk is free; a failed run goes to runs/long/failures.txt.
# Disk: only the final checkpoint last.pt is kept per run (the duplicate step<N>.pt is removed after each run).
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-runs/long}; ROOT=${ROOT:-/kaggle/working/split_dataset}; SUBDIR=${SUBDIR:-dev}
MED_STEPS=${MED_STEPS:-4000}; LARGE_STEPS=${LARGE_STEPS:-6000}; MIN_FREE_GB=${MIN_FREE_GB:-15}; RUN_TIMEOUT_H=${RUN_TIMEOUT_H:-4}
mkdir -p "$OUT"

# in order of peaks/est, lowest first: "base extras size"
CONFIGS=(
  "temporal_stack none medium" "temporal_diff dog medium" "temporal_full dog medium" "temporal_full none large" "temporal_stack edge large"
  "raw dog large" "temporal_stack dog medium" "temporal_stack watershed large" "temporal_stack dog+edge+erode medium" "temporal_diff none medium"
  "raw none medium" "temporal_full watershed large" "raw erode large" "temporal_diff edge large" "temporal_diff dog+edge+erode medium"
  "temporal_stack watershed medium" "raw_only erode large" "temporal_full erode medium" "temporal_diff dog+edge+erode large" "temporal_full none medium"
)

guards() {
  python3 -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null \
    || { echo "STOP: no GPU visible to torch (restart the container, then re-run: finished runs are skipped)"; exit 1; }
  local free; free=$(df --output=avail -BG "$OUT" | tail -1 | tr -dc '0-9')
  [[ "$free" -ge "$MIN_FREE_GB" ]] || { echo "STOP: only ${free} GB free disk (< ${MIN_FREE_GB}); free some space, then re-run"; exit 1; }
}

run_one() {
  local base=$1 ex=$2 size=$3 steps=$4
  local exs="${ex//+/,}"; [[ "$ex" == "none" ]] && exs=""      # "none" = no extra channels
  local name="${base}+${ex}@${size}-${steps}" dir="$OUT/${base}+${ex}@${size}-${steps}"
  if [[ "${DRY:-0}" == "1" ]]; then echo "$name"; return; fi
  [[ -f "$dir/eval_val.json" ]] && { echo "[skip] $name"; return; }
  guards
  mkdir -p "$dir"
  echo "[$(date +%H:%M:%S)] start $name"
  local t0=$SECONDS
  if timeout "${RUN_TIMEOUT_H}h" python3 -m biohub.cli train --config configs/base.yaml --run-dir "$dir" \
      --set "name=$name" --set "features.kind=$base" --set "features.extras=[${exs}]" --set "model.size=$size" \
      --set "data.root=$ROOT" --set "data.subdir=$SUBDIR" --set "train.steps=$steps" --set "train.ckpt_every=$steps" \
      --set "eval.dup_um=3.0" >> "$dir/train.log" 2>&1; then
    rm -f "$dir"/ckpt/step*.pt
    python3 -c "import json,sys; s=json.load(open(sys.argv[1])); print('           done in %.1f min | recall@1.0x %.3f  recall_all %.3f  peaks/est %.2f' % (float(sys.argv[2])/60, s['recall@1.0x'], s['recall_all'], s['peaks_per_est']))" "$dir/eval_val.json" "$((SECONDS - t0))"
  else
    echo "           FAILED $name (see $dir/train.log)"
    echo "$name" >> "$OUT/failures.txt"
  fi
}

for c in "${CONFIGS[@]}"; do
  read -r base ex size <<< "$c"
  if [[ "$size" == "medium" ]]; then steps=$MED_STEPS; else steps=$LARGE_STEPS; fi
  run_one "$base" "$ex" "$size" "$steps"
done 2>&1 | tee -a "$OUT.log"
