#!/usr/bin/env bash
# The whole grid, unattended. Run it inside tmux, after the seed runs have finished (they share the GPU).
#   bash scripts/run_grid.sh                       # 6 h budget, 2000 steps per run, data /kaggle/working/split_dataset/dev
#   HOURS=10 bash scripts/run_grid.sh              # longer budget
#   DRY=1 bash scripts/run_grid.sh                 # only print the plan and parameter counts
# Resumable: finished runs are skipped, a failed run is logged to runs/grid/failures.txt and skipped.
# Results: runs/grid/leaderboard.csv (rewritten after every run), per-run logs in runs/grid/<run>/train.log
set -uo pipefail
cd "$(dirname "$0")/.."
extra=""; [[ "${DRY:-0}" == "1" ]] && extra="--dry-run"
biohub grid --hours "${HOURS:-6}" --steps "${STEPS:-2000}" --data-root "${ROOT:-/kaggle/working/split_dataset}" --subdir "${SUBDIR:-dev}" \
  --root "${OUT:-runs/grid}" --dup-um "${DUP:-3}" --top-k "${TOPK:-4}" $extra 2>&1 | tee -a "${OUT:-runs/grid}.log"
