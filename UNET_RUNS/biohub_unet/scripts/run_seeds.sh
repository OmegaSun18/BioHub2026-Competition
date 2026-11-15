#!/usr/bin/env bash
# Seed noise: train every variant again with other seeds (the seed-8001 runs from run_combos.sh are the first seed), then score all seeds the same way.
#   bash scripts/run_seeds.sh                       # seeds 8002 8003, all five variants (~25-35 min at 2000 steps)
#   SEEDS="8002" VARIANTS="raw temporal_diff" bash scripts/run_seeds.sh
# Data: /kaggle/working/split_dataset/dev (override with ROOT=... SUBDIR=...). Finished runs are skipped on a re-run.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=${ROOT:-/kaggle/working/split_dataset}
SUBDIR=${SUBDIR:-dev}
OUT=${OUT:-runs/seeds}
COMBOS=${COMBOS:-runs/combos}
STEPS=${STEPS:-2000}
DUP=${DUP:-3}
read -r -a SEEDS <<< "${SEEDS:-8002 8003}"
read -r -a VARIANTS <<< "${VARIANTS:-raw_only p_only raw temporal_stack temporal_diff}"

dirs=()
for v in "${VARIANTS[@]}"; do
  [[ -f "$COMBOS/$v/ckpt/last.pt" ]] && dirs+=("$COMBOS/$v")            # seed 8001, trained earlier
  for s in "${SEEDS[@]}"; do
    biohub train --config "configs/$v.yaml" --run-dir "$OUT/${v}_s$s" --set "data.root=$ROOT" --set "data.subdir=$SUBDIR" \
      --set "train.steps=$STEPS" --set "train.seed=$s" --set "eval.dup_um=$DUP"
    dirs+=("$OUT/${v}_s$s")
  done
done
biohub seeds --run-dirs "${dirs[@]}" --dup-um "$DUP" --out "$OUT/seeds.csv"
