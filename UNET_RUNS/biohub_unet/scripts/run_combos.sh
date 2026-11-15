#!/usr/bin/env bash
# Train every input combination with the same settings and seed (8001), then draw them all on the same sampled cells.
#   ROOT=/path/to/split_dataset SUBDIR=dev bash scripts/run_combos.sh
# Re-running skips finished variants (stage markers in each run folder); add FORCE=1 to redo them.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=${ROOT:?set ROOT to the folder that contains the dev data folder}
SUBDIR=${SUBDIR:-dev}
OUT=${OUT:-runs/combos}
STEPS=${STEPS:-2000}
VARIANTS=(raw_only p_only raw temporal_stack temporal_diff)
force=""; [[ "${FORCE:-0}" == "1" ]] && force="--force"

dirs=()
for v in "${VARIANTS[@]}"; do
  biohub train --config "configs/$v.yaml" --run-dir "$OUT/$v" --set "data.root=$ROOT" --set "data.subdir=$SUBDIR" --set "train.steps=$STEPS" $force
  dirs+=("$OUT/$v")
done

biohub diagnose --run-dirs "${dirs[@]}" --out "$OUT/output.png" --n-samples "${N_SAMPLES:-8}" --seed 8001
echo "pictures: $OUT/output.png   table: $OUT/output.csv   per-variant metrics: $OUT/<variant>/eval_val.json"
