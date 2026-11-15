#!/usr/bin/env bash
# Inference of 10 finished SMALL models on the TEST half (frames 50-99): metrics on the annotated test videos (eval_all.json) + detections of every frame.
# Test videos without annotated cells are skipped by the metrics (and still get detections). Nothing is trained or tuned here.
# The 10 are the small grid runs with the best  recall_all x (1.1 - 0.1 x peaks/est)  on the DEV data.
#   bash scripts/run_test.sh              # run (inside tmux); resumable (finished stages are skipped)
#   DRY=1 bash scripts/run_test.sh        # print the plan only
#   STAGES=detect bash scripts/run_test.sh   # detections only
#   FRAC=0.2 bash scripts/run_test.sh        # a random 20% of the test VIDEOS (all 50 frames each, so tracking still works), into $VOL/test_0.2
# Every model is COPIED (ckpt/last.pt + config.yaml) into $OUT/<name>/ first, so the grid run folders stay untouched.
# Results: $OUT/<name>/eval_all.json, $OUT/<name>/detections/<video>.parquet, and the table $OUT/test_results.csv (dev next to test).
# The table is printed in the fixed dev order, not sorted by test score: do not pick the final model from the test numbers.
set -uo pipefail
cd "$(dirname "$0")/.."
STAGES=${STAGES:-evaluate,detect}
GRID=${GRID:-/kaggle/working/split_dataset/biohub_runs/grid}      # where the grid runs (and their ckpt/last.pt) are
FRAC=${FRAC:-1.0}                                                  # e.g. FRAC=0.2: a random 20% of the test videos (whole videos, same ones for every model, seed 8001)
OUT=${OUT:-/kaggle/working/split_dataset/biohub_runs/test$([[ "$FRAC" == "1.0" ]] || echo "_$FRAC")}
ROOT=${ROOT:-/kaggle/working/split_dataset}; SUBDIR=${SUBDIR:-test}

RUNS=(
  "temporal_diff+dog@small" "raw_only+dog@small" "temporal_diff+erode@small" "temporal_stack+erode@small" "p_only+dog+edge+erode@small"
  "temporal_full+erode@small" "raw_only+erode@small" "raw_only+dog+edge+erode@small" "p_only+dog@small" "temporal_full+none@small"
)

if [[ "${DRY:-0}" == "1" ]]; then printf '%s\n' "${RUNS[@]}"; echo "data: $ROOT/$SUBDIR"; exit 0; fi
[[ "$SUBDIR" != "dev" ]] || { echo "STOP: SUBDIR=dev; this script is for the test half"; exit 1; }
[[ -d "$ROOT/$SUBDIR" ]] || { echo "STOP: $ROOT/$SUBDIR does not exist"; exit 1; }
python3 -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null || { echo "STOP: no GPU visible to torch"; exit 1; }
mkdir -p "$OUT"

{
for n in "${RUNS[@]}"; do
  src="$GRID/$n"; dst="$OUT/$n"
  [[ -f "$src/ckpt/last.pt" && -f "$src/config.yaml" ]] || { echo "[missing] $n (no ckpt/last.pt or config.yaml in $src)"; continue; }
  mkdir -p "$dst/ckpt"
  [[ -f "$dst/ckpt/last.pt" ]] || cp "$src/ckpt/last.pt" "$dst/ckpt/last.pt"
  [[ -f "$dst/config.yaml" ]] || cp "$src/config.yaml" "$dst/config.yaml"
  echo "[$(date +%H:%M:%S)] start $n  ->  $ROOT/$SUBDIR  (stages: $STAGES)"
  if python3 -m biohub.cli run --run-dir "$dst" --stages "$STAGES" --split all \
      --frac "$FRAC" --set "data.root=$ROOT" --set "data.subdir=$SUBDIR" --set "eval.dup_um=3.0" >> "$dst/test.log" 2>&1; then
    echo "           done: $(ls "$dst/detections" 2>/dev/null | wc -l) videos with detections"
  else
    echo "           FAILED $n (see $dst/test.log)"; echo "$n" >> "$OUT/failures.txt"
  fi
done

python3 - "$GRID" "$OUT" "${RUNS[@]}" <<'PY'
import json, sys
from pathlib import Path
import pandas as pd
grid, out, names = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
keys = ["recall@1.0x", "recall_all", "peaks_per_est"]
rows = []
for n in names:
    t, d = out / n / "eval_all.json", grid / n / "eval_val.json"
    if t.exists() and d.exists():
        a, b = json.loads(d.read_text()), json.loads(t.read_text())
        rows.append({"name": n, "test_videos": b.get("n_videos"), **{f"dev_{k}": a[k] for k in keys}, **{f"test_{k}": b[k] for k in keys}})
if rows:
    df = pd.DataFrame(rows)
    df.to_csv(out / "test_results.csv", index=False)
    print("\n" + df.round(3).to_string(index=False))
PY
} 2>&1 | tee -a "$OUT.log"
