#!/usr/bin/env bash
# Inference on the TEST half of the bigger (medium and large) models of the dropout run, each config twice: the 2000-step grid run and the
# dropout-0.5 run (4000 steps for medium, 6000 for large). p_only+erode@medium is left out (the over-prediction spike, 2.93 peaks/est).
# Same videos as the small-model test table (FRAC=0.2, seed 8001), so the numbers are comparable. Nothing is trained or tuned here.
#   bash scripts/run_test_large.sh
#   SKIP="raw+dog" bash scripts/run_test_large.sh     # leave out every model whose name contains this text (several: SKIP="raw+dog temporal_full+none")
#   DRY=1 bash scripts/run_test_large.sh              # print the plan only
# Every model is COPIED (ckpt/last.pt + config.yaml) into $OUT/<label>/ first. Results: $OUT/<label>/eval_all.json, detections/, and $OUT/test_results.csv.
set -uo pipefail
cd "$(dirname "$0")/.."
VOL=${VOL:-/kaggle/working/split_dataset/biohub_runs}
STAGES=${STAGES:-evaluate,detect}; FRAC=${FRAC:-0.2}; SKIP=${SKIP:-}
OUT=${OUT:-$VOL/test_large_$FRAC}
ROOT=${ROOT:-/kaggle/working/split_dataset}; SUBDIR=${SUBDIR:-test}

CONFIGS=("temporal_full+dog|medium|4000" "temporal_stack+watershed|large|6000" "temporal_full+none|large|6000" "raw+dog|large|6000")   # config|size|steps of the dropout run
MODELS=()                                                              # "label|source folder"
for c in "${CONFIGS[@]}"; do
  IFS='|' read -r cfg size steps <<< "$c"
  MODELS+=("$cfg@$size-2000|$VOL/grid/$cfg@$size" "$cfg@$size-$steps-do50|$VOL/dropout/$cfg@$size-$steps-do50")
done
KEEP=()
for m in "${MODELS[@]}"; do
  skip=0; for s in $SKIP; do [[ "${m%%|*}" == *"$s"* ]] && skip=1; done
  [[ $skip == 0 ]] && KEEP+=("$m")
done

if [[ "${DRY:-0}" == "1" ]]; then printf '%s\n' "${KEEP[@]}"; echo "data: $ROOT/$SUBDIR  frac $FRAC  out $OUT"; exit 0; fi
[[ "$SUBDIR" != "dev" ]] || { echo "STOP: SUBDIR=dev; this script is for the test half"; exit 1; }
[[ -d "$ROOT/$SUBDIR" ]] || { echo "STOP: $ROOT/$SUBDIR does not exist"; exit 1; }
python3 -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null || { echo "STOP: no GPU visible to torch"; exit 1; }
mkdir -p "$OUT"

{
for m in "${KEEP[@]}"; do
  n="${m%%|*}"; src="${m#*|}"; dst="$OUT/$n"
  [[ -f "$src/ckpt/last.pt" && -f "$src/config.yaml" ]] || { echo "[missing] $n (no ckpt/last.pt or config.yaml in $src)"; continue; }
  mkdir -p "$dst/ckpt"
  [[ -f "$dst/ckpt/last.pt" ]] || cp "$src/ckpt/last.pt" "$dst/ckpt/last.pt"
  [[ -f "$dst/config.yaml" ]] || cp "$src/config.yaml" "$dst/config.yaml"
  [[ -f "$dst/dev_eval_val.json" ]] || cp "$src/eval_val.json" "$dst/dev_eval_val.json" 2>/dev/null
  echo "[$(date +%H:%M:%S)] start $n  ->  $ROOT/$SUBDIR  (stages: $STAGES, frac $FRAC)"
  if python3 -m biohub.cli run --run-dir "$dst" --stages "$STAGES" --split all --frac "$FRAC" \
      --set "data.root=$ROOT" --set "data.subdir=$SUBDIR" --set "eval.dup_um=3.0" >> "$dst/test.log" 2>&1; then
    echo "           done: $(ls "$dst/detections" 2>/dev/null | wc -l) videos with detections"
  else
    echo "           FAILED $n (see $dst/test.log)"; echo "$n" >> "$OUT/failures.txt"
  fi
done

python3 - "$OUT" "${KEEP[@]}" <<'PY'
import json, sys
from pathlib import Path
import pandas as pd
out, models = Path(sys.argv[1]), sys.argv[2:]
keys = ["recall@1.0x", "recall_all", "peaks_per_est"]
rows = []
for m in models:
    n = m.split("|")[0]
    t, d = out / n / "eval_all.json", out / n / "dev_eval_val.json"
    if t.exists() and d.exists():
        a, b = json.loads(d.read_text()), json.loads(t.read_text())
        rows.append({"name": n, "test_videos": b.get("n_videos"), **{f"dev_{k}": a[k] for k in keys}, **{f"test_{k}": b[k] for k in keys}})
if rows:
    df = pd.DataFrame(rows)
    df.to_csv(out / "test_results.csv", index=False)
    print("\n" + df.round(3).to_string(index=False))
PY
} 2>&1 | tee -a "$OUT.log"
