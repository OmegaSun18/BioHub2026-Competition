#!/usr/bin/env bash
# Tracker v1-v4 (appearance terms) vs v0, on the same dev val detections of the 4 small models. DEV only. Needs run_track_dev.sh to have run first (detections + sweep.csv).
#   1) features: intensity / DoG / erosion sampled at every detection with the model's own filters   (GPU, light; --device cpu possible)
#   2) for each variant v1..v4: the v0 best (thr, c_div, c_birth) from sweep/sweep.csv is held FIXED and only the appearance weight w_app is swept
#   3) table: v0 best vs the best of each variant, per model
#   bash scripts/run_variants_dev.sh        DRY=1 prints the plan      DEVICE=cpu to extract on the CPU      STAGES="features variants"
# v1 = +intensity (raw normalised)   v2 = +DoG   v3 = +DoG +intensity   v4 = +DoG +intensity +erosion.   Writes only below $OUT (default $VOL/track_dev).
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
BIOHUB=${BIOHUB:-/workspace/UNET_RUNS/biohub_unet}
VOL=${VOL:-/kaggle/working/split_dataset/biohub_runs}; ROOT=${ROOT:-/kaggle/working/split_dataset}
OUT=${OUT:-$VOL/track_dev}; STAGES=${STAGES:-"features variants"}; WAPP=${WAPP:-0.5,1,2}
DEVICE=${DEVICE:-$(python3 -c "import torch;print('cuda' if torch.cuda.is_available() else 'cpu')" 2>/dev/null || echo cpu)}
MODELS=("temporal_diff+dog@small" "temporal_stack+erode@small" "temporal_diff+erode@small" "raw_only+dog+edge+erode@small")
[[ "${DRY:-0}" == "1" ]] && { printf '%s\n' "${MODELS[@]}"; echo "out: $OUT  w_app=$WAPP  device=$DEVICE"; exit 0; }
export PYTHONPATH="$BIOHUB/src:$BIOHUB:$HERE:${PYTHONPATH:-}"
{
for n in "${MODELS[@]}"; do
  d="$OUT/$n"
  [[ -d "$d/detections" && -f "$d/sweep/sweep.csv" ]] || { echo "[missing] $n: run scripts/run_track_dev.sh first"; continue; }
  if [[ " $STAGES " == *" features "* ]]; then
    echo "[$(date +%H:%M:%S)] features $n"
    (cd "$HERE" && python3 -m tracker.features --run-dir "$d" --data-dir "$ROOT/dev" --device "$DEVICE") >> "$d/features.log" 2>&1 || { echo "  FAILED features $n (see $d/features.log)"; continue; }
  fi
  if [[ " $STAGES " == *" variants "* ]]; then
    read -r THR CD CB < <(python3 -c "import pandas as pd; r=pd.read_csv('$d/sweep/sweep.csv').iloc[0]; print(r.thr, r.c_div, r.c_birth)")
    echo "[$(date +%H:%M:%S)] variants $n  (v0 best: thr=$THR c_div=$CD c_birth=$CB)"
    for v in v1 v2 v3 v4; do
      (cd "$HERE" && python3 -m tracker.cli sweep --variant $v --feat-dir "$d/features" --det-dir "$d/detections" --gt-dir "$ROOT/dev" --out-dir "$d/variants/$v" \
          --thr "$THR" --c-div "$CD" --c-birth "$CB" --w-app "$WAPP") >> "$d/variants.log" 2>&1 || echo "  FAILED $v $n (see $d/variants.log)"
    done
  fi
done
python3 - "$OUT" "${MODELS[@]}" <<'PY'
import sys, pandas as pd
from pathlib import Path
out = Path(sys.argv[1]); rows = []
for n in sys.argv[2:]:
    d = out / n
    try:
        v0 = pd.read_csv(d / "sweep" / "sweep.csv").iloc[0]
    except Exception:
        continue
    rows.append(dict(model=n, variant="v0", w_app=0.0, score=v0.score, edge_j=v0.edge_j, div_j=v0.div_j))
    for v in ("v1", "v2", "v3", "v4"):
        f = d / "variants" / v / "sweep.csv"
        if f.exists():
            b = pd.read_csv(f).iloc[0]
            rows.append(dict(model=n, variant=v, w_app=b.w_app, score=b.score, edge_j=b.edge_j, div_j=b.div_j))
if rows:
    df = pd.DataFrame(rows); df.to_csv(out / "variants_summary.csv", index=False)
    print("\n" + df.round(4).to_string(index=False))
PY
} 2>&1 | tee -a "$OUT/variants.log"
