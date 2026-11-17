#!/usr/bin/env bash
# KAGGLE side (GPU): cache detections AND appearance features of every packed model, on DEV (the model's own held-out val videos) and TEST (20% of the videos, seed 8001), with separate timings.
# Afterwards the trackers (v0-v4) need no GPU and no models: they only read $OUT.
#   MODELS_DIR   the extracted tarball's models/ folder     DEV / TEST   the split data trees     OUT   where the cache goes (default /kaggle/working/cache)
#   bash scripts/kaggle_cache.sh                  resumable: a model/split whose features folder is complete is skipped       DRY=1 prints the plan
#   ONLY="temporal_diff+erode@small" bash ...     only that model     SPLITS="test" or "dev" to do one half
# Layout: $OUT/<model>/dev_val/{detections,features,timing_detect.json}   $OUT/<model>/test_0.2/{detections,features,timing_detect.json}
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
BIOHUB=${BIOHUB:-/kaggle/working/UNET_RUNS/biohub_unet}
MODELS_DIR=${MODELS_DIR:-/kaggle/working/models}
DEV=${DEV:-/kaggle/working/split_dataset/dev}; TEST=${TEST:-/kaggle/working/split_dataset/test}
OUT=${OUT:-/kaggle/working/cache}; FRAC=${FRAC:-0.2}; SEED=${SEED:-8001}; SPLITS=${SPLITS:-"dev test"}; MIN_FREE_GB=${MIN_FREE_GB:-3}; DEVICE=${DEVICE:-cuda}
export PYTHONPATH="$BIOHUB/src:$BIOHUB:$HERE:${PYTHONPATH:-}"
if [[ -n "${ONLY:-}" ]]; then read -r -a MODELS <<< "$ONLY"; else mapfile -t MODELS < <(ls "$MODELS_DIR" 2>/dev/null); fi
if [[ "${DRY:-0}" == "1" ]]; then printf '%s\n' "${MODELS[@]}"; echo "models: ${#MODELS[@]}  dev=$DEV test=$TEST out=$OUT splits=$SPLITS"; exit 0; fi
[[ "$DEVICE" != cuda ]] || python3 -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "STOP: no GPU visible to torch (turn the GPU on in the Kaggle notebook settings)"; exit 1; }
mkdir -p "$OUT"
{
for n in "${MODELS[@]}"; do
  run="$MODELS_DIR/$n"; [[ -f "$run/ckpt/last.pt" ]] || { echo "[missing] $n"; continue; }
  for sp in $SPLITS; do
    if [[ "$sp" == "dev" ]]; then data="$DEV"; tag=dev_val; extra=(--split val); else data="$TEST"; tag="test_$FRAC"; extra=(--frac "$FRAC" --frac-seed "$SEED"); fi
    d="$OUT/$n/$tag"
    [[ -f "$d/features/.complete" ]] && { echo "[skip] $n $tag"; continue; }
    free=$(df --output=avail -BG "$OUT" | tail -1 | tr -dc '0-9'); [[ "$free" -ge "$MIN_FREE_GB" ]] || { echo "STOP: only ${free} GB free"; exit 1; }
    mkdir -p "$d"; echo "[$(date +%H:%M:%S)] detect   $n  $tag"
    (cd "$HERE" && python3 -m tracker.timed_detect --run-dir "$run" --data-dir "$data" --out-dir "$d" --device "$DEVICE" "${extra[@]}") 2>&1 | grep -v -i warn | tail -2 || { echo "  FAILED detect $n $tag"; continue; }
    [[ -f "$d/timing_detect.json" ]] || { echo "  FAILED detect $n $tag (no timing file)"; continue; }
    echo "[$(date +%H:%M:%S)] features $n  $tag"
    (cd "$HERE" && python3 -m tracker.features --run-dir "$run" --data-dir "$data" --det-dir "$d/detections" --out-dir "$d/features" --device "$DEVICE") > "$d/features.log" 2>&1 \
      && touch "$d/features/.complete" || echo "  FAILED features $n $tag (see $d/features.log)"
  done
done
echo; echo "cache size:"; du -sh "$OUT"
} 2>&1 | tee -a "$OUT.log"
