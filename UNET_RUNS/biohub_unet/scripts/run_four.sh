#!/usr/bin/env bash
# The 4 top small configs (by the test proxy) at medium and large size.
#   TRAIN=1 (default): train the missing LONG versions (medium 4000 steps, large 6000), skipping any that already have eval_val.json
#   then test inference (FRAC=0.2, seed 8001) on: the 2000-step grid medium/large runs AND the long runs, small ones are already in $VOL/test_0.2.
#   DRY=1 prints the plan.   TRAIN=0 only runs inference.   Resumable.
set -uo pipefail
cd "$(dirname "$0")/.."
VOL=${VOL:-/kaggle/working/split_dataset/biohub_runs}; ROOT=${ROOT:-/kaggle/working/split_dataset}
FRAC=${FRAC:-0.2}; TRAIN=${TRAIN:-1}; OUT=${OUT:-$VOL/test_four_$FRAC}
BASES=("temporal_diff dog" "temporal_stack erode" "temporal_diff erode" "raw_only dog+edge+erode")
declare -A STEPS=([medium]=4000 [large]=6000)

if [[ "${DRY:-0}" == "1" ]]; then
  for b in "${BASES[@]}"; do read -r base ex <<< "$b"; for s in medium large; do
    n="$base+$ex@$s"; l="$n-${STEPS[$s]}"
    echo "grid: $n  [$([[ -f $VOL/grid/$n/ckpt/last.pt ]] && echo ok || echo MISSING)]   long: $l  [$([[ -f $VOL/long/$l/ckpt/last.pt ]] && echo ok || echo TO TRAIN)]"
  done; done; exit 0
fi
python3 -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null || { echo "STOP: no GPU visible to torch"; exit 1; }
{
if [[ "$TRAIN" == "1" ]]; then
  for b in "${BASES[@]}"; do read -r base ex <<< "$b"; for s in medium large; do
    st=${STEPS[$s]}; name="$base+$ex@$s-$st"; dir="$VOL/long/$name"
    [[ -f "$dir/eval_val.json" ]] && { echo "[skip] $name"; continue; }
    free=$(df --output=avail -BG "$VOL" | tail -1 | tr -dc '0-9'); [[ "$free" -ge 15 ]] || { echo "STOP: $free GB free"; exit 1; }
    mkdir -p "$dir"; echo "[$(date +%H:%M:%S)] train $name"
    if python3 -m biohub.cli train --config configs/base.yaml --run-dir "$dir" --set "name=$name" --set "features.kind=$base" \
        --set "features.extras=[${ex//+/,}]" --set "model.size=$s" --set "data.root=$ROOT" --set "data.subdir=dev" \
        --set "train.steps=$st" --set "train.ckpt_every=$st" --set "eval.dup_um=3.0" >> "$dir/train.log" 2>&1; then rm -f "$dir"/ckpt/step*.pt
    else echo "           FAILED $name"; echo "$name" >> "$VOL/long/failures.txt"; fi
  done; done
fi
mkdir -p "$OUT"
for b in "${BASES[@]}"; do read -r base ex <<< "$b"; for s in medium large; do
  for src in "$VOL/grid/$base+$ex@$s" "$VOL/long/$base+$ex@$s-${STEPS[$s]}"; do
    n=$(basename "$src"); dst="$OUT/$n"
    [[ -f "$src/ckpt/last.pt" ]] || { echo "[missing] $n"; continue; }
    mkdir -p "$dst/ckpt"; cp -n "$src/ckpt/last.pt" "$dst/ckpt/"; cp -n "$src/config.yaml" "$dst/"
    echo "[$(date +%H:%M:%S)] test inference $n"
    python3 -m biohub.cli run --run-dir "$dst" --stages evaluate,detect --split all --frac "$FRAC" \
      --set "data.root=$ROOT" --set "data.subdir=test" --set "eval.dup_um=3.0" >> "$dst/test.log" 2>&1 || echo "           FAILED $n"
  done
done; done
python3 - "$VOL" "$OUT" <<'PY'
import json, sys, glob
from pathlib import Path
vol, out = Path(sys.argv[1]), Path(sys.argv[2])
print(f"{'run':52s} dev(r1.0 r_all p/est)   test(r1.0 r_all p/est)  proxy_dev proxy_test")
for d in sorted(out.glob("*@*")):
    t = d / "eval_all.json"
    s = vol / "grid" / d.name / "eval_val.json"
    if not s.exists(): s = vol / "long" / d.name / "eval_val.json"
    if t.exists() and s.exists():
        a, b = json.loads(s.read_text()), json.loads(t.read_text())
        px = lambda m: m["recall_all"] * (1.1 - 0.1 * m["peaks_per_est"])
        f = lambda m: f"{m['recall@1.0x']:.3f} {m['recall_all']:.3f} {m['peaks_per_est']:.2f}"
        print(f"{d.name:52s} {f(a)}   {f(b)}   {px(a):.3f}  {px(b):.3f}")
PY
} 2>&1 | tee -a "$OUT.log"
