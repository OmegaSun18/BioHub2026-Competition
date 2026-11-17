#!/usr/bin/env bash
# Cache the detections of the small models on the TEST half (20% of the videos, seed 8001) WITH separate timings, then track and score them.
# Tracker settings are the ones the DEV sweep chose (first row of track_dev/<model>/sweep/sweep.csv); nothing is tuned on test.
#   detection time per video = repeated-frame check + reading frames + feature channels + network + peak search. NOT included: model loading, one warm-up pass, writing the parquet.
#   tracking time per video  = linking + building the graph. NOT included: reading the detection file, writing the .geff, scoring.
#   MODELS="temporal_diff+erode@small" MAX_VIDEOS=3 bash scripts/run_small_test_timed.sh     # quick timing of one model on 3 videos first
#   bash scripts/run_small_test_timed.sh                                                    # the 4 small models, all 40 videos     DRY=1 prints the plan
# Output: $OUT/<model>/detections/*.parquet (cached, with dup_frame), timing_detect.json, tracks_v0/*.geff + timing_track.json + score.json   and   $OUT/summary.csv
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
BIOHUB=${BIOHUB:-/mnt/data/UNET_RUNS/biohub_unet}
VOL=${VOL:-/kaggle/working/split_dataset/biohub_runs}; ROOT=${ROOT:-/kaggle/working/split_dataset}
FRAC=${FRAC:-0.2}; SEED=${SEED:-8001}; DEVICE=${DEVICE:-cpu}; MAX_VIDEOS=${MAX_VIDEOS:-}; THREADS=${THREADS:-}
OUT=${OUT:-$VOL/test_small_timed_$FRAC$([[ -n "$MAX_VIDEOS" ]] && echo "_first$MAX_VIDEOS")}
read -r -a MODELS <<< "${MODELS:-temporal_diff+dog@small temporal_stack+erode@small temporal_diff+erode@small raw_only+dog+edge+erode@small}"
[[ "${DRY:-0}" == "1" ]] && { printf '%s\n' "${MODELS[@]}"; echo "out: $OUT  test data: $ROOT/test  frac=$FRAC seed=$SEED device=$DEVICE cpus=$(nproc) load=$(cut -d' ' -f1 /proc/loadavg)"; exit 0; }
[[ -d "$ROOT/test" ]] || { echo "STOP: $ROOT/test missing"; exit 1; }
export PYTHONPATH="$BIOHUB/src:$BIOHUB:$HERE:${PYTHONPATH:-}"
mkdir -p "$OUT"
echo "load average now: $(cut -d' ' -f1-3 /proc/loadavg) on $(nproc) cpus  (another heavy job running makes the timings slower than they really are)"
{
for n in "${MODELS[@]}"; do
  src="$VOL/grid/$n"; d="$OUT/$n"
  [[ -f "$src/ckpt/last.pt" && -f "$src/config.yaml" ]] || { echo "[missing] $n"; continue; }
  read -r THR CD CB < <(python3 -c "import pandas as pd; r=pd.read_csv('$VOL/track_dev/$n/sweep/sweep.csv').iloc[0]; print(r.thr, r.c_div, r.c_birth)") || { echo "[no dev sweep] $n"; continue; }
  mkdir -p "$d"
  if [[ -f "$d/timing_detect.json" ]]; then echo "[skip detect] $n (timing_detect.json exists)"; else
    echo "[$(date +%H:%M:%S)] detect $n  (test, frac $FRAC, seed $SEED, device $DEVICE)"
    (cd "$HERE" && python3 -m tracker.timed_detect --run-dir "$src" --data-dir "$ROOT/test" --out-dir "$d" --frac "$FRAC" --frac-seed "$SEED" --device "$DEVICE" \
        ${MAX_VIDEOS:+--max-videos $MAX_VIDEOS} ${THREADS:+--threads $THREADS}) 2>&1 | tee -a "$d/detect.log" | grep -v -i warn || { echo "  FAILED detect $n"; continue; }
  fi
  echo "[$(date +%H:%M:%S)] track $n  (dev-chosen: thr=$THR c_div=$CD c_birth=$CB)"
  (cd "$HERE" && python3 -m tracker.cli run --det-dir "$d/detections" --gt-dir "$ROOT/test" --out-dir "$d/tracks_v0" --thr "$THR" --c-div "$CD" --c-birth "$CB" --score) 2>&1 | grep -v -i warn | tee -a "$d/track.log" | grep -E "tracked|tracking|edge_jaccard|adj_edge|score|n_videos|node_recall" || echo "  FAILED track $n"
done
python3 - "$OUT" "${MODELS[@]}" <<'PY'
import json, sys
from pathlib import Path
import pandas as pd
out, rows = Path(sys.argv[1]), []
for n in sys.argv[2:]:
    d = out / n
    try:
        a, t, s = (json.loads((d / f).read_text()) for f in ("timing_detect.json", "tracks_v0/timing_track.json", "tracks_v0/score.json"))
    except Exception:
        continue
    rows.append(dict(model=n, videos=a["n_videos"], model_load_s=round(a["model_load_s"], 1), warmup_s=round(a["warmup_s"], 1), detect_s_per_video=round(a["mean_detect_s_per_video"], 1),
                     detect_s_per_frame=round(a["mean_detect_s_per_frame"], 2), track_s_per_video=round(t["mean_track_s_per_video"], 2), scored_videos=s["n_videos"],
                     edge_j=round(s["edge_jaccard"], 4), adj_edge_j=round(s["adj_edge_jaccard"], 4), score=round(s["score"], 4), load_start=a["loadavg_start"]))
if rows:
    df = pd.DataFrame(rows); df.to_csv(out / "summary.csv", index=False); print("\n" + df.to_string(index=False))
PY
} 2>&1 | tee -a "$OUT.log"
