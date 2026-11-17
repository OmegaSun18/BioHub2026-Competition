#!/usr/bin/env bash
# Moves the host's 12 detection sets (detections + features) into ONE directory, using the same layout as the Kaggle cache:
#     $DST/<model>/test_0.2/{detections,features}          (default DST=/mnt/data/cache, where the 8 Kaggle-cache models already are)
# After this, run_tracks_cache.sh (CACHE=$DST) handles all of them. Nothing is deleted or copied: the folders are MOVED (instant on the same disk), and a symlink is left at the old place so older scripts still find them.
# A model is moved only when its features are complete (.complete). Models already in $DST are left alone. DRY=1 prints the plan without changing anything. Run inside the container as root.
set -uo pipefail
VOL=${VOL:-/mnt/data/biohub_working/split_dataset/biohub_runs}; DST=${DST:-/mnt/data/cache}
ROOTS=${ROOTS:-"$VOL/test_four_0.2 $VOL/test_spots"}
for r in $ROOTS; do
  for md in "$r"/*/; do
    m=$(basename "$md"); det="$md"detections; feat="$md"features; tgt="$DST/$m/test_0.2"
    [[ -d "$det" && ! -L "$det" ]] || continue
    [[ -e "$tgt" ]] && { echo "[exists] $m: $tgt is already there, left alone"; continue; }
    [[ -f "$feat/.complete" ]] || { echo "[not moved] $m: features not complete yet"; continue; }
    nd=$(ls "$det" | wc -l); nf=$(ls "$feat" | grep -c parquet)
    echo "[$([[ ${DRY:-0} == 1 ]] && echo plan || echo move)] $m  det=$nd feat=$nf  ->  $tgt"
    [[ "${DRY:-0}" == "1" ]] && continue
    mkdir -p "$tgt" || continue
    mv "$det" "$tgt/detections" && mv "$feat" "$tgt/features" || { echo "  FAILED moving $m"; continue; }
    [[ -f "$md/timing_detect.json" ]] && mv "$md/timing_detect.json" "$tgt/"
    ln -s "$tgt/detections" "$det"; ln -s "$tgt/features" "$feat"
    [[ $(ls "$tgt/detections" | wc -l) -eq $nd && $(ls "$tgt/features" | grep -c parquet) -eq $nf ]] && echo "  ok (counts unchanged)" || echo "  CHECK: counts differ after the move"
  done
done
