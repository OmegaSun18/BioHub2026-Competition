#!/usr/bin/env bash
# HOST: packs what Kaggle needs to compute appearance features for detections that already exist here: the detections + the config.yaml of the run they belong to.
# (The features need only the filter settings from the config - not the checkpoint, not the network - so the upload is small.)
# Default: every detection set under ROOTS that has no complete features yet. ONLY="m1 m2" to pick. Output: $OUTF (default $VOL/kaggle_features_in.tar.gz). DRY=1 lists only.
# dog_detector / log_detector use the config of REF (default grid/temporal_diff+erode@small), as in features_existing.sh.
set -uo pipefail
VOL=${VOL:-/mnt/data/biohub_working/split_dataset/biohub_runs}; REF=${REF:-grid/temporal_diff+erode@small}
ROOTS=${ROOTS:-"$VOL/test_four_0.2 $VOL/test_spots"}; OUTF=${OUTF:-$VOL/kaggle_features_in.tar.gz}
run_of() { case "$1" in dog_detector|log_detector) echo "$VOL/$REF";; *) for k in grid long; do [[ -f "$VOL/$k/$1/config.yaml" ]] && { echo "$VOL/$k/$1"; return; }; done;; esac; }
mapfile -t DIRS < <(for r in $ROOTS; do [[ -d "$r" ]] && find "$r" -type d -name detections | sort; done)
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT; n=0
for det in "${DIRS[@]}"; do
  m=$(basename "$(dirname "$det")"); base=$(dirname "$det")
  if [[ -n "${ONLY:-}" ]]; then [[ " $ONLY " == *" $m "* ]] || continue; else [[ -f "$base/features/.complete" ]] && continue; fi
  r=$(run_of "$m"); [[ -n "$r" && -f "$r/config.yaml" ]] || { echo "[missing config] $m"; continue; }
  mkdir -p "$tmp/fin/$m/run"; ln -s "$det" "$tmp/fin/$m/detections"; ln -s "$r/config.yaml" "$tmp/fin/$m/run/config.yaml"
  echo "  $m  videos=$(ls "$det"/*.parquet | wc -l)  config: $r/config.yaml"; n=$((n+1))
done
echo "$n detection sets"
[[ "${DRY:-0}" == "1" || $n -eq 0 ]] && exit 0
tar chzf "$OUTF" -C "$tmp" fin && ls -la "$OUTF"
