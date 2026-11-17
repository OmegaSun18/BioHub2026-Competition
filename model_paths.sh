VOL=${VOL:-/mnt/data/biohub_working/split_dataset/biohub_runs}
cfgs=(temporal_diff+dog temporal_stack+erode temporal_diff+erode raw_only+dog+edge+erode)
list=()
for c in "${cfgs[@]}"; do
  list+=(grid/$c@small grid/$c@medium long/$c@medium-4000 grid/$c@large long/$c@large-6000)
done
for r in "${list[@]}"; do
  for f in ckpt/last.pt config.yaml; do
    p="$VOL/$r/$f"
    [[ -f "$p" ]] && echo "$p" || echo "MISSING  $p"
  done
done