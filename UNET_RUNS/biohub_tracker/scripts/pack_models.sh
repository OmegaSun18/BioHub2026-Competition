#!/usr/bin/env bash
# HOST/container side: bundle ONLY what inference needs (ckpt/last.pt + config.yaml) of the chosen runs into one tarball you can upload to Kaggle as a dataset.
#   bash scripts/pack_models.sh            -> $VOL/kaggle_models.tar.gz       DRY=1 lists the runs and their sizes
# Runs (10): the 4 small models + the 6 medium/large runs that were NOT cached on the other machine
#   small (grid, 2000 steps): temporal_diff+dog, temporal_stack+erode, temporal_diff+erode, raw_only+dog+edge+erode
#   temporal_diff+erode@large (grid) and @large-6000 (long),  raw_only+dog+edge+erode@medium (grid), @medium-4000 (long), @large (grid), @large-6000 (long)
#   RUNS="grid/<name> long/<name> ..." overrides the list.
q
