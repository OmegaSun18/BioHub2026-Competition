# biohub

Centroid detection (and, later, tracking) for the Biohub cell-tracking data. One config file = one experiment.

## Data protocol

* **Dev** = frames 0-49 of every video. This is the only data this code opens. Training and validation videos are a fixed random split of the dev videos (`data.n_val`, `data.split_seed`).
* **Test** = frames 50-99. It lives in a separate tree and the tuning code never opens it. There is no test split in the code. Evaluate on it once, at the end, by pointing `--set data.root=<test root>` at it and running `biohub evaluate --split all` for the finished checkpoint.
* `est` in the `.geff` metadata must be the cell count of the frames that the `.zarr` holds (the count loss and the peaks-per-estimate metric divide it by the number of frames).

## Variants

| config | input channels | what it tests |
|---|---|---|
| `configs/raw_only.yaml` | `[raw_t]` | input ablation: no engineered feature |
| `configs/p_only.yaml` | `[P_t]` | input ablation: the engineered feature alone |
| `configs/raw.yaml` | `[raw_t, P_t]` | single-frame baseline |
| `configs/temporal_stack.yaml` | `[raw_t-1, raw_t, raw_t+1, P_t]` | neighbouring frames stacked as channels |
| `configs/temporal_diff.yaml` | `[raw_t, raw_t - raw_t-1, raw_t+1 - raw_t, P_t]` | frame differencing (experiment) |

`P_t` = raw -> balance -> dehaze -> floor 0.02, always computed from frame t only. All three share every other setting, so a difference between runs comes from the input only.
A new variant is one class in `src/biohub/features/builders.py` plus one YAML.

## Grid: base input x extra channels x model size

```bash
DRY=1 bash scripts/run_grid.sh                   # print the 108-run plan and parameter counts, run nothing
biohub channels --video 6bba_ab78413d --t 10 --set data.root=/kaggle/working/split_dataset --set data.subdir=dev   # look at every extra channel of a real frame first
HOURS=6 bash scripts/run_grid.sh                 # unattended; resumable; tee'd to runs/grid.log
biohub leaderboard --root runs/grid              # ranking at any time (also rewritten to runs/grid/leaderboard.csv after every run)
```

Run name = `<base>+<extras>@<size>`. Bases: raw_only, p_only, raw, temporal_stack, temporal_diff, temporal_full (stack + differences). Extras (one channel each, computed from P_t, any combination via `features.extras=[dog,edge]`): dog, edge, erode, watershed. Sizes: small (3.7M, GroupNorm, as before), medium (14.7M, BatchNorm + dropout 0.1), large (33M, BatchNorm + dropout 0.2).
Order: 1 bases x small; 2 bases x {dog, edge, erode, dog+edge+erode} x small; 3 bases x medium/large; 4 bases x watershed x small; 5 the best 4 small runs (by recall@1.0x) x medium/large; 6 the rest of the cross. `HOURS` only stops NEW runs from starting. A failed run goes to `runs/grid/failures.txt` and the grid continues. All runs share seed and settings and are scored at `eval.dup_um=3`. One seed each: use `biohub seeds` on the best few before believing a gap.
Guesses, untested: every extra's constants (DoG 2.5 um, erosion +-2 um, watershed smoothing / marker distance / threshold), dropout 0.1 / 0.2, the widths of medium / large. The count-loss forward pass (one whole frame) runs in eval mode so it cannot disturb BatchNorm statistics.

## Repeated frames

Some videos contain frames that are bit-identical to the previous frame (the acquisition repeated it); a motion model sees the next real frame as a jump of two steps.
`biohub dupframes --data-dir <folder with the .zarr videos>` counts them per video (`dup_frames.csv`). The detect stage writes a boolean `dup_frame` column into every
`detections/<video>.parquet` (True = this frame equals the previous one) and, for single-frame models, reuses the previous frame's peaks instead of running the network again.
The tracker should treat a flagged frame as "no new observation".

## Seed

`train.seed = 8001` seeds torch, numpy, the patch sampler of every loader worker and the count-loss frame sampler. `biohub diagnose` draws its samples with `np.random.default_rng(8001)`. The train/validation split of the videos uses its own `data.split_seed = 7`, unchanged, so the validation videos stay the same ones. Evaluation frames are evenly spaced, not random.

## Run

```bash
pip install -e ".[dev]"
pytest -q                                                     # synthetic smoke test, CPU, ~30 s

biohub train --config configs/raw.yaml --set data.root=/data  # train + evaluate on the validation videos
biohub train --config configs/temporal_stack.yaml --set data.root=/data
biohub evaluate --run-dir runs/raw/<stamp>                    # re-evaluate (--split val|train|all)
biohub detect   --run-dir runs/raw/<stamp> --split val        # detections/<video>.parquet for every frame
biohub run --config configs/raw.yaml --stages train,evaluate,detect
```

All five input combinations, same settings, then the comparison picture:

```bash
ROOT=/path/to/split_dataset SUBDIR=dev bash scripts/run_combos.sh      # -> runs/combos/<variant>/ and runs/combos/output.png + output.csv
biohub diagnose --run-dirs runs/combos/raw runs/combos/temporal_stack --out out.png --n-samples 12   # any subset, any sample count
```

`output.png`: one row per sampled annotated cell (video, frame, cell drawn with `default_rng(8001)`; every variant sees the same cells); columns are the raw frame, P, and the raw heat output of each variant, in linear grey, nearest-neighbour, with the annotated cell ringed in cyan and every peak above the floor as a red x. `output.csv` gives hit/miss, nearest-peak distance, heat at the cell and highest heat within 7 um for each variant and cell.

Override anything: `--set train.steps=6000 --set train.amp=fp16 --set features.floor=0.03`.

Every run writes `runs/<name>/<timestamp>/`:

```
config.yaml           the resolved config
ckpt/step*.pt, last.pt  weights + config (the checkpoint rebuilds its own model and feature kind)
train_log.jsonl       heat loss, cells/estimate, bad steps every `train.log` steps
eval_val.csv/.json    per-video and summary metrics
detections/           one parquet per video (hand-over to tracking)
.done/                stage markers; a re-run skips finished stages unless --force
```

## Metrics (evaluate.py)

* `recall@Nx`: top-N peaks per frame (N = frac x estimated cells per frame), one-to-one within 7 um, over ANNOTATED cells only.
* `recall_all`: every peak above `eval.floor`, no cap.
* `peaks_per_est`: peaks per frame / estimated cells per frame. Recall is blind to false positives; this is the only false-positive signal here.
* The summary reports each recall as the mean over videos and pooled over all annotated cells.

## Pipeline

```
train -> checkpoint -> detect -> detections/<video>.parquet -> track -> tracks/<video>.geff
                 \-> evaluate -> eval_<split>.json
```

Stages (`src/biohub/pipeline/`) communicate only through files in the run directory, so each can run as a separate process on its own. A new stage is one class with `consumes`, `produces` and `run(ctx)`, registered with `@STAGES.register("name")`. `track.py` holds the empty tracking stage and its input/output contract.

## Layout

```
src/biohub/
  config.py        typed config, YAML inherit, --set overrides
  registry.py      FEATURES, MODELS, STAGES registries
  constants.py     voxel size
  data/            io (zarr, geff), norm (per-video percentile scale), targets (heat), dataset, split
  features/        filters (gauss, dehaze, balance, P), builders (raw, temporal_stack, temporal_diff)
  models/          unet3d
  losses.py        focal heat loss, soft count, band
  train.py  infer.py  evaluate.py
  pipeline/        stage.py (contract + runner), stages.py (train, evaluate, detect), track.py (placeholder)
  cli.py
configs/  tests/  Makefile  pyproject.toml
```
