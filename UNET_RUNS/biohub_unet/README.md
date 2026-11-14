# biohub

Centroid detection (and, later, tracking) for the Biohub cell-tracking data. One config file = one experiment.

## Data protocol

* **Dev** = frames 0-49 of every video. This is the only data this code opens. Training and validation videos are a fixed random split of the dev videos (`data.n_val`, `data.split_seed`).
* **Test** = frames 50-99. It lives in a separate tree and the tuning code never opens it. There is no test split in the code. Evaluate on it once, at the end, by pointing `--set data.root=<test root>` at it and running `biohub evaluate --split all` for the finished checkpoint.
* `est` in the `.geff` metadata must be the cell count of the frames that the `.zarr` holds (the count loss and the peaks-per-estimate metric divide it by the number of frames).

## Variants

| config | input channels | what it tests |
|---|---|---|
| `configs/raw.yaml` | `[raw_t, P_t]` | single-frame baseline |
| `configs/temporal_stack.yaml` | `[raw_t-1, raw_t, raw_t+1, P_t]` | neighbouring frames stacked as channels |
| `configs/temporal_diff.yaml` | `[raw_t, raw_t - raw_t-1, raw_t+1 - raw_t, P_t]` | frame differencing (experiment) |

`P_t` = raw -> balance -> dehaze -> floor 0.02, always computed from frame t only. All three share every other setting, so a difference between runs comes from the input only.
A new variant is one class in `src/biohub/features/builders.py` plus one YAML.

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
