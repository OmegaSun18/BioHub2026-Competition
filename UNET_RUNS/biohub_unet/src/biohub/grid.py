"""Experiment grid: base input x extra channels x model size, run one after another with a time budget, then rank.

Name of a run:  <base>+<extras>@<size>   e.g.  raw+dog+edge@medium   (extras 'none' = no extra channel)
Order of the plan (most informative first, so a stopped grid still answers the main questions):
  1. every base, no extras, small                       (the five inputs, same model as before; plus temporal_full)
  2. every base x {dog, edge, erode, dog+edge+erode}, small
  3. every base, no extras, medium and large
  4. every base x watershed, small                      (CPU-heavy, so later)
  5. the TOP_K runs so far (by recall@1.0x) x medium and large, same extras
  6. everything else of the full cross
All runs use the same seed (train.seed) and the same settings; every run is scored with distance suppression eval.dup_um.
A run that fails is logged and skipped; a run that finished is never repeated.
"""
from __future__ import annotations

import itertools
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import config as C

BASES = ["raw_only", "p_only", "raw", "temporal_stack", "temporal_diff", "temporal_full"]
EXTRA_SETS = {"none": [], "dog": ["dog"], "edge": ["edge"], "erode": ["erode"], "watershed": ["watershed"], "dog+edge+erode": ["dog", "edge", "erode"]}
SIZES = ["small", "medium", "large"]


@dataclass(frozen=True)
class Exp:
    base: str
    extras: str          # key of EXTRA_SETS
    size: str

    @property
    def name(self) -> str:
        return f"{self.base}+{self.extras}@{self.size}"


def plan(bases=BASES, extras=tuple(EXTRA_SETS), sizes=tuple(SIZES)):
    """static part of the plan (groups 1-4 and the rest of the cross); group 5 is inserted at run time from the results"""
    ex = [e for e in extras]
    g1 = [Exp(b, "none", "small") for b in bases if "none" in ex and "small" in sizes]
    g2 = [Exp(b, e, "small") for b in bases for e in ("dog", "edge", "erode", "dog+edge+erode") if e in ex and "small" in sizes]
    g3 = [Exp(b, "none", s) for b in bases for s in ("medium", "large") if "none" in ex and s in sizes]
    g4 = [Exp(b, "watershed", "small") for b in bases if "watershed" in ex and "small" in sizes]
    head = g1 + g2 + g3 + g4
    seen = set(head)
    rest = [Exp(b, e, s) for b, e, s in itertools.product(bases, ex, sizes) if Exp(b, e, s) not in seen]
    return head, rest


def overrides(e: Exp):
    return [f"features.kind={e.base}", f"features.extras=[{','.join(EXTRA_SETS[e.extras])}]", f"model.size={e.size}"]


def n_params(cfg: C.Config) -> float:
    from .features import build_features
    from .models import build_model
    b = build_features(cfg.features)
    return sum(p.numel() for p in build_model(cfg.model, b.out_ch).parameters()) / 1e6


def leaderboard(root, top=None, out_csv=None) -> pd.DataFrame:
    rows = []
    for d in sorted(Path(root).glob("*")):
        f = d / "eval_val.json"
        if not f.exists() or not (d / "config.yaml").exists():
            continue
        cfg = C.load_config(d / "config.yaml")
        s = json.loads(f.read_text())
        done = d / ".done" / "train.json"
        rows.append(dict(name=d.name, base=cfg.features.kind, extras="+".join(cfg.features.extras) or "none", size=cfg.model.size,
                         params_M=round(n_params(cfg), 1), train_min=round(json.loads(done.read_text())["seconds"] / 60, 1) if done.exists() else None,
                         **{k: s[k] for k in ("recall@0.8x", "recall@1.0x", "recall@1.2x", "recall_all", "peaks_per_est", "recall@1.0x_pooled", "recall_all_pooled") if k in s},
                         dup_um=cfg.eval.dup_um))
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("recall@1.0x", ascending=False).reset_index(drop=True)
        if out_csv:
            df.to_csv(out_csv, index=False)
    return df if top is None else df.head(top)


def _run_one(e: Exp, a, out: Path):
    run_dir = out / e.name
    args = ["train", "--config", a.config, "--run-dir", str(run_dir), "--set", f"name={e.name}", *sum((["--set", o] for o in overrides(e)), []),
            "--set", f"data.root={a.data_root}", "--set", f"data.subdir={a.subdir}", "--set", f"train.steps={a.steps}", "--set", f"eval.dup_um={a.dup_um}",
            *sum((["--set", o] for o in a.set), [])]
    if a.inprocess:
        from .cli import main
        main(args)
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "train.log", "a") as lf:
        r = subprocess.run([sys.executable, "-m", "biohub.cli", *args], stdout=lf, stderr=subprocess.STDOUT, timeout=a.run_timeout_min * 60)
    if r.returncode != 0:
        raise RuntimeError(f"exit code {r.returncode}, see {run_dir}/train.log")


def run_grid(a):
    out = Path(a.root)
    out.mkdir(parents=True, exist_ok=True)
    head, rest = plan(a.bases, a.extras, a.sizes)
    if a.dry_run:
        print(f"{len(head)} runs in the fixed part, {len(rest)} more in the rest of the cross (+ top-{a.top_k} x sizes chosen at run time)\n")
        for i, e in enumerate(head + rest, 1):
            cfg = C.load_config(a.config, overrides(e))
            print(f"{i:3d}  {e.name:45s} {n_params(cfg):6.1f}M params")
        return
    t_start, done, failed = time.time(), 0, []
    queue = list(head)
    scheduled = set(head)
    inserted_top = False

    def finished(e):
        return (out / e.name / "eval_val.json").exists()

    while queue or not inserted_top or rest:
        if not queue:
            if not inserted_top:
                inserted_top = True
                lb = leaderboard(out)
                small = lb[lb["size"] == "small"].head(a.top_k) if len(lb) else lb
                for _, r in small.iterrows():
                    for s in ("medium", "large"):
                        e = Exp(r["base"], r["extras"], s)                 # the extras label of a run is its EXTRA_SETS key
                        if s in a.sizes and e.extras in EXTRA_SETS and e not in scheduled:
                            queue.append(e)
                            scheduled.add(e)
                print(f"\nround 5: top-{a.top_k} small runs by recall@1.0x -> {len(queue)} medium/large runs queued\n")
                continue
            queue = [e for e in rest if e not in scheduled]
            scheduled.update(queue)
            rest = []
            continue
        e = queue.pop(0)
        if (time.time() - t_start) / 3600 > a.hours:
            print(f"time budget of {a.hours} h used up - stopping before {e.name} ({len(queue)} runs not started)")
            break
        if finished(e):
            print(f"[skip] {e.name} already finished")
            continue
        t0 = time.time()
        print(f"[{(time.time() - t_start) / 60:6.1f} min] start {e.name}", flush=True)
        try:
            _run_one(e, a, out)
            done += 1
            s = json.loads((out / e.name / "eval_val.json").read_text())
            print(f"           done in {(time.time() - t0) / 60:.1f} min | recall@1.0x {s['recall@1.0x']:.3f}  recall_all {s['recall_all']:.3f}  peaks/est {s['peaks_per_est']:.2f}", flush=True)
        except Exception as ex:                                              # a failed run must not stop the night
            failed.append((e.name, str(ex)))
            print(f"           FAILED {e.name}: {ex}", flush=True)
            with open(out / "failures.txt", "a") as f:
                f.write(f"{e.name}\t{ex}\n")
        leaderboard(out, out_csv=out / "leaderboard.csv")
    lb = leaderboard(out, out_csv=out / "leaderboard.csv")
    print(f"\nfinished {done} runs in {(time.time() - t_start) / 3600:.2f} h, {len(failed)} failed")
    if len(lb):
        print(lb.head(20).to_string(index=False))
    print(f"leaderboard: {out / 'leaderboard.csv'}")
