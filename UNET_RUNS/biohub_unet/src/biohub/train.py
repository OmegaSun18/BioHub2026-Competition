"""Training loop. Behaviour matches the single-file notebook version (focal heat loss + band count loss, AdamW + OneCycle, non-finite step guard)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import config as C
from .data import io
from .data.dataset import CentroidDataset
from .features import build_features
from .infer import amp_dtype, cut, full_stack, get_device, save_checkpoint
from .losses import band, centroid_loss, eroded, soft_count
from .models import build_model


def train(cfg: C.Config, names: list[str], data_dir: str, run_dir: Path):
    tc, dc = cfg.train, cfg.data
    device = get_device()
    torch.manual_seed(tc.seed)
    np.random.seed(tc.seed)
    builder = build_features(cfg.features)
    model = build_model(cfg.model, builder.out_ch).to(device)
    print(f"features '{cfg.features.kind}': {builder.n_frames} frame(s) -> {builder.out_ch} channels | params {sum(p.numel() for p in model.parameters())/1e6:.2f}M | device {device}")

    ds = CentroidDataset(names, data_dir, tc.steps * tc.bs, dc.patch, dc.margin, dc.sigma_um, builder.before, builder.after, tc.p_pos, tc.seed)
    dl = DataLoader(ds, batch_size=tc.bs, num_workers=tc.workers, drop_last=True, persistent_workers=tc.workers > 0)

    est_pf = {}
    if tc.count_w > 0 or tc.total_w > 0:
        for n in names:
            e = io.find_estimate(f"{data_dir}/{n}.geff")
            if e:
                est_pf[n] = e / ds.arr(n).shape[0]
        print(f"count loss: {len(est_pf)}/{len(names)} training videos have an estimate" + ("" if est_pf else " -> count loss OFF"))

    crng = np.random.default_rng(tc.seed)
    st = dict(r=[], raw=[], out=0, ema=None)
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tc.lr, total_steps=tc.steps, pct_start=0.05)
    use_amp = device == "cuda" and tc.amp != "off"
    scaler = torch.amp.GradScaler(enabled=use_amp and tc.amp == "fp16")
    adt = amp_dtype(tc.amp)
    ck_dir = run_dir / "ckpt"
    ck_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(run_dir / "train_log.jsonl", "a")

    model.train()
    run = 0.0
    ok_w = n_bad = bad_run = 0
    for step, (stack, heat) in enumerate(dl):
        stack, heat = stack.to(device, non_blocking=True), heat.to(device, non_blocking=True)
        why = None
        if not torch.isfinite(stack).all():
            why = "input"
        else:
            with torch.no_grad():
                xin = cut(builder(stack), dc.margin)
            with torch.autocast(device, dtype=adt, enabled=use_amp):
                out = model(xin)
            if not torch.isfinite(out).all():
                why = "forward"
        if why is None:
            loss = centroid_loss(out, heat, tc.w_bg)
            if not torch.isfinite(loss):
                why = "loss"
        if why is None:
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if est_pf and step >= tc.count_start and step % tc.count_every == 0:
                n = list(est_pf)[int(crng.integers(len(est_pf)))]
                carr = ds.arr(n)
                ct = int(crng.integers(carr.shape[0]))
                with torch.no_grad():
                    fc = builder(full_stack(carr, ct, ds.scale(n), builder).to(device))
                model.eval()                                                 # one whole frame must not update BatchNorm statistics nor use dropout
                with torch.autocast(device, dtype=adt, enabled=use_amp):
                    oc = model(fc)
                model.train()
                pc = torch.sigmoid(oc[0, 0].float())
                r = soft_count(eroded(pc, tc.erode)) / est_pf[n]
                st["ema"] = r.item() if st["ema"] is None else (1 - tc.ema_a) * st["ema"] + tc.ema_a * r.item()
                r_eff = r + (st["ema"] - r).detach()                       # value = running mean, gradient = this frame
                lc = tc.count_w * band(r, *tc.frame_band) + tc.total_w * band(r_eff, *tc.total_band)
                if torch.isfinite(lc):
                    if lc.requires_grad:
                        scaler.scale(lc).backward()
                    st["r"].append(r.item())
                    st["out"] += int(not tc.frame_band[0] <= r.item() <= tc.frame_band[1])
                    with torch.no_grad():
                        st["raw"].append((soft_count(pc) / est_pf[n]).item())
            scaler.unscale_(opt)
            if not torch.isfinite(nn.utils.clip_grad_norm_(model.parameters(), 5.0)):
                why = "grad"
            scaler.step(opt)
            scaler.update()
        if why is not None:
            n_bad += 1
            bad_run += 1
            if n_bad == 1:
                print(f"first non-finite step: {step+1} (in the {why})", flush=True)
            if bad_run >= tc.max_bad:
                raise RuntimeError(f"{tc.max_bad} non-finite steps in a row, last cause: {why}")
        else:
            bad_run = 0
            run += loss.item()
            ok_w += 1
        sched.step()
        if (step + 1) % tc.log == 0:
            k = max(ok_w, 1)
            rec = dict(step=step + 1, heat_loss=run / k, bad_steps=n_bad)
            cs = ""
            if st["r"]:
                rec.update(cells_per_est_eroded=float(np.mean(st["r"])), cells_per_est_raw=float(np.mean(st["raw"])), cells_per_est_running=st["ema"], frames_outside=st["out"], frames_counted=len(st["r"]))
                cs = f"  cells/estimate: eroded {rec['cells_per_est_eroded']:.2f} raw {rec['cells_per_est_raw']:.2f} running {st['ema']:.2f} | frames outside {tuple(tc.frame_band)}: {st['out']}/{len(st['r'])}"
            print(f"step {step+1}/{tc.steps}  heat loss {run/k:.4f}{cs}  bad_steps {n_bad}", flush=True)
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            run = 0.0
            ok_w = 0
            st["r"], st["raw"], st["out"] = [], [], 0
        if (step + 1) % tc.ckpt_every == 0 or step + 1 == tc.steps:
            if all(torch.isfinite(p).all() for p in model.parameters()):
                save_checkpoint(ck_dir / f"step{step+1}.pt", model, cfg, step + 1)
                save_checkpoint(ck_dir / "last.pt", model, cfg, step + 1)
            else:
                print(f"step {step+1}: weights contain NaN/inf - NOT saved", flush=True)
    log_f.close()
    return model
