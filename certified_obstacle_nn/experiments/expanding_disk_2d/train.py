"""Adam -> L-BFGS training, copy-adapted from training/trainer.py (same schedule semantics:
full batch, Adam lr 1e-3, then L-BFGS lr 1, max_iter=1 per step, strong_wolfe, warm-started;
same abort rules: non-finite loss, loss increase, stalled line search).

Every loss component is logged every step. Additionally E_true (U-norm) is monitored on a
small INDEPENDENT Sobol set every `monitor_every_*` steps -- used only for reporting
(overnight gate: "E_true clearly decreasing"), never fed back into training.
"""

import json
import math
import random
import resource
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

from certified_obstacle_nn.experiments.expanding_disk_2d import config as config_mod
from certified_obstacle_nn.experiments.expanding_disk_2d import evaluate, sampling
from certified_obstacle_nn.experiments.expanding_disk_2d.losses import compute_loss, compute_residuals
from certified_obstacle_nn.experiments.expanding_disk_2d.model import Triple2D, n_params


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def make_run_dir(cfg):
    d = config_mod.RESULTS_ROOT / cfg.run_name
    if d.exists():  # never overwrite a previous run
        d = config_mod.RESULTS_ROOT / f"{cfg.run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    (d / "figures").mkdir(parents=True, exist_ok=False)
    return d


def _record(step, phase, loss, elapsed):
    return {"step": step, "phase": phase, "loss_total": loss.total.item(), "loss_pde": loss.pde.item(),
            "loss_sigma": loss.sigma.item(), "loss_initial": loss.initial.item(),
            "loss_penetration": loss.penetration.item(), "loss_complementarity": loss.complementarity.item(),
            "wall_s": elapsed}


def train(cfg, run_dir, log=print):
    seed_everything(cfg.seed)
    torch.set_default_dtype(torch.float64)
    triple = Triple2D(cfg.depth, cfg.width, cfg.multiplier_head).double()
    t, x, y, x0, y0 = sampling.training_points(cfg)
    t.requires_grad_(True), x.requires_grad_(True), y.requires_grad_(True)
    W = dict(w_r=cfg.w_r, w_s=cfg.w_s, w_0=cfg.w_0, w_p=cfg.w_p, w_c=cfg.w_c)
    mon_seeds = (cfg.sobol_seeds["monitor"], cfg.sobol_seeds["monitor"] + 1, cfg.sobol_seeds["monitor"] + 2)
    records, monitor = [], []
    t_start = time.time()

    def do_monitor(step, phase):
        triple.eval()
        m = evaluate.evaluate_all(triple, cfg, n_eval=cfg.n_monitor, n_trace=cfg.n_monitor // 4, seeds=mon_seeds)
        triple.train()
        monitor.append({"step": step, "phase": phase, "E_true": m["E_true"], "rho_tilde": m["rho_tilde"],
                        "err_lambda_L2": m["err_lambda_L2"], "wall_s": time.time() - t_start})
        log(f"  [{phase} {step:5d}] loss={records[-1]['loss_total']:.3e} E_true~{m['E_true']:.4e} "
            f"rho~{m['rho_tilde']:.4e} |e_lam|~{m['err_lambda_L2']:.3e} ({time.time() - t_start:.0f}s)")

    opt = torch.optim.Adam(triple.parameters(), lr=cfg.adam_lr)
    # optional piecewise-constant lr: cfg.extra["adam_lr_schedule"] = [[step, lr], ...] (absent = constant)
    lr_at = {int(s): float(v) for s, v in cfg.extra.get("adam_lr_schedule", [])}
    nan_hit = False
    for step in range(cfg.n_adam):
        if step in lr_at:
            for grp in opt.param_groups:
                grp["lr"] = lr_at[step]
            log(f"  [adam {step:5d}] lr -> {lr_at[step]:g}")
        opt.zero_grad()
        loss = compute_loss(compute_residuals(triple, t, x, y, x0, y0, cfg.kappa), **W)
        if not torch.isfinite(loss.total):
            nan_hit = True
            log(f"non-finite loss at Adam step {step}")
            break
        loss.total.backward()
        opt.step()
        records.append(_record(step, "adam", loss, time.time() - t_start))
        if step % cfg.monitor_every_adam == 0 or step == cfg.n_adam - 1:
            do_monitor(step, "adam")
    adam_time = time.time() - t_start

    lbfgs = torch.optim.LBFGS(triple.parameters(), lr=cfg.lbfgs_lr, max_iter=1, line_search_fn="strong_wolfe")
    holder = {}

    def closure():
        lbfgs.zero_grad()
        ls = compute_loss(compute_residuals(triple, t, x, y, x0, y0, cfg.kappa), **W)
        ls.total.backward()
        holder["loss"] = ls
        return ls.total

    reason, done, prev = ("skipped" if cfg.n_lbfgs == 0 or nan_hit else "completed"), 0, None
    for k in range(0 if nan_hit else cfg.n_lbfgs):
        try:
            lbfgs.step(closure)
            closure()
        except RuntimeError as exc:
            reason = f"runtime_error: {exc}"
            break
        ls = holder["loss"]
        if not torch.isfinite(ls.total):
            reason = "non_finite_loss"
            break
        tot = ls.total.item()
        gn = math.sqrt(sum(p.grad.detach().pow(2).sum().item() for p in triple.parameters() if p.grad is not None))
        if prev is not None:
            if tot > prev + 1e-9:
                reason = "loss_increased"
                break
            if abs(tot - prev) < 1e-12 and gn > 1e-6:
                reason = "line_search_stalled"
                break
        prev = tot
        done += 1
        records.append(_record(cfg.n_adam + k, "lbfgs", ls, time.time() - t_start))
        if k % cfg.monitor_every_lbfgs == 0 or k == cfg.n_lbfgs - 1:
            do_monitor(cfg.n_adam + k, "lbfgs")
    if records and monitor and monitor[-1]["step"] != records[-1]["step"]:
        do_monitor(records[-1]["step"], "final")
    train_time = time.time() - t_start

    pd.DataFrame.from_records(records).to_csv(run_dir / "loss_history.csv", index=False)
    pd.DataFrame.from_records(monitor).to_csv(run_dir / "monitor_history.csv", index=False)
    torch.save(triple.state_dict(), run_dir / "checkpoint.pt")
    info = {"train_time_s": train_time, "adam_time_s": adam_time, "lbfgs_outcome": reason,
            "lbfgs_steps_completed": done, "nan_during_adam": nan_hit, "n_params": n_params(triple),
            "n_params_per_net": {"state": n_params(triple.state), "flux": n_params(triple.flux),
                                 "mult": n_params(triple.mult)},
            "n_train_interior": int(t.shape[0]), "n_train_initial": int(x0.shape[0]),
            "peak_rss_mb_after_train": peak_rss_mb(), "torch_threads": torch.get_num_threads(),
            "device": "cpu"}
    (run_dir / "train_info.json").write_text(json.dumps(info, indent=2))
    return triple, info
