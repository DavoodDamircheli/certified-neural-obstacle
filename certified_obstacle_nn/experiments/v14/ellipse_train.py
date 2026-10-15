"""v14 Stage 6: train the expanding-ellipse benchmark (problems/ellipse.py) with
the SAME model and schedule as the stored disk baseline (stage3_6x64_N20000_s0_k5):
6x64 tanh triple (Triple2D, mask 16 x(1-x) y(1-y), square multiplier head),
20000 scrambled-Sobol interior points + 2000 initial points (disk seed scheme
train 1000+s, init 2000+s), loss weights w_r = w_s = w_0 = 1, w_p = w_c = 10,
Adam lr 1e-3 for 5000 steps, then L-BFGS (lr 1, max_iter 1, strong Wolfe,
<= 300 steps, same abort rules). No retuning.

    nohup python3 -m certified_obstacle_nn.experiments.v14.ellipse_train --seed 0 --threads 6 \
        > logs/v14_ellipse_train_s0.log 2>&1 &

Writes results/v14_ellipse_train/<timestamp>/{config.json, losses.csv (every
component, every step), summary.csv, weights.pt (frozen state_dict), monitor.csv}.
The monitor uses the float64 diagnostics on a coarse independent Gauss grid,
for reporting only (never fed back into training).
"""

from __future__ import annotations

import argparse
import math
import random
import time

import numpy as np
import pandas as pd
import torch

from certified_obstacle_nn.certification import reliability as rel
from certified_obstacle_nn.errors.diagnostics_2d import diagnostics
from certified_obstacle_nn.experiments import results_io
from certified_obstacle_nn.experiments.expanding_disk_2d import sampling
from certified_obstacle_nn.experiments.expanding_disk_2d.losses import compute_loss, triple_fields
from certified_obstacle_nn.experiments.expanding_disk_2d.model import Triple2D, n_params
from certified_obstacle_nn.experiments.v14.common import git_commit, git_dirty, hardware, sha256_file
from certified_obstacle_nn.problems.ellipse import EllipseProblem

EXPERIMENT = "v14_ellipse_train"


def residuals(problem, triple, t, x, y, x0, y0):
    from certified_obstacle_nn.experiments.expanding_disk_2d.losses import Residuals2D

    u, u_t, grad_u, sig, div_sig, lam = triple_fields(triple, t, x, y, create_graph=True)
    f = problem.f(t, x, y)
    g = problem.g(t, x, y)
    R_PDE = f - u_t - div_sig + lam
    R_sig = sig + grad_u
    R_0 = problem.u0(x0, y0) - triple.u(torch.zeros_like(x0), x0, y0)
    return Residuals2D(R_PDE, R_sig, R_0, torch.relu(g - u), lam * torch.relu(u - g))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--n_int", type=int, default=20000)
    ap.add_argument("--n_init", type=int, default=2000)
    ap.add_argument("--n_adam", type=int, default=5000)
    ap.add_argument("--n_lbfgs", type=int, default=300)
    ap.add_argument("--w_p", type=float, default=10.0)
    ap.add_argument("--w_c", type=float, default=10.0)
    ap.add_argument("--kappa0_num", type=int, default=1)
    ap.add_argument("--kappa0_den", type=int, default=100)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--monitor_every", type=int, default=500)
    ap.add_argument("--experiment", default=EXPERIMENT, help="results/<experiment>/ (smoke tests use another name)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    k0 = a.kappa0_num / a.kappa0_den
    problem = EllipseProblem(kappa0=k0)
    variant = "main" if (a.kappa0_num, a.kappa0_den) == (1, 100) else f"VARIANT kappa0={a.kappa0_num}/{a.kappa0_den}"
    run_dir = results_io.make_run_dir(a.experiment)
    seeds = {"train": 1000 + a.seed, "init": 2000 + a.seed}
    W = dict(w_r=1.0, w_s=1.0, w_0=1.0, w_p=a.w_p, w_c=a.w_c)
    config = {
        "experiment": a.experiment, "problem": "ellipse", "variant": variant, "kappa0": [a.kappa0_num, a.kappa0_den],
        "seed": a.seed, "architecture": {"depth": a.depth, "width": a.width, "activation": "tanh",
                                         "multiplier_head": "square", "mask": "16 x(1-x) y(1-y)"},
        "n_int": a.n_int, "n_init": a.n_init, "sobol_seeds": seeds, "loss_weights": W,
        "optimizer": {"adam": {"lr": 1e-3, "steps": a.n_adam},
                      "lbfgs": {"lr": 1.0, "max_iter": 1, "line_search": "strong_wolfe", "max_steps": a.n_lbfgs}},
        "schedule_source": "identical to the disk baseline results/expanding_disk_2d/stage3_6x64_N20000_s0_k5",
        "problem_constants": rel.problem_constants_record(problem), "threads": a.threads,
        "hardware": hardware(), "git_commit": git_commit(), "git_dirty": git_dirty(),
    }
    results_io.write_config(run_dir, config)

    triple = Triple2D(a.depth, a.width, "square").double()
    t, x, y = sampling.interior(a.n_int, seeds["train"])
    x0, y0 = sampling.initial(a.n_init, seeds["init"])
    t.requires_grad_(True), x.requires_grad_(True), y.requires_grad_(True)
    records, monitor = [], []
    t0 = time.time()

    def log(msg):
        print(time.strftime("%H:%M:%S"), msg, flush=True)

    def rec(step, phase, L):
        records.append({"step": step, "phase": phase, "loss_total": L.total.item(), "loss_pde": L.pde.item(),
                        "loss_sigma": L.sigma.item(), "loss_initial": L.initial.item(),
                        "loss_penetration": L.penetration.item(), "loss_complementarity": L.complementarity.item(),
                        "wall_s": time.time() - t0})

    def mon(step, phase):
        triple.eval()
        d = diagnostics(problem, triple, n=12, q=3, n_trace=64)
        triple.train()
        monitor.append({"step": step, "phase": phase, "E_true": d["E_true"], "r2": d["r2"], "p2": d["p2"],
                        "c2": d["c2"], "E_mult": math.sqrt(d["E_mult"]), "wall_s": time.time() - t0})
        log(f"[{phase} {step}] loss={records[-1]['loss_total']:.3e} pde={records[-1]['loss_pde']:.2e} "
            f"sig={records[-1]['loss_sigma']:.2e} pen={records[-1]['loss_penetration']:.2e} "
            f"com={records[-1]['loss_complementarity']:.2e} E_true~{d['E_true']:.4e}")

    opt = torch.optim.Adam(triple.parameters(), lr=1e-3)
    nan_hit = False
    for step in range(a.n_adam):
        opt.zero_grad()
        L = compute_loss(residuals(problem, triple, t, x, y, x0, y0), **W)
        if not torch.isfinite(L.total):
            nan_hit = True
            log(f"non-finite loss at Adam step {step}")
            break
        L.total.backward()
        opt.step()
        rec(step, "adam", L)
        if step % a.monitor_every == 0 or step == a.n_adam - 1:
            mon(step, "adam")
    adam_time = time.time() - t0
    lb = torch.optim.LBFGS(triple.parameters(), lr=1.0, max_iter=1, line_search_fn="strong_wolfe")
    hold = {}

    def closure():
        lb.zero_grad()
        Lc = compute_loss(residuals(problem, triple, t, x, y, x0, y0), **W)
        Lc.total.backward()
        hold["L"] = Lc
        return Lc.total

    reason, done, prev = ("skipped" if a.n_lbfgs == 0 or nan_hit else "completed"), 0, None
    for k in range(0 if nan_hit else a.n_lbfgs):
        try:
            lb.step(closure)
            closure()
        except RuntimeError as exc:
            reason = f"runtime_error: {exc}"
            break
        L = hold["L"]
        if not torch.isfinite(L.total):
            reason = "non_finite_loss"
            break
        tot = L.total.item()
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
        rec(a.n_adam + k, "lbfgs", L)
        if k % 50 == 0:
            mon(a.n_adam + k, "lbfgs")
    train_time = time.time() - t0
    for p in triple.parameters():
        p.requires_grad_(False)
    triple.eval()
    pd.DataFrame(records).to_csv(results_io.losses_csv_path(run_dir), index=False)
    pd.DataFrame(monitor).to_csv(run_dir / "monitor.csv", index=False)
    torch.save(triple.state_dict(), run_dir / "weights.pt")
    d = diagnostics(problem, triple, n=32, q=3, n_trace=256)
    summary = {"problem": "ellipse", "variant": variant, "seed": a.seed, "train_time_s": train_time,
               "adam_time_s": adam_time, "lbfgs_outcome": reason, "lbfgs_steps_completed": done,
               "nan_during_adam": nan_hit, "n_params": n_params(triple), "final_loss": records[-1]["loss_total"],
               "E_true": d["E_true"], "E_true_vhat": d["E_true_vhat"], "E_grad": math.sqrt(d["E_grad"]),
               "E_flux": math.sqrt(d["E_flux"]), "E_graph": math.sqrt(d["E_graph"]),
               "E_mult_L2": math.sqrt(d["E_mult"]), "norm_grad_u": d["norm_grad_u"], "r2": d["r2"], "p2": d["p2"],
               "c2": d["c2"], "status_E_true": "diagnostic_float64", "weights_sha256": sha256_file(run_dir / "weights.pt"),
               "git_commit": git_commit(), "git_dirty": git_dirty()}
    results_io.write_summary(run_dir, summary)
    log(f"done: {run_dir} E_true={d['E_true']:.4e} train {train_time:.0f}s lbfgs {reason} ({done})")


if __name__ == "__main__":
    main()
