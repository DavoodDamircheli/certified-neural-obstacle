"""v19 Task C: controlled retraining of the ellipse multiplier (protocol results/v19_protocols/ellipse_multiplier_protocol.json).

    python3 -m certified_obstacle_nn.experiments.v19.multiplier_train --arm control|intervention --seed 3 --threads 3

The training is experiments/v14/ellipse_train.py (ellipse s0's script, same model, sampler, loss, Adam + L-BFGS schedule
and abort rules, same defaults), with two additions:
  * --arm intervention: after the (seeded) default initialisation, the output bias of M_psi is set to 1.0, so that
    lambda_psi = M_psi^2 ~ 1 at the start. The head stays lambda_psi = M_psi^2. --arm control changes nothing.
  * psi_monitor.csv, every step: max lambda_psi and the fraction of collocation points with lambda_psi > 1e-6
    (forward pass of M_psi on the training collocation points under no_grad), and ||grad_psi L_train|| (Euclidean norm
    of the multiplier network's gradient after backward). Read-only: no RNG draw, no change to any parameter or
    gradient, so the optimisation is the one of ellipse_train.py (checked by tests/test_v19_multiplier_train.py).
Writes results/v19_ellipse_multiplier_train/<timestamp>/ (config.json, losses.csv, monitor.csv, psi_monitor.csv,
summary.csv, weights.pt); the summary has the fields experiments/v14/ellipse_validated.load_triple reads.
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
from certified_obstacle_nn.experiments.expanding_disk_2d.losses import compute_loss
from certified_obstacle_nn.experiments.expanding_disk_2d.model import Triple2D, n_params
from certified_obstacle_nn.experiments.v14.common import git_commit, git_dirty, hardware, sha256_file
from certified_obstacle_nn.experiments.v14.ellipse_train import residuals
from certified_obstacle_nn.problems.ellipse import EllipseProblem

EXPERIMENT = "v19_ellipse_multiplier_train"
ARMS = {"control": None, "intervention": 1.0}  # arm -> output bias of M_psi after the default initialisation


def build(a):
    """Seeded set-up identical to ellipse_train.main (same order of RNG use), then the arm's bias change."""
    torch.set_num_threads(a.threads)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    triple = Triple2D(a.depth, a.width, "square").double()
    if ARMS[a.arm] is not None:
        with torch.no_grad():
            triple.mult.layers[-1].bias.fill_(ARMS[a.arm])
    return triple


def psi_stats(triple, t, x, y):
    with torch.no_grad():
        lam = triple.mult(torch.cat([t, x, y], dim=-1)) ** 2
        g2 = sum(p.grad.detach().pow(2).sum().item() for p in triple.mult.parameters() if p.grad is not None)
    return {"max_lambda_psi": lam.max().item(), "frac_lambda_psi_gt_1e-6": (lam > 1e-6).double().mean().item(),
            "grad_psi_norm": math.sqrt(g2)}


def train(a, run_dir=None, log=print):
    k0 = a.kappa0_num / a.kappa0_den
    problem = EllipseProblem(kappa0=k0)
    variant = "main"
    seeds = {"train": 1000 + a.seed, "init": 2000 + a.seed}
    W = dict(w_r=1.0, w_s=1.0, w_0=1.0, w_p=a.w_p, w_c=a.w_c)
    triple = build(a)
    config = {
        "experiment": a.experiment, "problem": "ellipse", "variant": variant, "kappa0": [a.kappa0_num, a.kappa0_den],
        "seed": a.seed, "architecture": {"depth": a.depth, "width": a.width, "activation": "tanh",
                                         "multiplier_head": "square", "mask": "16 x(1-x) y(1-y)"},
        "n_int": a.n_int, "n_init": a.n_init, "sobol_seeds": seeds, "loss_weights": W,
        "optimizer": {"adam": {"lr": 1e-3, "steps": a.n_adam},
                      "lbfgs": {"lr": 1.0, "max_iter": 1, "line_search": "strong_wolfe", "max_steps": a.n_lbfgs}},
        "schedule_source": "identical to the disk baseline results/expanding_disk_2d/stage3_6x64_N20000_s0_k5",
        "problem_constants": rel.problem_constants_record(problem), "threads": a.threads,
        "arm": a.arm, "mult_output_bias_init": ARMS[a.arm], "protocol": a.protocol,
        "training_script": "experiments/v19/multiplier_train.py (= experiments/v14/ellipse_train.py + arm + psi_monitor)",
        "hardware": hardware(), "git_commit": git_commit(), "git_dirty": git_dirty(),
    }
    if run_dir is not None:
        results_io.write_config(run_dir, config)
    t, x, y = sampling.interior(a.n_int, seeds["train"])
    x0, y0 = sampling.initial(a.n_init, seeds["init"])
    t.requires_grad_(True), x.requires_grad_(True), y.requires_grad_(True)
    records, monitor, psi = [], [], []
    t0 = time.time()

    def rec(step, phase, L):
        records.append({"step": step, "phase": phase, "loss_total": L.total.item(), "loss_pde": L.pde.item(),
                        "loss_sigma": L.sigma.item(), "loss_initial": L.initial.item(),
                        "loss_penetration": L.penetration.item(), "loss_complementarity": L.complementarity.item(),
                        "wall_s": time.time() - t0})
        psi.append({"step": step, "phase": phase, **psi_stats(triple, t, x, y)})

    def mon(step, phase):
        if a.monitor_every <= 0:
            return
        triple.eval()
        d = diagnostics(problem, triple, n=12, q=3, n_trace=64)
        triple.train()
        monitor.append({"step": step, "phase": phase, "E_true": d["E_true"], "r2": d["r2"], "p2": d["p2"],
                        "c2": d["c2"], "E_mult": math.sqrt(d["E_mult"]), "wall_s": time.time() - t0})
        log(f"[{phase} {step}] loss={records[-1]['loss_total']:.3e} com={records[-1]['loss_complementarity']:.2e} "
            f"max lam_psi={psi[-1]['max_lambda_psi']:.3e} |grad_psi|={psi[-1]['grad_psi_norm']:.2e} "
            f"E_true~{d['E_true']:.4e}")

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
        opt.step()  # leaves .grad untouched: psi_stats sees this step's gradient and the updated lambda_psi
        rec(step, "adam", L)
        if a.monitor_every > 0 and (step % a.monitor_every == 0 or step == a.n_adam - 1):
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
    out = {"triple": triple, "records": records, "monitor": monitor, "psi": psi, "reason": reason, "done": done,
           "nan_hit": nan_hit, "train_time": train_time, "adam_time": adam_time, "problem": problem,
           "variant": variant}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARMS), required=True)
    ap.add_argument("--seed", type=int, default=3)
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
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--monitor_every", type=int, default=500)
    ap.add_argument("--protocol", default="results/v19_protocols/ellipse_multiplier_protocol.json")
    ap.add_argument("--experiment", default=EXPERIMENT)
    a = ap.parse_args()
    run_dir = results_io.make_run_dir(a.experiment)

    def log(msg):
        print(time.strftime("%H:%M:%S"), msg, flush=True)

    o = train(a, run_dir, log)
    triple, problem = o["triple"], o["problem"]
    pd.DataFrame(o["records"]).to_csv(results_io.losses_csv_path(run_dir), index=False)
    pd.DataFrame(o["monitor"]).to_csv(run_dir / "monitor.csv", index=False)
    pd.DataFrame(o["psi"]).to_csv(run_dir / "psi_monitor.csv", index=False)
    torch.save(triple.state_dict(), run_dir / "weights.pt")
    d = diagnostics(problem, triple, n=32, q=3, n_trace=256)
    summary = {"problem": "ellipse", "variant": o["variant"], "seed": a.seed, "arm": a.arm,
               "train_time_s": o["train_time"], "adam_time_s": o["adam_time"], "lbfgs_outcome": o["reason"],
               "lbfgs_steps_completed": o["done"], "nan_during_adam": o["nan_hit"], "n_params": n_params(triple),
               "final_loss": o["records"][-1]["loss_total"], "E_true": d["E_true"], "E_true_vhat": d["E_true_vhat"],
               "E_grad": math.sqrt(d["E_grad"]), "E_flux": math.sqrt(d["E_flux"]), "E_graph": math.sqrt(d["E_graph"]),
               "E_mult_L2": math.sqrt(d["E_mult"]), "norm_grad_u": d["norm_grad_u"], "r2": d["r2"], "p2": d["p2"],
               "c2": d["c2"], "status_E_true": "diagnostic_float64",
               "weights_sha256": sha256_file(run_dir / "weights.pt"), "git_commit": git_commit(),
               "git_dirty": git_dirty()}
    results_io.write_summary(run_dir, summary)
    log(f"done: {run_dir} E_true={d['E_true']:.4e} train {o['train_time']:.0f}s lbfgs {o['reason']} ({o['done']})")


if __name__ == "__main__":
    main()
