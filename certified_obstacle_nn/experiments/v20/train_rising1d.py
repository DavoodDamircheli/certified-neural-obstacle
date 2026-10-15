"""v20 Task 3: train the rising1d triple under the frozen protocol (results/v20_protocols/rising1d_protocol.json).

    python3 -m certified_obstacle_nn.experiments.v20.train_rising1d --seed 0 --threads 8

The training is experiments/stefan_core._train_one_run (the script of the Stefan seed-0 checkpoint
results/stefan_m18/4x64/seed0) with the problem object replaced by Rising1DProblem: torch.manual_seed(seed),
NeuralTriple(width 64, depth 4, square head), TrainingSampler(5000 interior, 300 initial, seed), training.trainer.train
(5000 Adam epochs lr 1e-3, up to 300 L-BFGS epochs lr 1, strong Wolfe, unit loss weights). Added, read-only:
psi_monitor.csv every 500 Adam steps and after L-BFGS (loss components, max lambda_psi and the fraction of the 5000
interior collocation points with lambda_psi > 1e-6; no_grad forward of M_psi, no RNG use).
Writes results/v20_rising1d_train/<timestamp>/ (config.json, history.csv, psi_monitor.csv, weights.pt).
"""

from __future__ import annotations

import argparse
import json
import time

import pandas as pd
import torch

from certified_obstacle_nn.experiments import results_io
from certified_obstacle_nn.experiments.v14.common import REPO_ROOT, git_commit, git_dirty, hardware, sha256_file, sha256_json
from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.problems.rising1d import Rising1DProblem
from certified_obstacle_nn.training.sampler import TrainingSampler
from certified_obstacle_nn.training.trainer import train

EXPERIMENT = "v20_rising1d_train"
PROTOCOL = REPO_ROOT / "results/v20_protocols/rising1d_protocol.json"
# stefan_core constants (identical)
WIDTH, DEPTH = 64, 4
N_INTERIOR, N_INITIAL = 5000, 300
N_ADAM_EPOCHS, ADAM_LR = 5000, 1e-3
N_LBFGS_EPOCHS, LBFGS_LR = 300, 1.0


def make_monitor(triple, t, x, rows, wall0):
    def monitor(step, phase, loss):
        with torch.no_grad():
            lam = triple.multiplier_net(t, x)
        row = {"step": step, "phase": phase, "wall_s": time.perf_counter() - wall0,
               "max_lambda_psi": lam.max().item(), "frac_lambda_psi_gt_1e-6": (lam > 1e-6).double().mean().item()}
        if loss is not None:
            row.update({"loss_total": loss.total.item(), "loss_pde": loss.pde.item(), "loss_sigma": loss.sigma.item(),
                        "loss_initial": loss.initial.item(), "loss_penetration": loss.penetration.item(),
                        "loss_complementarity": loss.complementarity.item()})
        rows.append(row)
        print(time.strftime("%H:%M:%S"), {k: (f"{v:.4e}" if isinstance(v, float) else v) for k, v in row.items()},
              flush=True)
    return monitor


def train_one(seed: int, n_adam=N_ADAM_EPOCHS, n_lbfgs=N_LBFGS_EPOCHS, monitor_on=True):
    """Returns (triple, history, outcome, monitor rows, wall seconds). Same RNG order as stefan_core._train_one_run."""
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)
    problem = Rising1DProblem()
    triple = NeuralTriple(problem, width=WIDTH, depth=DEPTH)
    sampler = TrainingSampler(problem, n_interior=N_INTERIOR, n_initial=N_INITIAL, seed=seed)
    t, x, x0 = sampler.sample()
    rows = []
    wall0 = time.perf_counter()
    mon = make_monitor(triple, t.detach(), x.detach(), rows, wall0) if monitor_on else None
    history, outcome = train(problem, triple, t, x, x0, n_epochs=n_adam, lr=ADAM_LR, n_lbfgs_epochs=n_lbfgs,
                             lbfgs_lr=LBFGS_LR, monitor=mon)
    return triple, history, outcome, rows, time.perf_counter() - wall0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    protocol_sha = sha256_file(PROTOCOL)
    d = results_io.make_run_dir(EXPERIMENT)
    triple, history, outcome, rows, wall = train_one(a.seed)
    history.to_csv(d / "history.csv", index=False)
    pd.DataFrame(rows).to_csv(d / "psi_monitor.csv", index=False)
    torch.save(triple.state_dict(), d / "weights.pt")
    stefan_cfg = REPO_ROOT / "results/stefan_m18/4x64/seed0/config.json"
    config = {
        "experiment": EXPERIMENT, "problem": "rising1d", "seed": a.seed,
        "architecture": {"width": WIDTH, "depth": DEPTH, "multiplier_head": triple.multiplier_net.head},
        "optimizer_schedule": {
            "adam": {"lr": ADAM_LR, "n_epochs": N_ADAM_EPOCHS},
            "lbfgs": {"lr": LBFGS_LR, "n_epochs_requested": N_LBFGS_EPOCHS, "line_search": "strong_wolfe",
                      "outcome_reason": outcome.reason, "n_steps_completed": outcome.n_steps_completed},
        },
        "grid_sizes": {"n_interior": N_INTERIOR, "n_initial": N_INITIAL},
        "training_time_seconds": wall,
        "torch_threads": a.threads,
        "copied_from": {"config": str(stefan_cfg.relative_to(REPO_ROOT)), "config_file_sha256": sha256_file(stefan_cfg),
                        "config_hash_sha256_json": sha256_json(json.loads(stefan_cfg.read_text()))},
        "protocol": str(PROTOCOL.relative_to(REPO_ROOT)), "protocol_sha256": protocol_sha,
        "weights_sha256": sha256_file(d / "weights.pt"),
        "git_commit": git_commit(), "git_dirty": git_dirty(), "hardware": hardware(),
    }
    config["config_hash"] = sha256_json({k: v for k, v in config.items()
                                         if k not in ("hardware", "training_time_seconds", "weights_sha256")})
    results_io.write_config(d, config)
    print("run dir:", d.relative_to(REPO_ROOT), "weights", config["weights_sha256"][:12], f"{wall:.0f}s",
          outcome.reason, flush=True)


if __name__ == "__main__":
    main()
