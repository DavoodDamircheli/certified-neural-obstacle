"""Shared helpers for the final (freeze) phase: run registry and frozen-checkpoint loading.
All final-phase outputs go to results/expanding_disk_2d/final/ (existing results untouched)."""

import json
from pathlib import Path

import torch

from certified_obstacle_nn.experiments.expanding_disk_2d.config import RESULTS_ROOT, Config
from certified_obstacle_nn.experiments.expanding_disk_2d.model import Triple2D

FINAL_DIR = RESULTS_ROOT / "final"

# Stored runs behind the reported numbers (all kappa = 5, N_int = 20k, w_p = w_c = 10, 5000 Adam + L-BFGS).
# 6x64 seed 0 is the Stage-3 run (decision D5.1 in PROGRESS.md).
REFERENCE_6X64 = {0: "stage3_6x64_N20000_s0_k5", 1: "stage5_6x64_N20000_s1_k5", 2: "stage5_6x64_N20000_s2_k5",
                  3: "stage5_6x64_N20000_s3_k5", 4: "stage5_6x64_N20000_s4_k5"}
ARCH_RUNS = {"4x64": {0: "stage3_4x64_N20000_s0_k5"},
             "6x64": REFERENCE_6X64,
             "6x128": {0: "stage3_6x128_N20000_s0_k5"}}


def load_run(run_dir) -> tuple[Triple2D, Config]:
    """Frozen float64 triple (eval mode, no grad on parameters) + its Config."""
    run_dir = Path(run_dir)
    if not run_dir.is_absolute():
        run_dir = RESULTS_ROOT / run_dir
    cd = json.loads((run_dir / "config.json").read_text())
    cfg = Config(**{k: cd[k] for k in Config.__dataclass_fields__ if k in cd})
    torch.set_default_dtype(torch.float64)
    triple = Triple2D(cfg.depth, cfg.width, cfg.multiplier_head).double()
    triple.load_state_dict(torch.load(run_dir / "checkpoint.pt"))
    for p in triple.parameters():
        p.requires_grad_(False)
    triple.eval()
    return triple, cfg


def final_dir() -> Path:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    return FINAL_DIR
