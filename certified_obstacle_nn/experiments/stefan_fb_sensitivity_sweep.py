"""Appendix-only sensitivity sweep of E_FB over BOTH thresholds (alpha_u,
alpha_lambda), M18 2026-09-11. Loads the 20 already-trained M18 checkpoints
(results/stefan_m18/<arch>/seed<seed>/weights.pt) -- no retraining, no
recertification -- and recomputes errors/free_boundary.py's two detectors
across a grid of alternative threshold FRACTIONS, reusing one network-
evaluation grid per run (compute_free_boundary_error_sensitivity_sweep's own
design: eps_u/eps_lambda only rescale an already-computed grid).

This is diagnostic, not part of Table 1: the "official" per-run E_FB values
(results/stefan_m18/table1.csv) always use the FIXED ALPHA_U/ALPHA_LAMBDA
from errors/free_boundary.py, never a value from this sweep -- this script
exists to show how much that fixed choice matters, for the appendix.
"""

import json
from pathlib import Path

import pandas as pd
import torch

from certified_obstacle_nn.errors.free_boundary import (
    ALPHA_LAMBDA,
    ALPHA_U,
    compute_free_boundary_error_sensitivity_sweep,
)
from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.problems.stefan import StefanProblem

REPO_ROOT = Path(__file__).resolve().parents[2]
M18_DIR = REPO_ROOT / "results" / "stefan_m18"
OUT_CSV = REPO_ROOT / "results" / "stefan_m18_fb_sensitivity.csv"

# Spans roughly 4x below to 10x above the fixed default (0.02), for the
# appendix's threshold-robustness picture -- not used anywhere in Table 1.
ALPHA_U_VALUES = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
ALPHA_LAMBDA_VALUES = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]


def _load_frozen_triple(config_path: Path, weights_path: Path) -> NeuralTriple:
    config = json.loads(config_path.read_text())
    arch = config["architecture"]
    problem = StefanProblem()
    triple = NeuralTriple(problem, width=arch["width"], depth=arch["depth"], multiplier_head=arch.get("multiplier_head", "square"))
    state_dict = torch.load(weights_path, weights_only=True, map_location="cpu")
    triple.load_state_dict(state_dict)
    triple.eval()
    for p in triple.parameters():
        p.requires_grad_(False)
    return triple


def main() -> pd.DataFrame:
    problem = StefanProblem()
    rows = []

    for arch_dir in sorted(M18_DIR.iterdir()):
        if not arch_dir.is_dir() or arch_dir.name == "table1.csv":
            continue
        for seed_dir in sorted(arch_dir.iterdir()):
            weights_path = seed_dir / "weights.pt"
            config_path = seed_dir / "config.json"
            if not weights_path.exists() or not config_path.exists():
                continue
            seed = json.loads(config_path.read_text())["seed"]
            print(f"=== {arch_dir.name} seed={seed} ===", flush=True)

            triple = _load_frozen_triple(config_path, weights_path)
            sweep_rows = compute_free_boundary_error_sensitivity_sweep(
                problem, triple, problem.free_boundary_exact,
                ALPHA_U_VALUES, ALPHA_LAMBDA_VALUES, n_t=200, n_scan=401,
            )
            for row in sweep_rows:
                row["Architecture"] = arch_dir.name
                row["Seed"] = seed
                rows.append(row)

    df = pd.DataFrame.from_records(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nsensitivity sweep -> {OUT_CSV}  ({len(df)} rows, "
          f"{len(ALPHA_U_VALUES)}x{len(ALPHA_LAMBDA_VALUES)} thresholds x 20 runs)")
    print(f"fixed official values used in Table 1: ALPHA_U={ALPHA_U}, ALPHA_LAMBDA={ALPHA_LAMBDA}")
    return df


if __name__ == "__main__":
    main()
