"""Shared helpers for the v14 drivers: frozen checkpoint loading, hashes."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

import torch

from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.problems.stefan import StefanProblem

REPO_ROOT = Path(__file__).resolve().parents[3]

STEFAN_SEEDS = {s: REPO_ROOT / f"results/stefan_m18/4x64/seed{s}/weights.pt" for s in range(5)}
STEFAN_BASELINE = REPO_ROOT / "results/stefan_baseline_weights.pt"


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


# Hex digests are text. pandas 2.3.3 segfaults (C and python engines alike) when type inference
# meets a digest such as "5e72945346954026cc..." (parsed as a float with a 14-digit exponent),
# so every reader of v14 summary.csv files goes through read_summary().
HASH_COLUMNS = ("config_hash", "partition_hash", "trace0_hash", "traceT_hash", "checkpoint_sha256",
                "weights_sha256", "git_commit")


def read_summary(path):
    """pandas.read_csv of a v14 summary.csv with every hash column read as str."""
    import pandas as pd

    return pd.read_csv(path, dtype={c: str for c in HASH_COLUMNS})


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain", "certified_obstacle_nn"], cwd=REPO_ROOT, text=True)
        return bool(out.strip())
    except Exception:  # pragma: no cover
        return True


def hardware() -> dict:
    import os

    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gpu": False,
    }


def load_frozen_stefan(weights, width=64, depth=4, head="square"):
    """float64, eval(), requires_grad False on every parameter (specification section 12)."""
    p = StefanProblem()
    tr = NeuralTriple(p, width=width, depth=depth, multiplier_head=head)
    sd = torch.load(weights, map_location="cpu")
    tr.load_state_dict(sd)
    tr = tr.double().eval()
    for q in tr.parameters():
        q.requires_grad_(False)
    return p, tr
