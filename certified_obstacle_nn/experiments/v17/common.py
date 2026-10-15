"""v17 shared helpers: load a STORED validated run (v14 run directory) with its
frozen checkpoint, encloser, volume partition and trace partitions, every
piece checked against the hashes recorded in the run's summary.csv.

Nothing here trains or modifies a model (specification section 6): checkpoints are
loaded read-only, float64, eval(), requires_grad False.

Partitions:
  * Stefan uniform runs: uniform dyadic n x n volume partition, uniform 1-D traces
    (trace_n cells), reconstructed exactly.
  * 2-D runs: volume partition = lo/hi stored in cells.npz; trace partitions are
    not stored and are reconstructed by boxes.adaptive_traces with the stored
    trace_budget / theta (deterministic, numpy backend).
Every reconstructed partition must reproduce partition_hash / trace0_hash /
traceT_hash of the stored summary, otherwise StoredRunError is raised.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from certified_obstacle_nn.certification.enclose import boxes as Bx
from certified_obstacle_nn.certification.enclose import upper_sum as US
from certified_obstacle_nn.experiments.v14.common import REPO_ROOT, load_frozen_stefan, read_summary, sha256_file

V16_REF = REPO_ROOT / "docs/v16_reference/v16"


class StoredRunError(RuntimeError):
    pass


@dataclass
class StoredRun:
    run_dir: Path
    config: dict
    summary: dict  # one row (first backend)
    problem_name: str
    problem: object
    triple: object
    encloser: object
    checkpoint: Path
    checkpoint_sha256: str
    part: object
    tp0: object
    tpT: object

    @property
    def frozen(self) -> bool:
        return all(not q.requires_grad for q in self.triple.parameters()) and not self.triple.training

    def identity(self) -> dict:
        s = self.summary
        return {"checkpoint_sha256": self.checkpoint_sha256, "partition_hash": s["partition_hash"],
                "trace0_hash": s["trace0_hash"], "traceT_hash": s["traceT_hash"],
                "source_run": str(self.run_dir.relative_to(REPO_ROOT))}


def _freeze(triple):
    for p in triple.parameters():
        p.requires_grad_(False)
    triple.eval()
    return triple


def _find_weights(run_dir: Path, sha: str) -> Path:
    """The run's own weights.pt, else any stored file with the same SHA-256 (ellipse 1e6 runs
    did not keep a copy; their checkpoint is the 1e5 run's / the training run's weights.pt)."""
    cands = [run_dir / "weights.pt"]
    cands += sorted((REPO_ROOT / "results").glob(f"{run_dir.parent.name}/*/weights.pt"))
    cands += sorted((REPO_ROOT / "results/v14_ellipse_train").glob("*/weights.pt"))
    for c in cands:
        if c.exists() and sha256_file(c) == sha:
            return c
    raise StoredRunError(f"no stored checkpoint with sha256 {sha[:12]} for {run_dir}")


def load_stored_run(run_dir, *, backend_row: str | None = None, workers: int = 1) -> StoredRun:
    run_dir = Path(run_dir)
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    cfg = json.loads((run_dir / "config.json").read_text())
    rows = read_summary(run_dir / "summary.csv").to_dict("records")
    row = rows[0] if backend_row is None else next(r for r in rows if r["backend"] == backend_row)
    sha = row["checkpoint_sha256"]
    name = cfg.get("problem", "stefan")
    if cfg["experiment"] == "v14_stefan_validated":
        name = "stefan"
    torch.set_default_dtype(torch.float64)
    if name == "stefan":
        from certified_obstacle_nn.certification.enclose.engine import StefanAdapter, TripleEncloser

        ck = REPO_ROOT / cfg["checkpoint"]
        if sha256_file(ck) != sha:
            raise StoredRunError(f"checkpoint hash mismatch {ck}")
        problem, triple = load_frozen_stefan(ck)
        enc = TripleEncloser(triple, StefanAdapter(problem))
        if cfg["mode"] != "uniform":
            raise StoredRunError("only uniform Stefan runs are reconstructed")
        n = int(cfg["n"])
        part = US.uniform_2d(n, n)
        tp0 = US.uniform_1d(int(cfg["trace_n"]))
        tpT = US.uniform_1d(int(cfg["trace_n"]))
    elif name == "disk2d":
        from certified_obstacle_nn.certification.enclose.engine2d import DiskAdapter, TripleEncloser2D
        from certified_obstacle_nn.experiments.expanding_disk_2d.final_common import load_run
        from certified_obstacle_nn.problems.disk import DiskProblem

        ck = REPO_ROOT / cfg["checkpoint"]
        if sha256_file(ck) != sha:
            raise StoredRunError(f"checkpoint hash mismatch {ck}")
        triple, tcfg = load_run(ck.parent)
        problem = DiskProblem(tcfg.kappa)
        enc = TripleEncloser2D(triple, DiskAdapter(tcfg.kappa, tcfg.Lambda, tcfg.alpha))
        part = tp0 = tpT = None
    elif name == "ellipse":
        from certified_obstacle_nn.certification.enclose.engine2d import EllipseAdapter, TripleEncloser2D
        from certified_obstacle_nn.experiments.expanding_disk_2d.model import Triple2D
        from certified_obstacle_nn.problems.ellipse import EllipseProblem

        ck = _find_weights(run_dir, sha)
        arch = cfg["training_config"]["architecture"]
        triple = Triple2D(arch["depth"], arch["width"], arch["multiplier_head"]).double()
        triple.load_state_dict(torch.load(ck))
        _freeze(triple)
        k0 = tuple(cfg["kappa0"])
        problem = EllipseProblem(kappa0=k0[0] / k0[1])
        enc = TripleEncloser2D(triple, EllipseAdapter(*k0))
        part = tp0 = tpT = None
    else:
        raise StoredRunError(f"unknown problem {name}")
    if name != "stefan":
        cells = run_dir / "cells.npz"
        if not cells.exists():
            raise StoredRunError(f"{run_dir}: volume partition not stored (no cells.npz)")
        z = np.load(cells)
        part = Bx.BoxPartition(z["lo"].copy(), z["hi"].copy())
        frozen = sorted((REPO_ROOT / "results/v17_stage0").glob(f"*/traces/{run_dir.parent.name}__{run_dir.name}.npz"))
        if frozen:  # Stage 0 froze the reconstructed traces; hashes are re-checked below
            zt = np.load(frozen[-1])
            tp0, tpT = Bx.BoxPartition(zt["lo0"], zt["hi0"]), Bx.BoxPartition(zt["loT"], zt["hiT"])
        else:
            tps = Bx.adaptive_traces(enc, D=3, theta=float(cfg["theta"]), backend="numpy", workers=workers,
                                     trace_start=64, trace_budget=int(cfg["trace_budget"]))
            tp0, tpT = tps[0.0], tps[1.0]
    for nm, p, key in (("volume", part, "partition_hash"), ("trace0", tp0, "trace0_hash"), ("traceT", tpT, "traceT_hash")):
        if p.hash() != row[key]:
            raise StoredRunError(f"{run_dir.name}: reconstructed {nm} partition hash {p.hash()[:12]} != stored {row[key][:12]}")
    return StoredRun(run_dir, cfg, row, name, problem, triple, enc, ck, sha, part, tp0, tpT)
