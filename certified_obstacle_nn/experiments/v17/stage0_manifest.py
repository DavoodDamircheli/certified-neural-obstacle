"""v17 Stage 0 (roadmap R3 = cleanup C0): freeze and hash the inputs.

    python3 -m certified_obstacle_nn.experiments.v17.stage0_manifest --workers 20

Writes results/v17_stage0/<timestamp>/ with
  manifest.json  -- SHA-256 of the three long-disk checkpoints and training configs, of every stored validated run
                    used by v16 (checkpoint, config.json, summary.csv, cells.npz), the recorded partition / trace
                    hashes, the v16 provenance cross-check, library versions, hardware, git commit;
  traces/<experiment>__<run>.npz  -- the reconstructed 2-D trace partitions (lo/hi at t = 0 and t = T), each
                    verified against the stored trace0_hash / traceT_hash (they were never saved by v14).
Never touches an existing run directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np

from certified_obstacle_nn.experiments import results_io
from certified_obstacle_nn.experiments.v14.common import REPO_ROOT, git_commit, git_dirty, hardware, sha256_file
from certified_obstacle_nn.experiments.v17.common import V16_REF, load_stored_run

LONG = [REPO_ROOT / f"results/expanding_disk_2d/final/long_6x64_N20000_s{s}_k5" for s in range(3)]


def versions() -> dict:
    import flint
    import matplotlib
    import mpmath
    import pandas
    import scipy
    import sympy
    import torch

    return {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
            "python-flint": flint.__version__, "mpmath": mpmath.__version__, "sympy": sympy.__version__,
            "scipy": scipy.__version__, "pandas": pandas.__version__, "matplotlib": matplotlib.__version__}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=20)
    a = ap.parse_args()
    t0 = time.time()
    out = results_io.make_run_dir("v17_stage0")
    (out / "traces").mkdir()
    prov = list(csv.DictReader((V16_REF / "data/provenance.csv").open()))
    man = {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git_commit": git_commit(), "git_dirty": git_dirty(),
           "versions": versions(), "hardware": hardware(), "long_disk": [], "runs": [], "checks": {}}
    for d in LONG:
        man["long_disk"].append({
            "dir": str(d.relative_to(REPO_ROOT)),
            **{f"{f}_sha256": sha256_file(d / f) for f in ("checkpoint.pt", "config.json", "train_info.json",
                                                            "metrics.json", "loss_history.csv")}})
    v16_long0 = {p["checkpoint_sha256"] for p in prov if p["problem"] == "disk2d" and p["schedule"] == "long"
                 and p["seed"] == "0"}
    man["checks"]["long_seed0_matches_v16_provenance"] = man["long_disk"][0]["checkpoint.pt_sha256"] in v16_long0
    if not man["checks"]["long_seed0_matches_v16_provenance"]:
        raise SystemExit("STOP: long-disk seed-0 checkpoint hash differs from the v16 provenance")
    seen = set()
    for p in prov:
        run_dir = REPO_ROOT / Path(p["source_summary"]).parent
        if run_dir in seen:
            continue
        seen.add(run_dir)
        rec = {"run_dir": str(run_dir.relative_to(REPO_ROOT)), "problem": p["problem"], "schedule": p["schedule"],
               "seed": int(p["seed"]), "mode": p["mode"], "size": int(p["size"]),
               "v16_checkpoint_sha256": p["checkpoint_sha256"], "v16_partition_hash": p["partition_hash"],
               "v16_trace0_hash": p["trace0_hash"], "v16_traceT_hash": p["traceT_hash"],
               "files": {f.name: sha256_file(f) for f in sorted(run_dir.iterdir()) if f.is_file()}}
        stefan_ok = p["problem"] == "stefan" and p["mode"] == "uniform" and p["size"] == "262144"
        two_d_ok = p["problem"] != "stefan" and (run_dir / "cells.npz").exists()
        if stefan_ok or two_d_ok:
            tic = time.time()
            r = load_stored_run(run_dir, workers=a.workers)
            rec["reconstructed"] = {**r.identity(), "checkpoint_file": str(r.checkpoint.relative_to(REPO_ROOT)),
                                    "n_cells": int(r.part.n), "n_trace0": int(r.tp0.n), "n_traceT": int(r.tpT.n),
                                    "seconds": time.time() - tic}
            rec["reconstructed"]["matches_v16"] = (r.checkpoint_sha256 == p["checkpoint_sha256"]
                                                   and r.summary["partition_hash"] == p["partition_hash"])
            if r.problem_name != "stefan":
                f = out / "traces" / f"{run_dir.parent.name}__{run_dir.name}.npz"
                np.savez_compressed(f, lo0=r.tp0.lo, hi0=r.tp0.hi, loT=r.tpT.lo, hiT=r.tpT.hi)
                rec["reconstructed"]["traces_file"] = str(f.relative_to(REPO_ROOT))
                rec["reconstructed"]["traces_file_sha256"] = sha256_file(f)
            print(time.strftime("%H:%M:%S"), rec["run_dir"], "ok", f"{time.time() - tic:.0f}s", flush=True)
        else:
            rec["reconstructed"] = None
            rec["not_reconstructed_reason"] = ("volume partition not stored (no cells.npz)" if p["problem"] != "stefan"
                                               else "not a Stefan 512^2 headline run")
        man["runs"].append(rec)
    man["wall_s"] = time.time() - t0
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print("manifest:", (out / "manifest.json").relative_to(REPO_ROOT), f"{man['wall_s']:.0f}s")


if __name__ == "__main__":
    main()
