"""v17 Stage 5 (roadmap R3): pilot at equal partition and at equal cost, long-disk seed 0.

    python3 -m certified_obstacle_nn.experiments.v17.stage5_pilot --mode same-partition --budget 10000 \
        --candidates TM1 TM1+PROD TM1+AFF32+PROD ... --workers 20
    python3 -m certified_obstacle_nn.experiments.v17.stage5_pilot --mode equal-cost --seconds <T> \
        --candidates TM1 TM1+AFF64+PROD ... --workers 20

Partitions: adaptive dyadic cubes refined by Doerfler marking (theta = 1/2) on the cell contributions to the DIRECT
FLUX bound F_dir^2 = U_S + U_I + 2 U_C + cF2_up U_R (objective "flux"): contribution |K| (Sh_term + 2 c + cF2_up
Rh_term), traces = the frozen Stage-0 trace partitions of the checkpoint (TM1, budget 16384). A partition is frozen,
hashed and saved (results/v17_partitions/<ts>/partition.npz) before any reported evaluation; every reported number is
a from-scratch evaluation on the frozen partition (split_certify.run).

same-partition: one partition, built with the TM1 encloser to the given budget, evaluated by every candidate.
equal-cost: for every candidate its own adaptive partition (marking with that candidate's densities) refined until
the selection time + final evaluation time reaches --seconds (time per cell measured on the fly); reports the
complete direct bound against total wall time.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time

import numpy as np

from certified_obstacle_nn.certification import reliability as rel
from certified_obstacle_nn.certification.enclose import boxes as Bx
from certified_obstacle_nn.experiments import results_io
from certified_obstacle_nn.experiments.v14.common import REPO_ROOT, sha256_file
from certified_obstacle_nn.experiments.v17 import split_certify as SC
from certified_obstacle_nn.experiments.v17.common import load_stored_run

LONG0_1E5 = "results/v14_disk_validated/20261007T212554Z"


def flux_contrib(vol, area, cf):
    return area * (vol["Sh_term"] + 2.0 * vol["c"] + cf * vol["Rh_term"])


def adaptive_flux(enc, *, budget=None, seconds=None, start=16, theta=0.5, workers=20, max_level=12, log=print,
                  eval_s_per_cell=None):
    """Doerfler refinement on the direct-flux contributions. Stops at `budget` cells, or (equal-cost mode) when the
    elapsed selection time plus the predicted final evaluation time (cells x eval_s_per_cell, measured from the
    loop when not given) would exceed `seconds`."""
    cf = rel.cF2_upper(2)["cF2_up"]
    tic = time.time()
    part = Bx.uniform(3, start)
    vol = Bx.evaluate_volume(enc, part, "numpy", workers=workers)
    n_eval = part.n
    hist = []
    stop = None
    while True:
        contrib = flux_contrib(vol, part.volume(), cf)
        elapsed = time.time() - tic
        spc = eval_s_per_cell or elapsed / max(n_eval, 1)
        hist.append({"n_cells": int(part.n), "flux_estimate_F2": float(contrib.sum()), "elapsed_s": elapsed})
        log(f"  adaptive: n={part.n} F2~{contrib.sum():.5g} elapsed {elapsed:.0f}s")
        lev = part.level()
        refinable = lev < max_level
        order = np.argsort(-np.where(refinable, contrib, -1.0))
        csum = np.cumsum(contrib[order])
        k = int(np.searchsorted(csum, theta * csum[-1]) + 1)
        k = max(1, min(k, int(refinable.sum())))
        if budget is not None:
            k = min(k, (budget - part.n) // 7)
            if k <= 0:
                stop = "cell budget reached"
                break
        if seconds is not None:
            # next step costs ~7k new cells now plus the final evaluation of the larger partition
            k = min(k, int(max(0.0, (seconds - elapsed) / spc - part.n) // 14))
            if k <= 0:
                stop = "time budget reached"
                break
        if not refinable.any():
            stop = "max level everywhere"
            break
        mark = np.zeros(part.n, dtype=bool)
        mark[order[:k]] = True
        part = Bx.refine(part, mark)
        n_keep = int((~mark).sum())
        new = Bx.evaluate_volume(enc, part.subset(np.arange(n_keep, part.n)), "numpy", workers=workers)
        vol = {key: np.concatenate([vol[key][~mark], new[key]]) for key in vol}
        n_eval += part.n - n_keep
    return part, {"stopping_reason": stop, "history": hist, "selection_s": time.time() - tic, "theta": theta,
                  "start": start, "objective": "flux: |K| (Sh_term + 2 c + cF2_up Rh_term)",
                  "level_histogram": {int(l): int((part.level() == l).sum()) for l in np.unique(part.level())}}


def freeze(part, tp0, tpT, meta) -> dict:
    d = results_io.make_run_dir("v17_partitions")
    f = d / "partition.npz"
    np.savez_compressed(f, lo=part.lo, hi=part.hi, lo0=tp0.lo, hi0=tp0.hi, loT=tpT.lo, hiT=tpT.hi)
    ok, msg = part.verify_coverage()
    info = {"partition_hash": part.hash(), "trace0_hash": tp0.hash(), "traceT_hash": tpT.hash(),
            "n_cells": int(part.n), "coverage": msg, "file": str(f.relative_to(REPO_ROOT)),
            "file_sha256": sha256_file(f), **meta}
    (d / "partition.json").write_text(json.dumps(info, indent=1, default=str))
    if not ok:
        raise RuntimeError(f"frozen partition does not cover Q: {msg}")
    return info


def evaluate(r0, part, tp0, tpT, info, enclosure, workers, q, diag=None, log=print):
    r = dataclasses.replace(r0, part=part, tp0=tp0, tpT=tpT)
    config, rows, cells, r, diag = SC.run(r0.run_dir, ["numpy"], workers, q, log=log, enclosure=enclosure, r=r,
                                          diag=diag, partition_info=info)
    d = SC.write_run(config, rows, cells, r, experiment="v17_pilot")
    return d, rows[0], diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("same-partition", "equal-cost"), required=True)
    ap.add_argument("--candidates", nargs="+", required=True)
    ap.add_argument("--budget", type=int, default=10000)
    ap.add_argument("--seconds", type=float, default=600.0)
    ap.add_argument("--start", type=int, default=16)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--q", type=int, default=2)
    ap.add_argument("--source", default=LONG0_1E5)
    a = ap.parse_args()

    def log(msg):
        print(time.strftime("%H:%M:%S"), msg, flush=True)

    r0 = load_stored_run(a.source, workers=a.workers)
    tp0, tpT = r0.tp0, r0.tpT
    if a.mode == "same-partition":
        tic = time.time()
        part, rec = adaptive_flux(r0.encloser, budget=a.budget, start=a.start, workers=a.workers, log=log)
        info = freeze(part, tp0, tpT, {"built_with": "TM1", "mode": a.mode, "budget": a.budget, "refinement": rec})
        log(f"frozen partition {info['partition_hash'][:12]} n={part.n} ({time.time() - tic:.0f}s)")
        diag = None
        for spec in a.candidates:
            d, row, diag = evaluate(r0, part, tp0, tpT, info, spec, a.workers, a.q, diag, log)
            log(f"{spec}: F_dir/N={row['flux_bound_direct_split_rel']:.4f} G_dir/N={row['grad_bound_direct_split_rel']:.4f}"
                f" U_S={row['U_S']:.4g} 2U_C={2 * row['U_C']:.4g} L={row['L']:.4g} t_eval={row['t_eval_total_s']:.0f}s"
                f" -> {d.relative_to(REPO_ROOT)}")
    else:
        for spec in a.candidates:
            enc = SC.encloser_for(r0, spec)
            tic = time.time()
            part, rec = adaptive_flux(enc, seconds=a.seconds, start=a.start, workers=a.workers, log=log)
            info = freeze(part, tp0, tpT, {"built_with": spec, "mode": a.mode, "seconds": a.seconds,
                                           "refinement": rec})
            d, row, _ = evaluate(r0, part, tp0, tpT, info, spec, a.workers, a.q, None, log)
            total = time.time() - tic
            log(f"{spec}: n={part.n} total {total:.0f}s F_dir/N={row['flux_bound_direct_split_rel']:.4f} "
                f"G_dir/N={row['grad_bound_direct_split_rel']:.4f} -> {d.relative_to(REPO_ROOT)}")
            with (d / "equal_cost.json").open("w") as fh:
                json.dump({"spec": spec, "seconds_target": a.seconds, "total_wall_s": total,
                           "selection_s": rec["selection_s"], "n_cells": int(part.n)}, fh, indent=1)


if __name__ == "__main__":
    main()
