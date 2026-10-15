"""v17 Stage 6 (roadmap R3): refine the measured bottleneck with the most cost-effective enclosure (TM1, D24),
declared objective = direct FLUX bound F_dir^2 = U_S + U_I + 2 U_C + cF2_up U_R.

    # near-uniform: uniform dyadic n^3 partition (frozen + hashed), then chunked_certify
    python3 -m certified_obstacle_nn.experiments.v17.stage6 uniform --n 128 [--source <stored run>]
    # adaptive (Doerfler theta = 1/2 on the flux contributions, stage5_pilot.adaptive_flux); resumable: --resume
    # continues refining a partition saved by an earlier call (its per-cell TM1 densities are saved with it)
    python3 -m certified_obstacle_nn.experiments.v17.stage6 adaptive --budget 2000000 [--resume <partition dir>]

Both write results/v17_partitions/<ts>/ (partition.npz with lo/hi and the traces, partition.json with hashes and the
refinement record; adaptive also selection.npz with the TM1 densities used for marking). The reported certificate is
always a from-scratch evaluation of the frozen partition (chunked_certify init/chunk/finalize).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from certified_obstacle_nn.certification import reliability as rel
from certified_obstacle_nn.certification.enclose import boxes as Bx
from certified_obstacle_nn.experiments.v14.common import REPO_ROOT
from certified_obstacle_nn.experiments.v17.common import load_stored_run
from certified_obstacle_nn.experiments.v17.stage5_pilot import LONG0_1E5, flux_contrib, freeze


def uniform(source, n, workers):
    r = load_stored_run(source, workers=workers)
    part = Bx.uniform(3, n)
    info = freeze(part, r.tp0, r.tpT, {"built_with": "uniform", "n_per_axis": n, "source": source,
                                       "checkpoint_sha256": r.checkpoint_sha256})
    print(info["file"].rsplit("/", 1)[0], part.n, info["partition_hash"][:12])


def grad_contrib(vol, area, cf):
    """v17b protocol v2 marking objective: first-order cell contribution to G_dir^2 = (sqrt L + sqrt(D + L))^2 at the
    current global values. With d_K = |K| (Sh_term + 2 c), l_K = |K| cF2_up Rh_term, D ~ sum d_K, L ~ sum l_K:
      dG^2/dL : dG^2/dD = 1 + sqrt((D + L)/L) : 1,
    so mu_K = d_K + (1 + sqrt((D + L)/L)) l_K (a positive multiple of the derivative-weighted contribution; U_I, a
    trace term independent of the volume cells, is omitted from D). Returns (mu, weight on l)."""
    d = area * (vol["Sh_term"] + 2.0 * vol["c"])
    l = area * (cf * vol["Rh_term"])
    D, L = float(d.sum()), float(l.sum())
    w = 1.0 + float(np.sqrt((D + L) / L))
    return d + w * l, w


def adaptive(source, budget, workers, resume=None, theta=0.5, max_level=12, start=16, objective="flux"):
    r = load_stored_run(source, workers=workers)
    enc = r.encloser
    cf = rel.cF2_upper(2)["cF2_up"]
    tic = time.time()
    hist, prev_sel = [], 0.0
    if resume:
        z = np.load(REPO_ROOT / resume / "partition.npz")
        part = Bx.BoxPartition(z["lo"], z["hi"])
        if part.hash() != json.loads((REPO_ROOT / resume / "partition.json").read_text())["partition_hash"]:
            raise RuntimeError("resume partition hash mismatch")
        s = np.load(REPO_ROOT / resume / "selection.npz")
        vol = {k: s[k] for k in s.files}
        old = json.loads((REPO_ROOT / resume / "partition.json").read_text())
        hist = old["refinement"]["history"]
        prev_sel = old["refinement"]["selection_s"]
    else:
        part = Bx.uniform(3, start)
        vol = Bx.evaluate_volume(enc, part, "numpy", workers=workers)
    stop = None
    while True:
        fc = flux_contrib(vol, part.volume(), cf)
        h = {"n_cells": int(part.n), "flux_estimate_F2": float(fc.sum()), "elapsed_s": prev_sel + time.time() - tic}
        if objective == "grad":
            contrib, w = grad_contrib(vol, part.volume(), cf)
            D = float((part.volume() * (vol["Sh_term"] + 2.0 * vol["c"])).sum())
            L = float((part.volume() * cf * vol["Rh_term"]).sum())
            h.update({"grad_weight_L": w, "D_estimate": D, "L_estimate": L,
                      "grad_estimate_G2": (np.sqrt(L) + np.sqrt(D + L)) ** 2})
        else:
            contrib = fc
        hist.append(h)
        print(time.strftime("%H:%M:%S"), f"adaptive: n={part.n} F2~{fc.sum():.5g}"
              + (f" G2~{h['grad_estimate_G2']:.5g} wL={h['grad_weight_L']:.4f}" if objective == "grad" else ""),
              flush=True)
        lev = part.level()
        refinable = lev < max_level
        k = (budget - part.n) // 7
        if k <= 0:
            stop = "cell budget reached"
            break
        if not refinable.any():
            stop = "max level everywhere"
            break
        order = np.argsort(-np.where(refinable, contrib, -1.0))
        csum = np.cumsum(contrib[order])
        k = max(1, min(k, int(np.searchsorted(csum, theta * csum[-1]) + 1), int(refinable.sum())))
        mark = np.zeros(part.n, dtype=bool)
        mark[order[:k]] = True
        part = Bx.refine(part, mark)
        n_keep = int((~mark).sum())
        new = Bx.evaluate_volume(enc, part.subset(np.arange(n_keep, part.n)), "numpy", workers=workers)
        vol = {key: np.concatenate([vol[key][~mark], new[key]]) for key in vol}
    rec = {"stopping_reason": stop, "history": hist, "selection_s": prev_sel + time.time() - tic, "theta": theta,
           "start": start, "resumed_from": resume,
           "objective": ("flux: |K| (Sh_term + 2 c + cF2_up Rh_term)" if objective == "flux" else
                         "grad: |K| (Sh_term + 2 c) + (1 + sqrt((D + L)/L)) |K| cF2_up Rh_term (stage6.grad_contrib)"),
           "level_histogram": {int(l): int((part.level() == l).sum()) for l in np.unique(part.level())}}
    info = freeze(part, r.tp0, r.tpT, {"built_with": f"TM1 adaptive {objective}", "budget": budget, "source": source,
                                       "checkpoint_sha256": r.checkpoint_sha256, "refinement": rec})
    d = REPO_ROOT / info["file"].rsplit("/", 1)[0]
    np.savez_compressed(d / "selection.npz", **vol)
    print(str(d.relative_to(REPO_ROOT)), part.n, info["partition_hash"][:12], f"selection {rec['selection_s']:.0f}s")


def seed_from_chunks(job):
    """A resumable adaptive state from a finished chunked TM1 evaluation: the frozen partition of the job and its
    per-cell TM1 densities (the evaluation cost is counted once, as selection cost of the refined partition)."""
    import shutil

    from certified_obstacle_nn.experiments import results_io

    jd = REPO_ROOT / job
    meta = json.loads((jd / "job.json").read_text())
    if meta["enclosure"] != "TM1":
        raise ValueError("adaptive refinement state must come from TM1 densities")
    n = len(meta["edges"]) - 1
    zs = [np.load(jd / f"chunk_{i:03d}.npz") for i in range(n)]
    keys = [k for k in zs[0].files if k not in ("start", "end", "seconds")]
    vol = {k: np.concatenate([z[k] for z in zs]) for k in keys}
    secs = float(sum(float(z["seconds"]) for z in zs))
    d = results_io.make_run_dir("v17_partitions")
    shutil.copyfile(REPO_ROOT / meta["partition"] / "partition.npz", d / "partition.npz")
    src = json.loads((REPO_ROOT / meta["partition"] / "partition.json").read_text())
    src["refinement"] = {"history": [], "selection_s": secs, "seeded_from_chunks": job}
    (d / "partition.json").write_text(json.dumps(src, indent=1, default=str))
    np.savez_compressed(d / "selection.npz", **vol)
    print(str(d.relative_to(REPO_ROOT)), vol["r"].size)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("uniform")
    u.add_argument("--n", type=int, required=True)
    u.add_argument("--source", default=LONG0_1E5)
    u.add_argument("--workers", type=int, default=20)
    ad = sub.add_parser("adaptive")
    ad.add_argument("--budget", type=int, required=True)
    ad.add_argument("--source", default=LONG0_1E5)
    ad.add_argument("--resume", default=None)
    ad.add_argument("--workers", type=int, default=20)
    ad.add_argument("--objective", choices=("flux", "grad"), default="flux")
    sd = sub.add_parser("seed-from-chunks")
    sd.add_argument("--job", required=True)
    a = ap.parse_args()
    if a.cmd == "seed-from-chunks":
        seed_from_chunks(a.job)
    elif a.cmd == "uniform":
        uniform(a.source, a.n, a.workers)
    else:
        adaptive(a.source, a.budget, a.workers, a.resume, objective=a.objective)


if __name__ == "__main__":
    main()
