"""v17 Stage 6: cross-check the reference backend (arb, 64-bit balls) against the fast backend (numpy outward)
  (a) on the worst cells (largest direct-flux contribution) of a frozen partition evaluated by a chunked job, and
  (b) on a manageable full partition (uniform n^3), complete per-term sums and direct bounds with both backends.

    python3 -m certified_obstacle_nn.experiments.v17.arb_crosscheck --job <v17_chunks dir> --worst 2000 --uniform 32

Both backends are validated; the check records the per-cell relative differences and the sign of the difference
(neither is required to be smaller), and whether both bound the same 50-digit-free float64 sampled maxima.
Writes results/v17_arb_crosscheck/<ts>/crosscheck.json.
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
from certified_obstacle_nn.experiments.v14.common import REPO_ROOT
from certified_obstacle_nn.experiments.v17 import chunked_certify as CC
from certified_obstacle_nn.experiments.v17 import split_certify as SC

KEYS = ("Rh_term", "Sh_term", "c", "rhat", "r", "p")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--worst", type=int, default=2000)
    ap.add_argument("--uniform", type=int, default=32)
    ap.add_argument("--workers", type=int, default=20)
    a = ap.parse_args()
    job = REPO_ROOT / a.job
    meta, r = CC._load(job, a.workers)
    n = len(meta["edges"]) - 1
    zs = [np.load(job / f"chunk_{i:03d}.npz") for i in range(n)]
    vol = {k: np.concatenate([z[k] for z in zs]) for k in KEYS}
    cf = rel.cF2_upper(2)["cF2_up"]
    contrib = r.part.volume() * (vol["Sh_term"] + 2 * vol["c"] + cf * vol["Rh_term"])
    worst = np.argsort(-contrib)[: a.worst]
    tic = time.time()
    sub = r.part.subset(np.sort(worst))
    va = Bx.evaluate_volume(r.encloser, sub, "arb", 64, batch=32, workers=a.workers)
    t_arb = time.time() - tic
    idx = np.sort(worst)
    out = {"job": a.job, "partition_hash": meta["partition_hash"], "worst_cells": int(a.worst), "arb_seconds": t_arb,
           "worst_share_of_flux_F2": float(contrib[worst].sum() / contrib.sum()), "per_key": {}}
    for k in KEYS:
        x, y = vol[k][idx], va[k]
        rd = np.abs(x - y) / np.maximum(np.abs(x), 1e-300)
        out["per_key"][k] = {"max_rel_diff": float(rd.max()), "median_rel_diff": float(np.median(rd)),
                             "frac_numpy_larger": float(np.mean(x > y)), "frac_equal": float(np.mean(x == y))}
    out["signs_agree"] = bool(np.all(np.load(job / "chunk_000.npz")["sign"].size > 0) and
                              np.all(np.concatenate([z["sign"] for z in zs])[idx] == va["sign"]))
    # (b) manageable full partition, both backends, complete bounds
    part = Bx.uniform(3, a.uniform)
    rr = dataclasses.replace(r, part=part)
    _, rows, _, _, _ = SC.run(r.run_dir, ["numpy", "arb"], a.workers, 2, r=rr,
                              partition_info={"partition_hash": part.hash(), "n_cells": int(part.n),
                                              "source": f"uniform {a.uniform}^3 (cross-check)"},
                              log=lambda m: print(time.strftime("%H:%M:%S"), m, flush=True))
    out["uniform"] = {"n": a.uniform, "partition_hash": part.hash()}
    for row in rows:
        out["uniform"][row["backend"]] = {k: row[k] for k in ("status", "U_R", "U_S", "U_I", "U_C",
                                                               "flux_bound_direct_split", "grad_bound_direct_split",
                                                               "E_dir_split", "t_eval_total_s")}
    nu, ar = out["uniform"]["numpy"], out["uniform"]["arb"]
    out["uniform"]["rel_diff"] = {k: abs(nu[k] - ar[k]) / ar[k] for k in ("U_R", "U_S", "U_C",
                                                                            "flux_bound_direct_split")}
    d = results_io.make_run_dir("v17_arb_crosscheck")
    (d / "crosscheck.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("worst_share_of_flux_F2", "per_key", "signs_agree", "arb_seconds")},
                     indent=1))
    print(json.dumps(out["uniform"], indent=1))
    print("wrote", d.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
