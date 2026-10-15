"""v17b Task 4: TM2 hybrid probe on a fixed cell set of a frozen partition (DECISIONS D26).

    python3 -m certified_obstacle_nn.experiments.v17.tm2_probe --set A|B|C --workers 16

  A  cells 0-145238 of the Stage-5 equal-cost TM1 partition (results/v17_partitions/20261009T212324Z, 580952 cells),
     the chunk of the affine hybrid results/v17_chunks/20261009T220235Z;
  B  145238 cells of the protocol-v1 seed-0 partition (results/v17_partitions/20261009T235639Z), default_rng(20261011);
  C  the 145238 cells of the protocol-v1 seed-0 partition with the largest direct-flux contribution.
For the set: TM1 and TM2 per-cell densities on the same cells (numpy outward; TM2 = "TM1+TM2", the TM1 range
intersected with the TM2 range), wall ms/cell, per-term contributions sum |K| x (float64 sums; diagnostic) and a
COMPLETE validated certificate of the whole partition with the TM2 densities on the set and TM1 elsewhere (arb
reduction, split_certify.run). Writes results/v17_tm2_probe/<ts>/probe.json and the certificate run dir.
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
from certified_obstacle_nn.experiments.v17 import chunked_certify as CC
from certified_obstacle_nn.experiments.v17 import split_certify as SC
from certified_obstacle_nn.experiments.v17.common import load_stored_run
from certified_obstacle_nn.experiments.v17.stage5_pilot import LONG0_1E5

PILOT_PART = "results/v17_partitions/20261009T212324Z"
V1_JOB = "results/v17_chunks/20261009T235700Z"
N_SET = 145238
TERMS = {"U_R": "Rh_term", "U_S": "Sh_term", "U_C": "c"}


def _load_partition(r, pdir):
    z = np.load(REPO_ROOT / pdir / "partition.npz")
    return dataclasses.replace(r, part=Bx.BoxPartition(z["lo"], z["hi"]), tp0=Bx.BoxPartition(z["lo0"], z["hi0"]),
                               tpT=Bx.BoxPartition(z["loT"], z["hiT"]))


def contributions(vol, area, cf):
    c = {k: float((area * vol[v]).sum()) for k, v in TERMS.items()}
    c["L"] = cf * c["U_R"]
    c["D_vol"] = c["U_S"] + 2 * c["U_C"]
    c["F2_vol"] = c["D_vol"] + c["L"]
    c["unresolved_frac"] = float(np.mean(vol["sign"] == 0))
    return c


def run(which, workers, seed=20261011, log=print):
    cf = rel.cF2_upper(2)["cF2_up"]
    out = {"set": which, "seed": seed, "workers": workers}
    if which == "A":
        r = _load_partition(load_stored_run(LONG0_1E5, workers=workers), PILOT_PART)
        tic = time.time()
        vol1 = Bx.evaluate_volume(r.encloser, r.part, "numpy", workers=workers)
        out["tm1_full_eval_s"] = time.time() - tic
        idx = np.arange(N_SET)
        out["partition"] = PILOT_PART
    else:
        meta, r = CC._load(REPO_ROOT / V1_JOB, workers)
        zs = [np.load(REPO_ROOT / V1_JOB / f"chunk_{i:03d}.npz") for i in range(len(meta["edges"]) - 1)]
        keys = [k for k in zs[0].files if k not in ("start", "end", "seconds")]
        vol1 = {k: np.concatenate([z[k] for z in zs]) for k in keys}
        out["tm1_full_eval_s"] = float(sum(float(z["seconds"]) for z in zs))
        out["partition"] = meta["partition"]
        if which == "B":
            idx = np.sort(np.random.default_rng(seed).choice(r.part.n, N_SET, replace=False))
        else:
            fc = r.part.volume() * (vol1["Sh_term"] + 2 * vol1["c"] + cf * vol1["Rh_term"])
            idx = np.sort(np.argsort(-fc, kind="stable")[:N_SET])
    out["partition_hash"] = r.part.hash()
    out["n_cells_partition"] = int(r.part.n)
    out["n_set"] = int(idx.size)
    sub = r.part.subset(idx)
    area = sub.volume()
    tic = time.time()
    s1 = Bx.evaluate_volume(r.encloser, sub, "numpy", workers=workers)
    t1 = time.time() - tic
    if not all(np.array_equal(s1[k], vol1[k][idx]) for k in TERMS.values()):
        raise RuntimeError("TM1 re-evaluation differs from the partition's TM1 densities")
    enc2 = SC.encloser_for(r, "TM2")
    tic = time.time()
    s2 = Bx.evaluate_volume(enc2, sub, "numpy", workers=workers)
    t2 = time.time() - tic
    for k in ("r", "p", "c", "rhat", "Rh_term", "Sh_term", "R0_term"):
        if np.any(s2[k] > s1[k]):
            raise RuntimeError(f"TM2 density above TM1 on some cell ({k}): intersection broken")
    out["levels"] = {int(l): int((sub.level() == l).sum()) for l in np.unique(sub.level())}
    out["tm1"] = {"wall_s": t1, "ms_per_cell": t1 / idx.size * 1e3, **contributions(s1, area, cf)}
    out["tm2"] = {"wall_s": t2, "ms_per_cell": t2 / idx.size * 1e3, **contributions(s2, area, cf)}
    out["cost_ratio_tm2_over_tm1"] = t2 / t1
    out["reduction"] = {k: out["tm1"][k] / out["tm2"][k] if out["tm2"][k] > 0 else float("inf")
                        for k in ("U_R", "U_S", "U_C", "L", "D_vol", "F2_vol")}
    full = contributions(vol1, r.part.volume(), cf)
    out["set_share_of_partition_F2_vol"] = out["tm1"]["F2_vol"] / full["F2_vol"]
    log(f"set {which}: {idx.size} cells, TM1 {out['tm1']['ms_per_cell']:.2f} ms/cell, TM2 "
        f"{out['tm2']['ms_per_cell']:.2f} ms/cell; F2_vol {out['tm1']['F2_vol']:.4g} -> {out['tm2']['F2_vol']:.4g}")
    # complete certificate: TM2 densities on the set, TM1 elsewhere
    hyb = {k: vol1[k].copy() for k in vol1}
    for k in hyb:
        hyb[k][idx] = s2[k]
    label = f"TM1 + TM2 on probe set {which} ({idx.size} cells)"
    info = {"partition_hash": r.part.hash(), "trace0_hash": r.tp0.hash(), "traceT_hash": r.tpT.hash(),
            "n_cells": int(r.part.n), "source": out["partition"], "tm2_probe_set": which,
            "tm2_probe_indices_sha256": None}
    d = results_io.make_run_dir("v17_tm2_probe")
    np.savez_compressed(d / "set.npz", idx=idx, **{f"tm1_{k}": s1[k] for k in TERMS.values()},
                        **{f"tm2_{k}": s2[k] for k in TERMS.values()})
    info["tm2_probe_indices_sha256"] = sha256_file(d / "set.npz")
    config, rows, cells, r2, _ = SC.run(r.run_dir, ["numpy"], workers, 2, enclosure="TM2", r=r, partition_info=info,
                                        volumes={"numpy": (hyb, out["tm1_full_eval_s"] - t1 + t2)},
                                        log=lambda m: log(m))
    config["enclosure"] = label
    rows[0]["enclosure"] = label
    cd = SC.write_run(config, rows, cells, r2)
    row = rows[0]
    out["certificate"] = {"run": CC._rel(cd), "status": row["status"],
                          **{k: row[k] for k in ("U_R", "U_S", "U_I", "U_C", "flux_bound_direct_split",
                                                 "grad_bound_direct_split", "E_dir_split",
                                                 "flux_bound_direct_split_rel", "grad_bound_direct_split_rel", "D",
                                                 "L")}}
    (d / "probe.json").write_text(json.dumps(out, indent=1))
    log(f"certificate {row['status']} F_dir/N={row['flux_bound_direct_split_rel']:.4f} "
        f"G_dir/N={row['grad_bound_direct_split_rel']:.4f} -> {CC._rel(cd)}; probe {CC._rel(d)}")
    return d, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=("A", "B", "C"), required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    run(a.set, a.workers, log=lambda m: print(time.strftime("%H:%M:%S"), m, flush=True))


if __name__ == "__main__":
    main()
