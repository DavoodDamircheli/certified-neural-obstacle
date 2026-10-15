"""v18 Task 3 (referee item M1): reproduce one reported certificate row from its frozen checkpoint and frozen partition,
from scratch, and compare it bit for bit with the stored record.

    python3 -m certified_obstacle_nn.experiments.v18.reproduce_row --run results/v17_split/<ts> [--workers 20]

The stored row (results/v17_split/<ts>/summary.csv, numpy backend) names its source run (checkpoint, problem) and its
frozen partition (config.json "partition": uniform Stefan grid, a results/v17_partitions/<ts>/partition.npz, or the
lo/hi of the source run's cells.npz with the Stage-0 frozen traces). Every partition hash and the checkpoint hash are
checked. All per-cell densities and traces are evaluated again with the TM1 engine (numpy_outward_float64), reduced in
arb (upper_sum.reduce_upper_sums) and turned into the direct bounds (reliability.direct_projected_bounds). Compared
EXACTLY (binary64 equality): every per-cell array stored in cells.npz (numpy_*), the per-term sums, the aggregates and
every direct bound and ratio. Float64 diagnostics (quadrature, slacks) are not recomputed. Prints REPRODUCED or the
differing quantities; writes results/v18_reproduce/<ts>/ (config.json, comparison.csv, log.txt); never touches the
stored run.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time

import numpy as np
import pandas as pd

from certified_obstacle_nn.certification import reliability as rel  # also sets arb precision >= 128 (D36)
from certified_obstacle_nn.certification.enclose import boxes as Bx
from certified_obstacle_nn.certification.enclose import upper_sum as US
from certified_obstacle_nn.experiments import results_io
from certified_obstacle_nn.experiments.v14.common import REPO_ROOT, git_commit, git_dirty, hardware
from certified_obstacle_nn.certification.enclose.engine2d import parse_enclosure
from certified_obstacle_nn.experiments.v17 import split_certify as SC
from certified_obstacle_nn.experiments.v17.common import load_stored_run

DIM = {"stefan": 1, "disk2d": 2, "ellipse": 2}
TERMS = {"U_R": "vol_Rhat", "U_S": "vol_Shat", "U_I": "init_mismatch_hat", "U_R0": "vol_R0", "U_S0": "vol_S_term",
         "U_init": "init_mismatch", "U_gradw": "vol_gradw", "U_w": "vol_w2", "U_wt": "vol_wt2", "U_w0": "w0",
         "U_wT": "wT"}
BOUNDS = ("D", "L", "flux_bound_direct_split", "grad_bound_direct_split", "grad_bound_direct_check_theta_half",
          "E_dir_split", "graph_bound_direct", "flux_bound_direct_split_rel", "grad_bound_direct_split_rel",
          "flux_bound_direct", "grad_bound_direct", "E_U_direct")


def load(run: str, workers: int):
    d = REPO_ROOT / run
    cfg = json.loads((d / "config.json").read_text())
    enclosure = cfg.get("enclosure", "TM1")
    parse_enclosure(enclosure)  # raises for the TM2 probe hybrids (reproduced by experiments/v17/tm2_probe.py)
    row = pd.read_csv(d / "summary.csv", dtype=str).set_index("backend").loc["numpy"]
    r = load_stored_run(cfg["source_run"], workers=workers)
    src = str(cfg["partition"].get("source", ""))
    pfile = cfg["partition"].get("file") or (f"{src}/partition.npz" if src.startswith("results/v17_partitions") else None)
    if pfile:
        z = np.load(REPO_ROOT / pfile)
        r = dataclasses.replace(r, part=Bx.BoxPartition(z["lo"], z["hi"]), tp0=Bx.BoxPartition(z["lo0"], z["hi0"]),
                                tpT=Bx.BoxPartition(z["loT"], z["hiT"]))
    for nm, p in (("partition_hash", r.part), ("trace0_hash", r.tp0), ("traceT_hash", r.tpT)):
        if p.hash() != row[nm]:
            raise RuntimeError(f"{nm}: reconstructed {p.hash()[:12]} != stored {row[nm][:12]}")
    if r.checkpoint_sha256 != row["checkpoint_sha256"]:
        raise RuntimeError("checkpoint hash mismatch")
    return d, cfg, row, r, SC.encloser_for(r, enclosure)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--workers", type=int, default=20)
    a = ap.parse_args()
    if a.run.endswith("/"):
        a.run = a.run[:-1]
    out = results_io.make_run_dir("v18_reproduce")

    def log(m):
        line = time.strftime("%H:%M:%S ") + m
        print(line, flush=True)
        with open(out / "log.txt", "a") as fh:
            fh.write(line + "\n")

    tic = time.time()
    d, cfg, row, r, enc = load(a.run, a.workers)
    log(f"{a.run}: {r.problem_name} n={r.part.n} traces {r.tp0.n}/{r.tpT.n}, checkpoint {r.checkpoint_sha256[:12]}")
    if r.problem_name == "stefan":
        us, vol, _ = US.certify_partition(enc, r.part, r.tp0, r.tpT, backend="numpy", workers=a.workers,
                                          model_frozen=r.frozen, constants_validated=True)
    else:
        us, vol, _ = Bx.certify_boxes(enc, r.part, r.tp0, r.tpT, backend="numpy", workers=a.workers,
                                      model_frozen=r.frozen, constants_validated=True)
    t_eval = time.time() - tic
    log(f"evaluated in {t_eval:.0f}s, status {us.status}")
    cmp = []
    cz = np.load(d / "cells.npz")
    for k in cz.files:
        if not k.startswith("numpy_"):
            continue
        new, old = vol[k[len("numpy_"):]], cz[k]
        nbad = int((new != old).sum()) if new.shape == old.shape else -1
        cmp.append({"quantity": f"cells.npz:{k}", "stored": f"{old.size} cells", "new": f"{new.size} cells",
                    "equal": nbad == 0, "n_differ": nbad})
    P = us.parts
    vals = {"U_r": us.U_r, "U_p": us.U_p, "U_c": us.U_c, "U_r_hat": us.U_r_hat, "U_C": us.U_c,
            **{k: P[v] for k, v in TERMS.items()}}
    prob = r.problem
    try:
        ref = rel.reference_norm(r.problem_name)
    except KeyError:
        ref = None
    dp = rel.direct_projected_bounds(
        U_R=vals["U_R"], U_S=vals["U_S"], U_I=vals["U_I"], U_C=vals["U_C"],
        cF2_up=rel.cF2_upper(DIM[r.problem_name])["cF2_up"], status=us.status, alpha=float(prob.alpha),
        A_inf=float(prob.A_inf), U_r_hat=us.U_r_hat, Lambda=float(prob.Lambda), U_gradw=vals["U_gradw"],
        reference_norm_lower=ref["norm_lower"] if ref else None, reference_status=ref["status"] if ref else None)
    vals.update({k: dp[k] for k in BOUNDS if k in dp})
    for k, v in vals.items():
        if k not in row or not isinstance(row[k], str):
            continue
        s = float(row[k])
        cmp.append({"quantity": f"summary:{k}", "stored": float(s).hex(), "new": float(v).hex(), "equal": s == v,
                    "n_differ": int(s != v)})
    for k in ("status", "partition_hash", "trace0_hash", "traceT_hash"):
        new = {"status": us.status, "partition_hash": us.partition_hash, "trace0_hash": us.trace0_hash,
               "traceT_hash": us.traceT_hash}[k]
        cmp.append({"quantity": f"summary:{k}", "stored": row[k], "new": new, "equal": row[k] == new,
                    "n_differ": int(row[k] != new)})
    c = pd.DataFrame(cmp)
    c.to_csv(out / "comparison.csv", index=False)
    ok = bool(c["equal"].all())
    cfgo = {"experiment": "v18_reproduce", "run": a.run, "workers": a.workers, "eval_s": t_eval,
            "total_s": time.time() - tic, "reproduced": ok, "n_compared": len(c), "git_commit": git_commit(),
            "git_dirty": git_dirty(), "hardware": hardware(), "stored_git_commit": row.get("git_commit"),
            "doc": __doc__}
    (out / "config.json").write_text(json.dumps(cfgo, indent=1, default=str))
    for _, x in c[~c["equal"]].iterrows():
        log(f"DIFF {x['quantity']}: stored {x['stored']} new {x['new']} ({x['n_differ']})")
    log(f"{'REPRODUCED' if ok else 'NOT REPRODUCED'}: {len(c)} quantities compared, {int((~c['equal']).sum())} differ; "
        f"wall {time.time() - tic:.0f}s -> {out.relative_to(REPO_ROOT)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
