"""v14 Stage 6: free-boundary diagnostic of the trained ellipse networks by gap
classification (errors/ellipse_interface.py; diagnostic_float64, not controlled
by the theorem). Declared before looking at any network: tolerances 1e-4 and
1e-3, time slices 0, 1/4, 1/2, 3/4, 1, 360 rays, 2001 radial samples on
rho in (0, 2]. The identical classifier on the exact solution is reported in the
same table (its own bias for each tol).

    python3 -m certified_obstacle_nn.experiments.v14.ellipse_interface_study --seeds 0 1 2

Writes results/v14_ellipse_interface/<timestamp>/{config.json, summary.csv
(aggregates per seed/who/slice/tol), rays.csv (every ray)}.
"""

from __future__ import annotations

import argparse

import pandas as pd

from certified_obstacle_nn.errors.ellipse_interface import N_ANGLES, N_RHO, RHO_MAX, TIMES, interface_diagnostic
from certified_obstacle_nn.experiments import results_io
from certified_obstacle_nn.experiments.v14.common import REPO_ROOT, git_commit, git_dirty, sha256_file
from certified_obstacle_nn.experiments.v14.ellipse_validated import find_training_run, load_triple
from certified_obstacle_nn.problems.ellipse import EllipseProblem

EXPERIMENT = "v14_ellipse_interface"
TOLS = (1e-4, 1e-3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    aggs, rays, runs = [], [], []
    for seed in a.seeds:
        train_dir = find_training_run(seed)
        triple, tcfg, _, ck = load_triple(train_dir)
        problem = EllipseProblem(kappa0=tcfg["kappa0"][0] / tcfg["kappa0"][1])
        runs.append({"seed": seed, "training_run": str(train_dir.relative_to(REPO_ROOT)),
                     "checkpoint_sha256": sha256_file(ck)})
        for tol in TOLS:
            df, agg, meta = interface_diagnostic(problem, triple, tol=tol)
            df.insert(0, "seed", seed)
            df.insert(1, "tol", tol)
            agg.insert(0, "seed", seed)
            rays.append(df)
            aggs.append(agg)
            for _, r in agg[agg["t"] == "all"].iterrows():
                print(f"seed {seed} tol {tol:g} {r['who']}: missing {r['missing']}/{r['n_rays']} "
                      f"multiple {r['multiple']} mean dist {r['mean_dist_err']:.4g} max {r['max_dist_err']:.4g}")
    run_dir = results_io.make_run_dir(EXPERIMENT)
    results_io.write_config(run_dir, {
        "experiment": EXPERIMENT, "problem": "ellipse", "seeds": a.seeds, "tols": list(TOLS), "times": list(TIMES),
        "n_angles": N_ANGLES, "n_rho": N_RHO, "rho_max": RHO_MAX, "rule": meta["rule"], "runs": runs,
        "status": "diagnostic_float64", "git_commit": git_commit(), "git_dirty": git_dirty()})
    agg = pd.concat(aggs, ignore_index=True)
    agg["t"] = agg["t"].astype(str)
    agg["status"] = "diagnostic_float64"
    agg.to_csv(run_dir / "summary.csv", index=False)
    pd.concat(rays, ignore_index=True).to_csv(run_dir / "rays.csv", index=False)
    print(run_dir)


if __name__ == "__main__":
    main()
