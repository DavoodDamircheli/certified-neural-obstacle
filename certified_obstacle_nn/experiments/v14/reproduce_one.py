"""v14 Stage 7: re-run one full Stefan evaluation from its checkpoint and
compare it with a stored run (default: the latest committed-code run of
Stefan seed 0, uniform 32 x 32, in results/v14_stefan_validated/).

    python3 -m certified_obstacle_nn.experiments.v14.reproduce_one [--run-dir results/v14_stefan_validated/<ts>]

Re-evaluates the same frozen partition with both validated backends, checks
the checkpoint SHA-256 and the partition hash against the stored run, and
requires U_r, U_p, U_c, U_r_hat and every bound to agree to 1e-12 relative.
Exit code 0 on agreement, 1 otherwise. Writes a new run directory (never
overwrites the reference).
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import pandas as pd

from certified_obstacle_nn.experiments.v14.common import REPO_ROOT, STEFAN_SEEDS, read_summary, sha256_file
from certified_obstacle_nn.experiments.v14.stefan_validated import run, write_run

KEYS = ("U_r", "U_p", "U_c", "U_r_hat", "E_U_uniform", "E_U_weighted", "E_U_best", "E_U_projected",
        "E_U_original_via_projection", "grad_bound", "flux_bound")


def find_reference():
    best = None
    for f in sorted(glob.glob(str(REPO_ROOT / "results/v14_stefan_validated/*/summary.csv"))):
        df = read_summary(f)
        r = df.iloc[0]
        if r["seed"] == 0 and r["mode"] == "uniform" and r["n_cells"] == 1024 and not bool(r["git_dirty"]) \
                and set(df["backend"]) >= {"arb", "numpy"} and r["workers"] > 1:
            best = f
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--results-dir", default=None, help="write the new run here instead of results/")
    a = ap.parse_args()
    ref_csv = (REPO_ROOT / a.run_dir / "summary.csv") if a.run_dir else find_reference()
    if ref_csv is None:
        print("no committed-code reference run of Stefan seed 0 uniform 32 found")
        return 1
    ref = read_summary(ref_csv).set_index("backend")
    ref_csv = Path(ref_csv)
    cfg = json.loads((ref_csv.parent / "config.json").read_text())
    seed, n = int(cfg["seed"]), int(cfg["n"])
    assert sha256_file(STEFAN_SEEDS[seed]) == cfg["checkpoint_sha256"], "checkpoint changed"
    config, rows, part, vols, _ = run(seed, "uniform", n, n, "numpy", "arb", a.workers, int(cfg["arb_prec_bits"]),
                                      int(cfg["trace_n"]), float(cfg["theta"]))
    d = write_run(config, rows, part, vols, STEFAN_SEEDS[seed], a.results_dir)
    ok = True
    for r in rows:
        b = r["backend"]
        if r["partition_hash"] != ref.loc[b, "partition_hash"]:
            print(b, "partition hash differs")
            ok = False
        for k in KEYS:
            rv, sv = float(r[k]), float(ref.loc[b, k])
            relerr = abs(rv - sv) / abs(sv) if sv else abs(rv)
            flag = "OK" if relerr <= 1e-12 else "DIFF"
            ok &= flag == "OK"
            print(f"{b:6s} {k:30s} stored {sv:.17g} new {rv:.17g} rel {relerr:.1e} {flag}")
    print("reference:", ref_csv)
    print("new run:", d)
    print("REPRODUCED" if ok else "NOT REPRODUCED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
