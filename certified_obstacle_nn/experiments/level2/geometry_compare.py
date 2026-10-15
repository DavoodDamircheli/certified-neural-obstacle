"""Step F / Task F (Gate 2): legacy global constants with the global h vs. the local radius r_K,
through the per-cell implementation certification/level2/legacy_local.py. The 'global' mode must
reproduce the recorded legacy outputs (rtol 1e-10, TODO section 28).

python3 -m certified_obstacle_nn.experiments.level2.geometry_compare [--cases 1d,2d_base] [--grids2d 16,32,64]
"""

import argparse
import csv
import json

from certified_obstacle_nn.certification.level2.legacy_local import legacy_uniform
from certified_obstacle_nn.certification.level2.problems import make_problem
from certified_obstacle_nn.experiments.level2.common import RUNS, legacy_constants, load_case, write_json

KEYS = ("rho_tilde_h", "E_h_lip_old", "E_h_lip_new", "E_h_unr", "E_h_0T", "E_cert_U_old", "E_cert_U_new")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="1d,2d_base")
    ap.add_argument("--grids2d", default="16,32,64,128")
    a = ap.parse_args()
    rows = []
    for case in a.cases.split(","):
        model, meta = load_case(case)
        prob = make_problem(model, meta)
        consts = legacy_constants(model, meta)
        grids = [1024] if case == "1d" else [int(s) for s in a.grids2d.split(",")]
        for n in grids:
            counts = [n] * prob.m
            q = 4 if case == "1d" else 2
            out = {}
            for mode in ("global", "local"):
                r = legacy_uniform(prob, consts, counts, q, radius=mode)
                out[mode] = r
            leg = json.loads((RUNS / "legacy" / f"{case}_{n}.json").read_text())["row"]
            reg = max(abs(out["global"][k] - leg[k]) / abs(leg[k]) for k in KEYS)
            for mode, r in out.items():
                row = {"case": case, "grid": f"{n}^{prob.m}", "mode": f"global_constants+{mode}_"
                       + ("h" if mode == "global" else "radius"), "n_cells": r["n_cells"],
                       "V_unr_frac": r["V_unr_frac"], "E_h_lip": r["E_h_lip_new"], "E_h_unr": r["E_h_unr"],
                       "E_h_0T": r["E_h_0T"], "E_h": r["E_h_new"], "E_cert_U": r["E_cert_U_new"],
                       "E_true": meta["E_true"], "ratio": r["E_cert_U_new"] / meta["E_true"],
                       "rho_tilde_h": r["rho_tilde_h"], "wall_s": r["wall_s"],
                       "legacy_regression_max_rel_diff": reg if mode == "global" else ""}
                rows.append(row)
                print(json.dumps(row), flush=True)
                write_json(RUNS / "geometry" / f"{case}_{n}_{mode}.json", r)
    out = RUNS / "geometry" / "geometry_compare.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
