"""Steps A-C (TODO Task B): rerun the UNCHANGED legacy global Level-2 certificate on the frozen
checkpoints and record the outputs (regression baseline, TODO section 28).

  1D Stefan 4x64 seed 0, grid 1024^2, n_qp=4      -> certify_option_b (scripts/certify_cli path)
  2D baseline seed 0 at 16^3, 32^3, 64^3, 128^3   -> expanding_disk_2d/level2_diagnostic.certify
  2D long seed 0 at 32^3, 64^3, 128^3             -> same

Outputs: results/level2_local/runs/legacy/<case>_<grid>.json (+ legacy_summary.csv). Nothing
inside any existing result folder is written.

python3 -m certified_obstacle_nn.experiments.level2.reproduce_legacy [--cases 1d,2d_base,2d_long]
"""

import argparse
import csv
import json
import math
import resource
import time

from certified_obstacle_nn.experiments.level2.common import RUNS, git_hash, load_case, write_json

GRIDS_2D = {"2d_base": [16, 32, 64, 128], "2d_long": [32, 64, 128]}


def run_1d(grid=1024, n_qp=4):
    from certified_obstacle_nn.certification.certificate_option_b import certify_option_b
    from certified_obstacle_nn.certification.partition import build_partition
    model, meta = load_case("1d")
    problem = meta["problem"]
    t0 = time.time()
    part = build_partition(problem, grid, grid)
    r = certify_option_b(model, problem, part, training_points=None, n_qp=n_qp)
    wall = time.time() - t0
    from dataclasses import asdict
    d = asdict(r)
    row = {"case": "1d", "grid": f"{grid}^2", "n_cells": grid * grid, "E_true": meta["E_true"],
           "rho_tilde_h": r.rho_tilde_h, "rho_r": r.rho_r, "rho_p_tilde": r.rho_p_tilde, "rho_c": r.rho_c,
           "V_unr_frac": r.V_h_unr, "E_h_lip_old": r.E_h_lip_old, "E_h_lip_new": r.E_h_lip_new,
           "E_h_unr": r.E_h_unr, "E_h_0T": r.E_h_0T, "E_h_old": r.E_h_old, "E_h_new": r.E_h_new,
           "E_cert_U_old": r.E_cert_U_old, "E_cert_U_new": r.E_cert_U_new,
           "ratio_old": r.E_cert_U_old / meta["E_true"], "ratio_new": r.E_cert_U_new / meta["E_true"],
           "L_R": r.bounds.lip_R_PDE, "L_gamma": r.bounds.L_gamma, "m_R_apriori": r.bounds.m_R,
           "m_R_cert": r.residual_bound.m_R_cert, "L_sm_old": r.L_sm_old, "L_Psi": r.bounds.L_Psi,
           "wall_s": wall, "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    with open(f"{meta['checkpoint']}/certification/certificate_grid1024.json") as fh:
        rec = json.load(fh)["result"]
    row["regression_max_rel_diff"] = max(
        abs(row[k] - rec[k2]) / abs(rec[k2]) for k, k2 in
        (("E_cert_U_old", "E_cert_U_old"), ("E_cert_U_new", "E_cert_U_new"), ("rho_tilde_h", "rho_tilde_h"),
         ("E_h_unr", "E_h_unr"), ("E_h_lip_old", "E_h_lip_old")))
    write_json(RUNS / "legacy" / f"1d_{grid}.json", {"git": git_hash(), "row": row, "full": d})
    return row


def run_2d(case, grids, q=2):
    from certified_obstacle_nn.experiments.expanding_disk_2d.level2_diagnostic import certify
    model, meta = load_case(case)
    cfg = meta["cfg"]
    rows = []
    for n in grids:
        r = certify(model, cfg, n, q=q)
        bd = r["bounds"]
        row = {"case": case, "grid": f"{n}^3", "n_cells": n ** 3, "E_true": meta["E_true"],
               "rho_tilde_h": r["rho_tilde_h"], "rho_r": r["rho_r"], "rho_p_tilde": r["rho_p_tilde"],
               "rho_c": r["rho_c"], "V_unr_frac": r["V_unr_frac"], "E_h_lip_old": r["E_h_lip_old"],
               "E_h_lip_new": r["E_h_lip_new"], "E_h_unr": r["E_h_unr"], "E_h_0T": r["E_h_0T"],
               "E_h_old": r["E_h_old"], "E_h_new": r["E_h_new"], "E_cert_U_old": r["E_cert_U_old"],
               "E_cert_U_new": r["E_cert_U_new"], "ratio_old": r["E_cert_U_old"] / meta["E_true"],
               "ratio_new": r["E_cert_U_new"] / meta["E_true"], "L_R": bd["L_R"], "L_gamma": bd["L_gamma"],
               "m_R_apriori": bd["m_R_apriori"], "m_R_cert": r["m_R_cert"], "L_sm_old": bd["L_sm_old"],
               "L_Psi": bd["L_Psi"], "wall_s": r["wall_s"], "peak_rss_mb": r["peak_rss_mb"]}
        rows.append(row)
        write_json(RUNS / "legacy" / f"{case}_{n}.json", {"git": git_hash(), "row": row, "full": r})
        print(json.dumps(row), flush=True)
    return rows


def regression_2d_base(rows):
    """Compare against results/expanding_disk_2d/level2_diagnostic.csv (seed 0 rows)."""
    from certified_obstacle_nn.experiments.level2.common import REPO_ROOT
    rec = {r["grid"]: r for r in csv.DictReader(open(REPO_ROOT / "results/expanding_disk_2d/level2_diagnostic.csv"))
           if r["seed"] == "0"}
    for row in rows:
        old = rec.get(row["grid"])
        if old is None:
            continue
        diffs = [abs(row[a] - float(old[b])) / abs(float(old[b])) for a, b in
                 (("E_cert_U_new", "E_cert_U"), ("rho_tilde_h", "rho_tilde_h"), ("E_h_unr", "E_h_unr"),
                  ("E_h_lip_new", "E_h_lip"))]
        row["regression_max_rel_diff"] = max(diffs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="1d,2d_base,2d_long")
    ap.add_argument("--grids2d", default=None, help="override the 2D grids, e.g. 64")
    a = ap.parse_args()
    rows = []
    for c in a.cases.split(","):
        if c == "1d":
            rows.append(run_1d())
            print(json.dumps(rows[-1]), flush=True)
        else:
            grids = [int(g) for g in a.grids2d.split(",")] if a.grids2d else GRIDS_2D[c]
            rr = run_2d(c, grids)
            if c == "2d_base":
                regression_2d_base(rr)
            rows += rr
    out = RUNS / "legacy" / "legacy_summary.csv"
    keys = sorted({k for r in rows for k in r})
    old = list(csv.DictReader(open(out))) if out.exists() else []
    done = {(r["case"], r["grid"]) for r in rows}
    old = [r for r in old if (r["case"], r["grid"]) not in done]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in old + rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print("wrote", out, "max regression diff:",
          max((r.get("regression_max_rel_diff", 0.0) for r in rows), default=math.nan))


if __name__ == "__main__":
    main()
