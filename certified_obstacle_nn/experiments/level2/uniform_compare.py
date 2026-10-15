"""Step H (TODO Task L, Gate 4): global/uniform (legacy) vs local/uniform (upper sum, all tiers)
on the same grids. Writes results/level2_local/runs/uniform/<case>_<method>_<n>.json and rows of
docs/history/LEVEL2_COMPARISON_TABLE.csv (legacy rows are taken from the Step A-C / Step F runs).

python3 -m certified_obstacle_nn.experiments.level2.uniform_compare --case 1d --grids 64,128,256,512,1024 \
    --methods T1,T3,T2i,T2a,LL,T3+T2a
"""

import argparse
import json

from certified_obstacle_nn.certification.level2.problems import make_problem
from certified_obstacle_nn.certification.level2.upper_sum import upper_sum_uniform
from certified_obstacle_nn.experiments.level2.common import (RUNS, legacy_row, load_case, record, upper_sum_row,
                                                             write_json)

VARIANT = {"T3q": "upper-sum-T3q(TM2)", "T3+T3q": "upper-sum-T3∩T3q", "T3q+T2a": "upper-sum-T3q∩T2", "T3z": "upper-sum-T3z(zonotope)", "T1": "upper-sum-T1", "T3": "upper-sum-T3", "T2i": "upper-sum-T2(interval-jac)",
           "T2a": "upper-sum-T2(affine-jac)", "LL": "local-Lipschitz", "T3+T2a": "upper-sum-T3∩T2"}


def legacy_rows(case, meta):
    rows = []
    for p in sorted((RUNS / "legacy").glob(f"{case}_*.json")):
        if not p.stem[len(case) + 1:].isdigit():   # e.g. '2d_base_s1_64' is a different case
            continue
        d = json.loads(p.read_text())
        r = dict(d["row"])
        r["n_cells"] = r["n_cells"]
        rows.append(legacy_row(meta, r, "legacy", r["grid"], notes="Option-B legacy (E_cert_U_new)"))
    for p in sorted((RUNS / "geometry").glob(f"{case}_*_local.json")):
        if not p.stem[len(case) + 1:-len("_local")].isdigit():
            continue
        r = json.loads(p.read_text())
        g = f"{r['counts'][0]}^{len(r['counts'])}"
        rows.append(legacy_row(meta, r, "legacy+local-radius", g, method="global-constants/local-radius/uniform"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--grids", required=True)
    ap.add_argument("--methods", default="T3,T2a,LL")
    ap.add_argument("--batch", type=int, default=2048)
    a = ap.parse_args()
    model, meta = load_case(a.case)
    prob = make_problem(model, meta)
    record(legacy_rows(a.case, meta))
    for n in [int(s) for s in a.grids.split(",")]:
        for meth in a.methods.split(","):
            res = upper_sum_uniform(prob, [n] * prob.m, meth, cells_per_batch=a.batch)
            assert res["L_vol"] <= res["rho_mid_vol"] <= res["U_vol"], "midpoint value outside [L,U]: bug"
            grid = f"{n}^{prob.m}"
            write_json(RUNS / "uniform" / f"{a.case}_{meth}_{n}.json", res)
            row = upper_sum_row(meta, res, "local/uniform", VARIANT[meth], grid)
            record([row])
            print(json.dumps({k: row[k] for k in ("case", "method_variant", "grid", "U", "L", "E_cert_U",
                                                  "certificate_effectivity", "slack", "unresolved_fraction",
                                                  "runtime")}), flush=True)


if __name__ == "__main__":
    main()
