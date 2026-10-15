"""Steps I-L (TODO Tasks M-S): adaptive upper-sum certificate on a frozen checkpoint.

Writes results/level2_local/runs/adaptive/<case>_<method>_<tag>/{result.json, history.csv, cells.npz}
and one docs/history/LEVEL2_COMPARISON_TABLE.csv row (method 'local/adaptive').

python3 -m certified_obstacle_nn.experiments.level2.adaptive_run --case 1d --method T3 --init_n 16 \
    --max_cells 65536 --tau 0.02 --tag c65k
"""

import argparse
import csv
import json

import numpy as np

from certified_obstacle_nn.certification.level2.adaptive import adaptive_upper_sum
from certified_obstacle_nn.certification.level2.problems import make_problem
from certified_obstacle_nn.experiments.level2.common import RUNS, git_hash, load_case, record, upper_sum_row, write_json


def run(case, method, init_n, max_cells, tau, wall, frac, tag, batch=2048, init_n_space=None):
    model, meta = load_case(case)
    prob = make_problem(model, meta)
    out = RUNS / "adaptive" / f"{case}_{method}_{tag}"
    out.mkdir(parents=True, exist_ok=True)

    def log(h):
        if h["sweep"] % 5 == 0:
            print(json.dumps({k: (round(v, 7) if isinstance(v, float) else v) for k, v in h.items()}), flush=True)

    res = adaptive_upper_sum(prob, method, init_n=init_n, init_n_space=init_n_space, tau=tau, max_cells=max_cells,
                             wall_time_budget=wall, frac=frac, batch=batch, log=log)
    np.savez_compressed(out / "cells.npz", **res.pop("cells"), **{f"bdry_{k}": v for k, v in res.pop("bdry_cells").items()})
    hist = res.pop("history")
    with open(out / "history.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(hist[0].keys()))
        w.writeheader()
        w.writerows(hist)
    res["git"] = git_hash()
    res["case_meta"] = {k: meta[k] for k in ("case", "checkpoint", "E_true", "rho_ref", "rho_ref_source", "C_eq",
                                            "floor")}
    write_json(out / "result.json", res)
    variant = f"adaptive-{method}"
    row = upper_sum_row(meta, res, "local/adaptive", variant, f"adaptive<= {max_cells} ({tag})",
                        notes=f"stop={res['stop']} init={init_n}^{prob.m} tau={tau} frac={frac}")
    row["method_variant"] = f"{variant}-{tag}"
    record([row])
    print(json.dumps({k: row[k] for k in ("case", "method_variant", "cells", "U", "L", "E_cert_U",
                                          "certificate_effectivity", "slack", "runtime", "max_depth")}), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--method", default="T3")
    ap.add_argument("--init_n", type=int, default=8)
    ap.add_argument("--init_n_space", type=int, default=None)
    ap.add_argument("--max_cells", type=int, default=2_000_000)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--wall", type=float, default=7200.0)
    ap.add_argument("--frac", type=float, default=0.10)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--tag", default="default")
    a = ap.parse_args()
    run(a.case, a.method, a.init_n, a.max_cells, a.tau, a.wall, a.frac, a.tag, a.batch, a.init_n_space)


if __name__ == "__main__":
    main()
