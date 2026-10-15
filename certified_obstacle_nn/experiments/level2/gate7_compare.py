"""Gate 7: baseline vs long-schedule checkpoint (seed 0): legacy global constants (L_R, L_gamma, m_R) and
local enclosure widths of R_PDE and gamma (T3 and T3q) on the SAME 300 random cells of three sizes,
relative to the sampled range (2000 samples per cell, diagnostic only).

python3 -m certified_obstacle_nn.experiments.level2.gate7_compare
"""

import json

import torch

from certified_obstacle_nn.certification.level2.fields import fields_at
from certified_obstacle_nn.certification.level2.problems import make_problem
from certified_obstacle_nn.certification.level2.upper_sum import enclose_cells
from certified_obstacle_nn.experiments.level2.common import RUNS, legacy_constants, load_case, write_json


def main():
    out = {}
    for case in ("2d_base", "2d_long"):
        model, meta = load_case(case)
        prob = make_problem(model, meta)
        k = legacy_constants(model, meta)
        rec = {"L_R": k["L_R"], "L_gamma": k["L_gamma"], "m_R_apriori": k["m_R_apriori"], "L_sm": k["L_sm"],
               "E_true": meta["E_true"], "rho_ref": meta["rho_ref"], "floor": meta["floor"]}
        for w in (1 / 32, 1 / 64, 1 / 128):
            g = torch.Generator().manual_seed(7)
            lo = torch.rand(100, 3, generator=g) * (1 - w)
            hi = lo + w
            S = lo[:, None, :] + (hi - lo)[:, None, :] * torch.rand(100, 2000, 3, generator=g)
            F = fields_at(prob, S.reshape(-1, 3))
            for meth in ("T3", "T3q"):
                iv = enclose_cells(prob, lo, hi, meth)
                for f in ("R", "gamma"):
                    v = F[f].reshape(100, 2000)
                    sr = v.max(1).values - v.min(1).values
                    width = iv[f][1] - iv[f][0]
                    rec[f"{f}_width_median_{meth}_w{round(1 / w)}"] = float(width.median())
                    rec[f"{f}_width_over_sampled_{meth}_w{round(1 / w)}"] = float((width / sr).median())
                    rec[f"{f}_sampled_range_median_w{round(1 / w)}"] = float(sr.median())
        out[case] = rec
        print(case, json.dumps(rec), flush=True)
    write_json(RUNS / "gate7_baseline_vs_long.json", out)


if __name__ == "__main__":
    main()
