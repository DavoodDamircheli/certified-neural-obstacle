"""Task X (prompt section 4 'Floating point'): high-precision spot check of the float64 enclosures.

For 10 random cells per case (1D, 2D baseline) and 6 points per cell (2 opposite corners + 4 random),
evaluate every field (R_PDE, R_sig, gamma, d_t gamma, grad_x gamma, lambda) in mpmath with 40
significant digits (network weights/biases and the points are converted exactly from float64; the
forward-mode jet and the data formulas are evaluated in mp arithmetic), and check that the value lies
inside the float64 Tier-3 / Tier-2 enclosure WITHOUT any tolerance. Also reports the max relative
difference between the float64 pointwise value and the mp value (rounding level).

python3 -m certified_obstacle_nn.experiments.level2.mpmath_spotcheck
"""

import json

import mpmath
import torch

from certified_obstacle_nn.certification.level2.fields import fields_at
from certified_obstacle_nn.certification.level2.pointwise import mlp_layers
from certified_obstacle_nn.certification.level2.problems import make_problem
from certified_obstacle_nn.certification.level2.upper_sum import enclose_cells
from certified_obstacle_nn.experiments.level2.common import OUT_ROOT, load_case, write_json

mpmath.mp.dps = 40


def mp_mlp(mlp):
    return [([[mpmath.mpf(float(w)) for w in row] for row in W.tolist()], [mpmath.mpf(float(v)) for v in b.tolist()])
            for W, b in mlp_layers(mlp)]


def mp_jet(layers, p):
    """value [out], jac [out][m] (all directions) in mp arithmetic."""
    m = len(p)
    W, b = layers[0]
    a = [mpmath.fdot(row, p) + bi for row, bi in zip(W, b)]
    da = [[row[j] for j in range(m)] for row in W]
    for W, b in layers[1:]:
        z = [mpmath.tanh(v) for v in a]
        s = [1 - v * v for v in z]
        dz = [[s[i] * da[i][j] for j in range(m)] for i in range(len(z))]
        a = [mpmath.fdot(row, z) + bi for row, bi in zip(W, b)]
        da = [[mpmath.fdot(row, [dz[i][j] for i in range(len(z))]) for j in range(m)] for row in W]
    return a, da


def mp_fields(prob, nets, p):
    Nv, Nj = mp_jet(nets["state"], p)
    Sv, Sj = mp_jet(nets["flux"], p)
    Mv, _ = mp_jet(nets["mult"], p)
    d = prob.d
    net = {"N": Nv[0], "N_t": Nj[0][0], "N_x": [Nj[0][1 + i] for i in range(d)], "S": [Sv[i] for i in range(d)],
           "dS": [Sj[i][1 + i] for i in range(d)], "M": Mv[0]}
    F = prob.fields_from_net(list(p), net)
    out = {"R": F["R"], "gamma": F["gamma"], "gamma_t": F["gamma_t"], "lam": F["lam"]}
    for i in range(d):
        out[f"Rsig_{i}"] = F["Rsig"][i]
        out[f"ggrad_{i}"] = F["ggrad"][i]
    return out


def main():
    report = {}
    g = torch.Generator().manual_seed(2026)
    for case in ("1d", "2d_base"):
        model, meta = load_case(case)
        prob = make_problem(model, meta)
        nets = {k: mp_mlp(getattr(prob, k)) for k in ("state", "flux", "mult")}
        w = 0.02 if case == "1d" else 0.01
        lo = torch.rand(10, prob.m, generator=g) * (1 - w)
        hi = lo + w
        encl = {meth: enclose_cells(prob, lo, hi, meth) for meth in ("T3", "T2a")}
        n_checked = n_viol = 0
        max_rel_f64 = 0.0
        min_margin = float("inf")
        for c in range(10):
            pts = [lo[c], hi[c]] + [lo[c] + (hi[c] - lo[c]) * torch.rand(prob.m, generator=g) for _ in range(4)]
            for p in pts:
                pm = [mpmath.mpf(float(v)) for v in p.tolist()]
                Fm = mp_fields(prob, nets, pm)
                Ff = fields_at(prob, p[None, :])
                ff = {"R": Ff["R"], "gamma": Ff["gamma"], "gamma_t": Ff["gamma_t"], "lam": Ff["lam"]}
                for i in range(prob.d):
                    ff[f"Rsig_{i}"] = Ff["Rsig"][i]
                    ff[f"ggrad_{i}"] = Ff["ggrad"][i]
                for k, vm in Fm.items():
                    vf = float(ff[k][0])
                    max_rel_f64 = max(max_rel_f64, float(abs(mpmath.mpf(vf) - vm) / (abs(vm) + mpmath.mpf(1e-300))))
                    for meth, iv in encl.items():
                        l, h = mpmath.mpf(float(iv[k][0][c])), mpmath.mpf(float(iv[k][1][c]))
                        n_checked += 1
                        if not (l <= vm <= h):
                            n_viol += 1
                        min_margin = min(min_margin, float(min(vm - l, h - vm) / (h - l)))
        report[case] = {"cells": 10, "points_per_cell": 6, "checks": n_checked, "violations": n_viol,
                        "max_rel_diff_float64_vs_mp40": max_rel_f64,
                        "min_relative_margin_inside_enclosure": min_margin, "cell_width": w}
        print(case, report[case], flush=True)
    write_json(OUT_ROOT / "runs" / "mpmath_spotcheck.json", report)


if __name__ == "__main__":
    main()
