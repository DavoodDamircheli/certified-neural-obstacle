"""v17 Stage 1 gate (roadmap R3 Stage 1; prompt Section 6): the NEW reliability.py no-split path, fed the v16
input records docs/v16_reference/v16/data/certificates_input.csv, must reproduce the *_direct columns of
data/certificates.csv bit-for-bit, and regenerating Tables 2-4 from those values must give files byte-identical to
the bundle's tables/*.tex.

    python3 -m certified_obstacle_nn.experiments.v17.stage1_v16_gate --out <scratch dir>

Exit 0 and prints GATE PASS, or lists every differing value / file and exits 1. Writes only into --out.
"""

from __future__ import annotations

import argparse
import csv
import filecmp
import sys
from pathlib import Path

from certified_obstacle_nn.certification import reliability as rel
from certified_obstacle_nn.experiments.v17.common import V16_REF
from certified_obstacle_nn.experiments.v17.v16_tables import write_v16_tables
from certified_obstacle_nn.problems.disk import DiskProblem
from certified_obstacle_nn.problems.ellipse import EllipseProblem
from certified_obstacle_nn.problems.stefan import StefanProblem

DIM = {"stefan": 1, "disk2d": 2, "ellipse": 2}
PROBLEMS = {"stefan": StefanProblem, "disk2d": DiskProblem, "ellipse": EllipseProblem}
DIRECT_COLS = ("cF2_upper", "theta_direct", "flux_bound_direct", "grad_bound_direct", "E_U_direct",
               "flux_direct_relative_upper", "grad_direct_relative_upper", "reference_norm_lower",
               "E_U_weighted", "B_projected", "grad_bound_projected", "flux_bound_projected", "E_U_projected",
               "projected_relative_upper", "projected_ratio_status", "ratio_E_U_direct_vs_vhat")


def recompute(r: dict) -> dict:
    prob = r["problem"]
    p = PROBLEMS[prob]
    alpha, A_inf, Lam = float(p.alpha), float(p.A_inf), float(p.Lambda)
    cf = rel.cF2_upper(DIM[prob])["cF2_up"]
    Ceq = rel.C_eq_heat(alpha, A_inf, Lam)
    out = {"cF2_upper": cf}
    ns = rel.direct_nosplit_bounds(U_r_hat=float(r["U_r_hat"]), U_C=float(r["U_c"]), cF2_up=cf, Lambda=Lam,
                                   alpha=alpha)
    out.update({k: ns[k] for k in ("theta_direct", "flux_bound_direct", "grad_bound_direct", "E_U_direct")})
    out["E_U_weighted"] = rel.weighted_bound_heat(U_r=float(r["U_r"]), U_p=float(r["U_p"]), U_c=float(r["U_c"]),
                                                  C_eq=Ceq)
    pe = rel.projected_estimator_bounds(U_r_hat=float(r["U_r_hat"]), U_c=float(r["U_c"]), alpha=alpha, A_inf=A_inf,
                                        C_eq=Ceq)
    out.update(pe)
    out["ratio_E_U_direct_vs_vhat"] = out["E_U_direct"] / float(r["E_true_vhat"])
    if prob in ("stefan", "disk2d"):  # v16 had validated denominators for these two only (ellipse: diagnostic)
        ref = rel.reference_norm(prob)
        Nlo = ref["norm_lower"]
        d = rel.direct_projected_bounds(U_R=0.0, U_S=0.0, U_I=0.0, U_C=float(r["U_c"]), cF2_up=cf,
                                        status=r["status"], U_r_hat=float(r["U_r_hat"]), Lambda=Lam, alpha=alpha,
                                        A_inf=A_inf, reference_norm_lower=Nlo, reference_status=ref["status"])
        out["reference_norm_lower"] = Nlo
        out["flux_direct_relative_upper"] = d["flux_bound_direct_rel"]
        out["grad_direct_relative_upper"] = d["grad_bound_direct_rel"]
        out["projected_relative_upper"] = rel.relative_upper(pe["grad_bound_projected"], Nlo)
        out["projected_ratio_status"] = d["ratio_status"]
    else:
        ng = float(r["norm_grad_u"])
        out["reference_norm_lower"] = ""
        out["projected_relative_upper"] = out["grad_bound_projected"] / ng
        out["flux_direct_relative_upper"] = out["flux_bound_direct"] / ng
        out["grad_direct_relative_upper"] = out["grad_bound_direct"] / ng
        out["projected_ratio_status"] = "diagnostic_float64"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if V16_REF.resolve() in [out.resolve(), *out.resolve().parents]:
        raise SystemExit("refusing to write into docs/v16_reference")
    out.mkdir(parents=True, exist_ok=True)
    inp = list(csv.DictReader((V16_REF / "data/certificates_input.csv").open()))
    ref = list(csv.DictReader((V16_REF / "data/certificates.csv").open()))
    assert len(inp) == len(ref) == 42
    bad = []
    rows = []
    for r, s in zip(inp, ref):
        new = recompute(r)
        row = dict(r)
        row.update(new)
        rows.append(row)
        for k in DIRECT_COLS:
            # the v16 CSV stores repr(float) (csv module); compare the exact string the v16 writer produced
            if str(new[k]) != s[k]:
                bad.append((r["problem"], r["schedule"], r["seed"], r["size"], k, s[k], new[k]))
    write_v16_tables(rows, out / "tables")
    files = ("stefan.tex", "two_dimensional.tex", "disk_relative.tex")
    fbad = [f for f in files if not filecmp.cmp(out / "tables" / f, V16_REF / "tables" / f, shallow=False)]
    for b in bad[:30]:
        print("DIFF value", *b)
    for f in fbad:
        print("DIFF table", f)
    print(f"{42 * len(DIRECT_COLS)} values compared, {len(bad)} differ; tables 2-4 differing: {len(fbad)}")
    print("GATE PASS" if not bad and not fbad else "GATE FAIL")
    return 0 if not bad and not fbad else 1


if __name__ == "__main__":
    sys.exit(main())
