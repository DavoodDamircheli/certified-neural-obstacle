"""v19 Task A (referee item 5): substitution diagnostics on the ellipse, no training. diagnostic_float64, post_hoc.

    python3 -m certified_obstacle_nn.experiments.v19.substitution --run results/v17_split/<ts> [--slices-only]

For the frozen checkpoint and the frozen partition of a stored ellipse row (protocol v1 3.5e6 or v2 8.0e6), the
float64 exact integrals of the per-term densities (the "I" of the stored slack U / I: q = 2 Gauss points per axis
in every cell, experiments/v17/cell_quadrature.py, same chunking) for six substitutions of (flux, multiplier); the
state is always v_hat = max(v, g) of the trained network:

    A0  tau_phi        lambda_psi = M_psi^2   (no exact information; must reproduce the stored I_R, I_S, I_I, I_C)
    A1  tau_phi        lambda (exact)         oracle
    A2  tau_phi        mu*                    no exact information
    A3  sigma (exact)  lambda_psi             oracle
    A4  sigma (exact)  lambda (exact)         oracle
    A5  sigma (exact)  mu*                    oracle (flux only)

with R_hat = f - d_t v_hat - div(flux) + mu, S_hat = flux + grad v_hat, C = mu (v - g)_+ and the post-hoc multiplier
mu* = 1{v <= g} max(-R_hat_0, 0), R_hat_0 = f - d_t v_hat - div(flux) (so C = 0, R_hat = R_hat_0 on {v > g} and
max(R_hat_0, 0) on {v <= g}). Exact flux sigma = -grad u and div sigma = -Delta u, exact multiplier lambda: closed
forms of problems/ellipse.py. U_0 = I_I is the t = 0 trace integral (independent of flux and multiplier).
From the integrals: L = cF2_up I_R, D = I_S + I_I + 2 I_C, F = sqrt(D + L), G_dir = sqrt(L) + sqrt(D + L),
G_AR = (a + sqrt(a^2 + 2 (I_I + 2 I_C)))/2 with a = sqrt(I_S) + sqrt(L), each divided by N_lower; shares of
L, I_S, 2 I_C in F^2 = D + L. Volume integrals are also split by region (pointwise): exact contact set
{zeta < 0}, ring = symmetric difference of {v < g} and the contact set, and the rest.

Slices t = 0.25, 0.5, 0.75 (as fig_ellipse_solution), A0 and A2, on a uniform midpoint grid of the unit square:
area of {v < g} / area of the exact contact set, and the area integrals of |S_hat|^2 and R_hat^2 over the three
regions.

Writes results/v19_substitution/<ts>/ (config.json, substitution.csv, slices.csv, log.txt).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time

import numpy as np
import pandas as pd
import torch

from certified_obstacle_nn.certification import reliability as rel
from certified_obstacle_nn.certification.enclose import boxes as Bx
from certified_obstacle_nn.experiments import results_io
from certified_obstacle_nn.experiments.v14.common import REPO_ROOT, git_commit, git_dirty, hardware
from certified_obstacle_nn.experiments.v17 import cell_quadrature as CQ
from certified_obstacle_nn.experiments.v17.common import load_stored_run

CONFIGS = {  # id: (flux, multiplier, uses exact solution)
    "A0": ("tau_phi", "lambda_psi", "no"),
    "A1": ("tau_phi", "lambda_exact", "oracle"),
    "A2": ("tau_phi", "mu_star", "no"),
    "A3": ("sigma_exact", "lambda_psi", "oracle"),
    "A4": ("sigma_exact", "lambda_exact", "oracle"),
    "A5": ("sigma_exact", "mu_star", "oracle (flux only)"),
}
REGIONS = ("contact", "ring", "rest")
SLICES = (0.25, 0.5, 0.75)


def point_fields(problem, triple, cols, configs=CONFIGS):
    """Point densities Rh2, Sh2, C of every configuration, region masks. The A0 expressions are those of
    cell_quadrature._fields (same operations in the same order, hence the same binary64 values)."""
    t, x, y = cols
    X = torch.stack([t, x, y], -1)
    v = problem.mask(x, y) * triple.state(X)[:, 0]
    S = triple.flux(X)
    taus = [S[:, 0], S[:, 1]]
    divt = CQ._grad(S[:, 0], (x,))[0] + CQ._grad(S[:, 1], (y,))[0]
    mu = triple.mult(X)[:, 0] ** 2
    g = problem.g(t, x, y)
    f = problem.f(t, x, y)
    dv = CQ._grad(v, cols)
    dg = CQ._grad(g, cols)
    out = {}
    with torch.no_grad():
        sig = list(problem.sigma_exact(t, x, y))
        divs = problem.div_sigma_exact(t, x, y)
        lam = problem.lambda_exact(t, x, y)
        pos = (v > g)
        vh_t = torch.where(pos, dv[0], dg[0])
        vh_x = [torch.where(pos, dv[k], dg[k]) for k in range(1, len(cols))]
        fluxes = {"tau_phi": (taus, divt), "sigma_exact": (sig, divs)}
        mults = {"lambda_psi": mu, "lambda_exact": lam}
        for cid, (fl, mm, _) in configs.items():
            ts, dvg = fluxes[fl]
            if mm == "mu_star":
                Rh0 = f - vh_t - dvg
                Rh = torch.where(pos, Rh0, torch.clamp(Rh0, min=0))
                C = torch.zeros_like(Rh)
            else:
                m = mults[mm]
                Rh = f - vh_t - dvg + m
                C = m * torch.clamp(v - g, min=0)
            out[f"{cid}:Rh2"] = Rh**2
            out[f"{cid}:Sh2"] = sum((ts[k] + vh_x[k]) ** 2 for k in range(len(ts)))
            out[f"{cid}:C"] = C
        contact = problem.contact(t, x, y)
        under = v < g
        masks = {"contact": contact, "ring": contact ^ under, "rest": ~(contact | under),
                 "under": under, "under_not_contact": under & ~contact, "contact_not_under": contact & ~under,
                 "le": ~pos}
    return ({k: a.detach().numpy() for k, a in out.items()},
            {k: m.detach().numpy() for k, m in masks.items()},
            {"mu_psi": mu.detach().numpy(), "lam": lam.numpy()})


def volume_integrals(problem, triple, lo, hi, q=2, chunk=200_000, log=print):
    """Totals over the partition (per-cell sums first, as cell_quadrature.cell_integrals; then np.sum of the cells)
    of every configuration's densities, plus region splits (pointwise masks, diagnostic)."""
    n = lo.shape[1]
    nq = q ** lo.shape[0]
    keys = [f"{c}:{d}" for c in CONFIGS for d in ("Rh2", "Sh2", "C")]
    cell = {k: np.zeros(n) for k in keys}
    reg = {f"{k}@{r}": 0.0 for k in keys for r in (*REGIONS, "le")}
    vol = {r: 0.0 for r in ("contact", "ring", "rest", "under", "under_not_contact", "contact_not_under")}
    cells_per = max(1, chunk // nq)
    tic = time.time()
    for s in range(0, n, cells_per):
        e = min(n, s + cells_per)
        P, W = CQ._pts(lo[:, s:e], hi[:, s:e], q)
        cols = [torch.as_tensor(P[j]).clone().requires_grad_(True) for j in range(P.shape[0])]
        F, M, _ = point_fields(problem, triple, cols)
        for k in keys:
            cell[k][s:e] = (F[k] * W).reshape(e - s, nq).sum(1)
            for r in (*REGIONS, "le"):
                reg[f"{k}@{r}"] += float((F[k] * W)[M[r]].sum())
        for r in vol:
            vol[r] += float(W[M[r]].sum())
        if (s // cells_per) % 100 == 0:
            log(f"  {e}/{n} cells ({time.time() - tic:.0f}s)")
    return {k: float(v.sum()) for k, v in cell.items()}, reg, vol


def bounds(I_R, I_S, I_I, I_C, cf, N):
    """float64 diagnostic counterparts of F_dir, G_dir (split_certify) and G_AR (Cor. 4.12) on the integrals."""
    D = I_S + I_I + 2 * I_C
    L = cf * I_R
    F = math.sqrt(D + L)
    Gd = math.sqrt(L) + math.sqrt(D + L)
    a = math.sqrt(I_S) + math.sqrt(L)
    Ga = 0.5 * (a + math.sqrt(a * a + 2 * (I_I + 2 * I_C)))
    return {"F_dir/N": F / N, "G_dir/N": Gd / N, "G_AR/N": Ga / N, "F": F, "G_dir": Gd, "G_AR": Ga,
            "share L": L / (D + L), "share U_S": I_S / (D + L), "share 2U_c": 2 * I_C / (D + L),
            "share U_0": I_I / (D + L)}


def slices(problem, triple, n=1024, log=print):
    """Slice diagnostics t in SLICES for A0 and A2 on an n x n midpoint grid of (0,1)^2."""
    z = (np.arange(n) + 0.5) / n
    xx, yy = np.meshgrid(z, z, indexing="ij")
    dA = 1.0 / n**2
    rows = []
    sub = {k: CONFIGS[k] for k in ("A0", "A2")}
    for tv in SLICES:
        cols = [torch.full((n * n,), tv, dtype=torch.float64).requires_grad_(True),
                torch.as_tensor(xx.ravel()).clone().requires_grad_(True),
                torch.as_tensor(yy.ravel()).clone().requires_grad_(True)]
        F, M, X = point_fields(problem, triple, cols, sub)
        ac, au = M["contact"].sum() * dA, M["under"].sum() * dA
        base = {"t": tv, "grid": f"{n}x{n} midpoints", "area contact (exact)": ac, "area {v<g}": au,
                "area {v<g} / area contact": au / ac if ac > 0 else float("nan"),
                "area {v<g} minus contact": M["under_not_contact"].sum() * dA,
                "area contact minus {v<g}": M["contact_not_under"].sum() * dA,
                "max lambda_psi": float(X["mu_psi"].max()), "max lambda (exact)": float(X["lam"].max()),
                "max lambda_psi on contact": float(X["mu_psi"][M["contact"]].max()) if M["contact"].any() else 0.0}
        for cid in sub:
            row = {**base, "id": cid}
            for r in REGIONS:
                row[f"area {r}"] = M[r].sum() * dA
                for d, nm in (("Sh2", "|S_hat|^2"), ("Rh2", "R_hat^2")):
                    row[f"int {nm} on {r}"] = float(F[f"{cid}:{d}"][M[r]].sum() * dA)
            rows.append(row)
        log(f"  slice t={tv}: area(v<g)/area(contact) = {base['area {v<g} / area contact']:.4f}")
    return rows


def load_run(run: str):
    """Checkpoint, problem and frozen partitions (hash-checked) of a stored v17_split ellipse row."""
    d = REPO_ROOT / run
    cfg = json.loads((d / "config.json").read_text())
    row = pd.read_csv(d / "summary.csv", dtype=str).set_index("backend").loc["numpy"]
    r = load_stored_run(cfg["source_run"], workers=1)
    src = str(cfg["partition"].get("source", ""))
    # a frozen partition directory (results/v17_partitions/<ts> or, for Task C, results/v19_C_<arm>_v17_partitions/<ts>)
    pfile = cfg["partition"].get("file") or (f"{src}/partition.npz" if src and (REPO_ROOT / src / "partition.npz").exists()
                                             else None)
    if pfile:
        z = np.load(REPO_ROOT / pfile)
        r = dataclasses.replace(r, part=Bx.BoxPartition(z["lo"], z["hi"]), tp0=Bx.BoxPartition(z["lo0"], z["hi0"]),
                                tpT=Bx.BoxPartition(z["loT"], z["hiT"]))
    for nm, p in (("partition_hash", r.part), ("trace0_hash", r.tp0), ("traceT_hash", r.tpT)):
        if p.hash() != row[nm]:
            raise RuntimeError(f"{nm}: reconstructed {p.hash()[:12]} != stored {row[nm][:12]}")
    if r.checkpoint_sha256 != row["checkpoint_sha256"]:
        raise RuntimeError("checkpoint hash mismatch")
    if r.problem_name != "ellipse":
        raise RuntimeError("Task A is the ellipse study")
    return d, cfg, row, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--q", type=int, default=2)
    ap.add_argument("--slice-n", type=int, default=1024)
    a = ap.parse_args()
    run = a.run.rstrip("/")
    out = results_io.make_run_dir("v19_substitution")

    def log(m):
        line = time.strftime("%H:%M:%S ") + m
        print(line, flush=True)
        with open(out / "log.txt", "a") as fh:
            fh.write(line + "\n")

    tic = time.time()
    d, cfg, row, r = load_run(run)
    q_stored = int(float(row["diag_q"]))
    if a.q != q_stored:
        raise RuntimeError(f"q={a.q} but the stored slack used q={q_stored}")
    log(f"{run}: n={r.part.n}, checkpoint {r.checkpoint_sha256[:12]}, q={a.q}")
    cf = float(row["cF2_up"])
    N = float(row["reference_norm_lower"])
    lo, hi = r.part.lo, r.part.hi
    I, reg, vol = volume_integrals(r.problem, r.triple, lo, hi, q=a.q, log=log)
    l0, h0 = r.tp0.lo, r.tp0.hi
    I_I = float(CQ.trace_integrals("2d", r.problem, r.triple, l0, h0, 0.0, q=max(a.q, 3))["I0"].sum())
    t_vol = time.time() - tic
    log(f"volume + trace integrals in {t_vol:.0f}s")
    rows = []
    for cid, (fl, mm, ex) in CONFIGS.items():
        I_R, I_S, I_C = I[f"{cid}:Rh2"], I[f"{cid}:Sh2"], I[f"{cid}:C"]
        b = bounds(I_R, I_S, I_I, I_C, cf, N)
        x = {"id": cid, "run": run, "cells": r.part.n, "state": "v_hat (trained)", "flux": fl, "multiplier": mm,
             "uses exact solution": ex, "status": "diagnostic_float64",
             "label": "post_hoc" + (";oracle" if ex.startswith("oracle") else ""),
             "U_R (I)": I_R, "U_S (I)": I_S, "U_0 (I)": I_I, "U_c (I)": I_C, **b}
        for k, nm in (("Rh2", "U_R"), ("Sh2", "U_S")):
            for rg in (*REGIONS, "le"):
                x[f"{nm} on {rg if rg != 'le' else 'v<=g'}"] = reg[f"{cid}:{k}@{rg}"]
        x.update({f"volume {k}": v for k, v in vol.items()})
        rows.append(x)
    df = pd.DataFrame(rows)
    # gate: A0 = stored exact integrals (same code path; reported both ways)
    gate = {}
    a0 = df.set_index("id").loc["A0"]
    for k, sk in (("U_R (I)", "I_R"), ("U_S (I)", "I_S"), ("U_0 (I)", "I_I"), ("U_c (I)", "I_C")):
        st = float(row[sk])
        gate[sk] = {"stored": st, "new": float(a0[k]), "rel diff": abs(a0[k] - st) / abs(st), "bitwise": a0[k] == st}
    for k, sk in (("F", "diag_F_dir"), ("G_dir", "diag_G_dir")):
        st = float(row[sk])
        gate[sk] = {"stored": st, "new": float(a0[k]), "rel diff": abs(a0[k] - st) / abs(st), "bitwise": a0[k] == st}
    for k, gk in gate.items():
        log(f"A0 {k}: stored {gk['stored']!r} new {gk['new']!r} rel {gk['rel diff']:.2e} bitwise {gk['bitwise']}")
    df.to_csv(out / "substitution.csv", index=False)
    srows = slices(r.problem, r.triple, n=a.slice_n, log=log)
    sl = pd.DataFrame(srows)
    sl.insert(0, "run", run)
    sl["status"], sl["label"] = "diagnostic_float64", "post_hoc;oracle"  # contact set is the exact one
    sl.to_csv(out / "slices.csv", index=False)
    for _, x in df.iterrows():
        log(f"{x['id']}: U_R {x['U_R (I)']:.4e} U_S {x['U_S (I)']:.4e} U_c {x['U_c (I)']:.3e} "
            f"F/N {x['F_dir/N']:.4f} G_dir/N {x['G_dir/N']:.4f} G_AR/N {x['G_AR/N']:.4f}")
    cfgo = {"experiment": "v19_substitution", "run": run, "q": a.q, "slice_n": a.slice_n, "cF2_up": cf,
            "N_lower": N, "A0_gate": gate, "seconds": time.time() - tic, "git_commit": git_commit(),
            "git_dirty": git_dirty(), "hardware": hardware(), "doc": __doc__}
    (out / "config.json").write_text(json.dumps(cfgo, indent=1, default=str))
    log(f"wrote {out.relative_to(REPO_ROOT)} ({time.time() - tic:.0f}s)")


if __name__ == "__main__":
    main()
