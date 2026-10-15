"""v20 Task 4 diagnostics against the finite-difference reference (all diagnostic_float64; label `reference`).

    python3 -m certified_obstacle_nn.experiments.v20.diagnostics_rising1d --ref results/v20_reference/<ts> \
        --cert results/v20_rising1d_cert/<ts> [...]

Reference: the 4096 grid (fine) of experiments/v20/reference_rising1d.py; the 2048 grid (coarse) gives the
uncertainty |fine - coarse| of every reference quantity. u_h is the space-time interpolant that is piecewise linear in
x and in t (protocol), sigma_h = -d_x u_h, lambda_h the same interpolant of the nodal multiplier.

Per certification run (both budgets; the network quantities are evaluated on that run's partition):
  error norms with q = 3 Gauss points per axis per cell: ||sigma_h - tau||, ||d_x(u_h - v_hat)||, ||u_h - v_hat||_{L2(Q)},
  terminal ||(u_h - v_hat)(T)|| (4096 trace cells, q = 3), E(v_hat) = sqrt(flux^2 + grad^2 + terminal^2), each also
  / N_ref; effectivities F_dir/||sigma_h - tau||, G_AR/||d_x(u_h - v_hat)||; N_lo/N_ref.
  Post-hoc multiplier (v19 Task A2 definition, q = 2 as v19, t = 0 trace q = 3): exact-integral U_R, U_S, U_0, U_c and
  G_AR/N_ref for the trained lambda_psi and for mu* (mu* = max(-R_hat_0, 0) on {v <= g}, 0 on {v > g}, where
  R_hat_0 is R_hat with mu = 0; then R_hat(mu*) = R_hat_0 on {v > g}, max(R_hat_0, 0) on {v <= g}, and U_c = 0).
Per seed (from the 1024^2 run): contact table at t in {0.15, 0.25, 0.5, 0.75, 1} (reference interval; the set
  {v <= g} on 65537 x-nodes: min, max, measure, number of components; length ratio; max lambda_psi vs max lambda_h at
  that t), first t with {v <= g} non-empty (4096 time levels x 65537 x-nodes) vs t*.
Figure data (seed 0): 512 x 512 midpoint grid of (t, x).
Writes results/v20_rising1d_diag/<timestamp>/ (config.json, diagnostics.csv, contact.csv, posthoc_mu.csv,
rising1d_figure.npz, rising1d_figure_summary.csv).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from certified_obstacle_nn.certification import reliability as rel
from certified_obstacle_nn.certification.enclose import upper_sum as US
from certified_obstacle_nn.experiments import results_io
from certified_obstacle_nn.experiments.v14.common import REPO_ROOT, git_commit, git_dirty, hardware, sha256_json
from certified_obstacle_nn.experiments.v17 import cell_quadrature as CQ
from certified_obstacle_nn.experiments.v20.certify_rising1d import load_frozen, trace0_integral

EXPERIMENT = "v20_rising1d_diag"
T_CONTACT = (0.15, 0.25, 0.5, 0.75, 1.0)
NX_FINE = 65536


class Ref:
    """Space-time P1 x P1 interpolant of a stored reference grid."""

    def __init__(self, npz):
        z = np.load(npz)
        self.U, self.LAM = z["u"], z["lam"]
        self.x, self.t = z["x"], z["t"]
        self.xm, self.xp = z["x_minus"], z["x_plus"]
        self.nt, self.nx = self.U.shape[0] - 1, self.U.shape[1] - 1
        sc = z["scalars"]
        self.t_star, self.N, self.L2, self.max_lambda = (float(v) for v in sc)

    def _loc(self, t, x):
        st = np.clip(t * self.nt, 0, self.nt)
        k = np.minimum(np.floor(st).astype(np.int64), self.nt - 1)
        s = st - k
        sx = np.clip(x * self.nx, 0, self.nx)
        i = np.minimum(np.floor(sx).astype(np.int64), self.nx - 1)
        r = sx - i
        return k, s, i, r

    def _bilin(self, A, t, x):
        k, s, i, r = self._loc(t, x)
        a0 = (1 - r) * A[k, i] + r * A[k, i + 1]
        a1 = (1 - r) * A[k + 1, i] + r * A[k + 1, i + 1]
        return (1 - s) * a0 + s * a1

    def u(self, t, x):
        return self._bilin(self.U, t, x)

    def lam(self, t, x):
        return self._bilin(self.LAM, t, x)

    def ux(self, t, x):
        k, s, i, r = self._loc(t, x)
        h = 1.0 / self.nx
        return (1 - s) * (self.U[k, i + 1] - self.U[k, i]) / h + s * (self.U[k + 1, i + 1] - self.U[k + 1, i]) / h

    def contact(self, tq):
        k = int(round(tq * self.nt))
        return float(self.xm[k]), float(self.xp[k]), float(self.LAM[k].max())


def net_points(problem, triple, t, x):
    """Point values of the network fields needed here (float64)."""
    tt = torch.as_tensor(t).clone().requires_grad_(True)
    xx = torch.as_tensor(x).clone().requires_grad_(True)
    v, tau, mu = (a.reshape(-1) for a in triple(tt.reshape(-1, 1), xx.reshape(-1, 1)))
    divt = CQ._grad(tau, (xx,))[0]
    dv = CQ._grad(v, (tt, xx))
    g = problem.g(tt, xx)
    dg = CQ._grad(g, (tt, xx))
    f = problem.f(tt, xx)
    with torch.no_grad():
        pos = v > g
        out = {"v": v, "g": g, "tau": tau, "mu": mu, "pos": pos, "vh": torch.maximum(v, g),
               "vh_t": torch.where(pos, dv[0], dg[0]), "vh_x": torch.where(pos, dv[1], dg[1]), "divt": divt, "f": f}
    return {k: w.detach().numpy() for k, w in out.items()}


def error_integrals(problem, triple, refs, lo, hi, q, chunk=300_000):
    """Integrals over the partition (q^2 Gauss per cell) for each reference grid: flux, grad, L2 errors."""
    n = lo.shape[1]
    per = max(1, chunk // q**2)
    acc = {nm: {"flux": 0.0, "grad": 0.0, "L2": 0.0} for nm in refs}
    for s in range(0, n, per):
        e = min(n, s + per)
        P, W = CQ._pts(lo[:, s:e], hi[:, s:e], q)
        F = net_points(problem, triple, P[0], P[1])
        for nm, R in refs.items():
            ux, u = R.ux(P[0], P[1]), R.u(P[0], P[1])
            acc[nm]["flux"] += float((W * (-ux - F["tau"]) ** 2).sum())
            acc[nm]["grad"] += float((W * (ux - F["vh_x"]) ** 2).sum())
            acc[nm]["L2"] += float((W * (u - F["vh"]) ** 2).sum())
    return acc


def terminal_integral(problem, triple, R, q=3, n=4096):
    tp = US.uniform_1d(n)
    P, W = CQ._pts(tp.x0[None, :], tp.x1[None, :], q)
    with torch.no_grad():
        x = torch.as_tensor(P[0])
        t = torch.ones_like(x)
        v = triple(t.reshape(-1, 1), x.reshape(-1, 1))[0].reshape(-1)
        vh = torch.maximum(v, problem.g(t, x)).numpy()
    return float((W * (R.u(np.ones_like(P[0]), P[0]) - vh) ** 2).sum())


def posthoc_mu(problem, triple, lo, hi, q=2, chunk=300_000):
    """Exact-integral (float64 Gauss) U_R, U_S, U_c for the trained multiplier and for mu* (v19 A2)."""
    n = lo.shape[1]
    per = max(1, chunk // q**2)
    acc = {"U_R_trained": 0.0, "U_S": 0.0, "U_C_trained": 0.0, "U_R_mustar": 0.0}
    for s in range(0, n, per):
        e = min(n, s + per)
        P, W = CQ._pts(lo[:, s:e], hi[:, s:e], q)
        F = net_points(problem, triple, P[0], P[1])
        Rh0 = F["f"] - F["vh_t"] - F["divt"]
        Rh = Rh0 + F["mu"]
        Rs = np.where(F["pos"], Rh0, np.maximum(Rh0, 0.0))
        acc["U_R_trained"] += float((W * Rh**2).sum())
        acc["U_R_mustar"] += float((W * Rs**2).sum())
        acc["U_S"] += float((W * (F["tau"] + F["vh_x"]) ** 2).sum())
        acc["U_C_trained"] += float((W * F["mu"] * np.maximum(F["v"] - F["g"], 0.0)).sum())
    return acc


def g_ar_f64(U_R, U_S, U_0, U_C, cf):
    a = math.sqrt(U_S) + math.sqrt(cf * U_R)
    return (a + math.sqrt(a * a + 2 * (U_0 + 2 * U_C))) / 2


def contact_sets(problem, triple, tq, nxf=NX_FINE):
    x = np.linspace(0.0, 1.0, nxf + 1)
    with torch.no_grad():
        xt = torch.as_tensor(x)
        t = torch.full_like(xt, tq)
        v, _, mu = (a.reshape(-1).numpy() for a in triple(t.reshape(-1, 1), xt.reshape(-1, 1)))
        g = problem.g(t, xt).numpy()
    m = v <= g
    comps = int(np.sum(np.diff(m.astype(int)) == 1) + (1 if m[0] else 0))
    if m.any():
        return float(x[m].min()), float(x[m].max()), float(m.sum() / nxf), comps, float(mu.max())
    return math.nan, math.nan, 0.0, 0, float(mu.max())


def first_contact_time(problem, triple, nt=4096, nxf=NX_FINE):
    x = torch.as_tensor(np.linspace(0.0, 1.0, nxf + 1))
    lo, hi = 0, nt
    ts = np.arange(nt + 1) / nt
    # scan all levels in blocks (no monotonicity assumed)
    for k0 in range(0, nt + 1, 64):
        ks = range(k0, min(nt + 1, k0 + 64))
        with torch.no_grad():
            T = torch.as_tensor(np.repeat(ts[list(ks)], nxf + 1))
            X = x.repeat(len(ks))
            v = triple.state_net(T.reshape(-1, 1), X.reshape(-1, 1)).reshape(len(ks), -1).numpy()
            g = problem.g(T, X).numpy().reshape(len(ks), -1)
        hit = (v <= g).any(1)
        if hit.any():
            return float(ts[list(ks)[int(np.argmax(hit))]])
    del lo, hi
    return math.nan


def figure_data(problem, triple, R, n=512):
    tc = (np.arange(n) + 0.5) / n
    xc = (np.arange(n) + 0.5) / n
    TT, XX = np.meshgrid(tc, xc, indexing="ij")
    t, x = TT.ravel(), XX.ravel()
    F = net_points(problem, triple, t, x)
    Sh2 = (F["tau"] + F["vh_x"]) ** 2
    Rh2 = (F["f"] - F["vh_t"] - F["divt"] + F["mu"]) ** 2
    out = {"t": tc, "x": xc, "u_h": R.u(t, x).reshape(n, n), "v": F["v"].reshape(n, n), "g": F["g"].reshape(n, n),
           "v_hat": F["vh"].reshape(n, n), "dx_err": (R.ux(t, x) - F["vh_x"]).reshape(n, n),
           "Shat2": Sh2.reshape(n, n), "Rhat2": Rh2.reshape(n, n), "lambda_h": R.lam(t, x).reshape(n, n),
           "lambda_psi": F["mu"].reshape(n, n)}
    # contact boundaries per t midpoint: reference (nearest time level) and network {v <= g} on 65537 x nodes
    rk = np.rint(tc * R.nt).astype(int)
    out["ref_x_minus"], out["ref_x_plus"] = R.xm[rk], R.xp[rk]
    nm, npl = np.full(n, np.nan), np.full(n, np.nan)
    for j, tq in enumerate(tc):
        a, b, _, _, _ = contact_sets(problem, triple, float(tq), nxf=4096)
        nm[j], npl[j] = a, b
    out["net_x_minus"], out["net_x_plus"] = nm, npl
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cert", nargs="+", required=True)
    a = ap.parse_args()
    torch.set_num_threads(16)

    def log(msg):
        print(time.strftime("%H:%M:%S"), msg, flush=True)

    ref_dir = REPO_ROOT / a.ref
    refs = {"fine": Ref(ref_dir / "ref_4096.npz"), "coarse": Ref(ref_dir / "ref_2048.npz")}
    Rf, Rc = refs["fine"], refs["coarse"]
    N_ref, dN = Rf.N, abs(Rf.N - Rc.N)
    cf = rel.cF2_upper(1)["cF2_up"]
    d = results_io.make_run_dir(EXPERIMENT)
    diag_rows, contact_rows, mu_rows = [], [], []
    by_seed = {}
    for cd in a.cert:
        cdir = REPO_ROOT / cd
        row = pd.read_csv(cdir / "summary.csv", dtype=str).iloc[0]
        cfg = json.loads((cdir / "config.json").read_text())
        train_run = REPO_ROOT / cfg["train_run"]
        problem, triple = load_frozen(train_run)
        n = int(row["n"])
        part = US.uniform_2d(n, n)
        if part.hash() != row["partition_hash"]:
            raise RuntimeError("partition hash mismatch")
        lo, hi = np.stack([part.t0, part.x0]), np.stack([part.t1, part.x1])
        tic = time.time()
        acc = error_integrals(problem, triple, refs, lo, hi, q=3)
        term = {nm: terminal_integral(problem, triple, R) for nm, R in refs.items()}
        out = {"run_dir": cd, "seed": int(row["seed"]), "cells": int(row["cells"]), "status": "diagnostic_float64",
               "label": "reference", "N_ref": N_ref, "N_ref_unc": dN}
        for nm in ("flux", "grad", "L2"):
            vf, vc = math.sqrt(acc["fine"][nm]), math.sqrt(acc["coarse"][nm])
            out[f"err_{nm}"], out[f"err_{nm}_unc"] = vf, abs(vf - vc)
            out[f"err_{nm}_rel"] = vf / N_ref
        tf, tc_ = math.sqrt(term["fine"]), math.sqrt(term["coarse"])
        out["err_terminal"], out["err_terminal_unc"] = tf, abs(tf - tc_)
        Ef = math.sqrt(acc["fine"]["flux"] + acc["fine"]["grad"] + term["fine"])
        Ec = math.sqrt(acc["coarse"]["flux"] + acc["coarse"]["grad"] + term["coarse"])
        out.update({"E_vhat": Ef, "E_vhat_unc": abs(Ef - Ec), "E_vhat_rel": Ef / N_ref})
        F_dir, G_AR, N_lo = float(row["F_dir"]), float(row["G_AR"]), float(row["N_lo"])
        out.update({"F_dir": F_dir, "G_AR": G_AR, "E_dir": float(row["E_dir"]), "N_lo": N_lo,
                    "eff_F_dir": F_dir / out["err_flux"], "eff_G_AR": G_AR / out["err_grad"],
                    "eff_E_dir": float(row["E_dir"]) / Ef, "N_lo_over_N_ref": N_lo / N_ref,
                    "N_lo_over_N_ref_unc": abs(N_lo / N_ref - N_lo / Rc.N)})
        # effectivity uncertainties from the coarse grid
        out["eff_F_dir_unc"] = abs(out["eff_F_dir"] - F_dir / math.sqrt(acc["coarse"]["flux"]))
        out["eff_G_AR_unc"] = abs(out["eff_G_AR"] - G_AR / math.sqrt(acc["coarse"]["grad"]))
        out["seconds"] = time.time() - tic
        diag_rows.append(out)
        log(f"{cd} n={n}: flux {out['err_flux']:.5f} grad {out['err_grad']:.5f} L2 {out['err_L2']:.5f} "
            f"E {Ef:.5f} (rel {Ef / N_ref:.4f}) eff F {out['eff_F_dir']:.3f} G_AR {out['eff_G_AR']:.3f} "
            f"N_lo/N_ref {out['N_lo_over_N_ref']:.4f}")
        if n == 1024:
            pm = posthoc_mu(problem, triple, lo, hi, q=2)
            U0 = trace0_integral(problem, triple, US.uniform_1d(4096).x0, US.uniform_1d(4096).x1)
            for mult, UR, UC in (("trained lambda_psi", pm["U_R_trained"], pm["U_C_trained"]),
                                 ("mu* (post hoc)", pm["U_R_mustar"], 0.0)):
                G = g_ar_f64(UR, pm["U_S"], U0, UC, cf)
                mu_rows.append({"run_dir": cd, "seed": int(row["seed"]), "cells": int(row["cells"]),
                                "multiplier": mult, "U_R": UR, "U_S": pm["U_S"], "U_0": U0, "U_c": UC, "G_AR": G,
                                "G_AR_over_N_ref": G / N_ref, "G_AR_over_N_ref_unc": abs(G / N_ref - G / Rc.N),
                                "quadrature": "q = 2 Gauss per axis per cell (v19 A2), t = 0 trace q = 3",
                                "status": "diagnostic_float64", "label": "reference (N_ref only)"})
            log(f"  post-hoc mu: G_AR/N_ref trained {mu_rows[-2]['G_AR_over_N_ref']:.4f} "
                f"mu* {mu_rows[-1]['G_AR_over_N_ref']:.4f}")
            seed = int(row["seed"])
            by_seed[seed] = (problem, triple)
            tfirst = first_contact_time(problem, triple)
            for tq in T_CONTACT:
                rxm, rxp, rlam = Rf.contact(tq)
                cxm, cxp, _ = Rc.contact(tq)
                nxm, nxp, meas, comps, mumax = contact_sets(problem, triple, tq)
                contact_rows.append({
                    "seed": seed, "t": tq, "ref_x_minus": rxm, "ref_x_plus": rxp,
                    "ref_unc": max(abs(rxm - cxm), abs(rxp - cxp)), "ref_length": rxp - rxm,
                    "net_x_minus": nxm, "net_x_plus": nxp, "net_hull_length": nxp - nxm, "net_measure": meas,
                    "net_components": comps, "length_ratio_hull": (nxp - nxm) / (rxp - rxm),
                    "length_ratio_measure": meas / (rxp - rxm), "max_lambda_psi": mumax, "max_lambda_h": rlam,
                    "net_first_contact_t": tfirst, "ref_t_star": Rf.t_star, "ref_t_star_unc": abs(Rf.t_star - Rc.t_star),
                    "status": "diagnostic_float64", "label": "reference"})
            log(f"  contact: first t {tfirst:.5f} vs t* {Rf.t_star:.5f}")
    pd.DataFrame(diag_rows).to_csv(d / "diagnostics.csv", index=False)
    pd.DataFrame(contact_rows).to_csv(d / "contact.csv", index=False)
    pd.DataFrame(mu_rows).to_csv(d / "posthoc_mu.csv", index=False)
    if 0 in by_seed:
        fig = figure_data(*by_seed[0], Rf)
        np.savez_compressed(d / "rising1d_figure.npz", **fig)
        summ = [{"field": k, "shape": "x".join(map(str, np.shape(v))), "min": float(np.nanmin(v)),
                 "max": float(np.nanmax(v)), "mean": float(np.nanmean(v))} for k, v in fig.items()]
        pd.DataFrame(summ).to_csv(d / "rising1d_figure_summary.csv", index=False)
    config = {"experiment": EXPERIMENT, "ref": a.ref, "cert": a.cert, "doc": __doc__, "N_ref": N_ref, "N_ref_unc": dN,
              "git_commit": git_commit(), "git_dirty": git_dirty(), "hardware": hardware()}
    config["config_hash"] = sha256_json({k: v for k, v in config.items() if k != "hardware"})
    results_io.write_config(d, config)
    log(f"wrote {d.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
