"""v20 Task 1: finite-difference reference solution of rising1d (diagnostic_float64; label `reference`).

    python3 -m certified_obstacle_nn.experiments.v20.reference_rising1d

Scheme: implicit Euler in time, P1 finite elements with mass lumping in space (= the 3-point finite difference on the
uniform grid x_i = i h, h = 1/n_x, interior nodes i = 1..n_x-1, u_0 = u_{n_x} = 0). Each step k (t_k = k dt) solves the
discrete obstacle problem exactly by a primal-dual active-set (PDAS) iteration:
    (u - u_prev)/dt + K u - lam = f(t_k),   u >= g(t_k),  lam >= 0,  lam (u - g) = 0,     K = tridiag(-1, 2, -1)/h^2,
    active set  A = {i : lam_i + c (g_i - u_i) > 0}  (c = 1);  on A: u = g, lam from the equation; off A: lam = 0.
The iteration starts from the previous step's active set and stops when the active set repeats (exact solution of the
discrete complementarity system). Gates per step: min(u - g) >= -1e-12, min lam >= -1e-10, max |lam (u - g)| <= 1e-10.

Stored per grid (npz + one CSV row): u (n_t + 1, n_x + 1) at the time levels, the discrete multiplier lam (n_t + 1,
n_x + 1; lam = 0 at t = 0 and on the boundary nodes), the contact interval [x_-(t_k), x_+(t_k)] (smallest / largest
active node, NaN if none), t* (first time level with a non-empty active set), N_ref = ||d_x u_h||_{L2(Q)},
||u_h||_{L2(Q)} and max lam_h. Norms use the space-time interpolant u_h that is piecewise linear in x and in t
(the protocol's interpolation): per time interval, int_{t_{k-1}}^{t_k} |a + (b - a) s|^2 = dt (a^2 + a b + b^2)/3
applied per spatial element to the element slopes (d_x u_h) and, for ||u_h||, the exact P1 x P1 integral.
Writes results/v20_reference/<timestamp>/ (config.json, summary.csv, ref_<nx>.npz).
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import pandas as pd
from scipy.linalg import solve_banded

from certified_obstacle_nn.experiments import results_io
from certified_obstacle_nn.experiments.v14.common import REPO_ROOT, git_commit, git_dirty, hardware, sha256_json

EXPERIMENT = "v20_reference"
GRIDS = ((2048, 2048), (4096, 4096))
C_PDAS = 1.0
SANITY = {"t_star": 0.11, "x_minus_T": 0.33, "x_plus_T": 0.67, "N": 0.865, "max_lambda": 12.6}
T_CHECK = (0.25, 0.5, 0.75, 1.0)


def g_fn(t, x):
    return -0.6 + 0.6 * t - 2.0 * (x - 0.5) ** 2


def f_fn(x):
    return -8.0 * np.sin(np.pi * x)


def pdas_step(u_prev, g, f, dt, h, active0, max_iter=200):
    """One implicit-Euler obstacle step on the interior nodes. Returns (u, lam, active, iterations)."""
    n = u_prev.size
    a = 1.0 / dt + 2.0 / h**2
    off = -1.0 / h**2
    rhs0 = u_prev / dt + f
    active = active0.copy()
    seen = []
    for it in range(1, max_iter + 1):
        # u = g exactly on the active set; the inactive rows (lam = 0) form a tridiagonal system whose couplings to
        # active neighbours move to the right-hand side (no row mixing, so u - g = 0 exactly on the active set)
        u = np.where(active, g, 0.0)
        ina = np.nonzero(~active)[0]
        if ina.size:
            rhs = rhs0[ina].copy()
            left, right = ina - 1, ina + 1
            lm = (left >= 0) & active[np.clip(left, 0, n - 1)]
            rm = (right < n) & active[np.clip(right, 0, n - 1)]
            rhs[lm] -= off * g[left[lm]]
            rhs[rm] -= off * g[right[rm]]
            adj = np.diff(ina) == 1  # consecutive inactive nodes are coupled
            ab = np.zeros((3, ina.size))
            ab[1] = a
            ab[0, 1:] = np.where(adj, off, 0.0)
            ab[2, :-1] = np.where(adj, off, 0.0)
            u[ina] = solve_banded((1, 1), ab, rhs)
        Ku = a * u
        Ku[:-1] += off * u[1:]
        Ku[1:] += off * u[:-1]
        lam = np.where(active, Ku - rhs0, 0.0)
        new = lam + C_PDAS * (g - u) > 0
        if np.array_equal(new, active):
            return u, lam, active, it
        key = new.tobytes()
        if key in seen:
            raise RuntimeError("PDAS cycles")
        seen.append(active.tobytes())
        active = new
    raise RuntimeError("PDAS did not converge")


def solve(nx: int, nt: int, log=print) -> dict:
    h, dt = 1.0 / nx, 1.0 / nt
    x = np.arange(nx + 1) * h
    xi = x[1:-1]
    f = f_fn(xi)
    U = np.zeros((nt + 1, nx + 1))
    LAM = np.zeros((nt + 1, nx + 1))
    xm = np.full(nt + 1, np.nan)
    xp = np.full(nt + 1, np.nan)
    active = np.zeros(nx - 1, dtype=bool)
    its = np.zeros(nt + 1, dtype=int)
    worst = {"min_gap": math.inf, "min_lam": math.inf, "max_compl": 0.0}
    u = np.zeros(nx - 1)
    for k in range(1, nt + 1):
        g = g_fn(k * dt, xi)
        u, lam, active, it = pdas_step(u, g, f, dt, h, active)
        its[k] = it
        worst["min_gap"] = min(worst["min_gap"], float((u - g).min()))
        worst["min_lam"] = min(worst["min_lam"], float(lam.min()))
        worst["max_compl"] = max(worst["max_compl"], float(np.abs(lam * (u - g)).max()))
        U[k, 1:-1] = u
        LAM[k, 1:-1] = lam
        if active.any():
            idx = np.nonzero(active)[0]
            xm[k], xp[k] = xi[idx[0]], xi[idx[-1]]
    t = np.arange(nt + 1) * dt
    first = np.nonzero(~np.isnan(xm))[0]
    t_star = float(t[first[0]]) if first.size else math.nan
    # norms of the P1-in-x, P1-in-t interpolant
    s = np.diff(U, axis=1) / h  # (nt + 1, nx) element slopes
    a_, b_ = s[:-1], s[1:]
    N2 = float((dt * (a_**2 + a_ * b_ + b_**2) / 3.0).sum() * h)
    # ||u_h||^2: per element P1 x P1 exact integral = (dt h / 36) sum_{corners} M_t M_x weights; use the 1-D mass
    # matrices: int_e (P1)^2 = h (p^2 + p q + q^2)/3 per time level, then the same rule in time on the bilinear form
    def m_x(A, B):  # int over x of the product of two P1 functions with nodal values A, B (per time level)
        return (h / 6.0) * (2 * A[:, :-1] * B[:, :-1] + A[:, :-1] * B[:, 1:] + A[:, 1:] * B[:, :-1]
                            + 2 * A[:, 1:] * B[:, 1:]).sum(1)
    Ua, Ub = U[:-1], U[1:]
    L2sq = float((dt / 6.0 * (2 * m_x(Ua, Ua) + 2 * m_x(Ua, Ub) + 2 * m_x(Ub, Ub))).sum())
    out = {"nx": nx, "nt": nt, "h": h, "dt": dt, "x": x, "t": t, "u": U, "lam": LAM, "x_minus": xm, "x_plus": xp,
           "pdas_iterations": its, "t_star": t_star, "N_ref": math.sqrt(N2), "L2_u": math.sqrt(L2sq),
           "max_lambda": float(LAM.max()), **worst}
    log(f"nx={nx} nt={nt}: t*={t_star:.6f} N={out['N_ref']:.8f} ||u||={out['L2_u']:.6f} max lam={out['max_lambda']:.6f} "
        f"max PDAS it={its.max()} min gap={worst['min_gap']:.2e} min lam={worst['min_lam']:.2e} "
        f"compl={worst['max_compl']:.2e}")
    return out


def contact_at(sol, tq):
    k = int(round(tq / sol["dt"]))
    return sol["x_minus"][k], sol["x_plus"][k]


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()

    def log(msg):
        print(time.strftime("%H:%M:%S"), msg, flush=True)

    d = results_io.make_run_dir(EXPERIMENT)
    sols, rows = {}, []
    for nx, nt in GRIDS:
        tic = time.time()
        s = solve(nx, nt, log)
        s["seconds"] = time.time() - tic
        sols[nx] = s
        np.savez_compressed(d / f"ref_{nx}.npz", **{k: v for k, v in s.items() if isinstance(v, np.ndarray)},
                            scalars=np.array([s["t_star"], s["N_ref"], s["L2_u"], s["max_lambda"]]))
        row = {"label": "reference", "status": "diagnostic_float64", "nx": nx, "nt": nt, "t_star": s["t_star"],
               "N_ref": s["N_ref"], "L2_u": s["L2_u"], "max_lambda": s["max_lambda"],
               "min_u_minus_g": s["min_gap"], "min_lambda": s["min_lam"], "max_abs_lambda_gap": s["max_compl"],
               "max_pdas_iterations": int(s["pdas_iterations"].max()), "seconds": s["seconds"]}
        for tq in T_CHECK + (0.15,):
            row[f"x_minus_t{tq}"], row[f"x_plus_t{tq}"] = contact_at(s, tq)
        rows.append(row)
    c, f = sols[GRIDS[0][0]], sols[GRIDS[1][0]]
    gates = {}
    gates["pdas"] = all(r["min_u_minus_g"] >= -1e-12 and r["min_lambda"] >= -1e-10 and r["max_abs_lambda_gap"] <= 1e-10
                        for r in rows)
    gates["N_rel_diff"] = abs(f["N_ref"] - c["N_ref"]) / f["N_ref"]
    gates["N_agree_1e-3"] = gates["N_rel_diff"] <= 1e-3
    ep = max(max(abs(a - b) for a, b in zip(contact_at(c, tq), contact_at(f, tq))) for tq in T_CHECK)
    gates["contact_max_diff"] = ep
    gates["contact_agree_2/2048"] = ep <= 2 / 2048
    # sanity values to 2 digits (rounded to 2 significant digits)
    xm1, xp1 = contact_at(f, 1.0)
    vals = {"t_star": f["t_star"], "x_minus_T": xm1, "x_plus_T": xp1, "N": f["N_ref"], "max_lambda": f["max_lambda"]}
    gates["sanity_values"] = vals
    gates["sanity_2digits"] = {k: float(f"{v:.2g}") == float(f"{SANITY[k]:.2g}") for k, v in vals.items()}
    # lambda_h vs psi on the contact set away from the free boundary (>= 8 h from both contact endpoints)
    worst = 0.0
    x = f["x"]
    for k in range(1, f["nt"] + 1):
        if np.isnan(f["x_minus"][k]):
            continue
        m = (x >= f["x_minus"][k] + 8 * f["h"]) & (x <= f["x_plus"][k] - 8 * f["h"])
        if m.any():
            worst = max(worst, float(np.abs(f["lam"][k, m] - (4.6 + 8 * np.sin(np.pi * x[m]))).max()))
    gates["lambda_vs_psi_interior_max"] = worst
    gates["lambda_vs_psi_1e-2"] = worst <= 1e-2
    # DECISIONS D48: the same check with the margin also in time (time levels >= 8 dt after the first contact level):
    # at the nucleation level t* every contact node lies on the space-time free boundary
    k_star = int(np.nonzero(~np.isnan(f["x_minus"]))[0][0])
    worst_st = 0.0
    for k in range(k_star + 8, f["nt"] + 1):
        m = (x >= f["x_minus"][k] + 8 * f["h"]) & (x <= f["x_plus"][k] - 8 * f["h"])
        if m.any():
            worst_st = max(worst_st, float(np.abs(f["lam"][k, m] - (4.6 + 8 * np.sin(np.pi * x[m]))).max()))
    gates["lambda_vs_psi_spacetime_max"] = worst_st
    gates["lambda_vs_psi_spacetime_1e-2"] = worst_st <= 1e-2
    gates = {k: (bool(v) if isinstance(v, (bool, np.bool_)) else v) for k, v in gates.items()}
    gates["sanity_2digits"] = {k: bool(v) for k, v in gates["sanity_2digits"].items()}
    log(f"gates: {gates}")
    pd.DataFrame(rows).to_csv(d / "summary.csv", index=False)
    config = {"experiment": EXPERIMENT, "grids": GRIDS, "c_pdas": C_PDAS, "scheme": __doc__, "gates": gates,
              "sanity_prompt": SANITY, "git_commit": git_commit(), "git_dirty": git_dirty(), "hardware": hardware()}
    config["config_hash"] = sha256_json({k: v for k, v in config.items() if k != "hardware"})
    results_io.write_config(d, config)
    log(f"wrote {d.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
