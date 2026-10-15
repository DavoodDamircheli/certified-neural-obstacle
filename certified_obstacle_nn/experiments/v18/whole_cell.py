"""v18 Task 1c (referee item B1): independent whole-cell checks of the stored validated per-cell upper endpoints.

    python3 -m certified_obstacle_nn.experiments.v18.whole_cell --runs all --workers 20

Shares NO code with the TM1 engine (certification/enclose/tm.py, engine*.py, intervals_*.py) on the checking side:
networks, fields and problem data are re-implemented here, in arb (python-flint) and, for screening only, in NumPy
float64. The engine is called only to RECOMPUTE the numpy upper endpoints of the chosen cells, which must equal the
stored cells.npz values bit for bit (a reproduction check of the patched engine, DECISIONS D34/D35).

Per run (protocol-v2 8.0e6 partitions: disk long seeds 0-2, ellipse seed 0; Stefan 512^2 seeds 0-2) the cell sets
are (seeded, DECISIONS D37): random (1000), top500_R / top500_S / top500_C (largest |K| x stored upper of the term),
trace0 / traceT (500 each: 250 largest |S| x upper, 250 random others), adversarial (200: 67 largest midpoint
pre-activation magnitude over all hidden neurons of the three networks, 67 largest |W|-propagated radius, 66 random
cells cut by the exact free boundary).

Terms (volume): Rh = sup R_hat^2, Sh = sup |S_hat|^2 (v_hat = max(v, g): branch by the sign of gamma = g - v),
C = sup mu (v - g)_+;  (traces) r0hat = sup (v_hat(0) - u0)^2, w2_0 = sup ((g - v)(0)_+)^2, w2_T at t = T.

(i)  dense sampling (necessary condition): 33^D points per cell (D = cell dimension: 3 for 2-D space, 2 for Stefan,
     D - 1 on traces) = unscrambled Sobol plus the 2^D corners. Every point is screened in float64; the TOP_K largest
     float values of each term and every corner are re-evaluated in 192-bit arb (exact data rationals). A violation is
     arb_lower(sample) > numpy_upper. Reported: max / median of arb sample / numpy upper (DECISIONS D37).
(ii) independent rigorous enclosure: naive arb ball extension (no Taylor model, 128 bits) of the same fields on the
     8^D subcells of a depth-3 bisection of the cell; the cell bound is the max over subcells (hull). Reported:
     numpy_upper / arb_bisect_upper.
Writes results/v18_whole_cell/<timestamp>/ (config.json, summary.csv = the table, cells.csv.gz per cell and term).
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from flint import arb, arb_mat, ctx  # noqa: E402

from certified_obstacle_nn.experiments.v14.common import REPO_ROOT, git_commit, git_dirty, hardware, sha256_file  # noqa: E402
# imported for its side effect as in every production entry point: global arb precision >= 128 bits, which the
# numpy backend's arb-evaluated exp depends on (DECISIONS D36)
from certified_obstacle_nn.certification import reliability  # noqa: E402,F401

SEED = 20261013
TOP_K = 4
N_PTS = 33
BISECT = 3
PREC_POINT = 192
PREC_BISECT = 128
VOL_TERMS = ("Rh", "Sh", "C")
TR_TERMS = {"trace0": ("r0hat", "w2_0"), "traceT": ("w2_T",)}


# ============================================================================ runs
def run_specs() -> dict:
    reg = json.loads((REPO_ROOT / "docs/v17b_run_registry.json").read_text())
    specs = {}
    for s in ("0", "1", "2"):
        specs[f"disk_long_s{s}_v2_8e6"] = {"kind": "disk", "split": reg["v2"][s]["8.0e6"],
                                           "partition": reg["v2_partitions"][s]["8.0e6"]}
    specs["ellipse_s0_v2_8e6"] = {"kind": "ellipse", "split": reg["ellipse_v2"]["0"]["8.0e6"],
                                  "partition": reg["ellipse_v2_partitions"]["0"]["8.0e6"]}
    st = pd.read_csv(REPO_ROOT / "handoff_v17/tables/headline_stefan.csv")
    for i, d in enumerate(st.run_dir):
        specs[f"stefan_s{i}_512"] = {"kind": "stefan", "split": d, "partition": "uniform 512^2 (reconstructed)"}
    return specs


def _layers(mlp) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(L.weight.detach().double().numpy().copy(), L.bias.detach().double().numpy().copy()) for L in mlp.layers]


def load_run(spec: dict) -> dict:
    """Frozen networks, partitions (hash-checked against the split summary) and stored per-cell upper endpoints."""
    split = REPO_ROOT / spec["split"]
    cfg = json.loads((split / "config.json").read_text())
    row = pd.read_csv(split / "summary.csv", dtype=str).set_index("backend").loc["numpy"]
    sha = row["checkpoint_sha256"]
    src = REPO_ROOT / cfg["source_run"]
    scfg = json.loads((src / "config.json").read_text())
    torch.set_default_dtype(torch.float64)
    out = {"split": spec["split"], "kind": spec["kind"], "checkpoint_sha256": sha, "row": row}
    if spec["kind"] == "stefan":
        from certified_obstacle_nn.experiments.v17.common import load_stored_run

        r = load_stored_run(src)
        if r.checkpoint_sha256 != sha:
            raise RuntimeError("Stefan checkpoint mismatch")
        tr = r.triple
        out["nets"] = {"state": _layers(tr.state_net.net), "flux": _layers(tr.flux_net.net),
                       "mult": _layers(tr.multiplier_net.net)}
        p = r.part
        out["lo"], out["hi"] = np.stack([p.t0, p.x0]), np.stack([p.t1, p.x1])
        out["tlo0"], out["thi0"] = r.tp0.x0[None, :], r.tp0.x1[None, :]
        out["tloT"], out["thiT"] = r.tpT.x0[None, :], r.tpT.x1[None, :]
        hashes = (r.part.hash(), r.tp0.hash(), r.tpT.hash())
        out["encloser"], out["problem"] = r.encloser, r.problem
        out["checkpoint"] = str(r.checkpoint.relative_to(REPO_ROOT))
    else:
        from certified_obstacle_nn.certification.enclose import boxes as Bx
        from certified_obstacle_nn.certification.enclose.engine2d import TripleEncloser2D

        if spec["kind"] == "disk":
            from certified_obstacle_nn.certification.enclose.engine2d import DiskAdapter
            from certified_obstacle_nn.experiments.expanding_disk_2d.final_common import load_run as load_disk
            from certified_obstacle_nn.problems.disk import DiskProblem

            ck = REPO_ROOT / scfg["checkpoint"]
            triple, tcfg = load_disk(ck.parent)
            out["problem"] = DiskProblem(tcfg.kappa)
            ad = DiskAdapter(tcfg.kappa, tcfg.Lambda, tcfg.alpha)
        else:
            from certified_obstacle_nn.certification.enclose.engine2d import EllipseAdapter
            from certified_obstacle_nn.experiments.expanding_disk_2d.model import Triple2D
            from certified_obstacle_nn.experiments.v17.common import _find_weights
            from certified_obstacle_nn.problems.ellipse import EllipseProblem

            ck = _find_weights(src, sha)
            arch = scfg["training_config"]["architecture"]
            triple = Triple2D(arch["depth"], arch["width"], arch["multiplier_head"]).double()
            triple.load_state_dict(torch.load(ck))
            k0 = tuple(scfg["kappa0"])
            if k0 != (1, 100):
                raise RuntimeError("ellipse data below are written for kappa0 = 1/100")
            out["problem"] = EllipseProblem(kappa0=k0[0] / k0[1])
            ad = EllipseAdapter(*k0)
        if sha256_file(ck) != sha:
            raise RuntimeError(f"checkpoint hash mismatch {ck}")
        triple.eval()
        for q in triple.parameters():
            q.requires_grad_(False)
        out["nets"] = {"state": _layers(triple.state), "flux": _layers(triple.flux), "mult": _layers(triple.mult)}
        z = np.load(REPO_ROOT / spec["partition"] / "partition.npz")
        out["lo"], out["hi"] = z["lo"], z["hi"]
        out["tlo0"], out["thi0"], out["tloT"], out["thiT"] = z["lo0"], z["hi0"], z["loT"], z["hiT"]
        hashes = (Bx.BoxPartition(z["lo"], z["hi"]).hash(), Bx.BoxPartition(z["lo0"], z["hi0"]).hash(),
                  Bx.BoxPartition(z["loT"], z["hiT"]).hash())
        out["encloser"] = TripleEncloser2D(triple, ad)
        out["checkpoint"] = str(Path(ck).relative_to(REPO_ROOT))
    for h, key in zip(hashes, ("partition_hash", "trace0_hash", "traceT_hash")):
        if h != row[key]:
            raise RuntimeError(f"{spec['split']}: {key} {h[:12]} != stored {row[key][:12]}")
    cz = np.load(split / "cells.npz")
    out["stored"] = {"Rh": cz["numpy_Rh_term"], "Sh": cz["numpy_Sh_term"], "C": cz["numpy_c"]}
    out["kappa_mult"] = getattr(out["problem"], "kappa_mult", None)
    return out


# ============================================================================ float64 (screening / selection only)
def mlp_f64(layers, X, chans):
    """Value (n, m) and derivatives {ch: (n, m)} of a tanh MLP at float64 points X (n, d). Diagnostic only."""
    a = X
    J = {ch: np.broadcast_to(np.eye(X.shape[1])[ch], X.shape) for ch in chans}
    for W, b in layers[:-1]:
        z = a @ W.T + b
        a = np.tanh(z)
        s = 1.0 - a * a
        J = {ch: (J[ch] @ W.T) * s for ch in chans}
    W, b = layers[-1]
    return a @ W.T + b, {ch: J[ch] @ W.T for ch in chans}


def preact_max(layers, X):
    a, m = X, np.zeros(X.shape[0])
    for W, b in layers[:-1]:
        z = a @ W.T + b
        m = np.maximum(m, np.abs(z).max(1))
        a = np.tanh(z)
    return m


def _t(a):
    return torch.as_tensor(np.ascontiguousarray(a))


def fields_f64(run, P):
    """Point values of every term at float64 points P (n, D): dict term -> (n,). Volume points."""
    kind, nets, prob = run["kind"], run["nets"], run["problem"]
    if kind == "stefan":
        t, x = P[:, 0], P[:, 1]
        N, dN = mlp_f64(nets["state"], P, (0, 1))
        sg, dsg = mlp_f64(nets["flux"], P, (1,))
        M, _ = mlp_f64(nets["mult"], P, ())
        N, Nt, Nx, sg, sx, M = N[:, 0], dN[0][:, 0], dN[1][:, 0], sg[:, 0], dsg[1][:, 0], M[:, 0]
        B, dB = 4 * x * (1 - x), 4 - 8 * x
        e = np.exp(t)
        h = e - t - 1
        f, g, gt, gx = -1 - (e - 1) * (1 - x), -h * (1 - x), -(e - 1) * (1 - x), h
        u, ut, ux = B * N, B * Nt, dB * N + B * Nx
        lam = M * M
        gam = g - u
        R, Rp = f - ut - sx + lam, f - gt - sx + lam
        S2, Sp2 = (sg + ux) ** 2, (sg + gx) ** 2
    else:
        t, x, y = (_t(P[:, i]) for i in range(3))
        N, dN = mlp_f64(nets["state"], P, (0, 1, 2))
        sg, dsg = mlp_f64(nets["flux"], P, (1, 2))
        M, _ = mlp_f64(nets["mult"], P, ())
        N, Nt, Nx, Ny = N[:, 0], dN[0][:, 0], dN[1][:, 0], dN[2][:, 0]
        s1, s2, div = sg[:, 0], sg[:, 1], dsg[1][:, 0] + dsg[2][:, 1]
        M = M[:, 0]
        xn, yn = P[:, 1], P[:, 2]
        px, py = xn * (1 - xn), yn * (1 - yn)
        B, Bx, By = 16 * px * py, 16 * (1 - 2 * xn) * py, 16 * px * (1 - 2 * yn)
        f = prob.f(t, x, y).numpy()
        g = prob.g(t, x, y).numpy()
        gt = prob.g_t(t, x, y).numpy()
        gx, gy = (v.numpy() for v in prob.grad_g(t, x, y))
        u, ut, ux, uy = B * N, B * Nt, Bx * N + B * Nx, By * N + B * Ny
        lam = M * M
        gam = g - u
        R, Rp = f - ut - div + lam, f - gt - div + lam
        S2, Sp2 = (s1 + ux) ** 2 + (s2 + uy) ** 2, (s1 + gx) ** 2 + (s2 + gy) ** 2
    pos = gam > 0
    return {"Rh": np.where(pos, Rp * Rp, R * R), "Sh": np.where(pos, Sp2, S2), "C": lam * np.maximum(-gam, 0.0)}


def trace_f64(run, Xs, tval):
    """Trace point values at spatial float points Xs (n, D-1)."""
    kind, nets, prob = run["kind"], run["nets"], run["problem"]
    n = Xs.shape[0]
    P = np.concatenate([np.full((n, 1), tval), Xs], 1)
    N, _ = mlp_f64(nets["state"], P, ())
    N = N[:, 0]
    if kind == "stefan":
        x = Xs[:, 0]
        B = 4 * x * (1 - x)
        h = math.exp(tval) - tval - 1
        g, u0 = -h * (1 - x), np.zeros(n)
    else:
        xn, yn = Xs[:, 0], Xs[:, 1]
        B = 16 * xn * (1 - xn) * yn * (1 - yn)
        tt, xx, yy = _t(np.full(n, tval)), _t(xn), _t(yn)
        g = prob.g(tt, xx, yy).numpy()
        u0 = prob.u0(xx, yy).numpy()
    v = B * N
    w2 = np.maximum(g - v, 0.0) ** 2
    if tval == 0.0:
        return {"r0hat": (np.maximum(v, g) - u0) ** 2, "w2_0": w2}
    return {"w2_T": w2}


def interface_fn(kind, P):
    """Exact free-boundary level function (selection of interface-cut cells): Stefan x - t, disk R^2 - r^2, ellipse
    zeta (rational constants as floats; selection only)."""
    if kind == "stefan":
        return P[:, 1] - P[:, 0]
    t, x, y = P[:, 0], P[:, 1], P[:, 2]
    if kind == "disk":
        R = 0.15 + 0.2 * t
        return R * R - (x - 0.5) ** 2 - (y - 0.5) ** 2
    a, d = 0.12 + 0.1 * t, 0.08 + 0.07 * t
    return (x - 0.43) ** 2 / a**2 + (y - 0.57) ** 2 / d**2 - 1.0


# ============================================================================ arb (checking side)
ZERO, ONE = arb(0), arb(1)


def _arbnet(layers):
    return [(arb_mat([[arb(float(v)) for v in row] for row in W]), [arb(float(v)) for v in b]) for W, b in layers]


def arb_mlp(alayers, inp: list[list], chans: tuple[int, ...]):
    """Forward-mode arb ball evaluation of a tanh MLP (naive extension; rigorous for every point of the input balls).
    inp: d lists of P input balls. Returns (val[m][P], {ch: der[m][P]})."""
    d, P = len(inp), len(inp[0])
    nb = 1 + len(chans)
    rows = []
    for i in range(d):
        r = list(inp[i])
        for ch in chans:
            r += [ONE if i == ch else ZERO] * P
        rows.append(r)
    A = arb_mat(rows)
    nl = len(alayers)
    for k, (W, b) in enumerate(alayers):
        Z = W * A
        e = Z.entries()
        m, n = Z.nrows(), Z.ncols()
        if k == nl - 1:
            val = [[e[i * n + p] + b[i] for p in range(P)] for i in range(m)]
            der = {ch: [[e[i * n + (c + 1) * P + p] for p in range(P)] for i in range(m)] for c, ch in enumerate(chans)}
            return val, der
        new = []
        for i in range(m):
            row = e[i * n:(i + 1) * n]
            bi = b[i]
            z = [row[p] + bi for p in range(P)]
            a = [zz.tanh() for zz in z]
            if nb > 1:
                s = [zz.sech() for zz in z]
                s = [q * q for q in s]
                out = a + [s[p] * row[(c + 1) * P + p] for c in range(nb - 1) for p in range(P)]
            else:
                out = a
            new.append(out)
        A = arb_mat(new)


def _sel(sign_ball, on, off, pos_is_gt=True):
    """Branch value: on where the level is certainly > 0 (pos_is_gt) / < 0, off where certainly <= 0 / >= 0, else
    the ball union of both (naive enclosure of an indicator-switched term)."""
    if pos_is_gt:
        a, b = sign_ball > 0, sign_ball <= 0
    else:
        a, b = sign_ball < 0, sign_ball >= 0
    if a:
        return on
    if b:
        return off
    return on.union(off)


def _Q(p, q):
    return arb(p) / arb(q)


def data_arb(kind, kap, t, x, y=None):
    """Exact problem data on arb balls (decimal constants as the exact rationals they denote)."""
    if kind == "stefan":
        e = t.exp()
        h = e - t - 1
        return {"f": -1 - (e - 1) * (1 - x), "g": -h * (1 - x), "g_t": -(e - 1) * (1 - x), "g_x": h, "u0": ZERO}
    px, py = x * (1 - x), y * (1 - y)
    B, Bx, By, lapB = 16 * px * py, 16 * (1 - 2 * x) * py, 16 * px * (1 - 2 * y), -32 * (px + py)
    if kind == "disk":
        e = t.exp()
        h = e - t - 1
        R = _Q(3, 20) + t / 5
        r2 = (x - _Q(1, 2)) ** 2 + (y - _Q(1, 2)) ** 2
        psi = (R * R - r2) / (2 * R)
        psi_t = (R * R + r2) / (5 * 2 * R * R)
        g, gt, gx, gy, lapg = -h * B, -(e - 1) * B, -h * Bx, -h * By, -h * lapB
        k = arb(kap)
        bracket = 2 * psi * psi_t - 2 * psi * (-2 / R) + 2 - 2 * r2 / (R * R)
        f0 = gt - lapg - 2 * k
        f = _sel(psi, f0 + k * bracket, f0)
        # u0 = kap (psi(0)_+)^2 (only used at t = 0)
        u0 = _sel(psi, k * psi * psi, ZERO)
        return {"f": f, "g": g, "g_t": gt, "g_x": gx, "g_y": gy, "u0": u0}
    a, d = _Q(3, 25) + t / 10, _Q(2, 25) + 7 * t / 100
    X1, X2 = x - _Q(43, 100), y - _Q(57, 100)
    zeta = X1 * X1 / (a * a) + X2 * X2 / (d * d) - 1
    zx, zy = 2 * X1 / (a * a), 2 * X2 / (d * d)
    lapz = 2 / (a * a) + 2 / (d * d)
    zt = -2 * X1 * X1 / (10 * a * a * a) - 2 * 7 * X2 * X2 / (100 * d * d * d)
    c, L = _Q(1, 10) + t / 20, 1 + x / 5 - y / 10
    ub = B * c * L
    ub_t = B * L / 20
    ub_x, ub_y = c * (Bx * L + B / 5), c * (By * L - B / 10)
    lap_ub = c * (lapB * L + 2 * (Bx / 5 - By / 10))
    k0 = _Q(1, 100)
    g = ub - k0 * zeta * zeta
    gt, gx, gy = ub_t - 2 * k0 * zeta * zt, ub_x - 2 * k0 * zeta * zx, ub_y - 2 * k0 * zeta * zy
    base = ub_t - lap_ub
    f = _sel(zeta, base - 2 * k0 * zeta * (zt - lapz), base, pos_is_gt=False)
    u0 = _sel(zeta, ub - k0 * zeta * zeta, ub, pos_is_gt=False)
    return {"f": f, "g": g, "g_t": gt, "g_x": gx, "g_y": gy, "u0": u0}


def _sq_hi(b: arb) -> arb:
    m = abs(b).upper()
    return m * m


def _up(b: arb) -> float:
    from certified_obstacle_nn.certification.reliability import up

    return up(b)


def _down(b: arb) -> float:
    from certified_obstacle_nn.certification.reliability import down

    return down(b)


class ArbChecker:
    """Holds the arb networks of one run (built once per worker)."""

    def __init__(self, run):
        self.kind = run["kind"]
        self.kap = run["kappa_mult"]
        self.nets = {k: _arbnet(v) for k, v in run["nets"].items()}

    def volume(self, cols):
        """Arb field balls at input balls cols = [t, x(, y)] (lists of P balls); returns per point dict of balls."""
        k2 = self.kind != "stefan"
        ch_state = (0, 1, 2) if k2 else (0, 1)
        Nv, dN = arb_mlp(self.nets["state"], cols, ch_state)
        Sv, dS = arb_mlp(self.nets["flux"], cols, (1, 2) if k2 else (1,))
        Mv, _ = arb_mlp(self.nets["mult"], cols, ())
        out = []
        for p in range(len(cols[0])):
            t, x = cols[0][p], cols[1][p]
            y = cols[2][p] if k2 else None
            d = data_arb(self.kind, self.kap, t, x, y)
            lam = Mv[0][p] * Mv[0][p]
            if k2:
                px, py = x * (1 - x), y * (1 - y)
                B, Bx, By = 16 * px * py, 16 * (1 - 2 * x) * py, 16 * px * (1 - 2 * y)
                N, Nt, Nx, Ny = Nv[0][p], dN[0][0][p], dN[1][0][p], dN[2][0][p]
                u, ut, ux, uy = B * N, B * Nt, Bx * N + B * Nx, By * N + B * Ny
                div = dS[1][0][p] + dS[2][1][p]
                s1, s2 = Sv[0][p], Sv[1][p]
                S = (s1 + ux, s2 + uy)
                Sp = (s1 + d["g_x"], s2 + d["g_y"])
            else:
                B, dB = 4 * x * (1 - x), 4 - 8 * x
                N, Nt, Nx = Nv[0][p], dN[0][0][p], dN[1][0][p]
                u, ut, ux = B * N, B * Nt, dB * N + B * Nx
                div = dS[1][0][p]
                S = (Sv[0][p] + ux,)
                Sp = (Sv[0][p] + d["g_x"],)
            R = d["f"] - ut - div + lam
            Rp = d["f"] - d["g_t"] - div + lam
            out.append({"R": R, "Rp": Rp, "S": S, "Sp": Sp, "gam": d["g"] - u, "lam": lam, "M": Mv[0][p]})
        return out

    def trace(self, cols_sp, tval):
        P = len(cols_sp[0])
        tb = [arb(tval)] * P
        Nv, _ = arb_mlp(self.nets["state"], [tb] + cols_sp, ())
        out = []
        for p in range(P):
            x = cols_sp[0][p]
            y = cols_sp[1][p] if self.kind != "stefan" else None
            d = data_arb(self.kind, self.kap, tb[p], x, y)
            B = 4 * x * (1 - x) if y is None else 16 * x * (1 - x) * y * (1 - y)
            out.append({"v": B * Nv[0][p], "g": d["g"], "u0": d["u0"]})
        return out


def point_terms(F: dict) -> dict:
    """Term balls at an exact point (branch by the certain sign of gamma; ambiguous sign -> the smaller branch, so
    the sample stays a lower value)."""
    gam = F["gam"]
    R2, Rp2 = F["R"] * F["R"], F["Rp"] * F["Rp"]
    S2, Sp2 = sum(s * s for s in F["S"]), sum(s * s for s in F["Sp"])
    if gam > 0:
        Rh, Sh, amb = Rp2, Sp2, 0
    elif gam <= 0:
        Rh, Sh, amb = R2, S2, 0
    else:
        Rh = Rp2 if Rp2.upper() < R2.upper() else R2
        Sh = Sp2 if Sp2.upper() < S2.upper() else S2
        amb = 1
    negg = -gam
    C = F["lam"] * negg if negg > 0 else ZERO  # ambiguous sign: 0 is the smaller value
    return {"Rh": Rh, "Sh": Sh, "C": C, "amb": amb}


def box_terms(F: dict) -> dict:
    """Upper floats of the terms on a ball subcell (termwise hull on sign-unresolved subcells)."""
    gam = F["gam"]
    pos, nonpos = gam > 0, gam <= 0
    R = F["Rp"] if pos else (F["R"] if nonpos else F["R"].union(F["Rp"]))
    S = F["Sp"] if pos else (F["S"] if nonpos else tuple(a.union(b) for a, b in zip(F["S"], F["Sp"])))
    lam_hi = _sq_hi(F["M"])
    ng = max(_up(-gam), 0.0)
    return {"Rh": _up(_sq_hi(R)), "Sh": _up(sum((_sq_hi(s) for s in S), ZERO)), "C": _up(lam_hi * arb(ng))}


def trace_point_terms(F, tval):
    v, g, u0 = F["v"], F["g"], F["u0"]
    gv = g - v
    w2 = gv * gv if gv > 0 else (ZERO if gv <= 0 else ZERO)  # ambiguous sign: 0 is the smaller value
    if tval != 0.0:
        return {"w2_T": w2}
    if g > v:
        r0 = (g - u0) * (g - u0)
    elif v >= g:
        r0 = (v - u0) * (v - u0)
    else:  # ambiguous: the smaller of the two squares
        a2, b2 = (v - u0) * (v - u0), (g - u0) * (g - u0)
        r0 = a2 if a2.upper() < b2.upper() else b2
    return {"r0hat": r0, "w2_0": w2}


def trace_box_terms(F, tval):
    v, g, u0 = F["v"], F["g"], F["u0"]
    w = max(_up(g - v), 0.0)
    w2 = _up(arb(w) * arb(w))
    if tval != 0.0:
        return {"w2_T": w2}
    vm, gm = v - u0, g - u0
    lo = max(_down(vm), _down(gm))
    hi = max(_up(vm), _up(gm))
    m = max(abs(lo), abs(hi))
    return {"r0hat": _up(arb(m) * arb(m)), "w2_0": w2}


# ============================================================================ per-cell work (worker)
_W: dict = {}


def _init_worker(run_key):
    torch.set_num_threads(1)
    _W["run"] = _RUNS[run_key]
    _W["chk"] = ArbChecker(_RUNS[run_key])


_RUNS: dict = {}


def _unit_points(D: int) -> np.ndarray:
    from scipy.stats import qmc

    n = N_PTS**D - 2**D
    sob = qmc.Sobol(d=D, scramble=False).random_base2(int(math.ceil(math.log2(n))))[:n]
    corners = np.array([[(c >> j) & 1 for j in range(D)] for c in range(2**D)], dtype=float)
    return np.concatenate([corners, sob])  # the first 2^D rows are the corners


def _balls_from_points(P):
    return [[arb(float(v)) for v in P[:, j]] for j in range(P.shape[1])]


def check_cell(task):
    """task = (kind 'vol'|'trace0'|'traceT', lo (D,), hi (D,), numpy upper {term: float}). Returns per-term record."""
    kind, lo, hi, upper = task
    run, chk = _W["run"], _W["chk"]
    tic = time.time()
    D = lo.size
    U = _unit_points(D)
    P = lo[None, :] + U * (hi - lo)[None, :]
    tval = {"trace0": 0.0, "traceT": 1.0}.get(kind)
    vals = fields_f64(run, P) if kind == "vol" else trace_f64(run, P, tval)
    terms = list(vals)
    cand = set(range(2**D))
    for k in terms:
        cand.update(np.argsort(-vals[k])[:TOP_K].tolist())
    cand = sorted(cand)
    ctx.prec = PREC_POINT
    balls = _balls_from_points(P[cand])
    if kind == "vol":
        pts = [point_terms(F) for F in chk.volume(balls)]
    else:
        pts = [trace_point_terms(F, tval) for F in chk.trace(balls, tval)]
    rec = {"kind": kind, "lo": lo.tolist(), "hi": hi.tolist()}
    for k in terms:
        b = max(range(len(cand)), key=lambda i: float(pts[i][k].mid()))
        best = pts[b][k]
        relerr = max(abs(float(pts[i][k].mid()) - vals[k][cand[i]]) / max(abs(float(pts[i][k].mid())), 1e-300)
                     for i in range(len(cand)))
        rec[k] = {"upper": float(upper[k]), "f64_max": float(vals[k].max()), "arb_max_mid": float(best.mid()),
                  "arb_max_lower": _down(best), "violation": bool(best > arb(float(upper[k]))),
                  "f64_vs_arb_rel": float(relerr)}
    if kind == "vol":
        rec["ambiguous_sign_points"] = int(sum(p["amb"] for p in pts))
    # (ii) depth-3 bisection, naive arb balls
    ctx.prec = PREC_BISECT
    m = 2**BISECT
    w = (hi - lo) / m
    grids = np.meshgrid(*([np.arange(m)] * D), indexing="ij")
    I = np.stack([g.ravel() for g in grids], 1)
    sub_lo = lo[None, :] + I * w[None, :]
    mid = sub_lo + 0.5 * w[None, :]  # dyadic, exact
    rad = 0.5 * w
    cols = [[arb(float(mid[i, j]), float(rad[j])) for i in range(mid.shape[0])] for j in range(D)]
    if kind == "vol":
        boxes = [box_terms(F) for F in chk.volume(cols)]
    else:
        boxes = [trace_box_terms(F, tval) for F in chk.trace(cols, tval)]
    for k in terms:
        rec[k]["bisect_upper"] = max(b[k] for b in boxes)
    rec["seconds"] = time.time() - tic
    return rec


# ============================================================================ selection
def select_cells(run, rng) -> dict:
    lo, hi = run["lo"], run["hi"]
    n = lo.shape[1]
    vol = np.prod(hi - lo, axis=0)
    st = run["stored"]
    sets = {"random": rng.choice(n, 1000, replace=False)}
    for k, nm in (("Rh", "top500_R"), ("Sh", "top500_S"), ("C", "top500_C")):
        sets[nm] = np.argsort(-(vol * st[k]), kind="stable")[:500]
    # adversarial
    mid = (0.5 * (lo + hi)).T
    pre = np.zeros(n)
    for s in range(0, n, 500000):
        X = mid[s:s + 500000]
        pre[s:s + 500000] = np.max([preact_max(run["nets"][k], X) for k in ("state", "flux", "mult")], axis=0)
    hw, inv = np.unique(0.5 * (hi - lo), axis=1, return_inverse=True)  # radius depends on the half-widths only
    rad_u = np.zeros(hw.shape[1])
    for k in ("state", "flux", "mult"):
        r = hw
        for W, _ in run["nets"][k][:-1]:
            r = np.abs(W) @ r
            rad_u = np.maximum(rad_u, r.max(0))
    radius = rad_u[inv.ravel()]
    tie = rng.random(n)
    a1 = np.argsort(-pre, kind="stable")[:67]
    s1 = set(a1.tolist())
    a2 = []
    for i in np.lexsort((tie, -radius)):
        if int(i) not in s1:
            a2.append(int(i))
            if len(a2) == 67:
                break
    D = lo.shape[0]
    g3 = np.stack(np.meshgrid(*([np.array([0.0, 0.5, 1.0])] * D), indexing="ij"), -1).reshape(-1, D)
    cut = np.zeros(n, dtype=bool)
    for s in range(0, n, 200000):
        l, h = lo[:, s:s + 200000].T, hi[:, s:s + 200000].T
        vals = np.stack([interface_fn(run["kind"], l + u * (h - l)) for u in g3], 1)
        cut[s:s + 200000] = (vals.min(1) < 0) & (vals.max(1) > 0)
    pool = np.setdiff1d(np.nonzero(cut)[0], np.concatenate([a1, a2]))
    a3 = rng.choice(pool, min(66, pool.size), replace=False)
    sets["adversarial"] = np.concatenate([a1, np.array(a2, dtype=np.int64), a3])
    adv_parts = {"preactivation": a1.tolist(), "radius": list(map(int, a2)), "interface_cut": a3.tolist(),
                 "n_interface_cut_cells": int(cut.sum()), "max_preactivation": float(pre.max())}
    return sets, adv_parts


def select_traces(run, enc_tr, rng):
    out = {}
    for nm, (lo, hi), key in (("trace0", (run["tlo0"], run["thi0"]), "r0hat"), ("traceT", (run["tloT"], run["thiT"]),
                                                                                   "w2")):
        vol = np.prod(hi - lo, axis=0)
        c = vol * enc_tr[nm][key]
        top = np.argsort(-c, kind="stable")[:250]
        rest = np.setdiff1d(np.arange(lo.shape[1]), top)
        out[nm] = np.concatenate([top, rng.choice(rest, min(250, rest.size), replace=False)])
    return out


def engine_recompute(run, idx):
    """Recompute the numpy upper endpoints of cells idx with the (patched) engine; must equal cells.npz bitwise."""
    from certified_obstacle_nn.certification.enclose import intervals_np as NB

    enc = run["encloser"]
    lo, hi = run["lo"][:, idx], run["hi"][:, idx]
    outs = {"Rh": [], "Sh": [], "C": []}
    for s in range(0, idx.size, 256):
        l, h = lo[:, s:s + 256], hi[:, s:s + 256]
        if run["kind"] == "stefan":
            d = enc.densities(NB, l[0], h[0], l[1], h[1])
        else:
            d = enc.densities(NB, l, h)
        outs["Rh"].append(d["Rh_term"])
        outs["Sh"].append(d["Sh_term"])
        outs["C"].append(d["c"])
    return {k: np.concatenate(v) for k, v in outs.items()}


def engine_traces(run):
    from certified_obstacle_nn.certification.enclose import intervals_np as NB

    enc = run["encloser"]
    out = {}
    for nm, lo, hi, tv in (("trace0", run["tlo0"], run["thi0"], 0.0), ("traceT", run["tloT"], run["thiT"], 1.0)):
        acc = {"w2": [], "r0hat": []}
        for s in range(0, lo.shape[1], 1024):
            l, h = lo[:, s:s + 1024], hi[:, s:s + 1024]
            d = enc.trace_densities(NB, tv, l[0], h[0]) if run["kind"] == "stefan" else enc.trace_densities(NB, tv, l, h)
            acc["w2"].append(d["w2"])
            acc["r0hat"].append(d["r0hat"])
        out[nm] = {k: np.concatenate(v) for k, v in acc.items()}
    return out


# ============================================================================ driver
def run_one(key, spec, workers, log, out_dir, limit=None):
    tic = time.time()
    run = load_run(spec)
    _RUNS[key] = run
    rng = np.random.default_rng(SEED)
    sets, adv = select_cells(run, rng)
    tr = engine_traces(run)
    tsets = select_traces(run, tr, rng)
    if limit:
        sets = {k: v[:limit] for k, v in sets.items()}
        tsets = {k: v[:limit] for k, v in tsets.items()}
    uniq = np.unique(np.concatenate(list(sets.values())))
    rec = engine_recompute(run, uniq)
    mism = {k: int((rec[k] != run["stored"][k][uniq]).sum()) for k in rec}
    log(f"{key}: {uniq.size} volume cells; engine recompute vs stored cells.npz mismatches {mism}; "
        f"adversarial: max preact {adv['max_preactivation']:.3g}, cut cells {adv['n_interface_cut_cells']}")
    if any(mism.values()):
        raise RuntimeError(f"{key}: recomputed numpy upper endpoints differ from the stored cells.npz {mism}")
    tasks, tags = [], []
    for i in uniq:
        tasks.append(("vol", run["lo"][:, i].copy(), run["hi"][:, i].copy(),
                      {k: run["stored"][k][i] for k in VOL_TERMS}))
        tags.append(("vol", int(i)))
    for nm, idx in tsets.items():
        lo, hi = (run["tlo0"], run["thi0"]) if nm == "trace0" else (run["tloT"], run["thiT"])
        for i in idx:
            up = {"r0hat": tr[nm]["r0hat"][i], "w2_0": tr[nm]["w2"][i]} if nm == "trace0" else {"w2_T": tr[nm]["w2"][i]}
            tasks.append((nm, lo[:, i].copy(), hi[:, i].copy(), up))
            tags.append((nm, int(i)))
    ctxm = mp.get_context("fork")
    res = {}
    with ctxm.Pool(workers, initializer=_init_worker, initargs=(key,)) as pool:
        for j, r in enumerate(pool.imap(check_cell, tasks, chunksize=1)):
            res[tags[j]] = r
            if (j + 1) % 500 == 0:
                log(f"  {key}: {j + 1}/{len(tasks)} cells ({time.time() - tic:.0f}s)")
    rows, cells = [], []
    for nm, idx in list(sets.items()) + list(tsets.items()):
        kind = "vol" if nm in sets else nm
        terms = VOL_TERMS if kind == "vol" else TR_TERMS[nm]
        for k in terms:
            up = np.array([res[(kind, int(i))][k]["upper"] for i in idx])
            sm = np.array([res[(kind, int(i))][k]["arb_max_mid"] for i in idx])
            bi = np.array([res[(kind, int(i))][k]["bisect_upper"] for i in idx])
            vio = int(sum(res[(kind, int(i))][k]["violation"] for i in idx))
            rel = max(res[(kind, int(i))][k]["f64_vs_arb_rel"] for i in idx)
            mpos, bpos = up > 0, bi > 0
            r1 = sm[mpos] / up[mpos]
            r2 = up[bpos] / bi[bpos]
            rows.append({"run": key, "cell set": nm, "n": int(len(idx)), "term": k,
                         "max sample/upper": float(r1.max()) if r1.size else float("nan"),
                         "median sample/upper": float(np.median(r1)) if r1.size else float("nan"),
                         "min upper/bisect": float(r2.min()) if r2.size else float("nan"),
                         "median upper/bisect": float(np.median(r2)) if r2.size else float("nan"),
                         "violations": vio,
                         "q10 sample/upper": float(np.quantile(r1, 0.1)) if r1.size else float("nan"),
                         "q90 sample/upper": float(np.quantile(r1, 0.9)) if r1.size else float("nan"),
                         "max upper/bisect": float(r2.max()) if r2.size else float("nan"),
                         "frac upper > bisect": float((r2 > 1).mean()) if r2.size else float("nan"),
                         "n upper = 0": int((~mpos).sum()), "n sample > 0 where upper = 0": int((sm[~mpos] > 0).sum()),
                         "max rel |f64 - arb| at arb points": rel,
                         "split": spec["split"], "checkpoint_sha256": run["checkpoint_sha256"]})
        for i in idx:
            r = res[(kind, int(i))]
            for k in terms:
                cells.append({"run": key, "set": nm, "cell": int(i), "term": k, **{q: r[k][q] for q in r[k]},
                              "seconds": r["seconds"]})
    meta = {"selection_adversarial": adv, "n_volume_cells": int(uniq.size),
            "n_trace_cells": int(sum(v.size for v in tsets.values())), "engine_recompute_mismatches": mism,
            "seconds": time.time() - tic, "split": spec["split"], "partition": spec["partition"],
            "checkpoint": run["checkpoint"], "checkpoint_sha256": run["checkpoint_sha256"],
            "partition_hash": run["row"]["partition_hash"], "ambiguous_sign_points":
                int(sum(res[t].get("ambiguous_sign_points", 0) for t in res))}
    pd.DataFrame(cells).to_csv(out_dir / f"cells_{key}.csv.gz", index=False)
    del _RUNS[key]
    return rows, meta


def main():
    from certified_obstacle_nn.experiments import results_io

    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["all"])
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N cells of every set")
    ap.add_argument("--experiment", default="v18_whole_cell")
    a = ap.parse_args()
    specs = run_specs()
    keys = list(specs) if a.runs == ["all"] else a.runs
    d = results_io.make_run_dir(a.experiment)

    def log(msg):
        line = time.strftime("%H:%M:%S ") + msg
        print(line, flush=True)
        with open(d / "log.txt", "a") as fh:
            fh.write(line + "\n")

    cfg = {"experiment": a.experiment, "runs": {k: specs[k] for k in keys}, "seed": SEED, "top_k": TOP_K,
           "points_per_axis": N_PTS, "bisect_depth": BISECT, "prec_point_bits": PREC_POINT,
           "prec_bisect_bits": PREC_BISECT, "limit": a.limit, "workers": a.workers, "git_commit": git_commit(),
           "git_dirty": git_dirty(), "hardware": hardware(), "doc": __doc__}
    (d / "config.json").write_text(json.dumps(cfg, indent=1, default=str))
    rows, metas = [], {}
    for k in keys:
        r, m = run_one(k, specs[k], a.workers, log, d, a.limit)
        rows += r
        metas[k] = m
        pd.DataFrame(rows).to_csv(d / "summary.csv", index=False)
        (d / "meta.json").write_text(json.dumps(metas, indent=1, default=str))
        v = sum(x["violations"] for x in r)
        log(f"{k}: done in {m['seconds']:.0f}s, violations {v}")
    log(f"wrote {d}")


if __name__ == "__main__":
    main()
