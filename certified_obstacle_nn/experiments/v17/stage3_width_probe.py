"""v17 Stage 3 diagnostic: where do TM1 enclosure widths come from? (diagnostic only; no certificate)

    python3 -m certified_obstacle_nn.experiments.v17.stage3_width_probe --run-dir <stored 2-D run> --n 2000

On a random sample of cells of a stored partition, re-runs the TM1 forward pass of the three networks layer by layer
(the same operations as engine2d.enclose_mlp_vec, numpy backend) and records, per layer: the mean radius of the
pre-activation range, the share of it carried by the TM remainder, the tanh Taylor remainder |tanh''(Z)| r^2/2
against the trivial tanh range width, and the same for the derivative channels. Also compares the final field
ranges with sampled float64 variation on the cell (q^3 points), per field (u_t, div sigma, mu, S components,
gamma). Writes one JSON into results/v17_width_probe/<timestamp>/.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from certified_obstacle_nn.certification.enclose import tm as T
from certified_obstacle_nn.certification.enclose import upper_sum as US
from certified_obstacle_nn.experiments import results_io
from certified_obstacle_nn.experiments.v17.common import load_stored_run


def _rad(iv):
    return 0.5 * (iv.upper() - iv.lower())


def probe_mlp(layers, inputs, channels, bk):
    n = inputs[0].n
    B = inputs[0].c.shape[1]
    a = T.tm_stack(inputs, axis=0)
    W0, b0 = layers[0]
    z = T.tm_linmap(W0, a, b0)
    dz = {}
    for ch in channels:
        col = np.broadcast_to(W0[:, ch].reshape(-1, 1), (W0.shape[0], B))
        dz[ch] = T.tm_const(bk, col, n, col.shape)
    rec = []
    for li, (W, b) in enumerate(layers[1:], start=1):
        Z = z.range()
        r = _rad(Z)
        rem = _rad(z.rem)
        lin = z.lin_mag()
        S_Z = bk.sech2(Z)
        T_Z = bk.tanh(Z)
        d2 = bk.mag(T_Z * S_Z * -2.0)
        taylor_rem = d2 * r * r / 2.0
        trivial = (T_Z.upper() - T_Z.lower())
        e = {"layer": li, "z_rad_mean": float(r.mean()), "z_rad_max": float(r.max()),
             "z_lin_mean": float(lin.mean()), "z_rem_mean": float(rem.mean()),
             "tanh_taylor_rem_mean": float(taylor_rem.mean()), "tanh_trivial_width_mean": float(trivial.mean()),
             "frac_taylor_rem_gt_trivial": float((taylor_rem > trivial).mean())}
        act, s = T.tm_tanh_sech2(z)
        e["act_rad_mean"] = float(_rad(act.range()).mean())
        e["sech2_rad_mean"] = float(_rad(s.range()).mean())
        da = {ch: T.tm_mul(s, dz[ch]) for ch in channels}
        for ch in channels:
            e[f"dz{ch}_rad_mean"] = float(_rad(dz[ch].range()).mean())
            e[f"da{ch}_rad_mean"] = float(_rad(da[ch].range()).mean())
            e[f"da{ch}_rem_mean"] = float(_rad(da[ch].rem).mean())
        rec.append(e)
        z = T.tm_linmap(W, act, b)
        dz = {ch: T.tm_linmap(W, da[ch]) for ch in channels}
    out = {"layer": len(layers), "z_rad_mean": float(_rad(z.range()).mean()), "z_rem_mean": float(_rad(z.rem).mean())}
    for ch in channels:
        out[f"dz{ch}_rad_mean"] = float(_rad(dz[ch].range()).mean())
        out[f"dz{ch}_rem_mean"] = float(_rad(dz[ch].rem).mean())
    rec.append(out)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    r = load_stored_run(a.run_dir)
    bk = US._backend("numpy", 64)
    from certified_obstacle_nn.certification.enclose import intervals_np

    intervals_np.tables()
    rng = np.random.default_rng(a.seed)
    idx = np.sort(rng.choice(r.part.n, size=min(a.n, r.part.n), replace=False))
    lo, hi = r.part.lo[:, idx], r.part.hi[:, idx]
    c, h = 0.5 * (lo + hi), 0.5 * (hi - lo)
    tt = T.tm_variable(bk, c[0], h[0], 1, 3)
    xx = T.tm_variable(bk, c[1], h[1], 2, 3)
    yy = T.tm_variable(bk, c[2], h[2], 3, 3)
    enc = r.encloser
    res = {"run": str(a.run_dir), "n_cells": int(idx.size), "cell_width_median": float(np.median(2 * h[0])),
           "state": probe_mlp(enc.Ws, [tt, xx, yy], (0, 1, 2), bk),
           "flux": probe_mlp(enc.Wf, [tt, xx, yy], (1, 2), bk),
           "mult": probe_mlp(enc.Wm, [tt, xx, yy], (), bk)}
    # field ranges vs sampled float64 variation on the same cells
    F = enc.fields(bk, lo, hi)
    from certified_obstacle_nn.experiments.v17.cell_quadrature import _pts

    q = 3
    P, _ = _pts(lo, hi, q)
    cols = [torch.as_tensor(P[j]).clone().requires_grad_(True) for j in range(3)]
    t, x, y = cols
    X = torch.stack(cols, -1)
    tr = r.triple
    prob = r.problem
    N = tr.state(X)[:, 0]
    Bm = prob.mask(x, y)
    v = Bm * N
    v_t, v_x, v_y = torch.autograd.grad(v.sum(), cols, retain_graph=True)
    S = tr.flux(X)
    div = torch.autograd.grad(S[:, 0].sum(), x, retain_graph=True)[0] + torch.autograd.grad(S[:, 1].sum(), y)[0]
    mu = tr.mult(X)[:, 0] ** 2
    with torch.no_grad():
        g = prob.g(t, x, y)
        f = prob.f(t, x, y)
        samp = {"R": f - v_t - div + mu, "S1": S[:, 0] + v_x, "S2": S[:, 1] + v_y, "gam": g - v, "lam": mu}
    nq = q**3
    res["fields"] = {}
    for k, val in samp.items():
        arr = val.detach().numpy().reshape(idx.size, nq)
        var = 0.5 * (arr.max(1) - arr.min(1))
        rad = _rad(F[k].range())
        res["fields"][k] = {"enclosure_rad_mean": float(rad.mean()), "sampled_halfvariation_mean": float(var.mean()),
                            "ratio_median": float(np.median(rad / np.maximum(var, 1e-300))),
                            "abs_mean_value": float(np.abs(arr).mean())}
    d = results_io.make_run_dir("v17_width_probe")
    (d / "probe.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["fields"], indent=1))
    for net in ("state", "flux", "mult"):
        print(net, [(e["layer"], round(e["z_rad_mean"], 3), round(e.get("tanh_taylor_rem_mean", 0), 3),
                     round(e.get("frac_taylor_rem_gt_trivial", 0), 3)) for e in res[net]])
    print("wrote", d)


if __name__ == "__main__":
    main()
