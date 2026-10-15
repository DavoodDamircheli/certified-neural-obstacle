"""v14 Stage 4 gate: on 10^3 random dyadic cells of the Stefan seed-0 4x64
network, the fast backend's enclosure must contain the arb backend's, or agree
with it to within its own width (structural check on interval end points).
Random float64 point samples are recorded as a diagnostic only.

    python3 -m certified_obstacle_nn.experiments.v14.stage4_backend_gate
"""

import json
import time

import numpy as np
import torch

from certified_obstacle_nn.autograd.derivatives import eval_derivatives_detached, eval_gap_derivatives
from certified_obstacle_nn.certification.enclose import intervals_arb as AB
from certified_obstacle_nn.certification.enclose import intervals_np as NB
from certified_obstacle_nn.certification.enclose.engine import StefanAdapter, TripleEncloser
from certified_obstacle_nn.experiments.v14.common import REPO_ROOT, STEFAN_SEEDS, load_frozen_stefan, sha256_file

FIELDS = ("R", "S", "gam", "gam_t", "gam_x", "lam", "Rhat_pos", "Shat_pos")
DENS = ("r", "p", "c", "rhat")


def random_cells(k, seed=0, lmin=3, lmax=10):
    rng = np.random.default_rng(seed)
    lev = rng.integers(lmin, lmax + 1, size=k)
    h = 2.0 ** -lev
    i = np.floor(rng.random(k) / h)
    j = np.floor(rng.random(k) / h)
    return i * h, (i + 1) * h, j * h, (j + 1) * h


def main(k=1000, prec=64):
    NB.tables()
    AB.set_precision(prec)
    p, tr = load_frozen_stefan(STEFAN_SEEDS[0])
    enc = TripleEncloser(tr, StefanAdapter(p))
    t0, t1, x0, x1 = random_cells(k)
    tic = time.time()
    Fn = enc.fields(NB, t0, t1, x0, x1)
    Dn = enc.densities(NB, t0, t1, x0, x1)
    tn = time.time() - tic
    tic = time.time()
    Fa = enc.fields(AB, t0, t1, x0, x1)
    Da = enc.densities(AB, t0, t1, x0, x1)
    ta = time.time() - tic
    out = {"n_cells": k, "arb_prec": prec, "seconds_numpy_fields+densities": tn, "seconds_arb_fields+densities": ta,
           "checkpoint_sha256": sha256_file(STEFAN_SEEDS[0]), "fields": {}, "densities": {}}
    ok = True
    for f in FIELDS:
        rn, ra = Fn[f].range(), Fa[f].range()
        nl, nh, al, ah = rn.lower(), rn.upper(), ra.lower(), ra.upper()
        w = nh - nl
        contain = (nl <= al) & (nh >= ah)
        close = (np.abs(nl - al) <= w) & (np.abs(nh - ah) <= w)
        out["fields"][f] = {"strict_containment": int(contain.sum()), "within_own_width_only": int((~contain & close).sum()),
                            "fail": int((~contain & ~close).sum()),
                            "max_rel_endpoint_diff": float(np.max(np.maximum(np.abs(nl - al), np.abs(nh - ah)) / np.maximum(w, 1e-300)))}
        ok &= bool(np.all(contain | close))
    for d in DENS:
        an, aa = Dn[d], Da[d]
        ge = an >= aa
        close = np.abs(an - aa) <= np.maximum(an, aa) * 1e-6
        out["densities"][d] = {"fast_ge_arb": int(ge.sum()), "within_1e-6_rel_only": int((~ge & close).sum()), "fail": int((~ge & ~close).sum()),
                               "max_rel_diff": float(np.max(np.abs(an - aa) / np.maximum(np.maximum(an, aa), 1e-300)))}
    # diagnostic: float64 point samples (NOT a proof)
    rng = np.random.default_rng(1)
    m = 20
    tt = (t0[:, None] + (t1 - t0)[:, None] * rng.random((k, m))).reshape(-1, 1)
    xx = (x0[:, None] + (x1 - x0)[:, None] * rng.random((k, m))).reshape(-1, 1)
    T_ = torch.tensor(tt)
    X_ = torch.tensor(xx)
    v, s, lam, vt, vx, sx = eval_derivatives_detached(tr.state_net, tr.flux_net, tr.multiplier_net, T_, X_)
    gt, gx = eval_gap_derivatives(p, T_, X_, vt, vx)
    R = (p.f(T_, X_) - vt - sx + lam).numpy().reshape(k, m)
    rnR = Fn["R"].range()
    inside = (R >= rnR.lower()[:, None]) & (R <= rnR.upper()[:, None])
    out["diagnostic_float64_samples_R_inside"] = f"{int(inside.sum())}/{inside.size}"
    out["gate_passed"] = ok
    dst = REPO_ROOT / "results/v14/stage4/backend_gate.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
