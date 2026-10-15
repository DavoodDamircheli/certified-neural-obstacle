"""A1 diagnostic (v14): pointwise comparison of the true-error graph-term
integrand d_t e_u + div_x e_s - e_l with R_PDE on the same points, and the
global-rule convergence of ||.||^2 at n=160 vs 320.

For Stefan (b=c=0) the identity  d_t e_u + d_x e_s - e_l == R_PDE  holds
pointwise off t=x (exact solution satisfies u_t + sigma_x - lambda = f on both
sides), so the integrand is smooth and global Gauss quadrature must converge
spectrally.
"""
import argparse
import json

import numpy as np
import torch

from certified_obstacle_nn.autograd.derivatives import eval_derivatives_detached
from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.problems.stefan import StefanProblem


def load_triple(weights, width=64, depth=4, head="square"):
    p = StefanProblem()
    tr = NeuralTriple(p, width=width, depth=depth, multiplier_head=head).double()
    tr.load_state_dict(torch.load(weights, map_location="cpu"))
    tr.eval()
    for q in tr.parameters():
        q.requires_grad_(False)
    return p, tr


def gauss_grid(n):
    z, w = np.polynomial.legendre.leggauss(n)
    z = 0.5 * (z + 1.0)
    w = 0.5 * w
    T, X = np.meshgrid(z, z, indexing="ij")
    W = np.outer(w, w)
    return (torch.tensor(T.reshape(-1, 1)), torch.tensor(X.reshape(-1, 1)), torch.tensor(W.reshape(-1, 1)))


def graph_and_rpde(p, tr, t, x):
    u_th, s_th, l_th, ut_th, ux_th, sx_th = eval_derivatives_detached(tr.state_net, tr.flux_net, tr.multiplier_net, t, x)
    u_rf, s_rf, l_rf, ut_rf, ux_rf, sx_rf = eval_derivatives_detached(p.u_exact, p.sigma_exact, p.lambda_exact, t, x)
    graph = (ut_rf - ut_th) + (sx_rf - sx_th) - (l_rf - l_th)
    r_pde = p.f(t, x) - ut_th - sx_th + l_th
    exact_res = ut_rf + sx_rf - l_rf - p.f(t, x)
    return graph, r_pde, exact_res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="results/stefan_baseline_weights.pt")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    p, tr = load_triple(a.weights)
    out = {}
    for n in (40, 80, 160, 320):
        t, x, w = gauss_grid(n)
        graph, r_pde, exact_res = graph_and_rpde(p, tr, t, x)
        out[n] = {
            "max_abs_graph_minus_RPDE": float((graph - r_pde).abs().max()),
            "max_abs_exact_residual": float(exact_res.abs().max()),
            "graph_sq_integral": float((w * graph**2).sum()),
            "RPDE_sq_integral": float((w * r_pde**2).sum()),
        }
        print(n, out[n])
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
