"""True error in the paper's weak (U) norm, plus the raw estimator, on independent Sobol sets.

E_true^2 = ||grad e_u||^2 + ||e_u(0)||^2 + ||e_u(T)||^2 + ||e_sigma||^2 + ||R_PDE||^2
(last term via the section-2.5 identity div_tx(e_u,e_sigma) - e_lambda = R_PDE, so the
discontinuous exact lambda is never differenced). Integrals are |Q| x mean, |Q| = 1.
Quasi-Monte Carlo (prompt section 5), not the 1D Gauss-Legendre rule: the free boundary is a
moving circle, see NOTES_existing_code.md conflict 3.
"""

import math

import torch

from certified_obstacle_nn.experiments.expanding_disk_2d import exact_solution as ex
from certified_obstacle_nn.experiments.expanding_disk_2d import raw_estimator, sampling
from certified_obstacle_nn.experiments.expanding_disk_2d.losses import triple_fields

BAND = 0.02


def _volume_batch(triple, t, x, y, kappa, Lambda, alpha):
    t = t.clone().requires_grad_(True)
    x = x.clone().requires_grad_(True)
    y = y.clone().requires_grad_(True)
    fields = triple_fields(triple, t, x, y, create_graph=False)
    fields = tuple(v.detach() for v in fields)
    t, x, y = t.detach(), x.detach(), y.detach()
    u, u_t, grad_u, sig, div_sig, lam = fields
    ux, uy = ex.grad_u_exact(t, x, y, kappa)
    e_grad2 = (ux - grad_u[:, 0:1]) ** 2 + (uy - grad_u[:, 1:2]) ** 2
    e_sig2 = (-ux - sig[:, 0:1]) ** 2 + (-uy - sig[:, 1:2]) ** 2
    e_u2 = (ex.u_exact(t, x, y, kappa) - u) ** 2
    e_l2 = (ex.lambda_exact(t, x, y, kappa) - lam) ** 2
    dens = raw_estimator.volume_densities(t, x, y, fields, kappa, Lambda, alpha)
    E_vol = e_grad2 + e_sig2 + dens["R_PDE_sq"]
    r = torch.sqrt(ex.r_squared(x, y))
    in_band = ((r - ex.R(t)).abs() < BAND).to(u.dtype)
    out = {"E_grad": e_grad2, "E_sig": e_sig2, "E_pde": dens["R_PDE_sq"], "e_u_L2": e_u2,
           "e_lam_L2": e_l2, "band_frac": in_band,
           "E_vol_band": in_band * E_vol, "E_vol_away": (1 - in_band) * E_vol,
           "e_u_band": in_band * e_u2, "e_u_away": (1 - in_band) * e_u2,
           "e_lam_band": in_band * e_l2, "e_lam_away": (1 - in_band) * e_l2}
    out.update({k: v for k, v in dens.items() if k != "R_PDE_sq"})
    return {k: v.sum().item() for k, v in out.items()}


@torch.no_grad()
def _trace_batch(triple, t_val, x, y, kappa):
    t = torch.full_like(x, t_val)
    u = triple.u(t, x, y)
    e2 = (ex.u_exact(t, x, y, kappa) - u) ** 2
    w2 = torch.relu(ex.g(t, x, y) - u) ** 2
    return e2.sum().item(), w2.sum().item()


def evaluate_all(triple, cfg, n_eval=None, n_trace=None, seeds=None) -> dict:
    """Single streaming pass: E_true (+ split errors) and rho_tilde components on the
    same independent Sobol sets."""
    n_eval = n_eval or cfg.n_eval
    n_trace = n_trace or cfg.n_eval_trace
    seeds = seeds or (cfg.sobol_seeds["eval"], cfg.sobol_seeds["eval_t0"], cfg.sobol_seeds["eval_T"])
    triple.eval()
    t, x, y = sampling.interior(n_eval, seeds[0])
    acc: dict = {}
    for i in range(0, n_eval, cfg.eval_batch):
        sl = slice(i, i + cfg.eval_batch)
        b = _volume_batch(triple, t[sl], x[sl], y[sl], cfg.kappa, cfg.Lambda, cfg.alpha)
        for k, v in b.items():
            acc[k] = acc.get(k, 0.0) + v
    m = {k: v / n_eval for k, v in acc.items()}
    tr = {}
    for name, t_val, sd in (("0", 0.0, seeds[1]), ("T", cfg.T, seeds[2])):
        x0, y0 = sampling.initial(n_trace, sd)
        se, sw = 0.0, 0.0
        for i in range(0, n_trace, cfg.eval_batch):
            a, b = _trace_batch(triple, t_val, x0[i:i + cfg.eval_batch], y0[i:i + cfg.eval_batch], cfg.kappa)
            se += a
            sw += b
        tr[name] = (se / n_trace, sw / n_trace)
    m["E_u0"], m["w0_sq"] = tr["0"]
    m["E_uT"], m["wT_sq"] = tr["T"]
    m["R0_sq"] = m["E_u0"]  # u0 - u_theta(0) = e_u(0) since u(0) = u0
    E2 = m["E_grad"] + m["E_u0"] + m["E_uT"] + m["E_sig"] + m["E_pde"]
    out = {
        "E_true": math.sqrt(E2),
        "E_true_grad": math.sqrt(m["E_grad"]), "E_true_init": math.sqrt(m["E_u0"]),
        "E_true_final": math.sqrt(m["E_uT"]), "E_true_sigma": math.sqrt(m["E_sig"]),
        "E_true_pde": math.sqrt(m["E_pde"]),
        "err_u_L2": math.sqrt(m["e_u_L2"]), "err_sigma_L2": math.sqrt(m["E_sig"]),
        "err_lambda_L2": math.sqrt(m["e_lam_L2"]),
        "band_halfwidth": BAND, "band_volume_fraction": m["band_frac"],
        "E_vol_band": math.sqrt(m["E_vol_band"]), "E_vol_away": math.sqrt(m["E_vol_away"]),
        "err_u_band": math.sqrt(m["e_u_band"]), "err_u_away": math.sqrt(m["e_u_away"]),
        "err_lambda_band": math.sqrt(m["e_lam_band"]), "err_lambda_away": math.sqrt(m["e_lam_away"]),
        "n_eval": n_eval, "n_eval_trace": n_trace,
    }
    est = raw_estimator.assemble(m)
    out.update(est)
    out["I_eff_raw"] = est["rho_tilde"] / out["E_true"]
    return out
