"""Raw continuum estimator rho_tilde (paper Def. 4.6 / specification section 4), 2D port.

Pointwise densities, integrated by the caller (evaluate.evaluate_all) as |Q| x mean over an
independent Sobol set (|Q| = 1, |Omega| = 1):
  rho_r^2       = (Lambda/alpha) ||R_PDE||^2 + ||R_sig||^2 + ||R_0||^2_{L2(Omega)}
  rho_p_tilde^2 = ||grad w||^2 + kappa_p ||w||^2 + (Lambda/alpha) ||d_t w||^2
                  + ||w(0)||^2 + ||w(T)||^2,     kappa_p = alpha/Lambda (b = c = 0)
  rho_c^2       = int lambda_psi (u_theta - g)^+
with w = (g - u_theta)^+, grad w = 1{g>u_theta} grad(g - u_theta), d_t w likewise
(g's derivatives are closed form, u_theta's come from autograd).
"""

import math

import torch

from certified_obstacle_nn.experiments.expanding_disk_2d import exact_solution as ex


def volume_densities(t, x, y, fields, kappa: float, Lambda: float, alpha: float) -> dict:
    u, u_t, grad_u, sig, div_sig, lam = fields
    f = ex.f_exact(t, x, y, kappa)
    g = ex.g(t, x, y)
    gx, gy = ex.grad_g(t, x, y)
    gt = ex.g_t(t, x, y)
    R_PDE = f - u_t - div_sig + lam
    R_sig = sig + grad_u
    gamma = g - u
    ind = (gamma > 0).to(u.dtype)
    w = torch.relu(gamma)
    grad_w2 = ind * ((gx - grad_u[:, 0:1]) ** 2 + (gy - grad_u[:, 1:2]) ** 2)
    dtw2 = ind * (gt - u_t) ** 2
    La = Lambda / alpha
    return {
        "rho_r_vol": La * R_PDE ** 2 + (R_sig ** 2).sum(-1, keepdim=True),
        "rho_r_pde": La * R_PDE ** 2,
        "rho_r_sig": (R_sig ** 2).sum(-1, keepdim=True),
        "rho_p_vol": grad_w2 + (alpha / Lambda) * w ** 2 + La * dtw2,
        "rho_c": lam * torch.relu(u - g),
        "R_PDE_sq": R_PDE ** 2,
    }


def assemble(m: dict) -> dict:
    """m: integrated means (|Q| = |Omega| = 1). Returns the reported components."""
    rho_r2 = m["rho_r_vol"] + m["R0_sq"]
    rho_p2 = m["rho_p_vol"] + m["w0_sq"] + m["wT_sq"]
    rho_c2 = m["rho_c"]
    return {
        "rho_r": math.sqrt(rho_r2),
        "rho_p_tilde": math.sqrt(rho_p2),
        "rho_c": math.sqrt(max(rho_c2, 0.0)),
        "rho_tilde": math.sqrt(rho_r2 + rho_p2 + rho_c2),
        "rho_r_pde": math.sqrt(m["rho_r_pde"]),
        "rho_r_sig": math.sqrt(m["rho_r_sig"]),
        "rho_r_init": math.sqrt(m["R0_sq"]),
    }
