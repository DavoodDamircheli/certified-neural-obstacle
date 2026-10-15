"""Residuals and training loss (paper eq. `loss`, specification sections 2-3), copy-adapted from
residuals/residuals.py + training/losses.py to two spatial dimensions. Same signs, same
weights semantics (R_com enters un-squared, all others squared means). Only first
derivatives of networks: u_t, grad_x u, div sigma = d_x sigma_1 + d_y sigma_2."""

from typing import NamedTuple

import torch

from certified_obstacle_nn.experiments.expanding_disk_2d import exact_solution as ex


class Residuals2D(NamedTuple):
    R_PDE: torch.Tensor  # [N,1]
    R_sig: torch.Tensor  # [N,2]
    R_0: torch.Tensor    # [N0,1]
    R_obs: torch.Tensor  # [N,1]
    R_com: torch.Tensor  # [N,1]


def _grad(out, inp, create_graph, retain_graph=None):
    return torch.autograd.grad(out, inp, grad_outputs=torch.ones_like(out), create_graph=create_graph,
                               retain_graph=retain_graph)


def triple_fields(triple, t, x, y, create_graph: bool):
    """u, u_t, grad u [N,2], sigma [N,2], div sigma [N,1], lambda. Inputs must require grad."""
    u = triple.u(t, x, y)
    u_t, u_x, u_y = _grad(u, (t, x, y), create_graph)
    sig = triple.sigma(t, x, y)
    (s1_x,) = _grad(sig[:, 0:1], (x,), create_graph, retain_graph=True)
    (s2_y,) = _grad(sig[:, 1:2], (y,), create_graph)
    lam = triple.lam(t, x, y)
    grad_u = torch.cat([u_x, u_y], dim=-1)
    div_sig = s1_x + s2_y
    n = t.shape[0]
    assert grad_u.shape == (n, 2) and sig.shape == (n, 2), (grad_u.shape, sig.shape)
    assert div_sig.shape == (n, 1) and lam.shape == (n, 1) and u.shape == (n, 1)
    return u, u_t, grad_u, sig, div_sig, lam


def compute_residuals(triple, t, x, y, x0, y0, kappa: float, create_graph: bool = True) -> Residuals2D:
    u, u_t, grad_u, sig, div_sig, lam = triple_fields(triple, t, x, y, create_graph)
    f = ex.f_exact(t, x, y, kappa)
    g = ex.g(t, x, y)
    R_PDE = f - u_t - div_sig + lam
    R_sig = sig + grad_u  # A = I
    t0 = torch.zeros_like(x0)
    R_0 = ex.u0_exact(x0, y0, kappa) - triple.u(t0, x0, y0)
    R_obs = torch.relu(g - u)
    R_com = lam * torch.relu(u - g)
    return Residuals2D(R_PDE, R_sig, R_0, R_obs, R_com)


class Loss2D(NamedTuple):
    total: torch.Tensor
    pde: torch.Tensor
    sigma: torch.Tensor
    initial: torch.Tensor
    penetration: torch.Tensor
    complementarity: torch.Tensor


def compute_loss(r: Residuals2D, w_r=1.0, w_s=1.0, w_0=1.0, w_p=1.0, w_c=1.0) -> Loss2D:
    pde = r.R_PDE.pow(2).mean()
    sigma = r.R_sig.pow(2).sum(dim=-1).mean()  # |R_sig|^2, Euclidean in R^2
    initial = r.R_0.pow(2).mean()
    pen = r.R_obs.pow(2).mean()
    com = r.R_com.mean()
    total = w_r * pde + w_s * sigma + w_0 * initial + w_p * pen + w_c * com
    return Loss2D(total, pde, sigma, initial, pen, com)
