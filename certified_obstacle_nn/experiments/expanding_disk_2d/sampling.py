"""Scrambled-Sobol point sets (prompt section 5). Each set has its own scramble seed, so
training, monitoring, evaluation and estimation points are independent."""

import torch

from certified_obstacle_nn.experiments.expanding_disk_2d import exact_solution as ex

DTYPE = torch.float64


def sobol(n: int, dim: int, seed: int) -> torch.Tensor:
    engine = torch.quasirandom.SobolEngine(dimension=dim, scramble=True, seed=seed)
    return engine.draw(n, dtype=DTYPE)


def interior(n: int, seed: int):
    """(t, x, y), each [n,1], in Q = (0,1)^3."""
    p = sobol(n, 3, seed)
    return p[:, 0:1], p[:, 1:2], p[:, 2:3]


def initial(n: int, seed: int):
    """(x, y), each [n,1], in Omega = (0,1)^2 (the t=0 slice)."""
    p = sobol(n, 2, seed)
    return p[:, 0:1], p[:, 1:2]


def band(n: int, seed: int, halfwidth: float):
    """n points in the space-time band |r - R(t)| < halfwidth (Stage-2 fallback 5). Sobol
    (t, s, theta) mapped to r = R(t) + halfwidth*(2s-1), area-weighted via r-rejection-free
    sampling of r^2 so the density is uniform in (x,y) within the annulus."""
    p = sobol(n, 3, seed)
    t = p[:, 0:1]
    Rt = ex.R(t)
    r_lo, r_hi = Rt - halfwidth, Rt + halfwidth
    r = torch.sqrt(r_lo ** 2 + p[:, 1:2] * (r_hi ** 2 - r_lo ** 2))
    th = 2.0 * torch.pi * p[:, 2:3]
    return t, ex.C0 + r * torch.cos(th), ex.C0 + r * torch.sin(th)


def training_points(cfg):
    n_band = int(round(cfg.band_fraction * cfg.n_int))
    t, x, y = interior(cfg.n_int - n_band, cfg.sobol_seeds["train"])
    if n_band > 0:
        tb, xb, yb = band(n_band, cfg.sobol_seeds["band"], cfg.band_halfwidth)
        t, x, y = torch.cat([t, tb]), torch.cat([x, xb]), torch.cat([y, yb])
    x0, y0 = initial(cfg.n_init, cfg.sobol_seeds["init"])
    return t, x, y, x0, y0
