"""Uniform tensor-product cells, Gauss-Legendre nodes and pointwise Level-2 densities.

Cell ids are row-major over counts = (n_t, n_x1, .., n_xd) (t slowest), exactly like the legacy
2D `_volume_pass` (id -> (it, ix, iy)) and the legacy 1D partition (i-slowest/j-fastest).
"""

from __future__ import annotations

import math

import numpy as np
import torch


def gauss(q: int):
    """Nodes in [0,1] and weights summing to 1."""
    x, w = np.polynomial.legendre.leggauss(q)
    return torch.tensor(0.5 * (x + 1)), torch.tensor(0.5 * w)


def gauss_max_offset(q: int) -> float:
    """max |node - 1/2| / (1/2) for the q-point rule (relative distance of the outermost node)."""
    x, _ = np.polynomial.legendre.leggauss(q)
    return float(np.abs(x).max()) if q > 1 else 0.0


def cell_boxes(ids: torch.Tensor, counts, box_lo, box_hi):
    """ids [B] -> lo, hi [B, m]."""
    idx = []
    rem = ids.clone()
    for n in reversed(counts):
        idx.append(rem % n)
        rem = rem // n
    idx = torch.stack(list(reversed(idx)), -1).to(torch.float64)
    lo_t = torch.tensor(box_lo, dtype=torch.float64)
    sz = torch.tensor([(b - a) / n for a, b, n in zip(box_lo, box_hi, counts)], dtype=torch.float64)
    lo = lo_t + idx * sz
    hi = lo + sz
    return lo, hi


def tensor_nodes(lo, hi, q: int):
    """Gauss nodes of every box: returns points [B*q^m, m] and weights [B*q^m] (sum = |box|)."""
    nodes, wts = gauss(q)
    m = lo.shape[1]
    grids = torch.meshgrid(*([nodes] * m), indexing="ij")
    off = torch.stack(grids, -1).reshape(-1, m)                         # [q^m, m]
    wq = torch.ones(off.shape[0], dtype=torch.float64)
    wgrids = torch.meshgrid(*([wts] * m), indexing="ij")
    for g_ in wgrids:
        wq = wq * g_.reshape(-1)
    size = hi - lo
    pts = lo[:, None, :] + size[:, None, :] * off[None]
    vol = size.prod(-1)
    W = vol[:, None] * wq[None]
    return pts.reshape(-1, m), W.reshape(-1)


def point_densities(prob, F):
    """Pointwise Level-2 densities (specification section 5) from exact fields F (tensors)."""
    La = prob.Lambda / prob.alpha
    gam = F["gamma"]
    Psi = prob.A * sum(v * v for v in F["ggrad"]) + La * F["gamma_t"] ** 2
    ind = (gam > 0).to(gam.dtype)
    return {"P_R": La * F["R"] ** 2, "P_sig": sum(v * v for v in F["Rsig"]) * 1.0,
            "P_kw": prob.kappa * torch.relu(gam) ** 2, "P_c": F["lam"] * torch.relu(-gam),
            "P_int": ind * Psi, "Psi": Psi}


def n_cells(counts) -> int:
    return int(math.prod(counts))
