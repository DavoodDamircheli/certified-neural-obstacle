"""Uniform tensor-product certification partition, specification section 7.

T_h is a uniform N_t x N_x grid of cells over Q = (0,T) x Omega, with midpoint
z_K, |K| the cell area, h_K = sqrt(dt^2 + dx^2), and h = max_K h_K (which, on
a uniform grid, is just h_K for every cell).

This module only builds geometry -- it knows nothing about residuals, models,
or training. Its nodes are the deterministic grid edges/midpoints of a fixed
tensor-product partition, never randomly sampled, so they are disjoint from
any training collocation set by construction (specification section 12 invariant
1): this module must never be handed, or need, the training sampler.
"""

import math
from typing import NamedTuple

import torch

from certified_obstacle_nn.problems.base_problem import BaseProblem


class Partition(NamedTuple):
    n_t: int
    n_x: int
    t_edges: torch.Tensor  # (n_t+1,)
    x_edges: torch.Tensor  # (n_x+1,)
    t_center: torch.Tensor  # (n_t*n_x,) flattened cell midpoint, i-slowest/j-fastest
    x_center: torch.Tensor  # (n_t*n_x,)
    i_index: torch.Tensor  # (n_t*n_x,) time-row index of each flattened cell
    j_index: torch.Tensor  # (n_t*n_x,) space-column index of each flattened cell
    dt: float
    dx: float
    area: float  # |K|, uniform across cells
    h_K: float  # sqrt(dt^2 + dx^2), uniform across cells
    h: float  # max_K h_K = h_K on a uniform grid


def build_partition(problem: BaseProblem, n_t: int, n_x: int) -> Partition:
    a, b = problem.omega_bounds
    T = problem.T

    t_edges = torch.linspace(0.0, T, n_t + 1, dtype=torch.float64)
    x_edges = torch.linspace(a, b, n_x + 1, dtype=torch.float64)

    dt = T / n_t
    dx = (b - a) / n_x
    area = dt * dx
    h_K = math.sqrt(dt**2 + dx**2)

    t_mid = 0.5 * (t_edges[:-1] + t_edges[1:])  # (n_t,)
    x_mid = 0.5 * (x_edges[:-1] + x_edges[1:])  # (n_x,)

    i_grid, j_grid = torch.meshgrid(torch.arange(n_t), torch.arange(n_x), indexing="ij")
    t_center_grid, x_center_grid = torch.meshgrid(t_mid, x_mid, indexing="ij")

    return Partition(
        n_t=n_t,
        n_x=n_x,
        t_edges=t_edges,
        x_edges=x_edges,
        t_center=t_center_grid.reshape(-1),
        x_center=x_center_grid.reshape(-1),
        i_index=i_grid.reshape(-1),
        j_index=j_grid.reshape(-1),
        dt=dt,
        dx=dx,
        area=area,
        h_K=h_K,
        h=h_K,
    )
