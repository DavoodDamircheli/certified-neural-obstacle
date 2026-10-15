"""Neural triple for the 2D benchmark (paper Definition 5.1, specification section 1).

Three separate tanh MLPs (models/mlp.py, reused unchanged), raw physical input (t,x,y) in
[0,1]^3 -- the 1D Stefan code applies no input normalization, so neither do we:
  u_theta   = B(x,y) * N_theta(t,x,y),   B = 16 x(1-x) y(1-y)  (physical coordinates)
  sigma_phi = S_phi(t,x,y) in R^2        (free)
  lambda_psi = M_psi(t,x,y)^2            (head 'square'; 'softplus'/'relu_square' optional)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from certified_obstacle_nn.models.mlp import MLP
from certified_obstacle_nn.experiments.expanding_disk_2d import exact_solution as ex


class Triple2D(nn.Module):
    def __init__(self, depth: int, width: int, head: str = "square"):
        super().__init__()
        self.head = head
        self.state = MLP(3, 1, hidden_width=width, hidden_depth=depth)
        self.flux = MLP(3, 2, hidden_width=width, hidden_depth=depth)
        self.mult = MLP(3, 1, hidden_width=width, hidden_depth=depth)

    def u(self, t, x, y):
        return ex.B(x, y) * self.state(torch.cat([t, x, y], dim=-1))

    def sigma(self, t, x, y):
        return self.flux(torch.cat([t, x, y], dim=-1))

    def lam(self, t, x, y):
        m = self.mult(torch.cat([t, x, y], dim=-1))
        if self.head == "square":
            return m ** 2
        if self.head == "softplus":
            return F.softplus(m)
        return F.relu(m) ** 2


def n_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
