"""Multiplier network lambda_psi(t,x), specification section 1 and section 6.

Head selects how nonnegativity is made structural:
  'square'      lambda_psi = M_psi^2                  (default)
  'softplus'    lambda_psi = softplus(M_psi)
  'relu_square' lambda_psi = relu(M_psi)^2             (exact open zero set)

All three guarantee lambda_psi >= 0 exactly, with no clamping or penalty.

x_rescale: see models/state_net.py's module docstring -- same optional
input-rescaling hook (default None = identity), same chain-rule guarantee.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from certified_obstacle_nn.models.mlp import MLP

_HEADS = ("square", "softplus", "relu_square")


class MultiplierNet(nn.Module):
    def __init__(self, width: int = 64, depth: int = 4, head: str = "square", x_rescale=None):
        super().__init__()
        if head not in _HEADS:
            raise ValueError(f"unknown multiplier head {head!r}, expected one of {_HEADS}")
        self.head = head
        self.x_rescale = x_rescale
        self.net = MLP(in_features=2, out_features=1, hidden_width=width, hidden_depth=depth)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x_in = x if self.x_rescale is None else self.x_rescale(x)
        m = self.net(torch.cat([t, x_in], dim=-1))
        if self.head == "square":
            return m ** 2
        if self.head == "softplus":
            return F.softplus(m)
        return F.relu(m) ** 2
