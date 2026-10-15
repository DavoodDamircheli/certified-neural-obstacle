"""Flux network sigma_phi(t,x) = S_phi(t,xi), specification section 1.

A free network, deliberately NOT tied to -A grad u_theta.

x_rescale: see models/state_net.py's module docstring -- same optional
input-rescaling hook (default None = identity), same chain-rule guarantee.
"""

import torch
import torch.nn as nn

from certified_obstacle_nn.models.mlp import MLP


class FluxNet(nn.Module):
    def __init__(self, width: int = 64, depth: int = 4, x_rescale=None):
        super().__init__()
        self.x_rescale = x_rescale
        self.net = MLP(in_features=2, out_features=1, hidden_width=width, hidden_depth=depth)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x_in = x if self.x_rescale is None else self.x_rescale(x)
        return self.net(torch.cat([t, x_in], dim=-1))
