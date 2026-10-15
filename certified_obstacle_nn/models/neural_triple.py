"""Assembled admissible neural triple (u_theta, sigma_phi, lambda_psi), specification section 1.

x_rescale: passed through to all three sub-networks (models/state_net.py's
module docstring has the full rationale) -- default None preserves exact
prior behavior for Stefan; American put passes problem.xi.
"""

import torch
import torch.nn as nn

from certified_obstacle_nn.models.flux_net import FluxNet
from certified_obstacle_nn.models.multiplier_net import MultiplierNet
from certified_obstacle_nn.models.state_net import StateNet
from certified_obstacle_nn.problems.base_problem import BaseProblem


class NeuralTriple(nn.Module):
    def __init__(self, problem: BaseProblem, width: int = 64, depth: int = 4, multiplier_head: str = "square", x_rescale=None):
        super().__init__()
        self.state_net = StateNet(problem.omega_bounds, width=width, depth=depth, x_rescale=x_rescale)
        self.flux_net = FluxNet(width=width, depth=depth, x_rescale=x_rescale)
        self.multiplier_net = MultiplierNet(width=width, depth=depth, head=multiplier_head, x_rescale=x_rescale)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u = self.state_net(t, x)
        sigma = self.flux_net(t, x)
        lam = self.multiplier_net(t, x)
        return u, sigma, lam
