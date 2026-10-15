"""State network u_theta(t,x) = B(x) * N_theta(t,xi), specification section 1.

B is the polynomial boundary mask on the problem's 1D domain, B(x) =
4(x-a)(b-x)/(b-a)^2. It vanishes exactly (bit-for-bit) at x=a and x=b, which is
what makes the Dirichlet boundary condition structural rather than penalized:
there is no boundary term anywhere in this module. The mask ALWAYS uses
PHYSICAL x (never x_rescale(x)) -- it must vanish at the PHYSICAL boundary.

x_rescale (2026-09-12, specification section 10's American put normalized-
training convention): an optional callable applied to x before it is fed to
the underlying MLP, e.g. problems/american_put.py's own xi(x) =
2(x-L)/(R-L)-1. Default None (identity) preserves EXACT prior behavior for
every other problem (Stefan never sets this). Autograd still differentiates
w.r.t. the PHYSICAL x argument -- xi is computed FROM x via ordinary
differentiable tensor ops, never detached, so d(output)/dx automatically
picks up the chain-rule factor d(xi)/dx; see
tests/test_x_rescale_chain_rule.py's direct autograd-vs-finite-difference
check in PHYSICAL x, not a claim taken on faith.
"""

import torch
import torch.nn as nn

from certified_obstacle_nn.models.mlp import MLP


class StateNet(nn.Module):
    def __init__(self, omega_bounds: tuple[float, float], width: int = 64, depth: int = 4, x_rescale=None):
        super().__init__()
        self.omega_bounds = omega_bounds
        self.x_rescale = x_rescale
        self.net = MLP(in_features=2, out_features=1, hidden_width=width, hidden_depth=depth)

    def mask(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.omega_bounds
        return 4.0 * (x - a) * (b - x) / (b - a) ** 2

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x_in = x if self.x_rescale is None else self.x_rescale(x)
        n = self.net(torch.cat([t, x_in], dim=-1))
        return self.mask(x) * n
