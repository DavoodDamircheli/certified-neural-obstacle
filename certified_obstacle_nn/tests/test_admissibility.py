"""Structural admissibility checks, specification section 1.

u_theta's boundary vanishing comes from multiplying by a mask that is exactly
zero at the boundary, not from any penalty -- so the check is exact equality,
not a small-tolerance check. lambda_psi's nonnegativity likewise comes from the
head's algebraic form (square / softplus / relu-square), not clamping.
"""

import torch
import pytest

from certified_obstacle_nn.models.multiplier_net import MultiplierNet
from certified_obstacle_nn.models.state_net import StateNet
from certified_obstacle_nn.problems.stefan import StefanProblem

SEEDS = [0, 1, 2, 3, 4]
N_POINTS = 200
HEADS = ["square", "softplus", "relu_square"]
DOMAINS = [(0.0, 1.0), (-1.0, 7.0), (-2.5, 3.5)]


@pytest.mark.parametrize("omega_bounds", DOMAINS)
def test_state_net_boundary_conditions_exact(omega_bounds):
    for seed in SEEDS:
        torch.manual_seed(seed)
        net = StateNet(omega_bounds, width=32, depth=3)

        generator = torch.Generator().manual_seed(seed + 100)
        t = torch.rand(N_POINTS, 1, generator=generator, dtype=torch.float64)
        a, b = omega_bounds

        left = net(t, torch.full_like(t, a))
        right = net(t, torch.full_like(t, b))

        assert (left == 0.0).all()
        assert (right == 0.0).all()


def test_state_net_boundary_conditions_stefan_problem_bounds():
    problem = StefanProblem()
    for seed in SEEDS:
        torch.manual_seed(seed)
        net = StateNet(problem.omega_bounds, width=32, depth=3)

        generator = torch.Generator().manual_seed(seed + 100)
        t = torch.rand(N_POINTS, 1, generator=generator, dtype=torch.float64)

        left = net(t, torch.zeros_like(t))
        right = net(t, torch.ones_like(t))

        assert (left == 0.0).all()
        assert (right == 0.0).all()


@pytest.mark.parametrize("head", HEADS)
def test_multiplier_head_nonnegative_everywhere(head):
    for seed in SEEDS:
        torch.manual_seed(seed)
        net = MultiplierNet(width=32, depth=3, head=head)

        generator = torch.Generator().manual_seed(seed + 200)
        t = torch.rand(N_POINTS, 1, generator=generator, dtype=torch.float64)
        x = torch.rand(N_POINTS, 1, generator=generator, dtype=torch.float64)

        lam = net(t, x)
        assert (lam >= 0.0).all()
