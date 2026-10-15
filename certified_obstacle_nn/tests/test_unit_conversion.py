"""Tests for models/unit_conversion.py's PhysicalUnitsTriple."""

import pytest

pytestmark = pytest.mark.legacy

import torch

from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.models.unit_conversion import PhysicalUnitsTriple
from certified_obstacle_nn.problems.american_put import AmericanPutProblem


def test_physical_units_triple_scales_all_three_outputs_by_K():
    torch.manual_seed(0)
    problem = AmericanPutProblem()
    base = NeuralTriple(problem, width=8, depth=2, x_rescale=problem.xi)
    K = problem.K
    wrapped = PhysicalUnitsTriple(base, K)

    t = torch.rand(20, 1, dtype=torch.float64) * problem.T
    x = torch.rand(20, 1, dtype=torch.float64) * (problem.R - problem.L) + problem.L

    with torch.no_grad():
        u_base, sigma_base, lam_base = base.state_net(t, x), base.flux_net(t, x), base.multiplier_net(t, x)
        u_phys, sigma_phys, lam_phys = wrapped.state_net(t, x), wrapped.flux_net(t, x), wrapped.multiplier_net(t, x)

    assert torch.allclose(u_phys, K * u_base)
    assert torch.allclose(sigma_phys, K * sigma_base)
    assert torch.allclose(lam_phys, K * lam_base)


def test_physical_units_triple_delegates_eval_and_parameters():
    torch.manual_seed(0)
    problem = AmericanPutProblem()
    base = NeuralTriple(problem, width=8, depth=2, x_rescale=problem.xi)
    wrapped = PhysicalUnitsTriple(base, problem.K)

    wrapped.eval()  # must not raise
    assert list(wrapped.parameters()) == list(base.parameters())
