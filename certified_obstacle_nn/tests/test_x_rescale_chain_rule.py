"""Chain-rule verification for the x_rescale hook (models/state_net.py,
flux_net.py, multiplier_net.py), specification section 10: "feed the network
xi = 2(x-L)/(R-L)-1 but differentiate w.r.t. PHYSICAL x so autograd supplies
the chain-rule factor."

Direct evidence, not a claim taken on faith: autograd's d(output)/dx (with
x_rescale=problem.xi wired in) is compared against a CENTRAL FINITE
DIFFERENCE of the SAME network, perturbing PHYSICAL x directly -- these two
computations share no code path (one is symbolic/autograd, the other
black-box function evaluation), so agreement is real evidence the chain
rule is correctly threaded through, not a self-fulfilling check.
"""

import pytest
import torch

from certified_obstacle_nn.models.flux_net import FluxNet
from certified_obstacle_nn.models.multiplier_net import MultiplierNet
from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.models.state_net import StateNet
from certified_obstacle_nn.problems.american_put import AmericanPutProblem

pytestmark = pytest.mark.legacy

FD_H = 1e-6
N_POINTS = 500


def _sample_interior(problem, n, seed):
    generator = torch.Generator().manual_seed(seed)
    a, b = problem.omega_bounds
    margin = 1e-3 * (b - a)
    t = torch.rand(n, 1, generator=generator, dtype=torch.float64) * problem.T
    x = torch.rand(n, 1, generator=generator, dtype=torch.float64) * (b - a - 2 * margin) + a + margin
    return t, x


def _autograd_dx(net, t, x):
    x = x.clone().requires_grad_(True)
    out = net(t, x)
    (grad,) = torch.autograd.grad(out, x, grad_outputs=torch.ones_like(out))
    return grad.detach()


def _finite_diff_dx(net, t, x, h=FD_H):
    with torch.no_grad():
        return (net(t, x + h) - net(t, x - h)) / (2 * h)


@pytest.mark.parametrize("net_factory", [
    lambda problem: StateNet(problem.omega_bounds, width=16, depth=2, x_rescale=problem.xi),
    lambda problem: FluxNet(width=16, depth=2, x_rescale=problem.xi),
    lambda problem: MultiplierNet(width=16, depth=2, x_rescale=problem.xi),
])
def test_autograd_dx_matches_finite_difference_in_physical_x(net_factory):
    torch.manual_seed(0)
    problem = AmericanPutProblem()
    net = net_factory(problem)
    t, x = _sample_interior(problem, N_POINTS, seed=1)

    grad_ad = _autograd_dx(net, t, x)
    grad_fd = _finite_diff_dx(net, t, x)

    assert torch.allclose(grad_ad, grad_fd, atol=1e-5, rtol=1e-4), (
        f"max abs diff = {(grad_ad - grad_fd).abs().max().item():.3e}"
    )


def test_gradient_is_not_accidentally_scaled_by_missing_chain_rule_factor():
    """A deliberately BROKEN x_rescale (detached from the autograd graph,
    e.g. computed via a plain Python float / .item() round-trip) would make
    d(output)/dx come out as if x_rescale were the identity -- i.e. it would
    be off by exactly the factor d(xi)/dx = 2/(R-L). This test builds that
    broken variant explicitly and confirms it disagrees with the correct
    (autograd-graph-preserving) one -- proof the test above is actually
    sensitive to the chain-rule factor, not vacuously passing regardless."""
    torch.manual_seed(0)
    problem = AmericanPutProblem()
    correct_net = StateNet(problem.omega_bounds, width=16, depth=2, x_rescale=problem.xi)

    def broken_rescale(x):
        # detaches xi from the graph -- forward VALUE is identical, but
        # autograd sees x_rescale(x) as a CONSTANT w.r.t. x.
        return problem.xi(x).detach()

    broken_net = StateNet(problem.omega_bounds, width=16, depth=2, x_rescale=broken_rescale)
    broken_net.load_state_dict(correct_net.state_dict())  # SAME weights, only the rescale differs

    t, x = _sample_interior(problem, N_POINTS, seed=2)
    grad_correct = _autograd_dx(correct_net, t, x)
    grad_broken = _autograd_dx(broken_net, t, x)

    # forward values must be identical (broken_rescale only detaches, doesn't
    # change the VALUE) -- this isolates the difference to the GRADIENT path.
    with torch.no_grad():
        assert torch.allclose(correct_net(t, x), broken_net(t, x), atol=1e-12)

    assert not torch.allclose(grad_correct, grad_broken, atol=1e-3), (
        "broken (detached) x_rescale produced the SAME gradient as the correct "
        "one -- this test is not actually sensitive to the chain-rule factor"
    )
