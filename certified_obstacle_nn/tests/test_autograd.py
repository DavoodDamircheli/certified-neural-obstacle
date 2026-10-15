"""Tests for autograd/derivatives.py.

Order matters: first validate compute_derivatives against manufactured
closed-form functions (monkeypatching the networks entirely), so a bug in the
derivative-extraction routine itself cannot hide behind network complexity.
Only then check the real, randomly initialized networks against central
finite differences. Finally, verify requires_grad / create_graph / no-detach
end to end, since a mistake there silently corrupts every residual and loss
built downstream without necessarily raising an error.
"""

import math

import pytest
import torch

from certified_obstacle_nn.autograd.derivatives import compute_derivatives, eval_triple_derivatives
from certified_obstacle_nn.models.flux_net import FluxNet
from certified_obstacle_nn.models.multiplier_net import MultiplierNet
from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.models.state_net import StateNet
from certified_obstacle_nn.problems.stefan import StefanProblem

N_POINTS = 500
TOL_ANALYTIC = 1e-10
TOL_FD = 1e-5
FD_H = 1e-6


def _sample_points(seed):
    generator = torch.Generator().manual_seed(seed)
    t = torch.rand(N_POINTS, 1, generator=generator, dtype=torch.float64, requires_grad=True)
    x = torch.rand(N_POINTS, 1, generator=generator, dtype=torch.float64, requires_grad=True)
    return t, x


def _trivial_lambda(t, x):
    return torch.ones_like(t)


MANUFACTURED_CASES = [
    pytest.param(
        lambda t, x: t ** 2 + x ** 2,
        lambda t, x: 2 * t,
        lambda t, x: 2 * x,
        id="poly",
    ),
    pytest.param(
        lambda t, x: torch.exp(t - x),
        lambda t, x: torch.exp(t - x),
        lambda t, x: -torch.exp(t - x),
        id="exp",
    ),
    pytest.param(
        lambda t, x: torch.sin(math.pi * x) * torch.exp(t),
        lambda t, x: torch.sin(math.pi * x) * torch.exp(t),
        lambda t, x: math.pi * torch.cos(math.pi * x) * torch.exp(t),
        id="trig",
    ),
]


@pytest.mark.parametrize("fn,dt_fn,dx_fn", MANUFACTURED_CASES)
def test_derivatives_match_manufactured_analytic_functions(fn, dt_fn, dx_fn):
    t, x = _sample_points(seed=0)

    u, sigma, lam, u_t, u_x, sigma_x = compute_derivatives(fn, fn, _trivial_lambda, t, x)

    assert torch.equal(u, fn(t, x))
    assert torch.equal(sigma, fn(t, x))

    max_u_t = (u_t - dt_fn(t, x)).abs().max().item()
    max_u_x = (u_x - dx_fn(t, x)).abs().max().item()
    max_sigma_x = (sigma_x - dx_fn(t, x)).abs().max().item()
    print(f"max |u_t - exact| = {max_u_t:.3e}")
    print(f"max |u_x - exact| = {max_u_x:.3e}")
    print(f"max |sigma_x - exact| = {max_sigma_x:.3e}")

    assert max_u_t < TOL_ANALYTIC
    assert max_u_x < TOL_ANALYTIC
    assert max_sigma_x < TOL_ANALYTIC


def test_derivatives_thread_chain_rule_through_internal_input_rescaling():
    """Guards the specification section 10 pattern: a network may internally remap
    x -> xi before its nonlinearity, but derivatives must still be taken w.r.t.
    the physical x, with autograd supplying the chain-rule factor unaided.
    """

    def rescaled_sin(t, x):
        xi = 2.0 * x - 1.0
        return torch.sin(xi) + 0.0 * t  # trivial t-dependence: compute_derivatives always differentiates w.r.t. both

    t, x = _sample_points(seed=1)

    _, sigma, _, _, u_x, sigma_x = compute_derivatives(rescaled_sin, rescaled_sin, _trivial_lambda, t, x)

    expected_dx = 2.0 * torch.cos(2.0 * x - 1.0)
    assert (u_x - expected_dx).abs().max().item() < TOL_ANALYTIC
    assert (sigma_x - expected_dx).abs().max().item() < TOL_ANALYTIC


def test_real_networks_match_central_finite_differences():
    problem = StefanProblem()
    torch.manual_seed(0)
    state_net = StateNet(problem.omega_bounds, width=32, depth=3)
    flux_net = FluxNet(width=32, depth=3)
    multiplier_net = MultiplierNet(width=32, depth=3)

    t, x = _sample_points(seed=2)

    _, _, _, u_t, u_x, sigma_x = compute_derivatives(state_net, flux_net, multiplier_net, t, x)

    with torch.no_grad():
        u_t_fd = (state_net(t + FD_H, x) - state_net(t - FD_H, x)) / (2 * FD_H)
        u_x_fd = (state_net(t, x + FD_H) - state_net(t, x - FD_H)) / (2 * FD_H)
        sigma_x_fd = (flux_net(t, x + FD_H) - flux_net(t, x - FD_H)) / (2 * FD_H)

    max_u_t = (u_t.detach() - u_t_fd).abs().max().item()
    max_u_x = (u_x.detach() - u_x_fd).abs().max().item()
    max_sigma_x = (sigma_x.detach() - sigma_x_fd).abs().max().item()
    print(f"max |u_t - FD| = {max_u_t:.3e}")
    print(f"max |u_x - FD| = {max_u_x:.3e}")
    print(f"max |sigma_x - FD| = {max_sigma_x:.3e}")

    assert max_u_t < TOL_FD
    assert max_u_x < TOL_FD
    assert max_sigma_x < TOL_FD


def test_requires_grad_and_no_detach_along_path():
    problem = StefanProblem()
    torch.manual_seed(3)
    state_net = StateNet(problem.omega_bounds, width=16, depth=2)
    flux_net = FluxNet(width=16, depth=2)
    multiplier_net = MultiplierNet(width=16, depth=2)

    t, x = _sample_points(seed=3)
    assert t.requires_grad and x.requires_grad
    assert t.is_leaf and x.is_leaf

    u, sigma, lam, u_t, u_x, sigma_x = compute_derivatives(state_net, flux_net, multiplier_net, t, x)

    # inputs must be untouched by the call
    assert t.requires_grad and x.requires_grad
    assert t.is_leaf and x.is_leaf

    # outputs must be part of the graph, not detached
    for name, tensor in [("u", u), ("sigma", sigma), ("lam", lam)]:
        assert tensor.requires_grad, f"{name} lost requires_grad"
        assert tensor.grad_fn is not None, f"{name} appears detached (no grad_fn)"

    # first derivatives must remain differentiable (create_graph=True), so a
    # training loss built from them can still backpropagate into theta/phi/psi
    for name, tensor in [("u_t", u_t), ("u_x", u_x), ("sigma_x", sigma_x)]:
        assert tensor.requires_grad, f"{name} lost requires_grad (create_graph missing?)"
        assert tensor.grad_fn is not None, f"{name} appears detached (no grad_fn)"

    # end-to-end: a synthetic loss built from the derivatives must be able to
    # backpropagate all the way into every network parameter.
    loss = (
        u_t.pow(2).mean()
        + u_x.pow(2).mean()
        + sigma_x.pow(2).mean()
        + u.pow(2).mean()
        + sigma.pow(2).mean()
        + lam.mean()
    )
    loss.backward()

    any_nonzero = False
    for net in (state_net, flux_net, multiplier_net):
        for p in net.parameters():
            assert p.grad is not None, "parameter received no gradient: graph is broken somewhere"
            if p.grad.abs().max().item() > 0:
                any_nonzero = True
    assert any_nonzero, "all parameter gradients are exactly zero -- suspicious"


def _frozen_triple(seed):
    problem = StefanProblem()
    torch.manual_seed(seed)
    triple = NeuralTriple(problem, width=16, depth=2)
    triple.eval()
    for p in triple.parameters():
        p.requires_grad_(False)
    return triple


def test_eval_triple_derivatives_matches_training_path_to_1e_14():
    """The memory optimization (create_graph=False, single grad() call for
    u_t/u_x, no torch.no_grad()) must be numerically inert: same five values
    as compute_derivatives (create_graph=True), to near machine precision."""
    triple = _frozen_triple(seed=7)
    t, x = _sample_points(seed=7)

    u_train, sigma_train, lam_train, u_t_train, u_x_train, sigma_x_train = compute_derivatives(
        triple.state_net, triple.flux_net, triple.multiplier_net, t, x
    )
    u_cert, sigma_cert, lam_cert, u_t_cert, u_x_cert, sigma_x_cert = eval_triple_derivatives(triple, t, x)

    pairs = [
        ("u", u_train, u_cert),
        ("sigma", sigma_train, sigma_cert),
        ("lam", lam_train, lam_cert),
        ("u_t", u_t_train, u_t_cert),
        ("u_x", u_x_train, u_x_cert),
        ("sigma_x", sigma_x_train, sigma_x_cert),
    ]
    for name, train_val, cert_val in pairs:
        max_abs_diff = (train_val.detach() - cert_val).abs().max().item()
        print(f"max |{name}_train - {name}_cert| = {max_abs_diff:.3e}")
        assert max_abs_diff < 1e-14, f"{name}: training-path and certification-path derivatives disagree"


def test_eval_triple_derivatives_returns_detached_leaves_no_graph_survives():
    triple = _frozen_triple(seed=8)
    t, x = _sample_points(seed=8)

    u, sigma, lam, u_t, u_x, sigma_x = eval_triple_derivatives(triple, t, x)

    for name, tensor in [("u", u), ("sigma", sigma), ("lam", lam), ("u_t", u_t), ("u_x", u_x), ("sigma_x", sigma_x)]:
        assert not tensor.requires_grad, f"{name} still requires_grad: a graph survived eval_triple_derivatives"
        assert tensor.grad_fn is None, f"{name} still has a grad_fn: not actually detached"

    # the caller's original t, x must be untouched (still leaves, still requiring grad)
    assert t.requires_grad and x.requires_grad
    assert t.is_leaf and x.is_leaf


def test_eval_triple_derivatives_rejects_unfrozen_model():
    problem = StefanProblem()
    torch.manual_seed(9)
    triple = NeuralTriple(problem, width=8, depth=2)  # parameters default requires_grad=True, NOT frozen
    t, x = _sample_points(seed=9)

    with pytest.raises(AssertionError):
        eval_triple_derivatives(triple, t, x)
