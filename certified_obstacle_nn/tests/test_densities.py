"""Tests for certification/densities.py.

The central check: 0 <= phi_interface <= psi_majorant pointwise, from
grad_x w = 1_{gamma>0} grad_x gamma, d_t w = 1_{gamma>0} d_t gamma (specification
section 5) -- so phi_interface = 1_{gamma>0} * psi_majorant exactly. A
violation means grad w / d_t w (the autograd chain through the relu) is being
computed wrongly, not just imprecisely.
"""

import torch

from certified_obstacle_nn.certification.densities import phi_interface, phi_smooth, psi_majorant
from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.problems.stefan import StefanProblem


def _sample(n, seed):
    generator = torch.Generator().manual_seed(seed)
    t = torch.rand(n, 1, generator=generator, dtype=torch.float64, requires_grad=True)
    x = torch.rand(n, 1, generator=generator, dtype=torch.float64, requires_grad=True)
    return t, x


def test_phi_interface_bounded_by_psi_majorant_pointwise():
    problem = StefanProblem()
    torch.manual_seed(0)
    triple = NeuralTriple(problem, width=24, depth=3)

    t, x = _sample(20000, seed=0)

    phi_int = phi_interface(problem, triple, t, x)
    psi = psi_majorant(problem, triple, t, x)

    assert (phi_int >= -1e-12).all()
    assert (psi >= -1e-12).all()

    violation = (phi_int - psi).clamp(min=0.0)
    max_violation = violation.max().item()
    print(f"max(phi_interface - psi_majorant) = {max_violation:.3e}")
    assert max_violation < 1e-10


def test_phi_interface_equals_indicator_times_psi():
    """A stronger, exact version of the same fact: phi_interface should equal
    psi_majorant exactly where gamma > 0, and be exactly 0 where gamma <= 0
    (w = relu(gamma) is identically 0 there, so its derivatives are 0 too).
    """
    problem = StefanProblem()
    torch.manual_seed(1)
    triple = NeuralTriple(problem, width=24, depth=3)

    t, x = _sample(20000, seed=3)

    with torch.no_grad():
        u = triple.state_net(t, x)
        gamma = problem.g(t, x) - u

    phi_int = phi_interface(problem, triple, t, x)
    psi = psi_majorant(problem, triple, t, x)

    active = (gamma > 0).squeeze(-1)
    inactive = ~active

    max_diff_on_active = (phi_int[active] - psi[active]).abs().max().item() if active.any() else 0.0
    max_phi_int_on_inactive = phi_int[inactive].abs().max().item() if inactive.any() else 0.0

    print(f"max |phi_interface - psi_majorant| where gamma>0: {max_diff_on_active:.3e}")
    print(f"max |phi_interface| where gamma<=0: {max_phi_int_on_inactive:.3e}")

    assert max_diff_on_active < 1e-10
    assert max_phi_int_on_inactive < 1e-10


def test_phi_smooth_is_nonnegative():
    # Every term in Phi_sm is individually nonnegative by construction
    # (squares, plus lambda_psi*gamma_minus, a product of two nonnegative
    # quantities), so the sum is nonnegative too.
    problem = StefanProblem()
    torch.manual_seed(2)
    triple = NeuralTriple(problem, width=16, depth=2)

    t, x = _sample(5000, seed=1)
    phi_sm = phi_smooth(problem, triple, t, x)
    assert (phi_sm >= -1e-12).all()


def test_functions_are_independently_callable():
    """phi_smooth, phi_interface, psi_majorant must each be independently
    callable in isolation -- none should depend on the others having been
    called first, and none should share mutable state between calls."""
    problem = StefanProblem()
    torch.manual_seed(4)
    triple = NeuralTriple(problem, width=16, depth=2)

    t, x = _sample(500, seed=2)

    psi_first = psi_majorant(problem, triple, t, x)
    phi_int_alone = phi_interface(problem, triple, t, x)
    phi_sm_alone = phi_smooth(problem, triple, t, x)
    psi_again = psi_majorant(problem, triple, t, x)

    assert phi_sm_alone.shape == t.shape
    assert phi_int_alone.shape == t.shape
    assert psi_first.shape == t.shape
    assert torch.equal(psi_first, psi_again)  # deterministic, no shared/mutated state
