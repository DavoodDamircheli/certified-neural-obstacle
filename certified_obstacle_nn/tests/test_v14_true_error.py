"""v14 A1: the true-error graph term equals R_PDE pointwise for Stefan, so the
global Gauss rule converges and the unchanged 1e-3 self-check passes."""

from pathlib import Path

import numpy as np
import pytest
import torch

from certified_obstacle_nn.autograd.derivatives import eval_derivatives_detached
from certified_obstacle_nn.errors.true_error import compute_true_error
from certified_obstacle_nn.errors.true_error_v14 import converged_true_error, true_error_components
from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.problems.stefan import StefanProblem

REPO = Path(__file__).resolve().parents[2]
SEED0 = REPO / "results/stefan_m18/4x64/seed0/weights.pt"


def _random_triple(seed=0):
    torch.manual_seed(seed)
    p = StefanProblem()
    tr = NeuralTriple(p, width=16, depth=2).double().eval()
    for q in tr.parameters():
        q.requires_grad_(False)
    return p, tr


def _seed0():
    p = StefanProblem()
    tr = NeuralTriple(p, width=64, depth=4).double()
    tr.load_state_dict(torch.load(SEED0, map_location="cpu"))
    tr.eval()
    for q in tr.parameters():
        q.requires_grad_(False)
    return p, tr


def _gauss_tensor(n):
    z, _ = np.polynomial.legendre.leggauss(n)
    z = 0.5 * (z + 1)
    T, X = np.meshgrid(z, z, indexing="ij")
    return (torch.tensor(T.reshape(-1, 1)), torch.tensor(X.reshape(-1, 1)))


def test_exact_stefan_residual_vanishes_including_diagonal():
    p = StefanProblem()
    t, x = _gauss_tensor(41)  # identical t/x nodes: 41 points lie exactly on t == x
    assert int((t == x).sum()) == 41
    u, s, lam, u_t, u_x, s_x = eval_derivatives_detached(p.u_exact, p.sigma_exact, p.lambda_exact, t, x)
    assert float((u_t + s_x - lam - p.f(t, x)).abs().max()) < 1e-14


def test_contact_convention_consistent_on_diagonal():
    p = StefanProblem()
    t = torch.tensor([[0.25], [0.5]], dtype=torch.float64)
    assert torch.all(p.lambda_exact(t, t.clone()) == 1.0)


@pytest.mark.parametrize("n", [20, 41])
def test_graph_integrand_equals_R_PDE_pointwise(n):
    p, tr = _random_triple()
    t, x = _gauss_tensor(n)
    v, tau, mu, v_t, v_x, tau_x = eval_derivatives_detached(tr.state_net, tr.flux_net, tr.multiplier_net, t, x)
    u, s, lam, u_t, u_x, s_x = eval_derivatives_detached(p.u_exact, p.sigma_exact, p.lambda_exact, t, x)
    graph = (u_t - v_t) + (s_x - tau_x) - (lam - mu)
    r_pde = p.f(t, x) - v_t - tau_x + mu
    assert float((graph - r_pde).abs().max()) < 1e-13


def test_duffy_and_global_graph_terms_agree():
    p, tr = _random_triple(1)
    d = true_error_components(p, tr, 48, "duffy")
    g = true_error_components(p, tr, 48, "global")
    assert d["graph"] == pytest.approx(g["graph"], rel=1e-10)
    assert d["init"] == pytest.approx(g["init"], rel=1e-12)
    # grad/flux have C^0 kinks at t=x under the global rule: algebraic agreement only
    assert d["grad"] == pytest.approx(g["grad"], rel=1e-2)
    assert d["flux"] == pytest.approx(g["flux"], rel=1e-2)


def test_duffy_matches_existing_split_rule():
    p, tr = _random_triple(2)
    d = true_error_components(p, tr, 40, "duffy")
    old = compute_true_error(p, tr, 40, 40, free_boundary_fn=p.free_boundary_exact)
    assert d["E_true"] == pytest.approx(old.E_true, rel=1e-10)
    assert d["graph"] == pytest.approx(old.term_consistency, rel=1e-10)


@pytest.mark.skipif(not SEED0.exists(), reason="seed-0 checkpoint not present")
def test_global_rule_self_check_passes_for_stefan_at_practical_resolution():
    p, tr = _seed0()
    r = compute_true_error(p, tr, 60, 60, quadrature_tolerance=1e-3)  # global rule, unchanged tolerance
    assert r.quadrature_relative_change < 1e-3
    c = converged_true_error(p, tr, 48, "duffy")
    assert c["rel_change_E_true"] < 1e-12
    assert r.E_true == pytest.approx(c["E_true"], rel=1e-3)
