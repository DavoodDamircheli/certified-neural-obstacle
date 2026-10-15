"""v14 Stage 1: component evaluator."""

import types

import pytest
import torch

from certified_obstacle_nn.certification.components import components
from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.problems.stefan import StefanProblem


def test_exact_solution_has_zero_components():
    p = StefanProblem()
    exact = types.SimpleNamespace(state_net=p.u_exact, flux_net=p.sigma_exact, multiplier_net=p.lambda_exact)
    c = components(p, exact, n=32, q=3)
    for k in ("R_term", "S_term", "init_mismatch", "gradw", "w_term", "wt_term", "w0", "wT", "compl",
              "Rhat_term", "Shat_term", "init_mismatch_hat"):
        assert abs(c[k]) < 1e-26, (k, c[k])
    assert c["status"] == "diagnostic_float64"


def test_projection_of_state_below_obstacle():
    """v = g - 1 (constant shift below the obstacle): w = 1 everywhere, so
    w_t = w_x = 0, R_hat = R, S_hat = S, v_hat = g, and ||w||^2 = |Q|."""
    p = StefanProblem()

    def v(t, x):
        return p.g(t, x) - 1.0

    trip = types.SimpleNamespace(state_net=v, flux_net=p.sigma_exact, multiplier_net=p.lambda_exact)
    c = components(p, trip, n=16, q=4)
    assert c["w_term"] == pytest.approx(1.0, rel=1e-12)
    assert c["w0"] == pytest.approx(1.0, rel=1e-12) and c["wT"] == pytest.approx(1.0, rel=1e-12)
    assert c["gradw"] < 1e-26 and c["wt_term"] < 1e-26
    assert c["Rhat_term"] == pytest.approx(c["R_term"], rel=1e-12)
    assert c["Shat_term"] == pytest.approx(c["S_term"], rel=1e-12)
    assert c["init_mismatch_hat"] < 1e-26  # v_hat(0) = g(0) = 0 = u0
    assert c["compl"] == 0.0


def test_component_sums_and_weights():
    torch.manual_seed(0)
    p = StefanProblem()
    tr = NeuralTriple(p, width=8, depth=2).double().eval()
    c = components(p, tr, n=16, q=3)
    assert c["r2"] == pytest.approx(c["R_term"] + c["S_term"] + c["init_mismatch"])
    assert c["p2"] == pytest.approx(c["gradw"] + c["w_term"] + c["wt_term"] + c["w0"] + c["wT"])
    assert c["weights"]["R_term"] == 1.0 and c["weights"]["w_term"] == 1.0
    assert len(c["partition_hash"]) == 64
