"""v14 A3 (C_rel per problem, C_eq from formula), A4 (E_h split by component),
A7 (projection admissibility), A8 (Lambda recorded), Stage 2 formulas."""

import math
from fractions import Fraction

import pytest
import sympy as sp
import torch

from certified_obstacle_nn.certification import reliability as rel
from certified_obstacle_nn.certification.certificate import certify
from certified_obstacle_nn.certification.partition import build_partition
from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.problems.american_put import AmericanPutProblem
from certified_obstacle_nn.problems.stefan import StefanProblem


def _triple(seed=0, w=12, d=2):
    torch.manual_seed(seed)
    tr = NeuralTriple(StefanProblem(), width=w, depth=d).double().eval()
    for q in tr.parameters():
        q.requires_grad_(False)
    return tr


def test_C_rel_is_per_problem():
    assert rel.C_rel(StefanProblem()) == 36
    assert rel.C_rel(AmericanPutProblem()) == 144
    assert rel.HISTORICAL_C_REL == 216


def test_C_eq_from_formula():
    assert rel.C_eq(StefanProblem()) == 2.0

    class Disk:  # alpha=1, Lambda=2, A=I, b=c=0 (roadmap section 0 table)
        alpha, Lambda, A_inf, b, c = 1.0, 2.0, 1.0, 0.0, 0.0
    assert rel.C_eq(Disk()) == 1.0
    ap = AmericanPutProblem()
    expected = max(1 / ap.alpha + 4 * (ap.b**2 + ap.Lambda * ap.c**2) / ap.alpha, ap.A_inf, 1.0, 2 * ap.alpha / ap.Lambda)
    assert rel.C_eq(ap) == pytest.approx(expected, rel=1e-15)
    assert rel.C_eq(ap) >= expected


def test_component_split_sums_to_old_total():
    p = StefanProblem()
    tr = _triple()
    part = build_partition(p, 12, 12)
    r = certify(tr, p, part)
    c = r.components
    old_total = r.rho_tilde_h**2 + r.remainder.E_h
    new_total = c["U_r"] + c["U_p"] + c["U_c"]
    assert new_total == pytest.approx(old_total, rel=1e-12)
    assert c["E_h_r"] + c["E_h_p"] + c["E_h_c"] == pytest.approx(r.remainder.E_h, rel=1e-12)
    assert min(c["E_h_r"], c["E_h_p"], c["E_h_c"]) >= 0
    assert c["status"] == "unvalidated_enclosure"
    assert r.C_rel == 216.0 and r.C_rel_v14 == 36.0
    assert r.E_cert_U_v14 == pytest.approx(r.E_cert_U * math.sqrt(36 / 216), rel=1e-12)


def test_weighted_and_delta_formulas():
    p = StefanProblem()
    Ur, Up, Uc = 0.01, 0.04, 0.002
    b = rel.all_bounds(p, Ur, Up, Uc, status="unvalidated_enclosure")
    B = 18 * Ur + 36 * Up + 12 * Uc
    assert b["B_weighted"] == pytest.approx(B, rel=1e-15) and b["B_weighted"] >= B
    assert b["E_U_weighted"] == pytest.approx(math.sqrt(2 * B), rel=1e-15)
    assert b["E_U_uniform"] == pytest.approx(math.sqrt(2 * 36 * (Ur + Up + Uc)), rel=1e-15)
    assert b["E_U_uniform_216_comparison_convention"] == pytest.approx(math.sqrt(2 * 216 * (Ur + Up + Uc)), rel=1e-15)
    # heat delta family at 1/6 coincides with the weighted B
    assert b["B_delta_1_6"] == pytest.approx(B, rel=1e-14)
    # minimiser beats both fixed deltas and is inside the interval
    d = Fraction(b["delta_star_rational"])
    assert 0 < d < Fraction(1, 3)
    assert b["B_delta_star"] <= min(b["B_delta_1_12"], b["B_delta_1_6"]) * (1 + 1e-12)
    # closed form agrees with a brute-force 1-D minimisation
    grid = [i / 100000 for i in range(1, 33333)]
    F = [(9 * Ur + 6 * Uc + 3 * Up / x) / (1 - 3 * x) for x in grid]
    assert b["B_delta_star"] <= min(F) * (1 + 1e-6)
    assert b["grad_bound"] == pytest.approx(math.sqrt(B), rel=1e-15)
    assert b["flux_bound"] == pytest.approx(math.sqrt(B), rel=1e-15)


def test_general_delta_closed_form():
    Ur, Up, Uc = 0.3, 0.05, 0.01
    ds = rel.delta_star(Ur, Up, Uc, diffusion_only=False)
    grid = [i / 200000 for i in range(1, 33333)]
    F = lambda x: (9 * Ur + 6 * Uc + 6 * Up / x) / (1 - 6 * x)
    best = min(grid, key=F)
    assert ds == pytest.approx(best, abs=2e-5)
    # general B at delta = 1/12 equals 18 U_r + 144 U_p + 12 U_c
    Bd = rel.B_delta_arb(rel.a(Ur), rel.a(Up), rel.a(Uc), Fraction(1, 12), False)
    assert rel.up(Bd) == pytest.approx(18 * Ur + 144 * Up + 12 * Uc, rel=1e-14)
    assert not rel.B_delta_arb(rel.a(Ur), rel.a(Up), rel.a(Uc), Fraction(1, 6), False).is_finite()


def test_projection_bounds_and_status_propagation():
    p = StefanProblem()
    b = rel.all_bounds(p, 0.01, 0.04, 0.002, status="validated_upper_bound", U_r_hat=0.02)
    assert b["E_U_projected"] == pytest.approx(math.sqrt(6 * 2 * (0.02 + 0.002)), rel=1e-15)
    assert b["E_U_original_via_projection"] == pytest.approx(
        math.sqrt(2) * (math.sqrt(6 * 0.022) + math.sqrt(0.04)), rel=1e-15)
    assert b["status"] == "validated_upper_bound"
    bad = rel.all_bounds(p, float("nan"), 0.04, 0.002, status="validated_upper_bound")
    assert bad["status"] != "validated_upper_bound"
    gen = rel.all_bounds(AmericanPutProblem(), 0.01, 0.04, 0.002, status="unvalidated_enclosure", U_r_hat=0.02)
    assert gen["E_U_original_via_projection"] == "NOT_IMPLEMENTED"


def test_outward_rounding_helpers():
    x = rel.a(2).sqrt()
    assert rel.up(x) >= math.sqrt(2) and rel.down(x) <= math.sqrt(2)
    assert rel.up(x) > rel.down(x)


# ---------------------------------------------------------------- A7, A8
def test_stefan_obstacle_nonpositive_on_lateral_boundary_symbolic():
    t = sp.symbols("t", nonnegative=True)
    h = sp.exp(t) - t - 1
    g0 = -h * (1 - 0)
    g1 = -h * (1 - 1)
    assert sp.simplify(g1) == 0
    # h >= 0 on [0,1]: h(0)=0, h' = e^t - 1 >= 0
    assert h.subs(t, 0) == 0
    assert sp.solve(sp.Eq(sp.diff(h, t), 0), t) == [0]
    assert all(float(g0.subs(t, tv)) <= 0 for tv in [i / 50 for i in range(51)])


def test_projection_satisfies_lateral_and_initial_conditions():
    p = StefanProblem()
    tr = _triple(3)
    tt = torch.linspace(0, 1, 101, dtype=torch.float64).reshape(-1, 1)
    with torch.no_grad():
        for xb in (0.0, 1.0):
            x = torch.full_like(tt, xb)
            v = tr.state_net(tt, x)
            assert torch.all(v == 0.0)  # mask vanishes bit-for-bit
            vhat = torch.maximum(v, p.g(tt, x))
            assert torch.all(vhat == 0.0)
        x = torch.linspace(0, 1, 201, dtype=torch.float64).reshape(-1, 1)
        z = torch.zeros_like(x)
        assert torch.all(p.g(z, x) == 0.0)
        v0 = tr.state_net(z, x)
        assert torch.equal(torch.maximum(v0, p.g(z, x)), torch.clamp(v0, min=0.0))


def test_lambda_and_constants_recorded():
    rec = rel.problem_constants_record(StefanProblem())
    for k in ("alpha", "Lambda", "A_norm_upper", "b_norm_upper", "c_norm_upper", "kappa", "C_eq", "C_rel",
              "reliability_formula", "Lambda_justification"):
        assert k in rec
    assert rec["Lambda"] == 1.0 and rec["C_eq"] == 2.0 and rec["kappa"] == 1.0 and rec["C_rel"] == 36
    assert "diam" in rec["Lambda_justification"]
