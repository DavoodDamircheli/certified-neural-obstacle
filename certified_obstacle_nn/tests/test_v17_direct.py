"""v17 Stage 1 (roadmap R3): per-term split of the cell majorant and the direct bounds of reliability.py."""

import json
import math
from fractions import Fraction

import numpy as np
import pytest
from flint import arb

from certified_obstacle_nn.certification import reliability as rel
from certified_obstacle_nn.certification.enclose import boxes as Bx
from certified_obstacle_nn.certification.enclose import upper_sum as US
from certified_obstacle_nn.certification.enclose.engine import StefanAdapter, TripleEncloser
from certified_obstacle_nn.experiments.v14.common import REPO_ROOT, STEFAN_SEEDS, load_frozen_stefan
from certified_obstacle_nn.tests.stored_data import needs

V16 = REPO_ROOT / "docs/v16_reference/v16/data"
DISK_LONG_1E5 = "results/v14_disk_validated/20261007T212554Z"


# ------------------------------------------------------------------ constants
@needs('docs/v16_reference/v16/data')
def test_cF2_upper_matches_v16_and_brackets_truth():
    ref = json.loads((V16 / "friedrichs_constants.json").read_text())
    for d, key in ((1, "stefan"), (2, "disk2d")):
        c = rel.cF2_upper(d)
        assert c["cF2_up"] == ref[key]
        lo, hi = c["bracket"]
        assert lo <= 1 / (d * math.pi**2) <= hi and hi == c["cF2_up"]
        assert arb(c["cF2_up"]) >= 1 / (d * arb.pi() ** 2)


@needs('docs/v16_reference/v16/data')
def test_reference_norms_match_v16():
    for key, f in (("stefan", "stefan_reference.json"), ("disk2d", "disk_reference.json")):
        ref = json.loads((V16 / f).read_text())
        n = rel.reference_norm(key)
        assert (n["norm_lower"], n["norm_upper"]) == (ref["norm_lower"], ref["norm_upper"])
    with pytest.raises(KeyError):
        rel.reference_norm("american_put")


@needs('docs/v16_reference/v16/data')
def test_ellipse_reference_norm_closed_form():
    """v17 closed form (experiments/v17/ellipse_norm.py): the arb formula in reliability.py is the sympy result."""
    import sympy as sp

    from certified_obstacle_nn.experiments.v17 import ellipse_norm as E

    _, _, N2 = E.derive()
    n = rel.reference_norm("ellipse")
    v = float(sp.sqrt(N2).evalf(30))
    assert n["norm_lower"] <= v <= n["norm_upper"]
    assert n["norm_upper"] - n["norm_lower"] <= 1e-15
    stored = E._stored_numerical_norm()["values"]  # v14 diagnostic quadrature of ||grad u||
    assert len(stored) == 1 and abs(v - float(stored[0])) < 1e-6


# ------------------------------------------------------------------ direct bounds
@pytest.mark.parametrize("D,L", [(0.3, 0.17), (1e-3, 2.0), (2.0, 1e-4), (0.21, 0.05), (1.0, 0.0)])
def test_primary_gradient_form_is_theta_infimum(D, L):
    """(sqrt L + sqrt(D + L))^2 = min over theta in (0,1) of (D + L/theta)/(1 - theta): never below the value at
    any theta, attained at theta* = sqrt(L)/(sqrt(L) + sqrt(D + L))."""
    G2 = (math.sqrt(L) + math.sqrt(D + L)) ** 2
    th = np.linspace(1e-4, 1 - 1e-4, 200001)
    vals = (D + L / th) / (1 - th)
    assert vals.min() >= G2 * (1 - 1e-12)
    if L > 0:
        ts = math.sqrt(L) / (math.sqrt(L) + math.sqrt(D + L))
        assert abs((D + L / ts) / (1 - ts) - G2) <= 1e-12 * G2
    assert G2 <= 2 * D + 4 * L + 1e-15


def test_direct_bounds_formulas_and_status():
    cf = rel.cF2_upper(2)["cF2_up"]
    o = rel.direct_projected_bounds(U_R=1.0, U_S=0.1, U_I=0.01, U_C=0.05, cF2_up=cf, status="validated_upper_bound",
                                    U_r_hat=2.11, Lambda=2.0, U_gradw=0.0625, reference_norm_lower=0.5,
                                    reference_status="validated_upper_bound")
    D, L = 0.1 + 0.01 + 0.1, cf
    assert o["flux_bound_direct_split"] >= math.sqrt(D + L) and o["flux_bound_direct_split"] <= math.sqrt(D + L) * (1 + 1e-15)
    G = math.sqrt(L) + math.sqrt(D + L)
    assert G <= o["grad_bound_direct_split"] <= G * (1 + 1e-15)
    assert o["grad_bound_direct_split"] <= o["grad_bound_direct_check_theta_half"]
    assert o["graph_bound_direct"] == 1.0
    # theta scan: the reported value is the min of the 31 rational evaluations
    th = Fraction(o["E_dir_split_theta"])
    E2 = max(1 / (1 - float(th)), 1.0) * (D + L / float(th)) + 0.01 + 1.0
    assert abs(o["E_dir_split"] - math.sqrt(E2)) <= 1e-14
    for k in range(1, 32):
        t = k / 32
        assert o["E_dir_split"] <= math.sqrt(max(1 / (1 - t), 1.0) * (D + L / t) + 1.01) * (1 + 1e-14)
    assert o["grad_bound_direct_original"] >= o["grad_bound_direct_split"] + 0.25
    assert o["ratio_status"] == "validated_upper_bound"
    assert o["flux_bound_direct_split_rel"] >= o["flux_bound_direct_split"] / 0.5
    # weakest status wins; a diagnostic denominator makes the ratio diagnostic
    o2 = rel.direct_projected_bounds(U_R=1.0, U_S=0.1, U_I=0.01, U_C=0.05, cF2_up=cf, status="validated_upper_bound",
                                     reference_norm_lower=0.5, reference_status="diagnostic_float64")
    assert o2["ratio_status"] == "diagnostic_float64"
    o3 = rel.direct_projected_bounds(U_R=float("nan"), U_S=0.1, U_I=0.01, U_C=0.05, cF2_up=cf,
                                     status="validated_upper_bound")
    assert o3["status"] == "diagnostic_float64"


def test_split_beats_nosplit_when_Lambda_ge_cF2():
    """With U_r_hat = Lambda U_R + U_S + U_I, the split forms are never larger than the no-split forms."""
    cf = rel.cF2_upper(2)["cF2_up"]
    UR, US_, UI, UC = 3.35, 0.155, 2e-5, 0.076
    Urh = 2 * UR + US_ + UI
    o = rel.direct_projected_bounds(U_R=UR, U_S=US_, U_I=UI, U_C=UC, cF2_up=cf, status="validated_upper_bound",
                                    U_r_hat=Urh, Lambda=2.0)
    assert o["flux_bound_direct_split"] <= o["flux_bound_direct"]
    assert o["grad_bound_direct_split"] <= o["grad_bound_direct"]
    assert o["E_dir_split"] <= o["E_U_direct"]


# ------------------------------------------------------------------ per-term split of the engines
def _check_split(us, La, kappa):
    P = us.parts
    tol = 1e-14
    rh = La * P["vol_Rhat"] + P["vol_Shat"] + P["init_mismatch_hat"]
    r0 = La * P["vol_R0"] + P["vol_S_term"] + P["init_mismatch"]
    p0 = P["vol_gradw"] + kappa * P["vol_w2"] + La * P["vol_wt2"] + P["w0"] + P["wT"]
    for split, agg in ((rh, us.U_r_hat), (r0, us.U_r), (p0, us.U_p)):
        assert split <= agg * (1 + tol)  # aggregate densities round each cell upward once more
        assert split >= agg * (1 - tol)


def _check_cells(vol, La):
    # termwise: the aggregate density is >= the weighted sum of its separately enclosed terms, cell by cell
    assert np.all(vol["rhat"] >= La * vol["Rh_term"] + vol["Sh_term"] - 0.0)
    assert np.all(vol["r"] >= La * vol["R0_term"] + vol["S_term"])
    s = vol["sign"]
    assert np.all(vol["Rh_term"][s < 0] == vol["R0_term"][s < 0])
    assert np.all(vol["Sh_term"][s < 0] == vol["S_term"][s < 0])
    assert np.all(vol["Rh_term"][s == 0] >= vol["R0_term"][s == 0])  # hull of both branches


@pytest.mark.parametrize("backend", ["numpy", "arb"])
def test_split_consistency_stefan_32(backend):
    problem, triple = load_frozen_stefan(STEFAN_SEEDS[0])
    enc = TripleEncloser(triple, StefanAdapter(problem))
    part, tp = US.uniform_2d(32, 32), US.uniform_1d(256)
    us, vol, _ = US.certify_partition(enc, part, tp, tp, backend=backend, workers=8, model_frozen=True,
                                      constants_validated=True)
    assert us.status == "validated_upper_bound"
    assert set(np.unique(vol["sign"])) >= {-1, 0}
    _check_split(us, 1.0, 1.0)
    _check_cells(vol, 1.0)


@pytest.fixture(scope="module")
def disk_long0():
    from certified_obstacle_nn.experiments.expanding_disk_2d.final_common import load_run
    from certified_obstacle_nn.certification.enclose.engine2d import DiskAdapter, TripleEncloser2D

    triple, cfg = load_run(REPO_ROOT / "results/expanding_disk_2d/final/long_6x64_N20000_s0_k5")
    return TripleEncloser2D(triple, DiskAdapter(cfg.kappa, cfg.Lambda, cfg.alpha))


@needs('results/expanding_disk_2d/final/long_6x64_N20000_s0_k5', 'results/v14_disk_validated/20261007T212554Z')
@pytest.mark.parametrize("backend,n", [("numpy", 16), ("arb", 8)])
def test_split_consistency_disk_smoke(disk_long0, backend, n):
    part, tp = Bx.uniform(3, n), Bx.uniform(2, 2 * n)
    us, vol, _ = Bx.certify_boxes(disk_long0, part, tp, tp, backend=backend, workers=8, model_frozen=True,
                                  constants_validated=True)
    assert us.status == "validated_upper_bound"
    _check_split(us, 2.0, 0.5)
    _check_cells(vol, 2.0)
