"""Tests for experiments/stefan_stress.py's own logic (not the full S1-S4
subprocess pipeline, which is exercised by actually running the sweep)."""

import pytest
import torch

from certified_obstacle_nn.experiments.stefan_stress import BandExcludingTrainingSampler, _build_row
from certified_obstacle_nn.problems.stefan import StefanProblem

pytestmark = pytest.mark.legacy


def test_band_excluding_sampler_respects_band_width():
    problem = StefanProblem()
    band_width = 0.1
    sampler = BandExcludingTrainingSampler(problem, n_interior=2000, n_initial=100, seed=0, band_width=band_width)
    t, x, x0 = sampler.sample()

    assert t.shape == (2000, 1)
    assert x.shape == (2000, 1)
    diff = (t.detach() - x.detach()).abs()
    assert (diff >= band_width).all()


def test_band_excluding_sampler_deterministic_given_seed():
    problem = StefanProblem()
    s1 = BandExcludingTrainingSampler(problem, n_interior=500, n_initial=50, seed=42, band_width=0.05)
    s2 = BandExcludingTrainingSampler(problem, n_interior=500, n_initial=50, seed=42, band_width=0.05)
    t1, x1, _ = s1.sample()
    t2, x2, _ = s2.sample()
    assert torch.equal(t1, t2)
    assert torch.equal(x1, x2)


def test_band_excluding_sampler_interior_points_property_matches_base_class():
    problem = StefanProblem()
    sampler = BandExcludingTrainingSampler(problem, n_interior=300, n_initial=30, seed=0, band_width=0.05)
    t, x, _ = sampler.sample()
    t_exposed, x_exposed = sampler.interior_points
    assert torch.equal(t_exposed, t.detach())
    assert torch.equal(x_exposed, x.detach())


def test_build_row_extracts_expected_fields():
    cert_payload = {"result": {
        "rho_r": 0.1, "rho_p_tilde": 0.2, "rho_c": 0.05, "rho_tilde_h": 0.3, "E_cert_U_new": 100.0,
    }}
    te_payload = {"result": {"E_true": 0.05}}
    row = _build_row("S_test", cert_payload, te_payload, training_loss=0.001)
    assert row == {
        "test": "S_test", "rho_r": 0.1, "rho_p": 0.2, "rho_c": 0.05, "rho_total": 0.3,
        "E_true": 0.05, "E_cert_U": 100.0, "training_loss": 0.001,
    }


def test_build_row_raises_if_M16_violated():
    cert_payload = {"result": {
        "rho_r": 0.1, "rho_p_tilde": 0.2, "rho_c": 0.05, "rho_tilde_h": 0.3, "E_cert_U_new": 1.0,
    }}
    te_payload = {"result": {"E_true": 100.0}}  # deliberately violates E_true <= E_cert_U
    with pytest.raises(AssertionError, match="M16 VIOLATED"):
        _build_row("S_broken", cert_payload, te_payload, training_loss=0.001)
