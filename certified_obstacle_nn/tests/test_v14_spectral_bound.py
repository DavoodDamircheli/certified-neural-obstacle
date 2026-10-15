"""v14 A2: per-layer spectral norms are PROVED upper bounds, not SVD values."""

import mpmath
import pytest
import torch

from certified_obstacle_nn.certification.network_bounds import (
    guaranteed_spectral_norm_upper,
    spectral_norms,
    spectral_norms_svd,
)
from certified_obstacle_nn.models.mlp import MLP


def _mp_spectral(w):
    mpmath.mp.dps = 50
    A = mpmath.matrix(w.tolist())
    s = mpmath.svd_r(A, compute_uv=False)
    return max(s)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_bound_dominates_high_precision_spectral_norm(seed):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(7, 5, generator=g, dtype=torch.float64)
    assert guaranteed_spectral_norm_upper(w) >= _mp_spectral(w)


def test_rank_one_is_tight():
    u = torch.tensor([[1.0], [2.0], [-0.5]], dtype=torch.float64)
    v = torch.tensor([[0.25, -3.0]], dtype=torch.float64)
    w = u @ v  # exact in binary
    exact = mpmath.sqrt(mpmath.mpf(1 + 4 + 0.25)) * mpmath.sqrt(mpmath.mpf(0.0625 + 9))
    b = guaranteed_spectral_norm_upper(w)
    assert b >= exact
    assert b <= float(exact) * (1 + 1e-15)


def test_network_bounds_use_guaranteed_norms():
    torch.manual_seed(0)
    mlp = MLP(2, 1, hidden_width=16, hidden_depth=3)
    for w, guaranteed, svd in zip(mlp.weights, spectral_norms(mlp), spectral_norms_svd(mlp)):
        # the float SVD may sit an ulp ABOVE or BELOW the truth; the guaranteed
        # bound must dominate the 50-digit value
        assert guaranteed >= _mp_spectral(w.detach().double())
        assert guaranteed >= svd * (1 - 1e-14)
