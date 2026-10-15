"""v17b Task 2: the arb point lower bounds used by the backend cross-check (DECISIONS D25)."""

import mpmath
import numpy as np
import pytest
import torch

from certified_obstacle_nn.certification.enclose import boxes as Bx
from certified_obstacle_nn.experiments.v14.common import REPO_ROOT
from certified_obstacle_nn.experiments.v17 import backend_crosscheck as BC
from certified_obstacle_nn.tests.stored_data import needs


def test_sq_dn_is_lower_bound():
    rng = np.random.default_rng(0)
    lo = rng.normal(size=500)
    hi = lo + np.abs(rng.normal(size=500))
    v = BC._sq_dn(lo, hi)
    for a, b, s in zip(lo, hi, v):
        exact = 0 if a <= 0 <= b else min(mpmath.mpf(a) ** 2, mpmath.mpf(b) ** 2)
        assert mpmath.mpf(s) <= exact


@pytest.fixture(scope="module")
def disk_long0():
    from certified_obstacle_nn.certification.enclose.engine2d import DiskAdapter, TripleEncloser2D
    from certified_obstacle_nn.experiments.expanding_disk_2d.final_common import load_run

    triple, cfg = load_run(REPO_ROOT / "results/expanding_disk_2d/final/long_6x64_N20000_s0_k5")
    return triple, cfg, TripleEncloser2D(triple, DiskAdapter(cfg.kappa, cfg.Lambda, cfg.alpha))


@needs('results/expanding_disk_2d/final/long_6x64_N20000_s0_k5')
def test_point_lower_bounds_below_both_backends_and_match_float64(disk_long0):
    triple, cfg, enc = disk_long0
    part = Bx.uniform(3, 4)
    vn = Bx.evaluate_volume(enc, part, "numpy")
    va = Bx.evaluate_volume(enc, part, "arb", 192, batch=16)
    cen = 0.5 * (part.lo + part.hi)
    pl = BC._point_lower_chunk((enc, 192, cen))
    for t, k in BC.VOL_TERMS.items():
        assert np.all(pl[t] <= vn[k]) and np.all(pl[t] <= va[k]), t
    # tightness: the 192-bit point lower bounds match the float64 point values (autograd) to ~1e-9 relative
    from certified_obstacle_nn.experiments.v17.cell_quadrature import _fields
    from certified_obstacle_nn.problems.disk import DiskProblem

    cols = [torch.tensor(c, dtype=torch.float64, requires_grad=True) for c in cen]
    fv = _fields("disk2d", DiskProblem(cfg.kappa), triple, cols)
    for t, k in (("R0", "R02"), ("S", "S02"), ("c", "C")):
        assert np.allclose(pl[t], fv[k], rtol=1e-9, atol=1e-13), t


@needs('results/expanding_disk_2d/final/long_6x64_N20000_s0_k5')
def test_trace_point_lower_bounds(disk_long0):
    _, _, enc = disk_long0
    tp = Bx.uniform(2, 8)
    for tval in (0.0, 1.0):
        tn = Bx.evaluate_trace(enc, tp, tval, "numpy")
        pl = BC._trace_point_lower_chunk((enc, 192, tval, 0.5 * (tp.lo + tp.hi)))
        for k in BC.TR_TERMS:
            assert np.all(pl[k] <= tn[k]), k
