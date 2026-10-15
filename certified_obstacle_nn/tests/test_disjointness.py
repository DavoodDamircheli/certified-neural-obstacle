"""Tests for certification/disjointness.py.

Test (ii) (profile at 256^2 / 10k training points, RSS increase < 200 MiB) is
the regression test for the original bug: certificate.py's old
_assert_disjoint_from_training built an (N_train, N_cell, 2) broadcast array,
which grows without bound as the grid refines. This module's replacement must
not.
"""

import pytest
import torch

from certified_obstacle_nn.certification.disjointness import (
    DisjointnessViolationError,
    assert_disjoint,
    iter_disjoint_checked,
)
from certified_obstacle_nn.certification.memory import peak_rss
from certified_obstacle_nn.certification.partition import build_partition
from certified_obstacle_nn.certification.quadrature import CellChunk, build_quadrature_rule, iter_cell_chunks
from certified_obstacle_nn.problems.stefan import StefanProblem

MAX_RSS_INCREASE_BYTES = 200 * 1024 * 1024  # 200 MiB


def _fake_chunk(t_values, x_values, cell_ids, points_per_cell):
    t = torch.tensor(t_values, dtype=torch.float64).reshape(-1, 1)
    x = torch.tensor(x_values, dtype=torch.float64).reshape(-1, 1)
    w = torch.ones_like(t)
    assert t.shape[0] == len(cell_ids) * points_per_cell
    return CellChunk(cell_ids=torch.tensor(cell_ids, dtype=torch.int64), t=t, x=x, w=w, area=1.0)


def test_planted_duplicate_point_is_detected_exact_mode():
    t_train = torch.tensor([0.123456789012, 0.5, 0.75], dtype=torch.float64)
    x_train = torch.tensor([0.333333333333, 0.6, 0.9], dtype=torch.float64)

    # cell 5: two clean points. cell 7: one clean point, one point planted to
    # exactly equal training point 0.
    chunk = _fake_chunk(
        t_values=[0.10, 0.20, 0.30, 0.123456789012],
        x_values=[0.10, 0.20, 0.40, 0.333333333333],
        cell_ids=[5, 7],
        points_per_cell=2,
    )

    with pytest.raises(DisjointnessViolationError) as exc_info:
        assert_disjoint(iter([chunk]), (t_train, x_train), tol=0.0)

    message = str(exc_info.value)
    assert "1 certification quadrature point" in message
    assert "cell 7" in message or "{7:" in message  # the planted duplicate is in cell 7


def test_no_collision_does_not_raise_exact_mode():
    t_train = torch.tensor([0.9, 0.95], dtype=torch.float64)
    x_train = torch.tensor([0.9, 0.95], dtype=torch.float64)
    chunk = _fake_chunk(
        t_values=[0.1, 0.2, 0.3, 0.4],
        x_values=[0.1, 0.2, 0.3, 0.4],
        cell_ids=[1, 2],
        points_per_cell=2,
    )
    assert_disjoint(iter([chunk]), (t_train, x_train), tol=0.0)  # must not raise


def test_near_miss_within_tol_is_detected_kdtree_mode():
    t_train = torch.tensor([0.5], dtype=torch.float64)
    x_train = torch.tensor([0.5], dtype=torch.float64)
    chunk = _fake_chunk(
        t_values=[0.1, 0.500001],
        x_values=[0.1, 0.500001],
        cell_ids=[3],
        points_per_cell=2,
    )
    assert_disjoint(iter([chunk]), (t_train, x_train), tol=1e-9)  # too tight, must not raise

    with pytest.raises(DisjointnessViolationError):
        assert_disjoint(iter([chunk]), (t_train, x_train), tol=1e-3)  # loose enough to catch it


def test_train_points_none_is_a_no_op():
    chunk = _fake_chunk(t_values=[0.1, 0.2], x_values=[0.1, 0.2], cell_ids=[0], points_per_cell=2)
    assert_disjoint(iter([chunk]), None, tol=0.0)  # must not raise, must not error

    seen = list(iter_disjoint_checked(iter([chunk]), None, tol=0.0))
    assert len(seen) == 1
    assert seen[0] is chunk


def test_iter_disjoint_checked_yields_chunks_through_when_clean():
    t_train = torch.tensor([0.99], dtype=torch.float64)
    x_train = torch.tensor([0.99], dtype=torch.float64)
    chunks = [
        _fake_chunk(t_values=[0.1, 0.2], x_values=[0.1, 0.2], cell_ids=[0], points_per_cell=2),
        _fake_chunk(t_values=[0.3, 0.4], x_values=[0.3, 0.4], cell_ids=[1], points_per_cell=2),
    ]
    seen = list(iter_disjoint_checked(iter(chunks), (t_train, x_train), tol=0.0))
    assert seen == chunks


def test_iter_disjoint_checked_still_yields_clean_chunks_before_raising_on_a_later_one():
    t_train = torch.tensor([0.123456789012], dtype=torch.float64)
    x_train = torch.tensor([0.333333333333], dtype=torch.float64)
    clean_chunk = _fake_chunk(t_values=[0.1, 0.2], x_values=[0.1, 0.2], cell_ids=[0], points_per_cell=2)
    dirty_chunk = _fake_chunk(
        t_values=[0.123456789012, 0.4], x_values=[0.333333333333, 0.4], cell_ids=[1], points_per_cell=2
    )

    seen = []
    with pytest.raises(DisjointnessViolationError):
        for chunk in iter_disjoint_checked(iter([clean_chunk, dirty_chunk]), (t_train, x_train), tol=0.0):
            seen.append(chunk)
    assert seen == [clean_chunk, dirty_chunk]  # the dirty chunk was yielded before the error fired


def test_disjointness_check_memory_is_bounded_at_256_grid_10k_training_points():
    """Regression test for the original O(N_train x N_cell) bug: this must
    stay O(N_train), never grow with the certification grid size."""
    problem = StefanProblem()
    torch.manual_seed(0)
    n_train = 10_000
    t_train = torch.rand(n_train, dtype=torch.float64)
    x_train = torch.rand(n_train, dtype=torch.float64)

    partition = build_partition(problem, n_t=256, n_x=256)  # 65536 cells
    rule = build_quadrature_rule(q=4)  # 1,048,576 quadrature points total

    rss_before = peak_rss()
    assert_disjoint(iter_cell_chunks(partition, rule, points_per_chunk=65536), (t_train, x_train), tol=0.0)
    rss_after = peak_rss()

    increase = rss_after - rss_before
    print(f"disjointness check RSS increase at 256x256 / 10k training points: {increase / (1024*1024):.2f} MiB")
    assert increase < MAX_RSS_INCREASE_BYTES
