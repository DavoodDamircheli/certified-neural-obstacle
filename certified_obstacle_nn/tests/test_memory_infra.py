"""Tests for certification/memory.py, the observability/OOM-prevention
primitives specified in docs/history/MEMORY_OOM_DEBUGGING_AND_SAFE_RUN_POLICY.md.
"""

import numpy as np
import pytest

from certified_obstacle_nn.certification.memory import (
    CertificationMemoryError,
    MemoryGuard,
    estimate_chunk_bytes,
    gib_for_array,
    peak_rss,
    report_memory,
)


def test_guard_raises_below_artificially_low_limit():
    guard = MemoryGuard(max_rss_gb=1e-6, label_prefix="test")
    with pytest.raises(CertificationMemoryError):
        guard.check("artificially tiny limit")


def test_guard_does_not_raise_below_generous_limit():
    guard = MemoryGuard(max_rss_gb=1e6, label_prefix="test")
    guard.check("generous limit")  # must not raise


def test_guard_error_message_contains_required_fields():
    guard = MemoryGuard(max_rss_gb=1e-6, label_prefix="test", chunk_size=65536, n_qp=4, grid=(256, 256))
    with pytest.raises(CertificationMemoryError) as exc_info:
        guard.check("chunk 3/40")
    message = str(exc_info.value)
    assert "RSS=" in message
    assert "max_rss_gb=1e-06" in message
    assert "test:chunk 3/40" in message
    assert "chunk size (currently 65536 pts/chunk)" in message
    assert "n_qp (currently 4)" in message
    assert "grid resolution (currently 256x256)" in message
    assert "create_graph=False" in message


def test_peak_rss_monotone_non_decreasing_across_allocations():
    before = peak_rss()

    # allocate and touch a sizeable array so it is actually resident, not
    # just reserved virtual address space.
    a = np.ones((20_000_000,), dtype=np.float64)
    a[:] = 1.0
    after_first = peak_rss()

    b = np.ones((40_000_000,), dtype=np.float64)
    b[:] = 2.0
    after_second = peak_rss()

    assert after_first >= before
    assert after_second >= after_first

    del a, b


def test_gib_for_array_matches_closed_form():
    assert gib_for_array((50_000, 50_000)) == pytest.approx(18.6265, rel=1e-3)
    assert gib_for_array((1024, 1024, 1024), itemsize=1) == pytest.approx(1.0)


def test_estimate_chunk_bytes_scales_linearly_in_each_factor():
    base = estimate_chunk_bytes(n_points=1000, depth=4, width=64, n_networks=3)
    double_points = estimate_chunk_bytes(n_points=2000, depth=4, width=64, n_networks=3)
    double_width = estimate_chunk_bytes(n_points=1000, depth=4, width=128, n_networks=3)

    assert double_points == pytest.approx(2 * base, rel=1e-9)
    assert double_width == pytest.approx(2 * base, rel=1e-9)
    assert base > 0


def test_report_memory_returns_positive_gib():
    rss_gib = report_memory("test_report_memory")
    assert rss_gib > 0.0
