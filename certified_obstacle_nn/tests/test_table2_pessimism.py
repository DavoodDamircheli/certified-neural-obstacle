"""Tests for experiments/stefan_table2_pessimism.py's identity check."""

import numpy as np
import pandas as pd
import pytest

from certified_obstacle_nn.experiments.stefan_table2_pessimism import CONSTANT_FACTOR, build_table2


def _synthetic_table1(n=25, seed=0):
    """Constructs a Table-1-shaped frame from independently-random
    rho_tilde/E_h/E_true, then derives E_quad/E_cert_U FROM THEM via the
    real specification formulas -- so the identity is checked against genuinely
    independent quantities, not a self-fulfilling construction."""
    rng = np.random.default_rng(seed)
    rho_tilde = rng.uniform(0.01, 1.0, n)
    E_h = rng.uniform(0.0, 500.0, n)
    E_true = rng.uniform(0.01, 1.0, n)

    E_quad = np.sqrt(rho_tilde ** 2 + E_h)
    C_rel, C_eq = 216.0, 2.0
    E_cert = np.sqrt(C_rel * (rho_tilde ** 2 + E_h))
    E_cert_U = np.sqrt(C_eq * C_rel * (rho_tilde ** 2 + E_h))

    return pd.DataFrame({
        "Architecture": ["synthetic"] * n,
        "Seed": range(n),
        "param_count": [1000] * n,
        "rho_tilde": rho_tilde,
        "E_true": E_true,
        "E_quad": E_quad,
        "E_cert": E_cert,
        "E_cert_U": E_cert_U,
    })


def test_identity_holds_to_machine_precision_on_synthetic_data():
    df = _synthetic_table1()
    table2 = build_table2(df)
    assert table2["identity_rel_diff"].max() < 1e-9


def test_identity_holds_on_real_m18_table1():
    import json
    from pathlib import Path
    table1_path = Path(__file__).resolve().parents[2] / "results" / "stefan_m18" / "table1.csv"
    if not table1_path.exists():
        pytest.skip("results/stefan_m18/table1.csv not present in this checkout")
    df = pd.read_csv(table1_path)
    table2 = build_table2(df)
    assert table2["identity_rel_diff"].max() < 1e-9
    assert len(table2) == len(df)


def test_constant_factor_is_sqrt_432():
    assert CONSTANT_FACTOR == pytest.approx((432.0) ** 0.5, rel=1e-12)


def test_identity_violation_is_caught():
    """A deliberately corrupted E_cert_U (inconsistent with rho_tilde/E_h)
    must make the assertion fire -- confirms the check is load-bearing, not
    a no-op that would pass on any input."""
    df = _synthetic_table1(n=5)
    df.loc[0, "E_cert_U"] *= 1.5  # break the identity for one row
    with pytest.raises(AssertionError, match="Pessimism-decomposition identity failed"):
        build_table2(df)
