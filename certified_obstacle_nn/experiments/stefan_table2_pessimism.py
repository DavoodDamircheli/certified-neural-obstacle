"""Table 2: pessimism decomposition (2026-09-11), specification section 8.

E_cert_U/E_true (how pessimistic the certificate is relative to the true
error) decomposes exactly as:

    E_cert_U / E_true
        = I_eff_raw * (E_quad / rho_tilde_h) * sqrt(C_rel) * sqrt(C_eq)
        = (rho_tilde_h/E_true) * (sqrt(rho_tilde_h^2+E_h)/rho_tilde_h) * sqrt(216) * sqrt(2)

since E_cert_U = sqrt(C_eq * C_rel * (rho_tilde_h^2+E_h)) = sqrt(C_eq)*sqrt(C_rel)*E_quad
and rho_tilde_h cancels between the first two factors, leaving
(E_quad/E_true)*sqrt(C_rel)*sqrt(C_eq) = (E_cert_U/E_true) exactly. This is a
pure algebraic identity -- not a new bound -- so it must hold to machine
precision for every run; failing this check would mean Table 1's own
reported quantities (rho_tilde, E_quad, E_cert_U, E_true) are internally
inconsistent, a bug in the reporting pipeline, not a numerical curiosity.
Asserted in code below (not just eyeballed in the CSV).

The decomposition separates PESSIMISM into three independently-interesting
factors:
  I_eff_raw            -- how well the raw estimator alone tracks E_true
  E_quad / rho_tilde_h  -- how much the Level-2 remainder E_h inflates the
                           raw estimator (>=1 always, since E_h>=0)
  sqrt(C_rel)*sqrt(C_eq) = sqrt(432) -- the two FIXED universal constants
                           (specification section 9, Stefan benchmark), identical
                           for every run, contributing no architecture-
                           dependence of its own.
"""

import math
from pathlib import Path

import pandas as pd

C_REL = 216.0
C_EQ = 2.0
CONSTANT_FACTOR = math.sqrt(C_REL) * math.sqrt(C_EQ)  # sqrt(432)

REPO_ROOT = Path(__file__).resolve().parents[2]
TABLE1_PATH = REPO_ROOT / "results" / "stefan_m18" / "table1.csv"
OUT_CSV = REPO_ROOT / "results" / "stefan_table2_pessimism.csv"
OUT_SUMMARY_CSV = REPO_ROOT / "results" / "stefan_table2_pessimism_summary.csv"

IDENTITY_REL_TOL = 1e-9  # machine-precision-level: this is pure arithmetic, not a numerical estimate


def build_table2(table1_df: pd.DataFrame) -> pd.DataFrame:
    df = table1_df.copy()

    df["I_eff_raw_recomputed"] = df["rho_tilde"] / df["E_true"]
    df["inflation_factor"] = df["E_quad"] / df["rho_tilde"]
    df["pessimism_lhs"] = df["I_eff_raw_recomputed"] * df["inflation_factor"] * CONSTANT_FACTOR
    df["pessimism_rhs_E_cert_U_over_E_true"] = df["E_cert_U"] / df["E_true"]
    df["identity_abs_diff"] = (df["pessimism_lhs"] - df["pessimism_rhs_E_cert_U_over_E_true"]).abs()
    df["identity_rel_diff"] = df["identity_abs_diff"] / df["pessimism_rhs_E_cert_U_over_E_true"]

    max_rel_diff = df["identity_rel_diff"].max()
    assert max_rel_diff < IDENTITY_REL_TOL, (
        f"Pessimism-decomposition identity failed: max relative difference "
        f"{max_rel_diff:.3e} exceeds {IDENTITY_REL_TOL:.0e}. This is a pure "
        f"algebraic identity from already-reported Table 1 quantities -- a "
        f"failure here means the pipeline's own reported numbers are "
        f"internally inconsistent, not that the bound is loose. Do not "
        f"loosen IDENTITY_REL_TOL to make this pass; find the inconsistency."
    )

    return df[[
        "Architecture", "Seed", "param_count",
        "rho_tilde", "E_true", "E_quad", "E_cert_U",
        "I_eff_raw_recomputed", "inflation_factor",
        "pessimism_lhs", "pessimism_rhs_E_cert_U_over_E_true",
        "identity_abs_diff", "identity_rel_diff",
    ]]


def main() -> tuple[pd.DataFrame, pd.DataFrame]:
    table1_df = pd.read_csv(TABLE1_PATH)
    table2_df = build_table2(table1_df)
    table2_df.to_csv(OUT_CSV, index=False)

    arch_order = ["4x32", "4x64", "6x64", "6x128"]
    table2_df["Architecture"] = pd.Categorical(table2_df["Architecture"], categories=arch_order, ordered=True)
    g = table2_df.groupby("Architecture", observed=True)
    summary = pd.DataFrame({
        "param_count": g["param_count"].mean(),
        "I_eff_raw_mean": g["I_eff_raw_recomputed"].mean(),
        "I_eff_raw_std": g["I_eff_raw_recomputed"].std(),
        "inflation_factor_mean": g["inflation_factor"].mean(),
        "inflation_factor_std": g["inflation_factor"].std(),
        "pessimism_mean": g["pessimism_rhs_E_cert_U_over_E_true"].mean(),
        "pessimism_std": g["pessimism_rhs_E_cert_U_over_E_true"].std(),
        "max_identity_rel_diff": g["identity_rel_diff"].max(),
    })
    summary.to_csv(OUT_SUMMARY_CSV)

    print(f"Identity E_cert_U/E_true == I_eff_raw * inflation_factor * sqrt(432) "
          f"verified to machine precision (max rel diff = {table2_df['identity_rel_diff'].max():.3e}) "
          f"across all {len(table2_df)} runs.")
    print(f"\nPer-architecture summary:\n{summary}")
    print(f"\nTable 2 -> {OUT_CSV}")
    print(f"Table 2 summary -> {OUT_SUMMARY_CSV}")
    return table2_df, summary


if __name__ == "__main__":
    main()
