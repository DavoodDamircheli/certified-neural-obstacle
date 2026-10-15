"""Standalone diagnostic (2026-09-11) confirming S2's central claim directly
at the residual level, not just through rho_r (which combines R_PDE and
R_sig together, per specification section 4: rho_r^2 = (Lambda/alpha)*R_PDE^2 +
R_sig^2 + R_0^2 -- rho_r growing does not by itself say WHICH term grew).

Evaluates R_PDE and R_sig (residuals/residuals.py's own formulas, via
eval_triple_derivatives -- no certification/network_bounds involved, this is
a direct numerical comparison) at a dense random sample, for the SAME base
triple with and without the flux corruption, and reports the ratio.
"""

import json
from pathlib import Path

import torch

from certified_obstacle_nn.autograd.derivatives import eval_triple_derivatives
from certified_obstacle_nn.certification.stress_corruption import make_flux_add_linear_t_triple
from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.problems.stefan import StefanProblem
from certified_obstacle_nn.residuals.residuals import assemble_residuals

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_WEIGHTS = REPO_ROOT / "results" / "stefan_m18" / "4x64" / "seed0" / "weights.pt"
BASE_CONFIG = REPO_ROOT / "results" / "stefan_m18" / "4x64" / "seed0" / "config.json"
A_COEFFICIENT = 1.0  # must match experiments/stefan_stress.py's S2_A_COEFFICIENT
N_SAMPLE = 20000


def _load_frozen_triple(weights_path: Path, config_path: Path, problem: StefanProblem) -> NeuralTriple:
    config = json.loads(config_path.read_text())
    arch = config["architecture"]
    triple = NeuralTriple(problem, width=arch["width"], depth=arch["depth"], multiplier_head=arch.get("multiplier_head", "square"))
    triple.load_state_dict(torch.load(weights_path, weights_only=True, map_location="cpu"))
    triple.eval()
    for p in triple.parameters():
        p.requires_grad_(False)
    return triple


def _residual_rms(model, problem, t, x, x0) -> tuple[float, float]:
    u, sigma, lam, u_t, u_x, sigma_x = eval_triple_derivatives(model, t, x)
    with torch.no_grad():
        u0_pred = model.state_net(torch.zeros_like(x0), x0)
    res = assemble_residuals(problem, t, x, x0, u, sigma, lam, u_t, u_x, sigma_x, u0_pred)
    R_PDE_rms = res.R_PDE.pow(2).mean().sqrt().item()
    R_sig_rms = res.R_sig.pow(2).mean().sqrt().item()
    return R_PDE_rms, R_sig_rms


def main() -> dict:
    problem = StefanProblem()
    base_triple = _load_frozen_triple(BASE_WEIGHTS, BASE_CONFIG, problem)
    corrupted_triple = make_flux_add_linear_t_triple(base_triple, A_COEFFICIENT)

    torch.manual_seed(0)
    t = torch.rand(N_SAMPLE, 1, dtype=torch.float64)
    x = torch.rand(N_SAMPLE, 1, dtype=torch.float64)
    x0 = torch.rand(500, 1, dtype=torch.float64)

    base_R_PDE, base_R_sig = _residual_rms(base_triple, problem, t, x, x0)
    corr_R_PDE, corr_R_sig = _residual_rms(corrupted_triple, problem, t, x, x0)

    result = {
        "n_sample": N_SAMPLE,
        "a_coefficient": A_COEFFICIENT,
        "base_R_PDE_rms": base_R_PDE,
        "base_R_sig_rms": base_R_sig,
        "corrupted_R_PDE_rms": corr_R_PDE,
        "corrupted_R_sig_rms": corr_R_sig,
        "R_PDE_ratio": corr_R_PDE / base_R_PDE,
        "R_sig_ratio": corr_R_sig / base_R_sig,
    }
    assert abs(result["R_PDE_ratio"] - 1.0) < 1e-6, (
        "R_PDE should be numerically UNCHANGED by the flux corruption (d_x a = 0 exactly) -- "
        f"got ratio {result['R_PDE_ratio']}, expected ~1.0"
    )

    out_path = REPO_ROOT / "results" / "stefan_stress_s2_residual_diagnostic.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"R_PDE ratio (corrupted/base) = {result['R_PDE_ratio']:.6f}x  (expected: exactly 1, d_x a = 0)")
    print(f"R_sig ratio (corrupted/base) = {result['R_sig_ratio']:.4f}x")
    print(f"-> {out_path}")
    return result


if __name__ == "__main__":
    main()
