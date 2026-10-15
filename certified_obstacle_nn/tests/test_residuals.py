"""residuals/residuals.py validated against the exact Stefan triple.

The exact (u, sigma, lambda) from specification section 9 satisfy every one of the
five residual identities exactly (up to floating point) away from the free
boundary t=x. Feeding them through compute_residuals -- using problem.u_exact,
problem.sigma_exact, problem.lambda_exact as drop-in replacements for the
neural networks, since they share the same (t, x) -> tensor signature -- is
the strongest available test of the residual algebra: any sign error, missing
b/c term, or wrong A power shows up as a nonzero residual against an
independently-known-correct solution, not just internal self-consistency.
"""

import torch

from certified_obstacle_nn.problems.stefan import StefanProblem
from certified_obstacle_nn.residuals.residuals import compute_residuals

TOL = 1e-10
BAND = 1e-2
N_POINTS = 5000
N_INITIAL = 500


def _sample_interior(n, band, seed):
    generator = torch.Generator().manual_seed(seed)
    t_full = torch.rand(n, 1, generator=generator, dtype=torch.float64)
    x_full = torch.rand(n, 1, generator=generator, dtype=torch.float64)
    mask = (t_full - x_full).abs().squeeze(-1) > band
    t = t_full[mask].requires_grad_(True)
    x = x_full[mask].requires_grad_(True)
    return t, x


def test_residuals_vanish_for_exact_stefan_solution():
    problem = StefanProblem()
    t, x = _sample_interior(N_POINTS, BAND, seed=0)

    generator = torch.Generator().manual_seed(1)
    x0 = torch.rand(N_INITIAL, 1, generator=generator, dtype=torch.float64)

    residuals = compute_residuals(
        problem,
        problem.u_exact,
        problem.sigma_exact,
        problem.lambda_exact,
        t,
        x,
        x0,
    )

    for name in ("R_PDE", "R_sig", "R_0", "R_obs", "R_com"):
        value = getattr(residuals, name)
        max_violation = value.abs().max().item()
        print(f"max |{name}| = {max_violation:.3e}")
        assert max_violation < TOL, f"{name} violated identity by {max_violation:.3e}"
