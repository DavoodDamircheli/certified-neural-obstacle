"""Stage-2 fallback item 1: sign-convention check. Feeds the EXACT triple through the same
residual code used in training (losses.compute_residuals); every residual must vanish away from
the free boundary (|psi| > 1e-3). Exits non-zero on failure."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch  # noqa: E402

from certified_obstacle_nn.experiments.expanding_disk_2d import exact_solution as ex, sampling  # noqa: E402
from certified_obstacle_nn.experiments.expanding_disk_2d.losses import compute_residuals  # noqa: E402

K = 5.0


class ExactTriple:
    def u(self, t, x, y):
        return ex.u_exact(t, x, y, K)

    def sigma(self, t, x, y):
        sx, sy = ex.sigma_exact(t, x, y, K)
        return torch.cat([sx, sy], -1)

    def lam(self, t, x, y):
        return ex.lambda_exact(t, x, y, K)


t, x, y = sampling.interior(200_000, 777)
x0, y0 = sampling.initial(20_000, 778)
for v in (t, x, y):
    v.requires_grad_(True)
r = compute_residuals(ExactTriple(), t, x, y, x0, y0, K, create_graph=False)
keep = (ex.psi(t, x, y).abs() > 1e-3).squeeze(1).detach()
vals = {"R_PDE": r.R_PDE[keep].abs().max().item(), "R_sig": r.R_sig[keep].abs().max().item(),
        "R_0": r.R_0.abs().max().item(), "R_obs": r.R_obs.abs().max().item(), "R_com": r.R_com.abs().max().item()}
print(vals)
sys.exit(0 if max(vals.values()) < 1e-10 else 1)
