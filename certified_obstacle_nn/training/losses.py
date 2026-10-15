"""Training loss, specification section 3 -- NOT the certificate.

R_com is not squared: it equals lambda_psi * gamma_minus, a product of two
already-nonnegative quantities, so mean(R_com) alone is a valid nonnegative
penalty. Every other component is penalized via its squared L2 mean. All five
raw (unweighted) components are returned alongside the weighted total so that
every component can be logged every epoch, per section 3's instruction to
never log only the total.
"""

from typing import NamedTuple

import torch

from certified_obstacle_nn.residuals.residuals import Residuals


class LossComponents(NamedTuple):
    total: torch.Tensor
    pde: torch.Tensor
    sigma: torch.Tensor
    initial: torch.Tensor
    penetration: torch.Tensor
    complementarity: torch.Tensor


def compute_loss(
    residuals: Residuals,
    w_r: float = 1.0,
    w_s: float = 1.0,
    w_0: float = 1.0,
    w_p: float = 1.0,
    w_c: float = 1.0,
) -> LossComponents:
    pde = residuals.R_PDE.pow(2).mean()
    sigma = residuals.R_sig.pow(2).mean()
    initial = residuals.R_0.pow(2).mean()
    penetration = residuals.R_obs.pow(2).mean()
    complementarity = residuals.R_com.mean()

    total = w_r * pde + w_s * sigma + w_0 * initial + w_p * penetration + w_c * complementarity

    return LossComponents(
        total=total,
        pde=pde,
        sigma=sigma,
        initial=initial,
        penetration=penetration,
        complementarity=complementarity,
    )
