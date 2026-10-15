"""Stefan benchmark problem, specification section 9.

Omega=(0,1), T=1, A=1, b=c=0, alpha=1, Lambda=1. Gamma_data is empty.

The g_*/f_*/u0_* sup-norm and Lipschitz constants below are hand-derived exact
values from the closed-form g, f, u0 (specification section 6 needs them, but does
not itself give their values -- they are problem data, not part of the general
spec). Derivation, using h(t) = e^t - t - 1 (so h(0)=0, h increasing on [0,1],
h(1) = e-2), h'(t) = e^t - 1, h''(t) = e^t:
  g = -h(t)(1-x)         => g_x = h(t), g_t = -h'(t)(1-x)
  g_xx = 0, g_tx = h'(t), g_tt = -h''(t)(1-x) = -e^t(1-x)
  f = -1 - (e^t-1)(1-x)  => f_x = e^t-1, f_t = -e^t(1-x)
Both h(t)(1-x) and (e^t-1)(1-x) etc. are products of a t-only increasing
nonnegative factor and an x-only decreasing nonnegative factor on [0,1]^2, so
each sup-norm below is attained at the shared extreme (t,x)=(1,0) or (0,*)/(*,1)
as appropriate, hand-verified rather than sampled.
"""

import math

import torch

from certified_obstacle_nn.problems.base_problem import BaseProblem


class StefanProblem(BaseProblem):
    A = 1.0
    b = 0.0
    c = 0.0
    alpha = 1.0
    A_inf = 1.0
    Lambda = 1.0
    # v14 A8: Lambda >= c_F^2 with c_F <= diam(Omega) = 1 on the unit interval.
    Lambda_justification = "c_F <= diam(Omega) = 1 (Omega = (0,1)), so Lambda = 1 >= c_F^2"
    T = 1.0
    omega_bounds = (0.0, 1.0)
    gamma_data: tuple[float, ...] = ()

    g_inf = math.e - 2.0
    g_x_inf = math.e - 2.0
    g_t_inf = math.e - 1.0
    g_xx_inf = 0.0
    g_tx_inf = math.e - 1.0
    g_tt_inf = math.e

    f_inf = math.e
    f_x_inf = math.e - 1.0
    f_t_inf = math.e

    u0_inf = 0.0
    u0_lip = 0.0

    @staticmethod
    def h(t: torch.Tensor) -> torch.Tensor:
        return torch.exp(t) - t - 1.0

    @staticmethod
    def h_prime(t: torch.Tensor) -> torch.Tensor:
        return torch.exp(t) - 1.0

    def g(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return -self.h(t) * (1.0 - x)

    def f(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return -1.0 - (torch.exp(t) - 1.0) * (1.0 - x)

    def u0(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    def u_exact(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        active = t > x
        extra = torch.exp(t - x) + x - t - 1.0
        return -self.h(t) * (1.0 - x) + torch.where(active, extra, torch.zeros_like(extra))

    def sigma_exact(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        active = t > x
        noncontact = -self.h(t) + torch.exp(t - x) - 1.0
        contact = -self.h(t)
        return torch.where(active, noncontact, contact)

    def lambda_exact(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Contact set convention must match u_exact/sigma_exact, which take the
        # contact branch on t <= x (v14 A1, 2026-10-06). The previous strict
        # `t < x` made u_t + sigma_x - lambda - f == 1 exactly on the diagonal
        # t == x, which tensor Gauss grids with equal t/x nodes hit -- the
        # source of the spurious O(1/n) "convergence" of the graph term. The
        # change is on a null set, so lambda is the same L2 function.
        return (t <= x).to(t.dtype)

    def free_boundary_exact(self, t: torch.Tensor) -> torch.Tensor:
        return t
