"""v20 non-manufactured 1-D benchmark `rising1d` (protocol
results/v20_protocols/rising1d_protocol.json).

Omega = (0,1), T = 1, A = 1, b = c = 0, alpha = 1, Lambda = 1 (Stefan conventions):
    f(t,x) = -8 sin(pi x),   u0 = 0,   g(t,x) = -3/5 + 3/5 t - 2 (x - 1/2)^2.
No closed-form solution. psi = g_t - g_xx - f = 23/5 + 8 sin(pi x) > 0, so on the contact set the multiplier is
lambda = 23/5 + 8 sin(pi x). The forcing pushes u down onto the rising obstacle; the contact set nucleates at
x = 1/2 at some t* > 0.

Exact data constants (sup over [0,1]^2):
  g in [-11/10, 0] (min at t = 0, x in {0,1}; max at t = 1, x = 1/2)  => ||g|| = 11/10
  g_x = -4(x - 1/2) => 2;  g_t = 3/5;  g_xx = -4 => 4;  g_tx = g_tt = 0
  f in [-8, 0] => 8;  f_x = -8 pi cos(pi x) => 8 pi;  f_t = 0;  u0 = 0.
"""

import math

import torch

from certified_obstacle_nn.problems.base_problem import BaseProblem


class Rising1DProblem(BaseProblem):
    name = "rising1d"
    A = 1.0
    b = 0.0
    c = 0.0
    alpha = 1.0
    A_inf = 1.0
    Lambda = 1.0
    Lambda_justification = "c_F <= diam(Omega) = 1 (Omega = (0,1)), so Lambda = 1 >= c_F^2"
    T = 1.0
    omega_bounds = (0.0, 1.0)
    gamma_data: tuple[float, ...] = ()

    g_inf = 1.1
    g_x_inf = 2.0
    g_t_inf = 0.6
    g_xx_inf = 4.0
    g_tx_inf = 0.0
    g_tt_inf = 0.0

    f_inf = 8.0
    f_x_inf = 8.0 * math.pi
    f_t_inf = 0.0

    u0_inf = 0.0
    u0_lip = 0.0

    def g(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return -0.6 + 0.6 * t - 2.0 * (x - 0.5) ** 2

    def f(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return -8.0 * torch.sin(math.pi * x)

    def u0(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    @staticmethod
    def psi(x):
        """g_t - g_xx - f = 4.6 + 8 sin(pi x) (numpy or torch)."""
        sin = torch.sin if isinstance(x, torch.Tensor) else __import__("numpy").sin
        return 4.6 + 8.0 * sin(math.pi * x)
