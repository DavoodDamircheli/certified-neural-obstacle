"""Expanding-disk obstacle benchmark as a v14 problem record.

The formulas are those of experiments/expanding_disk_2d/exact_solution.py
(taken there verbatim from the 2D benchmark specification
section 2; constant-multiplier variant, kappa = 5 in every stored run):

    Omega = (0,1)^2, T = 1, A = I, b = c = 0,
    B = 16 x(1-x) y(1-y),  h = e^t - t - 1,  g = -h B,
    R(t) = 0.15 + 0.2 t,  psi = (R^2 - r^2)/(2R),  r^2 = (x-1/2)^2 + (y-1/2)^2,
    u = g + kappa psi_+^2,  sigma = -grad u,  lambda = 2 kappa 1{psi < 0},
    f = u_t - Delta u - lambda,  u0 = kappa psi(0,.)_+^2.

Contact set {psi < 0} (outside the disk), non-contact {psi > 0} (inside).
This class only adds the v14 record (alpha, Lambda and its justification,
norm bounds) and delegates to exact_solution; tests/test_disk_exact.py checks
the PDE and complementarity symbolically on both branches.
"""

from __future__ import annotations

from certified_obstacle_nn.experiments.expanding_disk_2d import exact_solution as ex


class DiskProblem:
    name = "disk2d"
    dim = 2
    T = 1.0
    omega_bounds = ((0.0, 1.0), (0.0, 1.0))
    A = 1.0
    A_inf = 1.0
    b = 0.0
    c = 0.0
    alpha = 1.0
    Lambda = 2.0
    Lambda_justification = "c_F <= diam(Omega) = sqrt(2) on the unit square, so Lambda = 2 >= c_F^2"
    # recovered for the old disk records from experiments/expanding_disk_2d/config.py
    # ("Lambda = diam(Omega)^2 = 2"), which every stored disk config.json carries

    def __init__(self, kappa_mult: float = ex.KAPPA_DEFAULT):
        self.kappa_mult = kappa_mult

    @property
    def kappa(self) -> float:
        """Estimator weight kappa = alpha/Lambda + (|b|^2 + Lambda |c|^2)/alpha (= 1/2)."""
        return self.alpha / self.Lambda + (self.b**2 + self.Lambda * self.c**2) / self.alpha

    def f(self, t, x, y):
        return ex.f_exact(t, x, y, self.kappa_mult)

    def g(self, t, x, y):
        return ex.g(t, x, y)

    def g_t(self, t, x, y):
        return ex.g_t(t, x, y)

    def grad_g(self, t, x, y):
        return ex.grad_g(t, x, y)

    def u0(self, x, y):
        return ex.u0_exact(x, y, self.kappa_mult)

    def u_exact(self, t, x, y):
        return ex.u_exact(t, x, y, self.kappa_mult)

    def u_t_exact(self, t, x, y):
        return ex.u_t_exact(t, x, y, self.kappa_mult)

    def grad_u_exact(self, t, x, y):
        return ex.grad_u_exact(t, x, y, self.kappa_mult)

    def sigma_exact(self, t, x, y):
        return ex.sigma_exact(t, x, y, self.kappa_mult)

    def div_sigma_exact(self, t, x, y):
        return -ex.lap_u_exact(t, x, y, self.kappa_mult)

    def lambda_exact(self, t, x, y):
        return ex.lambda_exact(t, x, y, self.kappa_mult)

    @staticmethod
    def mask(x, y):
        return ex.B(x, y)

    @staticmethod
    def grad_mask(x, y):
        return ex.grad_B(x, y)
