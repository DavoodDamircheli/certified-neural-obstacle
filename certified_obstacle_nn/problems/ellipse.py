"""Off-centre expanding-ellipse obstacle benchmark (roadmap v14 section 6).

Q = (0,1) x (0,1)^2, A = I, b = c = 0, alpha = 1, Lambda = 2 (c_F <= diam = sqrt 2).

    B   = 16 x1(1-x1) x2(1-x2)
    u_b = B (0.10 + 0.05 t)(1 + 0.2 x1 - 0.1 x2)
    a(t) = 0.12 + 0.10 t,  d(t) = 0.08 + 0.07 t
    zeta = (x1-0.43)^2/a^2 + (x2-0.57)^2/d^2 - 1,   kappa0 = 0.01
    g      = u_b - kappa0 zeta^2
    u      = u_b - kappa0 (zeta_-)^2,  zeta_- = max(-zeta, 0)
    sigma  = -grad u
    lambda = 2 kappa0 |grad zeta|^2 1{zeta < 0}
    f      = (d_t - Delta) u_b - 2 kappa0 zeta (zeta_t - Delta zeta) 1{zeta < 0}
    u0     = u(0, .)

Branches: contact (zeta < 0, u = g, lambda > 0) and non-contact (zeta >= 0,
u = u_b, lambda = 0). Every formula below is an explicit closed form; the
sympy checks live in tests/test_ellipse_exact.py. All functions take torch
tensors (t, x1, x2) of one shape and are autograd-safe (torch.where).

The obstacle effect is small relative to the load: the contact correction is
at most kappa0 = 0.01 against max u_b ~ 0.158 (see PROGRESS.md for the
measured magnitudes). kappa0 = 0.01 is the main benchmark; other values are
only for explicitly labelled variants.
"""

from __future__ import annotations

import math

import torch

KAPPA0 = 0.01
C1, C2 = 0.43, 0.57
A0, A1 = 0.12, 0.10
D0, D1 = 0.08, 0.07


class EllipseProblem:
    name = "ellipse"
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

    def __init__(self, kappa0: float = KAPPA0):
        self.kappa0 = kappa0

    @property
    def kappa(self) -> float:
        return self.alpha / self.Lambda + (self.b**2 + self.Lambda * self.c**2) / self.alpha

    # ------------------------------------------------------------- geometry
    @staticmethod
    def a(t):
        return A0 + A1 * t

    @staticmethod
    def d(t):
        return D0 + D1 * t

    def zeta(self, t, x1, x2):
        return (x1 - C1) ** 2 / self.a(t) ** 2 + (x2 - C2) ** 2 / self.d(t) ** 2 - 1.0

    def grad_zeta(self, t, x1, x2):
        return 2.0 * (x1 - C1) / self.a(t) ** 2, 2.0 * (x2 - C2) / self.d(t) ** 2

    def lap_zeta(self, t, x1, x2):
        return 2.0 / self.a(t) ** 2 + 2.0 / self.d(t) ** 2 + 0.0 * x1

    def zeta_t(self, t, x1, x2):
        return -2.0 * A1 * (x1 - C1) ** 2 / self.a(t) ** 3 - 2.0 * D1 * (x2 - C2) ** 2 / self.d(t) ** 3

    # ------------------------------------------------------------- u_b
    @staticmethod
    def mask(x1, x2):
        return 16.0 * x1 * (1 - x1) * x2 * (1 - x2)

    @staticmethod
    def grad_mask(x1, x2):
        return 16.0 * (1 - 2 * x1) * x2 * (1 - x2), 16.0 * x1 * (1 - x1) * (1 - 2 * x2)

    @staticmethod
    def _parts(t, x1, x2):
        p, q = x1 * (1 - x1), x2 * (1 - x2)
        dp, dq = 1 - 2 * x1, 1 - 2 * x2
        B = 16 * p * q
        Bx1, Bx2 = 16 * dp * q, 16 * p * dq
        lapB = -32 * (q + p)
        c = 0.10 + 0.05 * t
        L = 1 + 0.2 * x1 - 0.1 * x2
        return B, Bx1, Bx2, lapB, c, L

    def ub(self, t, x1, x2):
        B, _, _, _, c, L = self._parts(t, x1, x2)
        return B * c * L

    def ub_t(self, t, x1, x2):
        B, _, _, _, c, L = self._parts(t, x1, x2)
        return B * 0.05 * L

    def grad_ub(self, t, x1, x2):
        B, Bx1, Bx2, _, c, L = self._parts(t, x1, x2)
        return c * (Bx1 * L + 0.2 * B), c * (Bx2 * L - 0.1 * B)

    def lap_ub(self, t, x1, x2):
        B, Bx1, Bx2, lapB, c, L = self._parts(t, x1, x2)
        return c * (lapB * L + 2 * (0.2 * Bx1 - 0.1 * Bx2))

    # ------------------------------------------------------------- data
    def g(self, t, x1, x2):
        return self.ub(t, x1, x2) - self.kappa0 * self.zeta(t, x1, x2) ** 2

    def g_t(self, t, x1, x2):
        z = self.zeta(t, x1, x2)
        return self.ub_t(t, x1, x2) - 2 * self.kappa0 * z * self.zeta_t(t, x1, x2)

    def grad_g(self, t, x1, x2):
        z = self.zeta(t, x1, x2)
        z1, z2 = self.grad_zeta(t, x1, x2)
        u1, u2 = self.grad_ub(t, x1, x2)
        return u1 - 2 * self.kappa0 * z * z1, u2 - 2 * self.kappa0 * z * z2

    def contact(self, t, x1, x2):
        return self.zeta(t, x1, x2) < 0

    def f(self, t, x1, x2):
        z = self.zeta(t, x1, x2)
        corr = 2 * self.kappa0 * z * (self.zeta_t(t, x1, x2) - self.lap_zeta(t, x1, x2))
        base = self.ub_t(t, x1, x2) - self.lap_ub(t, x1, x2)
        return base - torch.where(z < 0, corr, torch.zeros_like(corr))

    def u0(self, x1, x2):
        return self.u_exact(torch.zeros_like(x1), x1, x2)

    # ------------------------------------------------------------- exact solution
    def u_exact(self, t, x1, x2):
        zm = torch.clamp(-self.zeta(t, x1, x2), min=0.0)
        return self.ub(t, x1, x2) - self.kappa0 * zm**2

    def u_t_exact(self, t, x1, x2):
        z = self.zeta(t, x1, x2)
        corr = 2 * self.kappa0 * z * self.zeta_t(t, x1, x2)
        return self.ub_t(t, x1, x2) - torch.where(z < 0, corr, torch.zeros_like(corr))

    def grad_u_exact(self, t, x1, x2):
        z = self.zeta(t, x1, x2)
        z1, z2 = self.grad_zeta(t, x1, x2)
        u1, u2 = self.grad_ub(t, x1, x2)
        k = torch.where(z < 0, 2 * self.kappa0 * z, torch.zeros_like(z))
        return u1 - k * z1, u2 - k * z2

    def lap_u_exact(self, t, x1, x2):
        z = self.zeta(t, x1, x2)
        z1, z2 = self.grad_zeta(t, x1, x2)
        corr = self.kappa0 * (2 * (z1**2 + z2**2) + 2 * z * self.lap_zeta(t, x1, x2))
        return self.lap_ub(t, x1, x2) - torch.where(z < 0, corr, torch.zeros_like(corr))

    def sigma_exact(self, t, x1, x2):
        g1, g2 = self.grad_u_exact(t, x1, x2)
        return -g1, -g2

    def div_sigma_exact(self, t, x1, x2):
        return -self.lap_u_exact(t, x1, x2)

    def lambda_exact(self, t, x1, x2):
        z = self.zeta(t, x1, x2)
        z1, z2 = self.grad_zeta(t, x1, x2)
        lam = 2 * self.kappa0 * (z1**2 + z2**2)
        return torch.where(z < 0, lam, torch.zeros_like(lam))

    # ------------------------------------------------------------- interface
    def ellipse_boundary(self, t, theta):
        """Points of {zeta = 0} at time t, angle theta (exact)."""
        return C1 + self.a(t) * math.cos(theta), C2 + self.d(t) * math.sin(theta)
