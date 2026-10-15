"""Level-2 problem adapters: the two frozen benchmarks behind one interface.

Each adapter exposes
  * constants: d, T, Lambda, alpha, A (scalar, A = a I), kappa (estimator weight of ||w||^2),
    C_eq, C_rel, box (space-time box [0,T] x Omega as lo/hi lists);
  * the three MLPs (state N: R^{1+d} -> R, flux S: R^{1+d} -> R^d, mult M: R^{1+d} -> R) and the
    multiplier head ('square' only is supported by the enclosures; asserted);
  * DATA formulas written with the type-generic ops (tensor / Aff / Jet):
        mask(x) = B, dmask(x)[i] = d_{x_i} B, g, g_t, grad_g, f, u0, and the gap factorisation
        gap(t, x, N, N_t, N_x) -> (gamma, gamma_t, [d_{x_i} gamma]).
Only data (f, g, u0, coefficients, mask) and weights are used: never the exact solution or the
exact interface (TODO section 31).
"""

from __future__ import annotations

import math

from certified_obstacle_nn.certification.level2 import ops


class Level2Problem:
    name: str
    d: int
    T: float = 1.0
    C_rel: float = 216.0

    @property
    def m(self) -> int:
        return 1 + self.d

    @property
    def box(self):
        return [0.0] * self.m, [self.T] + [1.0] * self.d

    # u = B N  =>  u_t = B N_t, d_{x_i} u = dB_i N + B N_{x_i}
    def fields_from_net(self, P, net):
        """P: list of coordinates [t, x1, .., xd] (tensor / Aff / Jet); net: dict with
        N, N_t, N_x (list d), S (list d), dS (list d: d_{x_i} S_i), M (same type).
        Returns dict of fields: R, Rsig (list), gamma, gamma_t, ggrad (list), lam, u, R0-free."""
        t, X = P[0], P[1:]
        B = self.mask(X)
        dB = self.dmask(X)
        N, N_t, N_x = net["N"], net["N_t"], net["N_x"]
        u_t = B * N_t
        grad_u = [dB[i] * N + B * N_x[i] for i in range(self.d)]
        lam = ops.square(net["M"])
        div_s = net["dS"][0]
        for i in range(1, self.d):
            div_s = div_s + net["dS"][i]
        R = self.f(t, X) - u_t - div_s + lam
        a = math.sqrt(self.A)
        Rsig = [net["S"][i] * (1.0 / a) + grad_u[i] * a for i in range(self.d)]
        gamma, gamma_t, ggrad = self.gap(t, X, B, dB, N, N_t, N_x)
        return {"R": R, "Rsig": Rsig, "gamma": gamma, "gamma_t": gamma_t, "ggrad": ggrad, "lam": lam,
                "M": net["M"]}


class StefanL2(Level2Problem):
    """1D Stefan, specification section 9. Omega = (0,1), A = 1, b = c = 0, alpha = 1, Lambda = 1."""
    name = "stefan1d"
    d = 1
    Lambda = 1.0
    alpha = 1.0
    A = 1.0
    kappa = 1.0     # alpha/Lambda + (|b|^2 + Lambda |c|^2)/alpha
    C_eq = 2.0

    def __init__(self, triple):
        self.state = triple.state_net.net
        self.flux = triple.flux_net.net
        self.mult = triple.multiplier_net.net
        assert triple.multiplier_net.head == "square"
        assert triple.state_net.omega_bounds == (0.0, 1.0)
        assert triple.state_net.x_rescale is None

    @staticmethod
    def _h(t):
        return ops.exp(t) - t - 1.0

    def mask(self, X):
        x = X[0]
        return (x * (1.0 - x)) * 4.0

    def dmask(self, X):
        return [4.0 - 8.0 * X[0]]

    def g(self, t, X):
        return -(self._h(t) * (1.0 - X[0]))

    def g_t(self, t, X):
        return -((ops.exp(t) - 1.0) * (1.0 - X[0]))

    def grad_g(self, t, X):
        return [self._h(t)]

    def f(self, t, X):
        return -1.0 - (ops.exp(t) - 1.0) * (1.0 - X[0])

    def u0(self, X):
        return 0.0 * X[0]

    def gap(self, t, X, B, dB, N, N_t, N_x):
        u = B * N
        gamma = self.g(t, X) - u
        gamma_t = self.g_t(t, X) - B * N_t
        ggrad = [self.grad_g(t, X)[0] - (dB[0] * N + B * N_x[0])]
        return gamma, gamma_t, ggrad


class DiskL2(Level2Problem):
    """2D expanding disk (experiments/expanding_disk_2d/exact_solution.py), constant-multiplier
    variant. Omega = (0,1)^2, A = I, b = c = 0, alpha = 1, Lambda = 2, kappa_p = 1/2.
    Data (verbatim from exact_solution.py):
        B = 16 x(1-x) y(1-y),  h = e^t - t - 1,  g = -h B,  R(t) = 0.15 + 0.2 t,
        psi = (R^2 - r^2)/(2R) = R/2 - r^2/(2R),  r^2 = (x-1/2)^2 + (y-1/2)^2,
        f = g_t - Delta g - 2 kap + kap 1{psi>0} [2 psi psi_t - 2 psi Delta psi + 2 - 2 r^2/R^2],
        psi_t = R'(R^2 + r^2)/(2R^2),  Delta psi = -2/R,  Delta g = -h Delta B,
        Delta B = -32 (x(1-x) + y(1-y)),  u0 = kap (psi(0,.)^+)^2.
    The bracket vanishes on {psi = 0} (r = R), so f is continuous; ind_mul handles the kink."""
    name = "disk2d"
    d = 2
    alpha = 1.0
    A = 1.0
    C_eq = 1.0

    def __init__(self, triple, cfg):
        from certified_obstacle_nn.experiments.expanding_disk_2d import exact_solution as ex
        self.state, self.flux, self.mult = triple.state, triple.flux, triple.mult
        assert triple.head == "square"
        self.kap = float(cfg.kappa)
        self.Lambda = float(cfg.Lambda)
        assert float(cfg.alpha) == 1.0
        self.kappa = self.alpha / self.Lambda
        self.R0, self.Rp, self.c0 = ex.R0, ex.R_SPEED, ex.C0

    @staticmethod
    def _h(t):
        return ops.exp(t) - t - 1.0

    def mask(self, X):
        x, y = X
        return ((x * (1.0 - x)) * (y * (1.0 - y))) * 16.0

    def dmask(self, X):
        x, y = X
        return [((1.0 - 2.0 * x) * (y * (1.0 - y))) * 16.0, ((x * (1.0 - x)) * (1.0 - 2.0 * y)) * 16.0]

    def _lapB(self, X):
        x, y = X
        return (x * (1.0 - x) + y * (1.0 - y)) * -32.0

    def g(self, t, X):
        return -(self._h(t) * self.mask(X))

    def g_t(self, t, X):
        return -((ops.exp(t) - 1.0) * self.mask(X))

    def grad_g(self, t, X):
        h = self._h(t)
        return [-(h * dBi) for dBi in self.dmask(X)]

    def _r2(self, X):
        x, y = X
        return ops.square(x - self.c0) + ops.square(y - self.c0)

    def f(self, t, X):
        Rt = t * self.Rp + self.R0
        iR = ops.recip_pos(Rt)
        r2 = self._r2(X)
        psi = Rt * 0.5 - (r2 * iR) * 0.5
        psi_t = (ops.square(iR) * r2) * (0.5 * self.Rp) + 0.5 * self.Rp
        lap_psi = iR * -2.0
        bracket = (psi * psi_t) * 2.0 - (psi * lap_psi) * 2.0 + 2.0 - (r2 * ops.square(iR)) * 2.0
        lap_g = -(self._h(t) * self._lapB(X))
        return self.g_t(t, X) - lap_g - 2.0 * self.kap + ops.ind_mul(psi, bracket) * self.kap

    def u0(self, X):
        r2 = self._r2(X)
        psi0 = 0.5 * self.R0 - r2 * (0.5 / self.R0)
        return ops.relu_sq(psi0) * self.kap

    def gap(self, t, X, B, dB, N, N_t, N_x):
        h = self._h(t)
        hN = h + N
        gamma = -(B * hN)
        gamma_t = -(B * ((ops.exp(t) - 1.0) + N_t))
        ggrad = [-(dB[i] * hN + B * N_x[i]) for i in range(2)]
        return gamma, gamma_t, ggrad


def make_problem(model, meta):
    if "cfg" in meta:
        return DiskL2(model, meta["cfg"])
    return StefanL2(model)
