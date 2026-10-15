"""Abstract interface for a first-order parabolic obstacle problem.

See specification section 0 for the PDE and admissibility conditions, and section 4
for how alpha, A_inf, Lambda, kappa enter the computable estimator. Concrete
problems (Stefan, American put) use constant coefficients A, b, c, so kappa can
be computed once here from those constants; a future spatially varying problem
would need to override `kappa` directly with its own sup-norm computation.
"""

from abc import ABC, abstractmethod

import torch


class BaseProblem(ABC):
    @property
    @abstractmethod
    def A(self) -> float:
        """Diffusion coefficient."""

    @property
    @abstractmethod
    def b(self) -> float:
        """Convection coefficient."""

    @property
    @abstractmethod
    def c(self) -> float:
        """Reaction coefficient."""

    @property
    @abstractmethod
    def alpha(self) -> float:
        """Ellipticity constant: A >= alpha."""

    @property
    @abstractmethod
    def A_inf(self) -> float:
        """||A||_inf = ess sup ||A(t,x)||_2."""

    @property
    @abstractmethod
    def Lambda(self) -> float:
        """User-chosen constant, Lambda >= c_F^2."""

    @property
    def kappa(self) -> float:
        """alpha/Lambda + (||b||_inf^2 + Lambda ||c||_inf^2)/alpha, specification section 4."""
        return self.alpha / self.Lambda + (self.b ** 2 + self.Lambda * self.c ** 2) / self.alpha

    @property
    @abstractmethod
    def T(self) -> float:
        """Time horizon."""

    @property
    @abstractmethod
    def omega_bounds(self) -> tuple[float, float]:
        """(low, high) bounds of the 1D spatial domain Omega."""

    @property
    @abstractmethod
    def gamma_data(self) -> tuple[float, ...]:
        """x-locations of interior data interfaces Gamma_data (empty if none)."""

    @abstractmethod
    def f(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Source term, evaluated pointwise."""

    @abstractmethod
    def g(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Obstacle, evaluated pointwise."""

    @abstractmethod
    def u0(self, x: torch.Tensor) -> torch.Tensor:
        """Initial condition, evaluated pointwise."""

    # -- Problem-data sup-norms and Lipschitz constants, specification section 6 --
    # These are exact analytic constants of the closed-form problem data (g, f,
    # u0), needed to combine network bounds into the triple-level m_*, L_gamma,
    # L_sm, L_Psi, L_Phi0, L_PhiT quantities in certification/network_bounds.py.
    # Never sampled/estimated -- for Stefan these are hand-derived exact values.

    @property
    @abstractmethod
    def g_inf(self) -> float:
        """||g||_inf."""

    @property
    @abstractmethod
    def g_x_inf(self) -> float:
        """||grad_x g||_inf."""

    @property
    @abstractmethod
    def g_t_inf(self) -> float:
        """||d_t g||_inf."""

    @property
    @abstractmethod
    def g_xx_inf(self) -> float:
        """||d^2 g / dx^2||_inf."""

    @property
    @abstractmethod
    def g_tx_inf(self) -> float:
        """||d^2 g / dt dx||_inf."""

    @property
    @abstractmethod
    def g_tt_inf(self) -> float:
        """||d^2 g / dt^2||_inf."""

    @property
    @abstractmethod
    def f_inf(self) -> float:
        """||f||_inf."""

    @property
    @abstractmethod
    def f_x_inf(self) -> float:
        """||grad_x f||_inf."""

    @property
    @abstractmethod
    def f_t_inf(self) -> float:
        """||d_t f||_inf."""

    @property
    @abstractmethod
    def u0_inf(self) -> float:
        """||u0||_inf."""

    @property
    @abstractmethod
    def u0_lip(self) -> float:
        """Lip(u0) = ||d(u0)/dx||_inf (u0 depends on x alone)."""
