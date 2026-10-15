"""American put benchmark problem, specification section 10 (Phase 4).

Log-price x = log(S), tau = time-to-maturity (so "t" throughout this
codebase's (t,x) convention IS tau here -- the standard backward-in-calendar-
time Black-Scholes PDE becomes a FORWARD initial-value problem under
tau = T - t_calendar, matching specification section 0's u(0)=u0 framework
exactly). varsigma=0.4, r=0.06, L=-1, R=7, K=100, T=1;
alpha=A=varsigma^2/2=0.08, b=-(r-alpha), c=r.

u here is the LIFTED unknown u = V - u_lift, where V is the true (unlifted)
American put value: u_lift(x) = (K-e^L)^+ * (R-x)/(R-L) is the unique linear
function with u_lift(L) = (K-e^L)^+, u_lift(R) = 0 (the American put's own
deep-ITM / deep-OTM truncated-domain boundary values), chosen so u = V -
u_lift satisfies HOMOGENEOUS Dirichlet conditions u(tau,L) = u(tau,R) = 0 --
required by specification section 0's general framework. Substituting V = u +
u_lift into the (linear, homogeneous-coefficient) Black-Scholes PDE and using
that u_lift is linear (u_lift_xx=0) and tau-independent (d_tau u_lift=0)
gives the closed-form source term f(x) = -(b*u_lift_x + c*u_lift) below --
verified algebraically in tests/test_american_put_exact.py, not just quoted.

g(x) = (K-e^x)^+ - u_lift(x) is the LIFTED obstacle (also tau-independent:
the American put's obstacle, the intrinsic value, does not depend on time).
u0(x) = g(x) exactly: at tau=0 (maturity) the option is worth exactly its
intrinsic value, i.e. u starts exactly ON the obstacle -- the standard
American-option terminal condition.

Data interface at x* = log(K) (specification section 10): (K-e^x)^+ has a kink
there (continuous value, discontinuous first derivative -- a jump of exactly
K in g_x, verified in tests/test_american_put_exact.py), so g_xx does not
exist classically at x*. Every *_xx_inf / *_tx_inf property below is an
ESS SUP (a single point is measure-zero, consistent with
BaseProblem.A_inf's own "ess sup" convention) over Omega \\ {x*}, computed
in closed form from the two smooth one-sided pieces -- never sampled, and
never silently including the kink itself as if g were smooth there.

gamma_data = (log(K),): certification/cell_classifier.py's data-cut handling
(specification section 7) is how the actual certification machinery accounts for
this interface; this problem class only reports its location and the
correct (finite) constants for the SMOOTH remainder of the domain.

Normalized training convention (2026-09-12), specification section 10: train on
ubar=u/K, gbar=g/K, fbar=f/K, feeding the network xi=2(x-L)/(R-L)-1 but
differentiating w.r.t. PHYSICAL x. `xi()` below is the rescaling formula
itself (a pure function of x, used by models/state_net.py's own optional
x_rescale hook -- see its module docstring for why autograd still supplies
the correct chain-rule factor). `NormalizedAmericanPutProblem` is the
TRAINING-only view (g/f/u0 divided by K); this class itself (g/f/u0) stays
PHYSICAL always, matching reference/american_put_reference.py's own
physical-unit PSOR solve, so E_true/certification (specification's "convert
back to physical units before certifying") compares like with like --
models/unit_conversion.py's PhysicalUnitsTriple is the corresponding
post-training K-scaling-back step for the trained triple's own outputs.

verify_boundary_and_initial_conditions() (2026-09-12, this milestone's own
"before training, verify numerically" instruction): a NUMERICAL, not just
algebraic, check that g(t,L)=g(t,R)=0 and u0=g(0,.) -- run once before any
training starts, raising loudly on violation rather than silently trusting
the closed-form derivation (which tests/test_american_put_exact.py already
checks statically, but a pre-flight runtime check catches e.g. a future
parameter change that breaks the lift's own boundary cancellation).
"""

import math

import torch

from certified_obstacle_nn.problems.base_problem import BaseProblem


class AmericanPutProblem(BaseProblem):
    varsigma = 0.4
    r = 0.06
    L = -1.0
    R = 7.0
    K = 100.0
    T = 1.0

    alpha = varsigma ** 2 / 2.0
    A = alpha
    A_inf = alpha
    b = -(r - alpha)
    c = r

    omega_bounds = (L, R)
    Lambda = (R - L) ** 2  # >= c_F^2 (c_F <= diam(Omega) = R-L = 8, so c_F^2 <= 64 = Lambda exactly)

    x_star = math.log(K)  # data interface, specification section 10
    gamma_data = (x_star,)

    C = max(K - math.exp(L), 0.0)  # (K - e^L)^+
    _x_crit = math.log(C / (R - L))  # where g_x=0 on the left (smooth) piece, if interior

    def normalized_view(self) -> "NormalizedAmericanPutProblem":
        """Generic hook scripts/certify_cli.py uses (via getattr, never a
        hardcoded problem-name check) to get the matching NORMALIZED problem
        view for computing a priori bounds on a normalized-output triple
        before certification/network_bounds.py's physical_units_bounds
        converts them back."""
        return NormalizedAmericanPutProblem()

    def f(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return -(self.C / (self.R - self.L)) * ((self.r - self.alpha) + self.r * (self.R - x))

    def g(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        payoff = torch.clamp(self.K - torch.exp(x), min=0.0)
        return payoff - self._u_lift(x)

    def u0(self, x: torch.Tensor) -> torch.Tensor:
        return self.g(torch.zeros_like(x), x)

    def _u_lift(self, x: torch.Tensor) -> torch.Tensor:
        return self.C * (self.R - x) / (self.R - self.L)

    def xi(self, x: torch.Tensor) -> torch.Tensor:
        """xi = 2(x-L)/(R-L) - 1: maps [L,R] -> [-1,1]. A plain differentiable
        function of x (no detach, no Python-float branch) -- this is what
        lets autograd supply the chain-rule factor d(xi)/dx = 2/(R-L)
        automatically when this is used as models/state_net.py's x_rescale."""
        return 2.0 * (x - self.L) / (self.R - self.L) - 1.0

    # -- Problem-data sup-norms and Lipschitz constants, specification section 6 --
    # g, f, u0 are all tau-independent here (both g and f depend on x alone --
    # verified in tests/test_american_put_exact.py), so every *_t*_inf is 0.

    @property
    def g_t_inf(self) -> float:
        return 0.0

    @property
    def g_tx_inf(self) -> float:
        return 0.0

    @property
    def g_tt_inf(self) -> float:
        return 0.0

    @property
    def f_t_inf(self) -> float:
        return 0.0

    @property
    def g_inf(self) -> float:
        L, R, K, C = self.L, self.R, self.K, self.C
        x_star, x_crit = self.x_star, self._x_crit
        candidates = [self._g_scalar(L), self._g_scalar(R), self._g_scalar(x_star)]
        if L < x_crit < x_star:
            candidates.append(self._g_scalar(x_crit))
        return max(abs(v) for v in candidates)

    @property
    def g_x_inf(self) -> float:
        # g_x(x) = -e^x + C/(R-L) for x < x_star (monotonically decreasing,
        # so its extremes on [L, x_star) are exactly the two endpoints), and
        # the constant C/(R-L) for x >= x_star.
        L, R, K, C = self.L, self.R, self.K, self.C
        g_x_at_L = -math.exp(L) + C / (R - L)
        g_x_kink_left_limit = -K + C / (R - L)  # lim x->x_star^-
        g_x_right = C / (R - L)
        return max(abs(g_x_at_L), abs(g_x_kink_left_limit), abs(g_x_right))

    @property
    def g_xx_inf(self) -> float:
        # ess sup over Omega \ {x_star}: |{-e^x}| on (L,x_star) is increasing,
        # approaching e^{x_star}=K in the limit; exactly 0 on (x_star,R).
        return self.K

    @property
    def f_inf(self) -> float:
        L, R = self.L, self.R
        f_L = float(self.f(torch.tensor(0.0), torch.tensor(L)).item())
        f_R = float(self.f(torch.tensor(0.0), torch.tensor(R)).item())
        return max(abs(f_L), abs(f_R))

    @property
    def f_x_inf(self) -> float:
        return abs(self.C * self.r / (self.R - self.L))  # f is exactly linear in x

    @property
    def u0_inf(self) -> float:
        return self.g_inf

    @property
    def u0_lip(self) -> float:
        return self.g_x_inf

    def _g_scalar(self, x: float) -> float:
        t = torch.tensor(0.0, dtype=torch.float64)
        return float(self.g(t, torch.tensor(x, dtype=torch.float64)).item())


def verify_boundary_and_initial_conditions(problem: "AmericanPutProblem", n_t_samples: int = 200, tol: float = 1e-9) -> None:
    """NUMERICAL (not just algebraic) pre-flight check, run once before any
    training starts: g(t,L)=g(t,R)=0 for a sample of t values, and
    u0(x)=g(0,x) over a dense x sample. Raises AssertionError on violation --
    never silently proceeds to train against a boundary condition that does
    not actually hold at runtime."""
    t = torch.linspace(0.0, problem.T, n_t_samples, dtype=torch.float64)
    L_col = torch.full_like(t, problem.L)
    R_col = torch.full_like(t, problem.R)

    g_at_L = problem.g(t, L_col)
    g_at_R = problem.g(t, R_col)
    assert g_at_L.abs().max().item() < tol, f"g(t,L) != 0: max|g(t,L)|={g_at_L.abs().max().item():.3e}"
    assert g_at_R.abs().max().item() < tol, f"g(t,R) != 0: max|g(t,R)|={g_at_R.abs().max().item():.3e}"

    x = torch.linspace(problem.L, problem.R, 2000, dtype=torch.float64)
    u0_val = problem.u0(x)
    g_at_0 = problem.g(torch.zeros_like(x), x)
    diff = (u0_val - g_at_0).abs().max().item()
    assert diff < tol, f"u0 != g(0,.): max|u0-g(0,.)|={diff:.3e}"


class NormalizedAmericanPutProblem(AmericanPutProblem):
    """TRAINING-only view: g/f/u0 divided by K (specification section 10's own
    "train on normalized ubar=u/K, gbar=g/K, fbar=f/K"). Every other
    property (A, b, c, alpha, domain, gamma_data, and the certification-only
    sup-norm/Lipschitz constants) is inherited UNCHANGED from
    AmericanPutProblem -- certification must always see the PHYSICAL
    problem (models/unit_conversion.py's PhysicalUnitsTriple converts the
    TRAINED triple back before it is ever certified), this subclass exists
    ONLY to feed training/trainer.py's loss the correctly-scaled targets.

    u0 is deliberately NOT overridden here: AmericanPutProblem.u0(x) is
    defined as self.g(0,x) (dynamic dispatch), so calling it on a
    NormalizedAmericanPutProblem instance already resolves self.g to THIS
    class's overridden (normalized) g -- u0 comes out correctly normalized
    for free. Adding a redundant "/ self.K" override here was tried and
    divided by K TWICE (gbar/K instead of gbar); tests/test_american_put_exact.py's
    test_normalized_problem_scales_g_f_u0_by_one_over_K caught it.

    The SAME dynamic-dispatch mechanism means g_inf, g_x_inf, f_inf, f_x_inf,
    u0_inf, u0_lip (all of which internally call self.g/self.f, directly or
    via self._g_scalar) also come out scaled by 1/K automatically on THIS
    class, not left at their physical values -- this is exactly why
    certification code must always instantiate the BASE AmericanPutProblem
    directly, never this class: a "sup-norm" of gbar is a real number, just
    not the one certification/network_bounds.py's formulas need."""

    def f(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return super().f(t, x) / self.K

    def g(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return super().g(t, x) / self.K
