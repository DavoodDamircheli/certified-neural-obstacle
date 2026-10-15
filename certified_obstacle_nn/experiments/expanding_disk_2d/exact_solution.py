"""Exact manufactured solution for the 2D expanding-disk obstacle benchmark.

Formulas are taken verbatim from the 2D benchmark specification section 2
(authoritative; see NOTES_existing_code.md for how this matches the existing 1D Stefan
conventions in specification). Problem: Omega=(0,1)^2, T=1, A=I, b=0, c=0.

u(t,x,y) = g(t,x,y) + kappa * psi_+(t,x,y)^2            (constant-multiplier variant,
                                                           specification-prompt section 2.3 default)
sigma     = -grad_x u
lambda    = 2*kappa * 1{psi < 0}                          (constant on the contact set)
f         = u_t - Delta u - lambda                        (closed form, derived in the prompt)

Every function is written with torch.where / torch.relu, never Python branching on tensor
values, so every function here is autograd-safe (verify_exact_problem.py differentiates
u_exact directly). All derivative formulas (grad_B, lap_B, grad_g, lap_g, grad_psi, lap_psi,
psi_t, grad_u_exact, u_t_exact, lap_u_exact) are EXPLICIT closed forms, never autograd --
autograd is used only in verify_exact_problem.py, as an independent cross-check (ground rule
4 in the prompt's Section 3).

All public functions accept (t, x, y) (or (x, y), or (x0, y0) for the t=0 slice) as same-shape
tensors (any shape, typically [N] or [N,1]) and return a tensor of that same shape, except the
vector-valued quantities (grad_B, grad_g, grad_psi, grad_u_exact, sigma_exact), which return a
pair (component_x, component_y) of same-shape tensors -- the 2D analogue of how the 1D Stefan
code (problems/stefan.py) returns bare scalars per spatial component.

lap_u_exact is not in the prompt's required function list for exact_solution.py (section 5
lists only R/h/B/gradB/lapB/g/g_t/grad_g/lap_g/psi/grad_psi/lap_psi/psi_t/u_exact/u_t_exact/
grad_u_exact/sigma_exact/lambda_exact/f_exact/u0_exact) but is included here as a small
closed-form helper: verify_exact_problem.py's PDE-residual check needs Delta u in explicit-
formula form (u_t - Delta u - f == lambda), and f_exact itself was manufactured FROM this
Laplacian (see its derivation in the module-level comment on f_exact below), so exposing it
is cheaper and less error-prone than re-deriving it inline in the verification script.
"""

import torch

C0 = 0.5
R0 = 0.15
R_SPEED = 0.20
KAPPA_DEFAULT = 5.0


def R(t: torch.Tensor) -> torch.Tensor:
    return R0 + R_SPEED * t


def R_prime(t: torch.Tensor) -> torch.Tensor:
    return torch.full_like(t, R_SPEED)


def r_squared(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x - C0) ** 2 + (y - C0) ** 2


def h(t: torch.Tensor) -> torch.Tensor:
    return torch.exp(t) - t - 1.0


def h_prime(t: torch.Tensor) -> torch.Tensor:
    return torch.exp(t) - 1.0


def B(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 16.0 * x * (1.0 - x) * y * (1.0 - y)


def grad_B(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    Bx = 16.0 * (1.0 - 2.0 * x) * y * (1.0 - y)
    By = 16.0 * x * (1.0 - x) * (1.0 - 2.0 * y)
    return Bx, By


def lap_B(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return -32.0 * (x * (1.0 - x) + y * (1.0 - y))


def g(t: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return -h(t) * B(x, y)


def g_t(t: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return -h_prime(t) * B(x, y)


def grad_g(t: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    Bx, By = grad_B(x, y)
    hh = h(t)
    return -hh * Bx, -hh * By


def lap_g(t: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return -h(t) * lap_B(x, y)


def psi(t: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    Rt = R(t)
    return (Rt ** 2 - r_squared(x, y)) / (2.0 * Rt)


def grad_psi(t: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    Rt = R(t)
    return -(x - C0) / Rt, -(y - C0) / Rt


def lap_psi(t: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return -2.0 / R(t) * torch.ones_like(x)


def psi_t(t: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    Rt = R(t)
    Rp = R_prime(t)
    r2 = r_squared(x, y)
    return Rp * (Rt ** 2 + r2) / (2.0 * Rt ** 2)


def _grad_psi_sq(t: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """|grad psi|^2 = r^2 / R^2 (prompt section 2.2) -- a scalar helper, not part of the
    required function list, used by f_exact and lap_u_exact below."""
    return r_squared(x, y) / R(t) ** 2


def u_exact(t: torch.Tensor, x: torch.Tensor, y: torch.Tensor, kappa: float = KAPPA_DEFAULT) -> torch.Tensor:
    psi_plus = torch.relu(psi(t, x, y))
    return g(t, x, y) + kappa * psi_plus ** 2


def u_t_exact(t: torch.Tensor, x: torch.Tensor, y: torch.Tensor, kappa: float = KAPPA_DEFAULT) -> torch.Tensor:
    psi_plus = torch.relu(psi(t, x, y))
    return g_t(t, x, y) + 2.0 * kappa * psi_plus * psi_t(t, x, y)


def grad_u_exact(
    t: torch.Tensor, x: torch.Tensor, y: torch.Tensor, kappa: float = KAPPA_DEFAULT
) -> tuple[torch.Tensor, torch.Tensor]:
    psi_plus = torch.relu(psi(t, x, y))
    gx, gy = grad_g(t, x, y)
    psix, psiy = grad_psi(t, x, y)
    return gx + 2.0 * kappa * psi_plus * psix, gy + 2.0 * kappa * psi_plus * psiy


def lap_u_exact(t: torch.Tensor, x: torch.Tensor, y: torch.Tensor, kappa: float = KAPPA_DEFAULT) -> torch.Tensor:
    """Delta u = Delta g + kappa * 1{psi>0} * [2*psi*Delta psi + 2|grad psi|^2], from the
    standard identity Delta(f^2) = 2 f Delta f + 2|grad f|^2 applied to f = psi_+ (the
    indicator drops out automatically outside the disk since psi_+ = 0 there, matching how
    u_t_exact/grad_u_exact need no explicit indicator either)."""
    psi_val = psi(t, x, y)
    indicator = (psi_val > 0).to(t.dtype)
    bracket = 2.0 * psi_val * lap_psi(t, x, y) + 2.0 * _grad_psi_sq(t, x, y)
    return lap_g(t, x, y) + kappa * indicator * bracket


def sigma_exact(
    t: torch.Tensor, x: torch.Tensor, y: torch.Tensor, kappa: float = KAPPA_DEFAULT
) -> tuple[torch.Tensor, torch.Tensor]:
    ux, uy = grad_u_exact(t, x, y, kappa)
    return -ux, -uy


def lambda_exact(t: torch.Tensor, x: torch.Tensor, y: torch.Tensor, kappa: float = KAPPA_DEFAULT) -> torch.Tensor:
    psi_val = psi(t, x, y)
    return torch.where(psi_val < 0, torch.full_like(psi_val, 2.0 * kappa), torch.zeros_like(psi_val))


def f_exact(t: torch.Tensor, x: torch.Tensor, y: torch.Tensor, kappa: float = KAPPA_DEFAULT) -> torch.Tensor:
    """f = g_t - Delta g - 2*kappa + kappa * 1{psi>0} * [2*psi*psi_t - 2*psi*Delta psi
    + 2 - 2|grad psi|^2], prompt section 2.3. Manufactured so that u_t - Delta u - f ==
    lambda identically on both {psi>0} and {psi<0} (hand-verified in the module docstring
    discussion and checked numerically by verify_exact_problem.py)."""
    psi_val = psi(t, x, y)
    indicator = (psi_val > 0).to(t.dtype)
    bracket = (
        2.0 * psi_val * psi_t(t, x, y)
        - 2.0 * psi_val * lap_psi(t, x, y)
        + 2.0
        - 2.0 * _grad_psi_sq(t, x, y)
    )
    return g_t(t, x, y) - lap_g(t, x, y) - 2.0 * kappa + kappa * indicator * bracket


def u0_exact(x: torch.Tensor, y: torch.Tensor, kappa: float = KAPPA_DEFAULT) -> torch.Tensor:
    """u_exact(0,.) = kappa * psi(0,.)_+^2, since g(0,.) = -h(0)*B = 0 (h(0)=0)."""
    t0 = torch.zeros_like(x)
    psi0 = psi(t0, x, y)
    return kappa * torch.relu(psi0) ** 2
