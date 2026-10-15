"""Consistency checks for the Stefan exact solution, specification section 9.

Every identity is checked pointwise on many random samples. Derivatives needed
for the PDE/flux identities are computed two independent ways -- autograd
("closed form", exact for whatever is coded) and central finite differences
("independent numerical estimate") -- and cross-checked against each other so
that a bug in how the test itself differentiates cannot masquerade as a passing
identity. The identities themselves are then checked with the autograd
derivatives against the independently transcribed f, g, u0, so an algebra slip
in the exact solution shows up as a nonzero residual rather than being absorbed
into a self-consistent (but wrong) derivative.

t = x is the free boundary: lambda_exact jumps there and sigma_x has a genuine
kink, so a thin band around it is excluded from every pointwise/derivative
check (finite differences straddling a kink are meaningless, not evidence of
a bug).
"""

import torch

from certified_obstacle_nn.problems.stefan import StefanProblem

TOL_IDENTITY = 1e-10
TOL_FD = 1e-6
BAND = 1e-2
FD_H = 1e-6
N_POINTS = 5000


def _sample_interior(n, band, seed):
    generator = torch.Generator().manual_seed(seed)
    t = torch.rand(n, generator=generator, dtype=torch.float64)
    x = torch.rand(n, generator=generator, dtype=torch.float64)
    mask = (t - x).abs() > band
    return t[mask], x[mask]


def _autograd_derivs(problem, t, x):
    t = t.clone().requires_grad_(True)
    x = x.clone().requires_grad_(True)
    u = problem.u_exact(t, x)
    sigma = problem.sigma_exact(t, x)
    u_t, u_x = torch.autograd.grad(u, (t, x), grad_outputs=torch.ones_like(u), retain_graph=True)
    (sigma_x,) = torch.autograd.grad(sigma, x, grad_outputs=torch.ones_like(sigma))
    return u_t.detach(), u_x.detach(), sigma_x.detach()


def _fd_derivs(problem, t, x, h):
    u_t = (problem.u_exact(t + h, x) - problem.u_exact(t - h, x)) / (2 * h)
    u_x = (problem.u_exact(t, x + h) - problem.u_exact(t, x - h)) / (2 * h)
    sigma_x = (problem.sigma_exact(t, x + h) - problem.sigma_exact(t, x - h)) / (2 * h)
    return u_t, u_x, sigma_x


def test_autograd_matches_finite_difference():
    problem = StefanProblem()
    t, x = _sample_interior(N_POINTS, BAND, seed=0)

    u_t_ag, u_x_ag, sigma_x_ag = _autograd_derivs(problem, t, x)
    u_t_fd, u_x_fd, sigma_x_fd = _fd_derivs(problem, t, x, FD_H)

    max_u_t = (u_t_ag - u_t_fd).abs().max().item()
    max_u_x = (u_x_ag - u_x_fd).abs().max().item()
    max_sigma_x = (sigma_x_ag - sigma_x_fd).abs().max().item()
    print(f"max |u_t autograd - FD| = {max_u_t:.3e}")
    print(f"max |u_x autograd - FD| = {max_u_x:.3e}")
    print(f"max |sigma_x autograd - FD| = {max_sigma_x:.3e}")

    assert max_u_t < TOL_FD
    assert max_u_x < TOL_FD
    assert max_sigma_x < TOL_FD


def test_pde_residual():
    problem = StefanProblem()
    t, x = _sample_interior(N_POINTS, BAND, seed=1)

    u_t, _, sigma_x = _autograd_derivs(problem, t, x)
    lam = problem.lambda_exact(t, x)
    f = problem.f(t, x)

    max_violation = (f - u_t - sigma_x + lam).abs().max().item()
    print(f"max |R_PDE| (u_t + sigma_x - lambda = f) = {max_violation:.3e}")
    assert max_violation < TOL_IDENTITY


def test_sigma_relation():
    problem = StefanProblem()
    t, x = _sample_interior(N_POINTS, BAND, seed=2)

    _, u_x, _ = _autograd_derivs(problem, t, x)
    sigma = problem.sigma_exact(t, x)

    max_violation = (sigma + u_x).abs().max().item()
    print(f"max |R_sig| (sigma + u_x = 0) = {max_violation:.3e}")
    assert max_violation < TOL_IDENTITY


def test_obstacle_and_complementarity():
    problem = StefanProblem()
    t, x = _sample_interior(N_POINTS, BAND, seed=3)

    u = problem.u_exact(t, x)
    g = problem.g(t, x)
    lam = problem.lambda_exact(t, x)

    gap_violation = (g - u).clamp(min=0.0).max().item()
    lam_violation = (-lam).clamp(min=0.0).max().item()
    complementarity_violation = (lam * (u - g)).abs().max().item()

    print(f"max obstacle violation (u >= g) = {gap_violation:.3e}")
    print(f"max lambda-sign violation (lambda >= 0) = {lam_violation:.3e}")
    print(f"max complementarity violation (lambda*(u-g) = 0) = {complementarity_violation:.3e}")

    assert gap_violation < TOL_IDENTITY
    assert lam_violation < TOL_IDENTITY
    assert complementarity_violation < TOL_IDENTITY


def test_boundary_and_initial_conditions():
    problem = StefanProblem()
    generator = torch.Generator().manual_seed(4)
    t = torch.rand(N_POINTS, generator=generator, dtype=torch.float64)
    x = torch.rand(N_POINTS, generator=generator, dtype=torch.float64)

    left = problem.u_exact(t, torch.zeros_like(t)).abs().max().item()
    right = problem.u_exact(t, torch.ones_like(t)).abs().max().item()
    initial = (problem.u_exact(torch.zeros_like(x), x) - problem.u0(x)).abs().max().item()

    print(f"max |u(t,0)| = {left:.3e}")
    print(f"max |u(t,1)| = {right:.3e}")
    print(f"max |u(0,x) - u0(x)| = {initial:.3e}")

    assert left < TOL_IDENTITY
    assert right < TOL_IDENTITY
    assert initial < TOL_IDENTITY
