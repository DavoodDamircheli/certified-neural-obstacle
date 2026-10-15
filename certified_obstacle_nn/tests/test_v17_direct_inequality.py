"""v17 Stage 1 item 5 (roadmap R3): the direct-certificate INEQUALITIES (Theorem 4.10 for theta in {1/4, 1/2, 1},
the primary gradient form, Corollary 4.11 / printed (43)) hold on perturbed admissible triples around the exact
Stefan and expanding-disk solutions and on the stored Stefan seed-0 checkpoint (float64 quadrature, two
resolutions, gap must exceed the quadrature drift; see experiments/v17/inequality_check.py)."""

import math

import pytest
import torch

from certified_obstacle_nn.experiments.v14.common import STEFAN_SEEDS, load_frozen_stefan
from certified_obstacle_nn.experiments.v17 import inequality_check as IC
from certified_obstacle_nn.problems.stefan import StefanProblem

PI = math.pi
CF2_STEFAN = 1 / PI**2  # exact Friedrichs constant on (0,1)
CF2_DISK = 1 / (2 * PI**2)  # exact on (0,1)^2

STEFAN_FAMILIES = {
    # phi (zero at x = 0, 1), psi, chi (mu = (lambda + eps chi)_+ >= 0)
    "a": (lambda t, x: torch.sin(PI * x) * (0.5 + t), lambda t, x: torch.cos(PI * x) * t,
          lambda t, x: 1 + torch.sin(2 * PI * x) * torch.cos(PI * t)),
    "b": (lambda t, x: -torch.sin(2 * PI * x) * torch.exp(-t) - 0.3 * x * (1 - x),
          lambda t, x: torch.sin(3 * PI * x) * (1 - t) + 0.3, lambda t, x: x**2 * t - 0.5),
    "c": (lambda t, x: x * (1 - x) * torch.cos(3 * PI * t), lambda t, x: x, lambda t, x: 0 * x),
    "d": (lambda t, x: -torch.sin(PI * x), lambda t, x: 0 * x, lambda t, x: -1 + 0 * x),  # v below g: projection
}


def _stefan_trial(problem, fam, eps):
    phi, psi, chi = STEFAN_FAMILIES[fam]

    def trial(t, x):
        return (problem.u_exact(t, x) + eps * phi(t, x), problem.sigma_exact(t, x) + eps * psi(t, x),
                torch.relu(problem.lambda_exact(t, x) + eps * chi(t, x)))

    return trial


def _assert_all(res, label):
    for k, r in res.items():
        assert r["holds"], f"{label} {k}: LHS {r['lhs']:.6e} > RHS {r['rhs']:.6e} -- INEQUALITY VIOLATED"
        assert r["conclusive"], f"{label} {k}: gap {r['gap_rel']:.2e} not > 2 x drift {r['drift_rel']:.2e}"


@pytest.mark.parametrize("fam", sorted(STEFAN_FAMILIES))
@pytest.mark.parametrize("eps", [0.3, 0.1, 0.03])
def test_direct_inequalities_stefan_perturbed(fam, eps):
    p = StefanProblem()
    tr = _stefan_trial(p, fam, eps)
    q1, q2 = IC.stefan_quantities(p, tr, 48), IC.stefan_quantities(p, tr, 96)
    assert q2["graph_defect_max"] < 1e-12
    _assert_all(IC.judge(IC.inequalities(q1, CF2_STEFAN), IC.inequalities(q2, CF2_STEFAN)), f"stefan {fam} {eps}")


def test_direct_inequalities_stefan_seed0_checkpoint():
    p, triple = load_frozen_stefan(STEFAN_SEEDS[0])

    def trial(t, x):
        v, tau, mu = triple(t.reshape(-1, 1), x.reshape(-1, 1))
        return v.reshape(-1), tau.reshape(-1), mu.reshape(-1)

    q1, q2 = IC.stefan_quantities(p, trial, 96), IC.stefan_quantities(p, trial, 192)
    res = IC.judge(IC.inequalities(q1, CF2_STEFAN), IC.inequalities(q2, CF2_STEFAN))
    _assert_all(res, "stefan seed-0 checkpoint")


def _B(x, y):
    return 16 * x * (1 - x) * y * (1 - y)


DISK_FAMILIES = {
    "a": (lambda t, x, y: _B(x, y) * (0.5 + t) * torch.sin(PI * x),
          lambda t, x, y: (torch.cos(PI * x) * t, torch.sin(PI * y)),
          lambda t, x, y: 1 + torch.sin(2 * PI * x) * torch.cos(PI * t)),
    "b": (lambda t, x, y: -_B(x, y) * torch.exp(-t), lambda t, x, y: (x - 0.5, (y - 0.5) * t),
          lambda t, x, y: -2 + 0 * x),
    "c": (lambda t, x, y: _B(x, y) * torch.cos(3 * PI * t) * (x - y), lambda t, x, y: (0 * x, 0 * y),
          lambda t, x, y: 3 * x * y),
}


@pytest.mark.parametrize("fam", sorted(DISK_FAMILIES))
@pytest.mark.parametrize("eps", [0.3, 0.03])
def test_direct_inequalities_disk_perturbed(fam, eps):
    from certified_obstacle_nn.experiments.expanding_disk_2d import exact_solution as ex

    phi, psi, chi = DISK_FAMILIES[fam]

    def trial(t, x, y):
        s1, s2 = ex.sigma_exact(t, x, y)
        p1, p2 = psi(t, x, y)
        return (ex.u_exact(t, x, y) + eps * phi(t, x, y), (s1 + eps * p1, s2 + eps * p2),
                torch.relu(ex.lambda_exact(t, x, y) + eps * chi(t, x, y)))

    q1, q2 = IC.disk_quantities(trial, 16), IC.disk_quantities(trial, 32)
    assert q2["graph_defect_max"] < 1e-9
    _assert_all(IC.judge(IC.inequalities(q1, CF2_DISK), IC.inequalities(q2, CF2_DISK)), f"disk {fam} {eps}")
