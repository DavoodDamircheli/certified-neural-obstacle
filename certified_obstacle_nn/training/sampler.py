"""Uniform interior and initial-time samplers, specification section 3.

Points are drawn from a local, explicitly seeded torch.Generator -- not the
global RNG -- so a given seed always reproduces the same batch regardless of
what other code ran before it. The most recently sampled points are exposed
(detached) via `interior_points` / `initial_points` so that certification code
can later assert its own quadrature points are disjoint from them, per
specification section 12 invariant 1. Certification must never be handed this
sampler object itself, only compare against the point sets it exposes.
"""

import torch

from certified_obstacle_nn.problems.base_problem import BaseProblem


class TrainingSampler:
    def __init__(self, problem: BaseProblem, n_interior: int, n_initial: int, seed: int):
        self.problem = problem
        self.n_interior = n_interior
        self.n_initial = n_initial
        self.seed = seed
        self._t: torch.Tensor | None = None
        self._x: torch.Tensor | None = None
        self._x0: torch.Tensor | None = None

    def sample(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(self.seed)
        a, b = self.problem.omega_bounds
        T = self.problem.T

        t = torch.rand(self.n_interior, 1, generator=generator, dtype=torch.float64) * T
        x = torch.rand(self.n_interior, 1, generator=generator, dtype=torch.float64) * (b - a) + a
        x0 = torch.rand(self.n_initial, 1, generator=generator, dtype=torch.float64) * (b - a) + a

        self._t, self._x, self._x0 = t, x, x0

        return t.clone().requires_grad_(True), x.clone().requires_grad_(True), x0.clone()

    @property
    def interior_points(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Detached (t, x) most recently sampled, for disjointness checks."""
        if self._t is None:
            raise RuntimeError("sample() has not been called yet")
        return self._t, self._x

    @property
    def initial_points(self) -> torch.Tensor:
        """Detached x0 most recently sampled, for disjointness checks."""
        if self._x0 is None:
            raise RuntimeError("sample() has not been called yet")
        return self._x0
