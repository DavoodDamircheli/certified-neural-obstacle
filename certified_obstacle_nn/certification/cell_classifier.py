"""Cell classification, specification section 7.

Every cell of the partition falls into exactly one of three classes:
  data-cut       : meets a known (closed-form) data interface Gamma_data
  sign-resolved  : not data-cut, and |gamma(z_K)| > L_gamma * h
  sign-unresolved: everything else

This is a purely geometric/algebraic classification: it evaluates the
network's own gamma = g - u_theta at each cell midpoint and compares against
the a priori L_gamma bound from certification/network_bounds.py -- it never
looks at the exact solution or the free boundary's true location. The claim
under test is that sign-unresolved cells nonetheless cluster around the true
free boundary, without ever being told where it is.
"""

from dataclasses import dataclass

import torch

from certified_obstacle_nn.certification.partition import Partition
from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.problems.base_problem import BaseProblem


@dataclass(frozen=True)
class CellClassification:
    data_cut: torch.Tensor  # bool, (n_cells,)
    sign_resolved: torch.Tensor  # bool, (n_cells,)
    sign_unresolved: torch.Tensor  # bool, (n_cells,)
    V_h_dat: float  # sum |K| over data-cut cells
    V_h_unr: float  # sum |K| over sign-unresolved cells


def classify_cells(problem: BaseProblem, triple: NeuralTriple, partition: Partition, L_gamma: float) -> CellClassification:
    triple.eval()
    for p in triple.parameters():
        p.requires_grad_(False)

    t_center = partition.t_center.reshape(-1, 1)
    x_center = partition.x_center.reshape(-1, 1)

    with torch.no_grad():
        u = triple.state_net(t_center, x_center)
        gamma = (problem.g(t_center, x_center) - u).reshape(-1)

    half_dx = partition.dx / 2.0
    x_flat = x_center.reshape(-1)
    x_lo = x_flat - half_dx
    x_hi = x_flat + half_dx

    data_cut = torch.zeros_like(gamma, dtype=torch.bool)
    for interface_x in problem.gamma_data:
        data_cut |= (x_lo <= interface_x) & (interface_x <= x_hi)

    threshold = L_gamma * partition.h
    resolved_candidate = gamma.abs() > threshold
    sign_resolved = resolved_candidate & ~data_cut
    sign_unresolved = (~resolved_candidate) & (~data_cut)

    V_h_dat = partition.area * data_cut.sum().item()
    V_h_unr = partition.area * sign_unresolved.sum().item()

    return CellClassification(
        data_cut=data_cut,
        sign_resolved=sign_resolved,
        sign_unresolved=sign_unresolved,
        V_h_dat=V_h_dat,
        V_h_unr=V_h_unr,
    )
