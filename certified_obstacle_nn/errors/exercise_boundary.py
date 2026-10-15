"""American put exercise-boundary extraction from a trained (physical-units)
neural triple, specification section 10 (2026-09-12): "Extract and plot the
exercise boundary against the frozen reference."

Mirrors reference/american_put_reference.py's own _extract_exercise_boundary
(same convention: the exercise/contact region is the LOW-x, deep-ITM set
{x <= x*(tau)}, a single contiguous block starting at x=L, so x*(tau) is the
FIRST contact-to-non-contact transition scanning left to right, not the last
contact index overall -- g decays back toward 0 near the right boundary too,
which would otherwise look like a second, spurious contact region), but
evaluates the CONTINUOUS trained triple on a fine scan grid instead of
reading off a dense finite-difference solution array.
"""

import numpy as np
import torch


def extract_neural_exercise_boundary(
    problem, triple, tau_values: np.ndarray, n_scan: int = 2001, contact_tol: float = 1e-3,
) -> np.ndarray:
    """triple must already produce PHYSICAL-unit output (e.g.
    models/unit_conversion.py's PhysicalUnitsTriple) -- this function does
    no unit conversion of its own. contact_tol is looser than the reference
    solver's own 1e-6 (that solver enforces u>=g to its own PSOR tolerance
    ~1e-11; a trained network's gap=u-g only approximately vanishes on the
    contact set, so a tighter tolerance would just find noise)."""
    a, b = problem.omega_bounds
    x_scan = torch.linspace(a, b, n_scan, dtype=torch.float64)
    boundary = np.full(len(tau_values), np.nan)

    for i, tau in enumerate(tau_values):
        t_col = torch.full_like(x_scan, float(tau))
        with torch.no_grad():
            u = triple.state_net(t_col.reshape(-1, 1), x_scan.reshape(-1, 1)).reshape(-1)
            g = problem.g(t_col, x_scan)
        gap = (u - g).numpy()
        x_np = x_scan.numpy()

        contact = gap <= contact_tol
        if not contact[0] or contact.all():
            continue
        first_noncontact_idx = np.where(~contact)[0].min()
        if first_noncontact_idx == 0:
            continue
        last_contact_idx = first_noncontact_idx - 1
        x_lo, x_hi = x_np[last_contact_idx], x_np[first_noncontact_idx]
        gap_lo, gap_hi = gap[last_contact_idx], gap[first_noncontact_idx]
        if gap_hi == gap_lo:
            boundary[i] = x_hi
        else:
            frac = (contact_tol - gap_lo) / (gap_hi - gap_lo)
            boundary[i] = x_lo + frac * (x_hi - x_lo)

    return boundary
