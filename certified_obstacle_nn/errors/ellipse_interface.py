"""Ellipse benchmark: free-boundary diagnostic by gap classification (roadmap
v14 section 6; a DIAGNOSTIC, not controlled by the theorem).

Declared rule (no multiplier threshold; the disk's mu = 5 is not reused):
  * time slices t in TIMES; rays from the ellipse centre (0.43, 0.57) at
    N_ANGLES equally spaced angles theta; along each ray the normalised
    radius rho in (0, RHO_MAX] on N_RHO equally spaced points, physical point
    (0.43 + rho a(t) cos theta, 0.57 + rho d(t) sin theta); the exact
    interface is rho = 1.
  * a point is classified contact iff the gap v - g <= tol (tol declared);
  * the inferred interface on a ray is the first contact -> non-contact
    transition going outward from the innermost point (the midpoint of the
    two samples); 'missing' if the ray has no contact point at all or never
    leaves contact; 'multiple' if more than one transition occurs (the first
    one is still used); the physical distance error is
    |rho* - 1| * sqrt((a cos)^2 + (d sin)^2).
  * the identical classifier applied to the EXACT solution (gap kappa0 zeta_+^2)
    is reported alongside: since the exact gap is quadratic in the distance,
    any tol > 0 moves the exact solution's inferred interface outward by
    ~ sqrt(tol / kappa0) / 2 in rho; that is the classifier's own bias.
Aggregation: mean and max distance error over found rays, per slice and overall.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from certified_obstacle_nn.problems.ellipse import C1, C2

TIMES = (0.0, 0.25, 0.5, 0.75, 1.0)
N_ANGLES = 360
N_RHO = 2001
RHO_MAX = 2.0


def _classify(gap: np.ndarray, rho: np.ndarray, tol: float):
    contact = gap <= tol
    out = []
    for row in contact:
        trans = np.nonzero(row[:-1] & ~row[1:])[0]
        if not row.any() or trans.size == 0:
            out.append((math.nan, "missing", int(trans.size)))
            continue
        i = trans[0]
        out.append((0.5 * (rho[i] + rho[i + 1]), "multiple" if trans.size > 1 else "ok", int(trans.size)))
    return out


def interface_diagnostic(problem, triple=None, tol: float = 1e-4, times=TIMES, n_angles=N_ANGLES, n_rho=N_RHO):
    rho = np.linspace(RHO_MAX / n_rho, RHO_MAX, n_rho)
    th = np.arange(n_angles) * 2 * math.pi / n_angles
    rows = []
    for t in times:
        a, d = problem.a(t), problem.d(t)
        X = C1 + rho[None, :] * a * np.cos(th)[:, None]
        Y = C2 + rho[None, :] * d * np.sin(th)[:, None]
        inside = (X > 0) & (X < 1) & (Y > 0) & (Y < 1)
        tt = torch.full((X.size,), float(t), dtype=torch.float64)
        xx = torch.tensor(np.clip(X, 0, 1).ravel())
        yy = torch.tensor(np.clip(Y, 0, 1).ravel())
        with torch.no_grad():
            g = problem.g(tt, xx, yy).numpy().reshape(X.shape)
            gaps = {"exact": problem.u_exact(tt, xx, yy).numpy().reshape(X.shape) - g}
            if triple is not None:
                v = (problem.mask(xx, yy) * triple.state(torch.stack([tt, xx, yy], -1))[:, 0]).numpy()
                gaps["network"] = v.reshape(X.shape) - g
        for who, gap in gaps.items():
            gap = np.where(inside, gap, np.inf)  # outside the square: never contact
            for k, (rs, flag, ntr) in enumerate(_classify(gap, rho, tol)):
                scale = math.hypot(a * math.cos(th[k]), d * math.sin(th[k]))
                rows.append({"who": who, "t": t, "theta": th[k], "rho_star": rs, "flag": flag, "n_transitions": ntr,
                             "dist_err": abs(rs - 1.0) * scale if flag != "missing" else math.nan})
    import pandas as pd

    df = pd.DataFrame(rows)
    agg = []
    for (who, t), g in list(df.groupby(["who", "t"])) + [((w, "all"), df[df["who"] == w]) for w in df["who"].unique()]:
        found = g[g["flag"] != "missing"]
        agg.append({"who": who, "t": t, "tol": tol, "n_rays": len(g), "missing": int((g["flag"] == "missing").sum()),
                    "multiple": int((g["flag"] == "multiple").sum()),
                    "mean_dist_err": float(found["dist_err"].mean()) if len(found) else math.nan,
                    "max_dist_err": float(found["dist_err"].max()) if len(found) else math.nan,
                    "mean_rho_star": float(found["rho_star"].mean()) if len(found) else math.nan})
    meta = {"times": list(times), "n_angles": n_angles, "n_rho": n_rho, "rho_max": RHO_MAX, "tol": tol,
            "rule": "contact iff v - g <= tol; first outward contact->non-contact transition per ray",
            "status": "diagnostic_float64"}
    return df, pd.DataFrame(agg), meta
