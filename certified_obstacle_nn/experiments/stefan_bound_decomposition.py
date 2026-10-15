"""Diagnostic 3 of the Stefan certification-mesh convergence study (see
experiments/stefan_quad_convergence.py): the 32^2...512^2 sweep found
E_h ~= C*h with C ~= 7.7e4, and E_h_lip = (L_sm + L_Psi) * h * Q_measure
dominates E_h at every tested grid (E_h_lip is >90% of E_h at every row of
Table 3) -- so C is essentially L_sm + L_Psi for this trained triple. This
script answers "which single factor contributes most of that 7.7e4":

  1. Reports L_sm, L_Psi, L_Phi0, L_PhiT and, per network (state/flux/
     multiplier), Sigma_N, Pi_N, Lip(grad N) -- all read directly from
     certification/network_bounds.py's compute_triple_bounds, no
     recomputation, no change to any formula or constant.
  2. Decomposes L_sm into its 5 additive terms and L_Psi into its 2
     (certification/network_bounds.py's decompose_L_sm_and_L_Psi, verified
     there to sum back exactly), ranks every term by magnitude.
  3. Runs the SAME dense-sampling validation methodology
     tests/test_network_bounds.py uses (the exact two-point secant property
     for Lipschitz claims, direct sampling for sup-norms) against THIS
     SPECIFIC trained checkpoint (not the throwaway briefly-trained network
     tests/test_network_bounds.py itself uses) for every quantity feeding the
     dominant L_sm/L_Psi terms, logging analytic-bound / max-sampled-value
     ratio per quantity -- the single number that shows which bound is
     loosest, i.e. contributing the most pure slack rather than real
     Lipschitz/sup-norm content.

No constant or bound is changed anywhere in this script. It reports what the
existing formulas produce for this network; it does not retune anything to
make the numbers smaller.
"""

import json
import math
from pathlib import Path

import torch

from certified_obstacle_nn.certification.densities import phi_smooth, psi_majorant
from certified_obstacle_nn.certification.network_bounds import (
    TripleBounds,
    compute_triple_bounds,
    decompose_L_sm_and_L_Psi,
)
from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.problems.stefan import StefanProblem
from certified_obstacle_nn.residuals.residuals import compute_residuals

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
BASELINE_WEIGHTS_PATH = RESULTS_DIR / "stefan_baseline_weights.pt"
BASELINE_CONFIG_PATH = RESULTS_DIR / "stefan_quad_convergence_reference_triple_config.json"

N_SUP_SAMPLES = 4000
N_LIP_PAIRS = 1000
MARGIN = 1e-3


# ---------------------------------------------------------------------------
# Dense-sampling helpers -- same methodology as tests/test_network_bounds.py
# (exact two-point secant property for Lipschitz claims, direct sampling for
# sup-norms), reimplemented standalone here (not imported from a test module)
# so this script has no dependency on the test suite.
# ---------------------------------------------------------------------------

def _sample_domain(n, seed, lo=MARGIN, hi=1.0 - MARGIN):
    generator = torch.Generator().manual_seed(seed)
    t = torch.rand(n, 1, generator=generator, dtype=torch.float64) * (hi - lo) + lo
    x = torch.rand(n, 1, generator=generator, dtype=torch.float64) * (hi - lo) + lo
    return t, x


def _random_directions(n, eps, seed):
    generator = torch.Generator().manual_seed(seed)
    theta = torch.rand(n, 1, generator=generator, dtype=torch.float64) * 2 * math.pi
    return eps * torch.cos(theta), eps * torch.sin(theta)


def _empirical_scalar_lipschitz(fn, t, x, eps, seed):
    dt, dx = _random_directions(t.shape[0], eps, seed)
    f1 = fn(t, x)
    f2 = fn(t + dt, x + dx)
    dist = torch.sqrt(dt**2 + dx**2)
    return ((f2 - f1).abs() / dist).max().item()


def _ratio_row(name: str, sampled_max: float, bound: float) -> dict:
    ratio = sampled_max / bound if bound > 0 else float("inf")
    print(f"  {name:32s} sampled_max={sampled_max:.6e}  bound={bound:.6e}  ratio={ratio:.4f}")
    return {"quantity": name, "sampled_max": sampled_max, "bound": bound, "ratio": ratio}


def _load_reference_triple() -> tuple[StefanProblem, NeuralTriple]:
    problem = StefanProblem()
    config = json.loads(BASELINE_CONFIG_PATH.read_text())
    arch = config["architecture"]
    triple = NeuralTriple(problem, width=arch["width"], depth=arch["depth"], multiplier_head=arch.get("multiplier_head", "square"))
    triple.load_state_dict(torch.load(BASELINE_WEIGHTS_PATH, weights_only=True))
    triple.eval()
    for p in triple.parameters():
        p.requires_grad_(False)
    return problem, triple


def _report_network_bounds(bounds: TripleBounds) -> dict:
    print("\nPer-network bounds (Sigma_N = ||N||_inf, Pi_N = ||grad N||_inf <= Lip(N), Lip(grad N)):")
    rows = {}
    for name, nb in [("state_net", bounds.state_bounds), ("flux_net", bounds.flux_bounds), ("multiplier_net", bounds.multiplier_bounds)]:
        print(f"  {name:16s} Sigma_N={nb.output_bound:.6e}  Pi_N={nb.gradient_bound:.6e}  Lip(grad N)={nb.gradient_lipschitz_bound:.6e}")
        rows[name] = {"Sigma_N": nb.output_bound, "Pi_N": nb.gradient_bound, "Lip_grad_N": nb.gradient_lipschitz_bound}
    return rows


def _report_top_level_constants(bounds: TripleBounds) -> None:
    print("\nTop-level Lipschitz constants feeding E_h_lip = (L_sm + L_Psi) * h * Q_measure "
          "and E_h_0T = (L_Phi0 + L_PhiT) * h * |Omega|:")
    print(f"  L_sm   = {bounds.L_sm:.6e}")
    print(f"  L_Psi  = {bounds.L_Psi:.6e}")
    print(f"  L_sm + L_Psi = {bounds.L_sm + bounds.L_Psi:.6e}   <-- this is C in E_h ~= C*h")
    print(f"  L_Phi0 = {bounds.L_Phi0:.6e}")
    print(f"  L_PhiT = {bounds.L_PhiT:.6e}")


def _report_decomposition(problem: StefanProblem, bounds: TripleBounds) -> dict:
    decomposition = decompose_L_sm_and_L_Psi(problem, bounds)
    C = bounds.L_sm + bounds.L_Psi

    all_terms = {**decomposition["L_sm_terms"], **decomposition["L_Psi_terms"]}
    ranked = sorted(all_terms.items(), key=lambda kv: kv[1], reverse=True)

    print(f"\nDecomposition of C = L_sm + L_Psi = {C:.6e}, ranked by contribution:")
    for name, value in ranked:
        print(f"  {name:20s} {value:.6e}   ({100.0 * value / C:5.1f}% of C)")

    dominant_name, dominant_value = ranked[0]
    print(f"\nDOMINANT SINGLE FACTOR: {dominant_name} = {dominant_value:.6e} "
          f"({100.0 * dominant_value / C:.1f}% of C = L_sm + L_Psi)")

    return {
        "L_sm_terms": decomposition["L_sm_terms"],
        "L_Psi_terms": decomposition["L_Psi_terms"],
        "ranked": [{"term": n, "value": v, "pct_of_C": 100.0 * v / C} for n, v in ranked],
        "dominant_term": dominant_name,
        "dominant_value": dominant_value,
        "dominant_pct_of_C": 100.0 * dominant_value / C,
    }


def _dense_sampling_validation(problem: StefanProblem, triple: NeuralTriple, bounds: TripleBounds) -> list[dict]:
    print("\nDense-sampling validation (this checkpoint, not a throwaway network) -- "
          "ratio = sampled_max / analytic_bound; ratio close to 1 means the bound is tight, "
          "ratio close to 0 means the bound is loose (pure slack):")
    rows = []

    # per-network sup-norms and gradient Lipschitz, sampled directly.
    t_sup, x_sup = _sample_domain(N_SUP_SAMPLES, seed=100)
    for name, net, nb in [
        ("state_net", triple.state_net.net, bounds.state_bounds),
        ("flux_net", triple.flux_net.net, bounds.flux_bounds),
        ("multiplier_net", triple.multiplier_net.net, bounds.multiplier_bounds),
    ]:
        t = t_sup.clone().requires_grad_(True)
        x = x_sup.clone().requires_grad_(True)
        out = net(torch.cat([t, x], dim=-1))
        grad_t, grad_x = torch.autograd.grad(out, (t, x), grad_outputs=torch.ones_like(out))
        rows.append(_ratio_row(f"{name}.output (Sigma_N)", out.abs().max().item(), nb.output_bound))
        rows.append(_ratio_row(f"{name}.gradient (Pi_N)", torch.sqrt(grad_t**2 + grad_x**2).max().item(), nb.gradient_bound))

        t2, x2 = _sample_domain(N_LIP_PAIRS, seed=101)

        def grad_fn(tt, xx, net=net):
            tt = tt.clone().requires_grad_(True)
            xx = xx.clone().requires_grad_(True)
            o = net(torch.cat([tt, xx], dim=-1))
            gt, gx = torch.autograd.grad(o, (tt, xx), grad_outputs=torch.ones_like(o))
            return gt.detach(), gx.detach()

        dt, dx = _random_directions(t2.shape[0], 5e-2, seed=102)
        a1, b1 = grad_fn(t2, x2)
        a2, b2 = grad_fn(t2 + dt, x2 + dx)
        dist = torch.sqrt(dt**2 + dx**2)
        sampled_lip = (torch.sqrt((a2 - a1) ** 2 + (b2 - b1) ** 2) / dist).max().item()
        rows.append(_ratio_row(f"{name}.grad_lipschitz", sampled_lip, nb.gradient_lipschitz_bound))

    # the quantities directly feeding L_sm/L_Psi's dominant terms.
    t_lip, x_lip = _sample_domain(N_LIP_PAIRS, seed=110)

    def gamma_fn(tt, xx):
        tt = tt.clone().requires_grad_(True)
        xx = xx.clone().requires_grad_(True)
        residuals = compute_residuals(problem, triple.state_net, triple.flux_net, triple.multiplier_net, tt, xx, xx)
        return residuals.gamma.detach()

    rows.append(_ratio_row("L_gamma", _empirical_scalar_lipschitz(gamma_fn, t_lip, x_lip, 5e-2, 111), bounds.L_gamma))

    def phi_sm_fn(tt, xx):
        tt = tt.clone().requires_grad_(True)
        xx = xx.clone().requires_grad_(True)
        return phi_smooth(problem, triple, tt, xx).detach()

    rows.append(_ratio_row("L_sm", _empirical_scalar_lipschitz(phi_sm_fn, t_lip, x_lip, 5e-2, 112), bounds.L_sm))

    def psi_fn(tt, xx):
        tt = tt.clone().requires_grad_(True)
        xx = xx.clone().requires_grad_(True)
        return psi_majorant(problem, triple, tt, xx).detach()

    rows.append(_ratio_row("L_Psi", _empirical_scalar_lipschitz(psi_fn, t_lip, x_lip, 5e-2, 113), bounds.L_Psi))

    t_sup2, x_sup2 = _sample_domain(N_SUP_SAMPLES, seed=120)
    t_sup2.requires_grad_(True)
    x_sup2.requires_grad_(True)
    x0 = _sample_domain(N_SUP_SAMPLES, seed=121)[1]
    residuals = compute_residuals(problem, triple.state_net, triple.flux_net, triple.multiplier_net, t_sup2, x_sup2, x0)
    rows.append(_ratio_row("gamma (m_gap)", residuals.gamma.abs().max().item(), bounds.m_gap))
    rows.append(_ratio_row("R_PDE (m_R)", residuals.R_PDE.abs().max().item(), bounds.m_R))
    rows.append(_ratio_row("R_sig (m_sig)", residuals.R_sig.abs().max().item(), bounds.m_sig))

    return rows


def main():
    problem, triple = _load_reference_triple()
    bounds = compute_triple_bounds(problem, triple)

    print(f"reference triple: {BASELINE_WEIGHTS_PATH}")
    network_bounds_report = _report_network_bounds(bounds)
    _report_top_level_constants(bounds)
    decomposition_report = _report_decomposition(problem, bounds)
    sampling_report = _dense_sampling_validation(problem, triple, bounds)

    out_path = RESULTS_DIR / "stefan_bound_decomposition.json"
    out_path.write_text(json.dumps({
        "weights_path": str(BASELINE_WEIGHTS_PATH),
        "network_bounds": network_bounds_report,
        "L_sm": bounds.L_sm,
        "L_Psi": bounds.L_Psi,
        "L_Phi0": bounds.L_Phi0,
        "L_PhiT": bounds.L_PhiT,
        "decomposition": decomposition_report,
        "dense_sampling_validation": sampling_report,
    }, indent=2))
    print(f"\nreport -> {out_path}")

    return bounds, decomposition_report, sampling_report


if __name__ == "__main__":
    main()
