"""Pre-M18 milestone T1 report: E_true^2's per-term breakdown (all five terms
of specification section 8) at base and 2x resolution, BEFORE (global tensor-
product rule) and AFTER (free-boundary-aware split along s(t)=t) the fix in
errors/true_error.py. Confirms which term carries the quadrature error and
by how much the fix improves it, using the same fixed reference triple as
the rest of this study (results/stefan_baseline_weights.pt).

Does not change the definition of E_true or any norm -- reports the SAME
five terms computed the OLD way and the NEW way, side by side.
"""

import json
from pathlib import Path

import torch

from certified_obstacle_nn.errors.true_error import _e_true_squared, _volume_and_boundary_terms
from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.problems.stefan import StefanProblem

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
BASELINE_WEIGHTS_PATH = RESULTS_DIR / "stefan_baseline_weights.pt"
TRIPLE_CONFIG_PATH = RESULTS_DIR / "stefan_quad_convergence_reference_triple_config.json"

N_T = N_X = 100


def _load_reference_triple():
    problem = StefanProblem()
    config = json.loads(TRIPLE_CONFIG_PATH.read_text())
    arch = config["architecture"]
    triple = NeuralTriple(problem, width=arch["width"], depth=arch["depth"], multiplier_head=arch.get("multiplier_head", "square"))
    triple.load_state_dict(torch.load(BASELINE_WEIGHTS_PATH, weights_only=True))
    triple.eval()
    for p in triple.parameters():
        p.requires_grad_(False)
    return problem, triple


def _report_for(problem, triple, free_boundary_fn, label: str) -> dict:
    kwargs = dict(points_per_chunk=65536, max_rss_gb=8.0, check_every_n_chunks=10, free_boundary_fn=free_boundary_fn)
    terms_1x = _volume_and_boundary_terms(
        problem, triple.state_net, triple.flux_net, triple.multiplier_net,
        problem.u_exact, problem.sigma_exact, problem.lambda_exact, N_T, N_X, **kwargs,
    )
    terms_2x = _volume_and_boundary_terms(
        problem, triple.state_net, triple.flux_net, triple.multiplier_net,
        problem.u_exact, problem.sigma_exact, problem.lambda_exact, 2 * N_T, 2 * N_X, **kwargs,
    )

    E1 = _e_true_squared(terms_1x) ** 0.5
    E2 = _e_true_squared(terms_2x) ** 0.5
    E_true_rel_change = abs(E2 - E1) / max(E2, 1e-300)

    print(f"\n=== {label} (free_boundary_fn={'given' if free_boundary_fn else 'None'}) ===")
    print(f"{'term':22s} {'base (nt=100)':>16s} {'2x (nt=200)':>16s} {'rel_change':>12s}")
    per_term = {}
    for k in terms_1x:
        a, b = terms_1x[k], terms_2x[k]
        rel = abs(b - a) / max(abs(b), 1e-300)
        print(f"{k:22s} {a:16.8e} {b:16.8e} {rel:12.6%}")
        per_term[k] = {"base": a, "2x": b, "rel_change": rel}
    print(f"{'E_true (sqrt sum)':22s} {E1:16.8e} {E2:16.8e} {E_true_rel_change:12.6%}")

    return {"per_term": per_term, "E_true_base": E1, "E_true_2x": E2, "E_true_rel_change": E_true_rel_change}


def main():
    problem, triple = _load_reference_triple()

    before = _report_for(problem, triple, None, "BEFORE (global tensor-product rule)")
    after = _report_for(problem, triple, problem.free_boundary_exact, "AFTER (free-boundary-aware split, s(t)=t)")

    dominant_term_before = max(before["per_term"].items(), key=lambda kv: kv[1]["rel_change"])

    out_path = RESULTS_DIR / "stefan_true_error_quadrature_diagnostic.json"
    out_path.write_text(json.dumps({
        "weights_path": str(BASELINE_WEIGHTS_PATH),
        "n_t": N_T, "n_x": N_X,
        "before": before,
        "after": after,
        "dominant_term_before_fix": dominant_term_before[0],
        "dominant_term_rel_change_before_fix": dominant_term_before[1]["rel_change"],
        "self_check_improvement_factor": before["E_true_rel_change"] / max(after["E_true_rel_change"], 1e-300),
    }, indent=2))

    print(f"\nDominant term (carries the pre-fix quadrature error): {dominant_term_before[0]} "
          f"({dominant_term_before[1]['rel_change']:.4%} relative change)")
    print(f"Self-check relative change: before={before['E_true_rel_change']:.4%}  "
          f"after={after['E_true_rel_change']:.2e}  "
          f"(improvement factor {before['E_true_rel_change'] / max(after['E_true_rel_change'], 1e-300):.3e}x)")
    print(f"\nreport -> {out_path}")


if __name__ == "__main__":
    main()
