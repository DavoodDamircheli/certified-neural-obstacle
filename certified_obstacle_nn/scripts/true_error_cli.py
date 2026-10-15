"""Process-isolated true-error (weak-norm E_true) entry point.

    python -m certified_obstacle_nn.scripts.true_error_cli \\
        --weights results/<run>/weights.pt \\
        --config  results/<run>/config.json \\
        --n-t 100 --n-x 100 --out results/<run>/certification/

Loads a FROZEN model from disk in a fresh process and computes E_true. This
exists so that E_true and the certificate (scripts/certify_cli.py) can be
computed in SEPARATE processes, each independently loading the frozen weights
from disk, rather than one process holding both computations' intermediate
tensors at once -- test_certificate.py's Test A (E_true <= E_cert_U) compares
the two JSON records this and certify_cli.py produce rather than calling
compute_true_error() and certify() back to back in the same process.

Shares load_frozen_triple/build_problem/PROBLEM_REGISTRY with certify_cli.py
rather than duplicating them -- both scripts' jobs are "load a frozen triple
from disk and run one certification-adjacent computation over it", they
should not drift into two different definitions of "frozen" or "which
problems are known".

Thread count is capped the same way and for the same reason as certify_cli.py
(torch.set_num_threads, default 4, policy doc section 16): certification-
adjacent work here is memory-bound, not compute-bound.

Free-boundary-aware quadrature (errors/true_error.py, Pre-M18 milestone T1):
if `problem` exposes `free_boundary_exact` (Stefan does; American put, Phase
4, does not -- its free boundary is not known in closed form), this script
passes it to compute_true_error automatically, splitting the quadrature
domain along the exact interface. This restores spectral convergence (the
self-check's relative change drops from ~92% on the dominant term to
~1e-15) and is why --quadrature-tolerance's default is back to 1e-3, not a
loosened value.

Free-boundary LOCATION error E_FB (errors/free_boundary.py, M18): computed in
this same process, immediately after E_true, gated on the same
`free_boundary_exact` availability check (E_FB's definition requires a known
closed-form interface; there is nothing to fall back to when one is not
available, unlike compute_true_error's global-grid fallback). Colocated here
rather than a separate CLI/process because both are frozen-triple-vs-
closed-form-reference diagnostics over the same loaded triple -- not because
of any memory pressure (E_FB's own footprint is a handful of (n_t, n_scan)
arrays, trivially small next to the certification/true-error grids this
policy exists for).

American put frozen-reference wiring (2026-09-12, specification section 12
invariant 4): detected generically via getattr(problem, "x_star", None) --
the SAME pattern certify_cli.py's load_frozen_triple already uses for
problem.xi/problem.K, never a hardcoded problem-name check, so Stefan's
behavior (no x_star attribute) is byte-identical to before. When present,
loads reference/american_put_reference.py's FROZEN solution from
--reference-dir (never recomputes it, never touches the network) and passes
its build_reference_callables() as u_ref/sigma_ref/lambda_ref plus
vertical_interface_x=problem.x_star to compute_true_error, instead of the
default problem.u_exact/sigma_exact/lambda_exact fallback (American put has
no closed-form solution to fall back to). Practical note: this reference is
only a piecewise interpolant of a finite grid, not a smooth closed form, so
it needs more quadrature points than Stefan to clear the same
quadrature_tolerance -- pass --n-t/--n-x accordingly (measured 2026-09-12:
n_t=n_x=320 cleared 1e-3 by roughly 2x margin against the frozen 6400x6400
reference); this is a resolution choice, not a loosened tolerance.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from certified_obstacle_nn.certification.memory import certification_stage, peak_rss, report_memory
from certified_obstacle_nn.certification.raw_estimator import DEFAULT_MAX_RSS_GB, DEFAULT_POINTS_PER_CHUNK
from certified_obstacle_nn.certification.stress_corruption import make_flux_add_linear_t_triple, make_lambda_add_relu_bump_triple
from certified_obstacle_nn.errors.free_boundary import compute_free_boundary_error
from certified_obstacle_nn.errors.true_error import compute_true_error
from certified_obstacle_nn.reference.american_put_reference import FROZEN_DIR, build_reference_callables, load_reference
from certified_obstacle_nn.scripts.certify_cli import _git_hash, build_problem, load_frozen_triple


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True, type=Path, help="path to a torch.save(triple.state_dict()) checkpoint")
    parser.add_argument("--config", required=True, type=Path, help="path to the run's config.json (needs 'problem', 'architecture')")
    parser.add_argument("--out", required=True, type=Path, help="output directory for the true-error result")
    parser.add_argument("--n-t", type=int, default=60)
    parser.add_argument("--n-x", type=int, default=60)
    parser.add_argument("--quadrature-tolerance", type=float, default=1e-3, help="restored to the specified 1e-3 (Pre-M18 T1) now that free-boundary-aware quadrature is used automatically when available")
    parser.add_argument("--points-per-chunk", type=int, default=DEFAULT_POINTS_PER_CHUNK)
    parser.add_argument("--max-rss-gb", type=float, default=DEFAULT_MAX_RSS_GB)
    parser.add_argument("--num-threads", type=int, default=4, help="torch.set_num_threads -- see module docstring")
    parser.add_argument("--fb-n-t", type=int, default=400, help="E_FB (errors/free_boundary.py) t-quadrature order (400: measured-safe against 6x64/seed4 in the M18 sweep, where 200 failed the 1e-2 self-check by a hair, rel change 1.17e-2 at n_t=200->400)")
    parser.add_argument("--fb-n-scan", type=int, default=401, help="E_FB x-scan resolution for the leftmost-crossing search")
    parser.add_argument("--fb-quadrature-tolerance", type=float, default=1e-2, help="E_FB self-check tolerance (looser than E_true's 1e-3 -- see errors/free_boundary.py docstring)")
    parser.add_argument("--skip-fb", action="store_true", help="skip E_FB even if problem.free_boundary_exact is available")
    parser.add_argument(
        "--stress-corruption-type", type=str, default=None, choices=["flux_add_linear_t", "lambda_add_relu_bump"],
        help="experiments/stefan_stress.py S2/S3: post-hoc-corrupt the loaded triple "
             "(certification/stress_corruption.py) before computing E_true/E_FB against it. "
             "No bound correction needed here (compute_true_error is a direct numerical "
             "comparison, not an a priori bound) -- must match --stress-corruption-type/-param "
             "passed to certify_cli.py for the same run.",
    )
    parser.add_argument("--stress-corruption-param", type=float, default=None)
    parser.add_argument(
        "--reference-dir", type=Path, default=FROZEN_DIR,
        help="reference/american_put_reference.py FROZEN solution directory -- only consulted "
             "when problem.x_star is present (American put); ignored for Stefan.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> Path:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.num_threads)

    wall_start = time.perf_counter()
    report_memory(f"true_error_cli start (fresh process, num_threads={args.num_threads})")

    with certification_stage("true_error_cli:load_model"):
        config = json.loads(args.config.read_text())
        problem = build_problem(config["problem"])
        triple = load_frozen_triple(problem, config, args.weights)

    if args.stress_corruption_type is not None:
        if args.stress_corruption_param is None:
            raise ValueError("--stress-corruption-type requires --stress-corruption-param")
        with certification_stage("true_error_cli:apply_stress_corruption"):
            if args.stress_corruption_type == "flux_add_linear_t":
                triple = make_flux_add_linear_t_triple(triple, args.stress_corruption_param)
            else:
                triple = make_lambda_add_relu_bump_triple(triple, args.stress_corruption_param)

    free_boundary_fn = getattr(problem, "free_boundary_exact", None)
    vertical_interface_x = getattr(problem, "x_star", None)
    u_ref = sigma_ref = lambda_ref = None
    if vertical_interface_x is not None:
        with certification_stage("true_error_cli:load_frozen_reference"):
            reference_solution = load_reference(args.reference_dir)
            u_ref, sigma_ref, lambda_ref = build_reference_callables(reference_solution)

    with certification_stage("true_error_cli:compute_true_error"):
        result = compute_true_error(
            problem, triple, n_t=args.n_t, n_x=args.n_x, quadrature_tolerance=args.quadrature_tolerance,
            points_per_chunk=args.points_per_chunk, max_rss_gb=args.max_rss_gb,
            free_boundary_fn=free_boundary_fn, vertical_interface_x=vertical_interface_x,
            u_ref=u_ref, sigma_ref=sigma_ref, lambda_ref=lambda_ref,
        )

    fb_result = None
    if free_boundary_fn is not None and not args.skip_fb:
        with certification_stage("true_error_cli:compute_free_boundary_error"):
            fb_result = compute_free_boundary_error(
                problem, triple, free_boundary_fn,
                n_t=args.fb_n_t, n_scan=args.fb_n_scan,
                quadrature_tolerance=args.fb_quadrature_tolerance,
            )

    wall_clock_seconds = time.perf_counter() - wall_start

    out_path = args.out / f"true_error_nt{args.n_t}_nx{args.n_x}.json"
    payload = {
        "git_hash": _git_hash(),
        "weights_path": str(args.weights),
        "config_path": str(args.config),
        "n_t": args.n_t,
        "n_x": args.n_x,
        "quadrature_tolerance": args.quadrature_tolerance,
        "points_per_chunk": args.points_per_chunk,
        "max_rss_gb": args.max_rss_gb,
        "num_threads": args.num_threads,
        "wall_clock_seconds": wall_clock_seconds,
        "peak_rss_gib": peak_rss() / (1024 ** 3),
        "exit_status": "success",
        "stress_corruption_type": args.stress_corruption_type,
        "stress_corruption_param": args.stress_corruption_param,
        "problem": config["problem"],
        "free_boundary_split": free_boundary_fn is not None,
        "vertical_interface_x": vertical_interface_x,
        "reference_dir": str(args.reference_dir) if vertical_interface_x is not None else None,
        "reference_config_hash": reference_solution.config_hash if vertical_interface_x is not None else None,
        "result": result._asdict(),
        "free_boundary_error": fb_result._asdict() if fb_result is not None else None,
    }
    out_path.write_text(json.dumps(payload, indent=2))

    report_memory("true_error_cli done")
    print(f"E_true={result.E_true:.6e}  quadrature_relative_change={result.quadrature_relative_change:.3e}  "
          f"free_boundary_split={free_boundary_fn is not None}  tolerance={args.quadrature_tolerance:.0e}")
    if fb_result is not None:
        print(f"E_FB_gap_only={fb_result.E_FB_gap_only:.6e}  E_FB_gap_and_multiplier={fb_result.E_FB_gap_and_multiplier:.6e}  "
              f"eps_u={fb_result.eps_u:.4e}  eps_lambda={fb_result.eps_lambda:.4e}  "
              f"tolerance={args.fb_quadrature_tolerance:.0e}")
    print(
        f"RUN RECORD: git_hash={payload['git_hash']} n_t={result.n_t} n_x={result.n_x} "
        f"points_per_chunk={args.points_per_chunk} peak_rss_gib={payload['peak_rss_gib']:.3f} "
        f"wall_clock_seconds={wall_clock_seconds:.2f} exit_status=success"
    )
    print(f"true_error -> {out_path}")

    return out_path


if __name__ == "__main__":
    main()
