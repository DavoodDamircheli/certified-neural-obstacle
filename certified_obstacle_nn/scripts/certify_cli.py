"""Process-isolated certification entry point.

    python -m certified_obstacle_nn.scripts.certify_cli \\
        --weights results/<run>/weights.pt \\
        --config  results/<run>/config.json \\
        --grid 128 --out results/<run>/certification/

Loads a FROZEN model from disk in a fresh process and certifies it. This
script deliberately imports nothing from `certified_obstacle_nn.training` --
training must not happen in the same process as certification
(docs/history/MEMORY_OOM_DEBUGGING_AND_SAFE_RUN_POLICY.md section 4, point 4). That is
primarily an invariant benefit (certification cannot see training state,
closing the loop on specification section 12 invariant 1 more strongly than a
runtime assertion can) and secondarily a large memory win: no optimizer
state, no collocation tensors, no retained training graph is resident while
certifying.

`experiments/*.py` scripts must invoke certification through this CLI as a
subprocess, or at minimum from a process that has never constructed an
optimizer -- never call `certify()` directly in the same process that just
called `train()`.

Thread count is capped (torch.set_num_threads, default 4, policy doc section
16): intra-op parallelism allocates per-thread workspace buffers that scale
with thread count, and certification is memory-bound, not compute-bound (the
whole point of the streaming rewrite is that wall clock is cheap and RSS is
the scarce resource) -- more threads here trades away memory headroom for a
speedup this workload doesn't need.

The JSON record written to --out includes everything
docs/history/MEMORY_OOM_DEBUGGING_AND_SAFE_RUN_POLICY.md section 18 asks be recorded for
every certification run: git hash, config, grid, q, points_per_chunk,
midpoint_chunk, n_cells, peak RSS, wall clock, exit status.
"""

import argparse
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import torch

from certified_obstacle_nn.certification.certificate import certify
from certified_obstacle_nn.certification.certificate_option_b import certify_option_b
from certified_obstacle_nn.certification.certified_residual_bound import DEFAULT_MODE as DEFAULT_OPTION_B_MODE
from certified_obstacle_nn.certification.certified_residual_bound import VALID_MODES as VALID_OPTION_B_MODES
from certified_obstacle_nn.certification.level2_remainder import DEFAULT_MIDPOINT_CHUNK
from certified_obstacle_nn.certification.memory import MemoryGuard, certification_stage, peak_rss, report_memory
from certified_obstacle_nn.certification.network_bounds import compute_triple_bounds, physical_units_bounds
from certified_obstacle_nn.certification.partition import build_partition
from certified_obstacle_nn.certification.raw_estimator import DEFAULT_POINTS_PER_CHUNK
from certified_obstacle_nn.certification.stress_corruption import (
    corrected_bounds_flux_add_linear_t,
    corrected_bounds_lambda_add_relu_bump,
    make_flux_add_linear_t_triple,
    make_lambda_add_relu_bump_triple,
)
from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.models.unit_conversion import PhysicalUnitsTriple
from certified_obstacle_nn.problems.american_put import AmericanPutProblem
from certified_obstacle_nn.problems.base_problem import BaseProblem
from certified_obstacle_nn.problems.stefan import StefanProblem

REPO_ROOT = Path(__file__).resolve().parents[2]

PROBLEM_REGISTRY: dict[str, type[BaseProblem]] = {
    "stefan": StefanProblem,
    "american_put": AmericanPutProblem,
}


def build_problem(name: str) -> BaseProblem:
    try:
        return PROBLEM_REGISTRY[name]()
    except KeyError:
        raise ValueError(f"unknown problem {name!r}; known problems: {sorted(PROBLEM_REGISTRY)}") from None


def load_frozen_triple(problem: BaseProblem, config: dict, weights_path: Path):
    """Builds the triple from config["architecture"], loads weights, freezes
    parameters. This is the only place this process ever touches an
    nn.Module's requires_grad -- to False, never to True (no optimizer is
    ever constructed here, specification section 12 invariant 2).

    American put (specification section 10, 2026-09-12): detected generically via
    getattr, exactly like true_error_cli.py's own getattr(problem,
    "free_boundary_exact", None) pattern -- never a hardcoded problem-name
    check, so Stefan's behavior (neither attribute exists) is byte-identical
    to before. problem.xi (if present) is wired in as the triple's x_rescale
    (models/state_net.py); the loaded triple is trained on NORMALIZED
    targets, so it is wrapped in PhysicalUnitsTriple(triple, problem.K) (if
    problem.K is present) before being returned -- specification's own "convert
    back to physical units before certifying"."""
    arch = config["architecture"]
    x_rescale = getattr(problem, "xi", None)
    triple = NeuralTriple(
        problem,
        width=arch["width"],
        depth=arch["depth"],
        multiplier_head=arch.get("multiplier_head", "square"),
        x_rescale=x_rescale,
    )
    state_dict = torch.load(weights_path, weights_only=True, map_location="cpu")
    triple.load_state_dict(state_dict)
    triple.eval()
    for p in triple.parameters():
        p.requires_grad_(False)

    K = getattr(problem, "K", None)
    if K is not None:
        return PhysicalUnitsTriple(triple, K)
    return triple


def _git_hash() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True, type=Path, help="path to a torch.save(triple.state_dict()) checkpoint")
    parser.add_argument("--config", required=True, type=Path, help="path to the run's config.json (needs 'problem', 'architecture')")
    parser.add_argument("--grid", required=True, type=int, help="n_t = n_x for the certification partition")
    parser.add_argument("--out", required=True, type=Path, help="output directory for the certification result")
    parser.add_argument("--n-qp", type=int, default=4)
    parser.add_argument("--max-rss-gb", type=float, default=8.0, help="MemoryGuard abort threshold (this CLI's own stage checks AND the chunked loops inside certify())")
    parser.add_argument("--points-per-chunk", type=int, default=DEFAULT_POINTS_PER_CHUNK, help="interior quadrature points per chunk inside certify()'s streamed evaluation")
    parser.add_argument("--midpoint-chunk", type=int, default=DEFAULT_MIDPOINT_CHUNK, help="cell-midpoints per chunk in the E_h Level-2 remainder pass (a separate knob from --points-per-chunk)")
    parser.add_argument("--num-threads", type=int, default=4, help="torch.set_num_threads -- certification is memory-bound, not compute-bound; see module docstring")
    parser.add_argument(
        "--option-b-mode", type=str, default=None, choices=VALID_OPTION_B_MODES,
        help=f"enable Option B (a posteriori sharpened m_R, OPTION_B_..._PLAN.md); omit to disable "
             f"entirely (default: identical to pre-Option-B certify()). Recommended: {DEFAULT_OPTION_B_MODE!r}",
    )
    parser.add_argument(
        "--save-cells", action="store_true",
        help="persist the per-cell rho_*_K arrays to --out (certification/cells_io.py: cells.csv below "
             "1e6 cells, cells.npz above). Off by default -- most runs don't need the per-cell detail.",
    )
    parser.add_argument(
        "--stress-corruption-type", type=str, default=None, choices=["flux_add_linear_t", "lambda_add_relu_bump"],
        help="experiments/stefan_stress.py S2/S3: post-hoc-corrupt the loaded triple "
             "(certification/stress_corruption.py) and certify it with hand-corrected a priori "
             "bounds. Requires --option-b-mode (bounds_override only applies there).",
    )
    parser.add_argument(
        "--stress-corruption-param", type=float, default=None,
        help="a_coefficient (flux_add_linear_t) or bump_height (lambda_add_relu_bump)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> Path:
    args = parse_args(argv)
    if args.stress_corruption_type is not None:
        if args.option_b_mode is None:
            raise ValueError("--stress-corruption-type requires --option-b-mode (bounds_override only applies to certify_option_b)")
        if args.stress_corruption_param is None:
            raise ValueError("--stress-corruption-type requires --stress-corruption-param")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.num_threads)

    guard = MemoryGuard(
        max_rss_gb=args.max_rss_gb,
        label_prefix="certify_cli",
        chunk_size=args.points_per_chunk,
        n_qp=args.n_qp,
        grid=(args.grid, args.grid),
    )

    wall_start = time.perf_counter()
    report_memory(f"certify_cli start (fresh process, no optimizer ever constructed, num_threads={args.num_threads})")
    guard.check("start")

    with certification_stage("certify_cli:load_model"):
        config = json.loads(args.config.read_text())
        problem = build_problem(config["problem"])
        triple = load_frozen_triple(problem, config, args.weights)
    guard.check("after load_model")

    bounds_override = None
    if args.stress_corruption_type is not None:
        with certification_stage("certify_cli:apply_stress_corruption"):
            base_bounds = compute_triple_bounds(problem, triple)
            if args.stress_corruption_type == "flux_add_linear_t":
                triple = make_flux_add_linear_t_triple(triple, args.stress_corruption_param)
                bounds_override = corrected_bounds_flux_add_linear_t(base_bounds, problem, args.stress_corruption_param)
            else:
                triple = make_lambda_add_relu_bump_triple(triple, args.stress_corruption_param)
                bounds_override = corrected_bounds_lambda_add_relu_bump(base_bounds, problem, args.stress_corruption_param)
        guard.check("after apply_stress_corruption")
    elif isinstance(triple, PhysicalUnitsTriple):
        # American put (specification section 10, 2026-09-12): load_frozen_triple
        # already wrapped the loaded triple for PHYSICAL evaluation (which
        # compute_raw_estimator/level2_remainder need), but its sub-networks
        # are __call__-only (no .net), so compute_triple_bounds cannot
        # introspect them directly here. Compute the a priori bounds on the
        # UNDERLYING normalized-output triple against the matching
        # NORMALIZED problem view (fully unit-consistent, no crash), then
        # convert to physical units (certification/network_bounds.py's
        # physical_units_bounds) -- generic via getattr, never a hardcoded
        # problem-name check, so Stefan (no normalized_view) is unaffected.
        if args.option_b_mode is None:
            raise ValueError("problem.K is set (normalized-training problem) but --option-b-mode was not given -- bounds_override only applies to certify_option_b")
        normalized_view = getattr(problem, "normalized_view", None)
        if normalized_view is None:
            raise ValueError(f"problem {config['problem']!r} has K but no normalized_view() -- cannot compute physical-units bounds")
        with certification_stage("certify_cli:physical_units_bounds"):
            bounds_bar = compute_triple_bounds(normalized_view(), triple.base_triple)
            bounds_override = physical_units_bounds(bounds_bar, triple.K)
        guard.check("after physical_units_bounds")

    with certification_stage("certify_cli:build_partition"):
        partition = build_partition(problem, n_t=args.grid, n_x=args.grid)
    guard.check("after build_partition")

    # training_points=None: this process never had a TrainingSampler, so the
    # disjointness check is skipped rather than silently assumed to pass
    # (certificate.py's own documented default; disjointness from a
    # deterministic partition is guaranteed by partition.py's own
    # construction regardless, see partition.py's module docstring).
    if args.option_b_mode is None:
        with certification_stage("certify_cli:certify"):
            result = certify(
                triple, problem, partition, training_points=None, n_qp=args.n_qp,
                points_per_chunk=args.points_per_chunk, midpoint_chunk=args.midpoint_chunk, max_rss_gb=args.max_rss_gb,
                save_cells_dir=args.out if args.save_cells else None,
            )
        guard.check("after certify")
        wall_clock_seconds = time.perf_counter() - wall_start

        out_path = args.out / f"certificate_grid{args.grid}.json"
        Q_measure = problem.T * (problem.omega_bounds[1] - problem.omega_bounds[0])
        payload = {
            # provenance (policy doc section 18): record for every certification run.
            "git_hash": _git_hash(),
            "weights_path": str(args.weights),
            "config_path": str(args.config),
            "grid": args.grid,
            "n_qp": args.n_qp,
            "points_per_chunk": args.points_per_chunk,
            "midpoint_chunk": args.midpoint_chunk,
            "max_rss_gb": args.max_rss_gb,
            "num_threads": args.num_threads,
            "n_cells": args.grid * args.grid,
            "wall_clock_seconds": wall_clock_seconds,
            "peak_rss_gib": peak_rss() / (1024 ** 3),
            "exit_status": "success",
            "option_b_mode": None,
            "save_cells": args.save_cells,
            # the certificate itself
            "problem": config["problem"],
            "h": partition.h,
            "Q_measure": Q_measure,
            "result": asdict(result),
        }
        out_path.write_text(json.dumps(payload, indent=2))

        report_memory("certify_cli done")
        print(f"rho_tilde_h={result.rho_tilde_h:.6e}  E_h={result.remainder.E_h:.6e}  "
              f"E_cert={result.E_cert:.6e}  E_cert_U={result.E_cert_U:.6e}")
        print(
            f"RUN RECORD: git_hash={payload['git_hash']} grid={args.grid} n_qp={args.n_qp} "
            f"points_per_chunk={args.points_per_chunk} midpoint_chunk={args.midpoint_chunk} "
            f"n_cells={payload['n_cells']} peak_rss_gib={payload['peak_rss_gib']:.3f} "
            f"wall_clock_seconds={wall_clock_seconds:.2f} exit_status=success"
        )
        print(f"certificate -> {out_path}")
        return out_path

    with certification_stage("certify_cli:certify_option_b"):
        result = certify_option_b(
            triple, problem, partition, training_points=None, n_qp=args.n_qp,
            points_per_chunk=args.points_per_chunk, midpoint_chunk=args.midpoint_chunk, max_rss_gb=args.max_rss_gb,
            option_b_mode=args.option_b_mode,
            save_cells_dir=args.out if args.save_cells else None,
            bounds_override=bounds_override,
        )
    guard.check("after certify_option_b")

    wall_clock_seconds = time.perf_counter() - wall_start

    out_path = args.out / f"certificate_grid{args.grid}.json"
    Q_measure = problem.T * (problem.omega_bounds[1] - problem.omega_bounds[0])
    payload = {
        "git_hash": _git_hash(),
        "weights_path": str(args.weights),
        "config_path": str(args.config),
        "grid": args.grid,
        "n_qp": args.n_qp,
        "points_per_chunk": args.points_per_chunk,
        "midpoint_chunk": args.midpoint_chunk,
        "max_rss_gb": args.max_rss_gb,
        "num_threads": args.num_threads,
        "n_cells": args.grid * args.grid,
        "wall_clock_seconds": wall_clock_seconds,
        "peak_rss_gib": peak_rss() / (1024 ** 3),
        "exit_status": "success",
        "option_b_mode": args.option_b_mode,
        "save_cells": args.save_cells,
        "stress_corruption_type": args.stress_corruption_type,
        "stress_corruption_param": args.stress_corruption_param,
        "problem": config["problem"],
        "h": partition.h,
        "Q_measure": Q_measure,
        "result": asdict(result),
    }
    out_path.write_text(json.dumps(payload, indent=2))

    report_memory("certify_cli done")
    print(f"[OLD] E_h={result.E_h_old:.6e}  E_cert_U={result.E_cert_U_old:.6e}")
    print(f"[NEW] E_h={result.E_h_new:.6e}  E_cert_U={result.E_cert_U_new:.6e}  "
          f"m_R_cert={result.residual_bound.m_R_cert:.6e} (vs m_R_apriori={result.bounds.m_R:.6e})")
    print(
        f"RUN RECORD: git_hash={payload['git_hash']} grid={args.grid} n_qp={args.n_qp} "
        f"option_b_mode={args.option_b_mode} points_per_chunk={args.points_per_chunk} "
        f"midpoint_chunk={args.midpoint_chunk} n_cells={payload['n_cells']} "
        f"peak_rss_gib={payload['peak_rss_gib']:.3f} wall_clock_seconds={wall_clock_seconds:.2f} exit_status=success"
    )
    print(f"certificate -> {out_path}")
    return out_path


if __name__ == "__main__":
    main()
