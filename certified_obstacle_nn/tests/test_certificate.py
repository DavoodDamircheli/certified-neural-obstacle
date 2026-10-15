"""Tests for certification/certificate.py.

certify() is the top-level entry point tying together the raw estimator
(M9/M10), network bounds (M12), and the Level-2 remainder (M14) into the
final E_cert / E_cert_U numbers of specification section 8. Two runtime invariants
are asserted inside certify() itself, not just tested here: no parameter has
requires_grad enabled during certification, and the partition points are
disjoint from whatever training points are supplied.
"""

import json
import math
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import certified_obstacle_nn
from certified_obstacle_nn.certification.cell_classifier import classify_cells
from certified_obstacle_nn.certification.certificate import CertificateResult, certify
from certified_obstacle_nn.certification.disjointness import DisjointnessViolationError
from certified_obstacle_nn.certification.level2_remainder import compute_level2_remainder
from certified_obstacle_nn.certification.network_bounds import compute_triple_bounds
from certified_obstacle_nn.certification.partition import build_partition
from certified_obstacle_nn.certification.raw_estimator import compute_raw_estimator
from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.problems.stefan import StefanProblem
from certified_obstacle_nn.training.sampler import TrainingSampler
from certified_obstacle_nn.training.trainer import train

pytestmark = pytest.mark.legacy

REPO_ROOT = Path(certified_obstacle_nn.__file__).resolve().parent.parent

# Pre-M18 milestone T1: t=x is a genuine discontinuity in the exact solution
# (lambda jumps, sigma_x jumps), so errors/true_error.py's global
# tensor-product rule only got first-order convergence across it -- this used
# to require loosening the self-check tolerance to 0.05 (previously measured
# as high as 24-92% depending on the triple). Since s(t)=t is known in closed
# form, true_error_cli.py now auto-splits the quadrature domain along it
# (problem.free_boundary_exact), restoring spectral convergence: the
# self-check's relative change is ~1e-15 (machine precision) for every triple
# measured so far, not merely "small enough". TRUE_ERROR_TOLERANCE is
# therefore back to the originally specified 1e-3, not a loosened value.
TRUE_ERROR_N = 100
TRUE_ERROR_TOLERANCE = 1e-3


def _trained_triple(problem, seed=0, n_epochs=200):
    torch.manual_seed(seed)
    triple = NeuralTriple(problem, width=24, depth=3)
    sampler = TrainingSampler(problem, n_interior=1000, n_initial=100, seed=seed)
    t, x, x0 = sampler.sample()
    train(problem, triple, t, x, x0, n_epochs=n_epochs, lr=1e-3)
    return triple, sampler


def test_certify_returns_fully_populated_record_and_freezes_model():
    problem = StefanProblem()
    triple, sampler = _trained_triple(problem)
    partition = build_partition(problem, n_t=20, n_x=20)

    result = certify(triple, problem, partition, training_points=sampler.interior_points)

    assert isinstance(result, CertificateResult)
    for field in (
        "rho_r",
        "rho_p_tilde",
        "rho_c",
        "rho_tilde_h",
        "bounds",
        "V_h_unr",
        "V_h_dat",
        "remainder",
        "C_rel",
        "C_eq",
        "E_cert",
        "E_cert_U",
    ):
        assert getattr(result, field) is not None

    for p in triple.parameters():
        assert p.requires_grad is False


def test_C_eq_equals_2_for_stefan():
    problem = StefanProblem()
    triple, sampler = _trained_triple(problem, n_epochs=50)
    partition = build_partition(problem, n_t=10, n_x=10)

    result = certify(triple, problem, partition, training_points=sampler.interior_points)

    assert result.C_eq == 2.0  # exact, per specification section 9
    assert result.C_rel == 216.0


def test_E_cert_and_E_cert_U_match_the_formula():
    problem = StefanProblem()
    triple, sampler = _trained_triple(problem, n_epochs=50)
    partition = build_partition(problem, n_t=10, n_x=10)

    result = certify(triple, problem, partition, training_points=sampler.interior_points)

    inner = result.rho_tilde_h**2 + result.remainder.E_h
    expected_E_cert = math.sqrt(result.C_rel * inner)
    expected_E_cert_U = math.sqrt(result.C_eq * result.C_rel * inner)

    assert result.E_cert == pytest.approx(expected_E_cert)
    assert result.E_cert_U == pytest.approx(expected_E_cert_U)
    assert result.E_cert_U == pytest.approx(math.sqrt(2.0) * result.E_cert)  # C_eq==2 for Stefan


def test_certify_skips_disjointness_check_when_no_training_points_given():
    problem = StefanProblem()
    triple, _ = _trained_triple(problem, n_epochs=50)
    partition = build_partition(problem, n_t=10, n_x=10)

    result = certify(triple, problem, partition, training_points=None)
    assert result is not None


def test_certify_raises_if_training_points_coincide_with_partition():
    """The disjointness check (certification/disjointness.py, M15b) runs
    against the actual interior QUADRATURE points compute_raw_estimator
    evaluates, not the cell centers -- specification's hard invariant 1 says
    "quadrature points", and that is also where the bulk of certification's
    computation (hence any real contamination risk) actually happens. n_qp=1
    is used here so the single Gauss-Legendre node per cell falls exactly at
    the cell center, letting this test corrupt a training point to a known,
    exact coincidence the same way the pre-M15b test did.
    """
    problem = StefanProblem()
    triple, sampler = _trained_triple(problem, n_epochs=50)
    partition = build_partition(problem, n_t=10, n_x=10)

    t_train, x_train = sampler.interior_points
    t_train = t_train.clone()
    x_train = x_train.clone()
    # deliberately corrupt one training point to exactly match a partition cell center
    t_train[0, 0] = partition.t_center[0]
    x_train[0, 0] = partition.x_center[0]

    with pytest.raises(DisjointnessViolationError, match="certification invariant violated"):
        certify(triple, problem, partition, training_points=(t_train, x_train), n_qp=1)


def test_certify_matches_manual_pipeline():
    """certify() must not silently diverge from calling the constituent
    functions directly with the same inputs -- it should just be gluing them
    together."""
    problem = StefanProblem()
    triple, sampler = _trained_triple(problem, n_epochs=50)
    partition = build_partition(problem, n_t=10, n_x=10)

    result = certify(triple, problem, partition, training_points=sampler.interior_points)

    bounds = compute_triple_bounds(problem, triple)
    _, estimator_summary = compute_raw_estimator(problem, triple, partition)
    classification = classify_cells(problem, triple, partition, bounds.L_gamma)
    remainder = compute_level2_remainder(problem, triple, partition, bounds, classification)

    assert result.rho_tilde_h == pytest.approx(estimator_summary.rho_tilde)
    assert result.remainder.E_h == pytest.approx(remainder.E_h)
    assert result.V_h_unr == pytest.approx(classification.V_h_unr)
    assert result.V_h_dat == pytest.approx(classification.V_h_dat)


# Test A / Test B diagnostic order, IF Test A ever fails. A chunking bug (the
# M15 streaming rewrite) now joins the original suspect list, so check in
# this order -- each step is meaningless until the one before it is ruled
# out:
#   0. Does the streamed estimator reproduce the 32x32 reference
#      implementation? (tests/test_certificate_streaming.py). If not, the bug
#      is in the refactor, not the theory, and nothing below is meaningful.
#   1. Which norm is E_true using? Must be the weak/U norm of specification
#      section 8, including ||d_t e_u + div_x e_s - e_l||^2.
#   2. Is C_eq present? For Stefan, E_cert_U = sqrt(2) * E_cert.
#   3. Sign conventions in R_PDE and R_sig (tests/test_residuals.py).
#   4. Is the certification grid resolving rho_tilde_h? Refine and re-run
#      (tests/test_certificate_streaming.py's chunk-size-independence test
#      does NOT check grid resolution, only chunk-size independence at a
#      fixed grid -- these are different questions).
#   5. Are the network bounds actually upper bounds?
#      (tests/test_network_bounds.py's dense sampling check.)


@pytest.fixture(scope="module")
def cert_and_true_error_records(tmp_path_factory):
    """certify() and compute_true_error() computed in SEPARATE processes
    (scripts/certify_cli.py, scripts/true_error_cli.py), each independently
    loading the same frozen checkpoint from disk, rather than one test
    process holding both computations' intermediate tensors at once. Module-
    scoped so Test A and Test B share one train+certify+true_error run
    instead of paying for it twice.
    """
    problem = StefanProblem()
    triple, _sampler = _trained_triple(problem, n_epochs=200)

    run_dir = tmp_path_factory.mktemp("cert_true_error_run")
    weights_path = run_dir / "weights.pt"
    config_path = run_dir / "config.json"
    torch.save(triple.state_dict(), weights_path)
    config_path.write_text(json.dumps({
        "problem": "stefan",
        "architecture": {"width": 24, "depth": 3, "multiplier_head": triple.multiplier_net.head},
    }))
    out_dir = run_dir / "out"
    out_dir.mkdir()

    cert_cmd = [
        sys.executable, "-m", "certified_obstacle_nn.scripts.certify_cli",
        "--weights", str(weights_path), "--config", str(config_path),
        "--grid", "20", "--out", str(out_dir),
    ]
    cert_proc = subprocess.run(cert_cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert cert_proc.returncode == 0, f"certify_cli failed:\n--- stdout ---\n{cert_proc.stdout}\n--- stderr ---\n{cert_proc.stderr}"
    cert_json = json.loads((out_dir / "certificate_grid20.json").read_text())

    true_error_cmd = [
        sys.executable, "-m", "certified_obstacle_nn.scripts.true_error_cli",
        "--weights", str(weights_path), "--config", str(config_path),
        "--n-t", str(TRUE_ERROR_N), "--n-x", str(TRUE_ERROR_N),
        "--quadrature-tolerance", str(TRUE_ERROR_TOLERANCE), "--out", str(out_dir),
    ]
    true_error_proc = subprocess.run(true_error_cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert true_error_proc.returncode == 0, (
        f"true_error_cli failed:\n--- stdout ---\n{true_error_proc.stdout}\n--- stderr ---\n{true_error_proc.stderr}"
    )
    true_error_json = json.loads((out_dir / f"true_error_nt{TRUE_ERROR_N}_nx{TRUE_ERROR_N}.json").read_text())

    print(f"certify_cli peak RSS:    {cert_json['peak_rss_gib']:.3f} GiB")
    print(f"true_error_cli peak RSS: {true_error_json['peak_rss_gib']:.3f} GiB")

    return cert_json, true_error_json


def test_E_true_leq_E_cert_U_stefan(cert_and_true_error_records):
    """Test A -- THE correctness test. specification section 8 is explicit that
    the theorem gives E_true <= E_cert_U (the weak/U-norm bound, with the
    C_eq factor), never E_true <= E_cert (energy norm only, see test C).

    E_true's quadrature self-check is loosened to TRUE_ERROR_TOLERANCE (see
    module-level comment) since Stefan's free boundary defeats the module's
    default 1e-3 tolerance structurally, not from a bug -- if this ever needs
    revisiting, redo the empirical measurement rather than raising the
    tolerance blindly. If this test fails, see the diagnostic order above.
    """
    cert_json, true_error_json = cert_and_true_error_records
    E_true = true_error_json["result"]["E_true"]
    E_cert_U = cert_json["result"]["E_cert_U"]

    assert E_true <= E_cert_U


def test_effectivity_cert_has_theoretical_floor(cert_and_true_error_records):
    """Test B. I_eff_cert = sqrt(rho_tilde_h^2 + E_h) / E_true must never dip
    below 1/sqrt(C_eq * C_rel) = 1/sqrt(432) for Stefan (specification section 8's
    floor on I_eff_cert) -- this is just E_true <= E_cert_U rearranged, so it
    is really the same guarantee as test A, checked in the other direction.
    """
    cert_json, true_error_json = cert_and_true_error_records
    result = cert_json["result"]
    E_true = true_error_json["result"]["E_true"]

    I_eff_cert = math.sqrt(result["rho_tilde_h"] ** 2 + result["remainder"]["E_h"]) / E_true
    floor = 1.0 / math.sqrt(result["C_eq"] * result["C_rel"])

    assert result["C_eq"] * result["C_rel"] == pytest.approx(432.0)  # specification section 9
    assert I_eff_cert >= floor


# Test C's forbidden patterns match "<=" between the two quantity names in
# either order (E_true<=E_cert / E_cert>=E_true), tolerant of whitespace.
# E_cert_U is excluded via the negative lookahead -- that comparison is the
# one thing the theorem *does* guarantee.
_FORBIDDEN_ENERGY_CLAIM = re.compile(r"E_true\s*<=\s*E_cert(?!_U)\b|E_cert(?!_U)\b\s*>=\s*E_true")
_FORBIDDEN_ESTIMATOR_CLAIM = re.compile(r"E_true\s*<=\s*rho_tilde_h\b|rho_tilde_h\b\s*>=\s*E_true")


def test_negative_control_no_unproven_inequality_is_claimed_in_source():
    """Test C. specification section 8 is explicit that E_true <= E_cert (missing
    the C_eq factor) and E_true <= rho_tilde_h (an empirical estimator with no
    guaranteed relation to E_true -- quadrature can under- or over-estimate it
    from either side) are both UNPROVEN and may be false; the only guaranteed
    inequality is E_true <= E_cert_U (test A/B above). This statically scans
    the production source (not this test file, which legitimately discusses
    the prohibition in prose) to guard against a future regression that
    quietly asserts either forbidden comparison.
    """
    package_root = Path(certified_obstacle_nn.__file__).parent
    source_files = [
        p for p in package_root.rglob("*.py") if "tests" not in p.relative_to(package_root).parts
    ]
    assert len(source_files) > 10  # sanity: the scan actually found the package

    for path in source_files:
        text = path.read_text()
        assert not _FORBIDDEN_ENERGY_CLAIM.search(text), f"{path} appears to claim E_true <= E_cert"
        assert not _FORBIDDEN_ESTIMATOR_CLAIM.search(text), f"{path} appears to claim E_true <= rho_tilde_h"
