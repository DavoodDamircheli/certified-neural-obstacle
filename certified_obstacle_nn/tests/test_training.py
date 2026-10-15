"""Tests for training/sampler.py, training/losses.py, training/trainer.py."""

import pandas as pd
import pytest
import torch

from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.problems.stefan import StefanProblem
from certified_obstacle_nn.residuals.residuals import Residuals
from certified_obstacle_nn.training.losses import compute_loss
from certified_obstacle_nn.training.sampler import TrainingSampler
from certified_obstacle_nn.training.trainer import LBFGSOutcome, train


def test_sampler_reproducible_and_in_bounds():
    problem = StefanProblem()
    sampler = TrainingSampler(problem, n_interior=1000, n_initial=200, seed=42)

    t1, x1, x01 = sampler.sample()
    t2, x2, x02 = sampler.sample()

    assert torch.equal(t1.detach(), t2.detach())
    assert torch.equal(x1.detach(), x2.detach())
    assert torch.equal(x01, x02)

    a, b = problem.omega_bounds
    assert (t1 >= 0).all() and (t1 <= problem.T).all()
    assert (x1 >= a).all() and (x1 <= b).all()
    assert (x01 >= a).all() and (x01 <= b).all()

    interior_t, interior_x = sampler.interior_points
    assert torch.equal(interior_t, t1.detach())
    assert torch.equal(interior_x, x1.detach())
    assert torch.equal(sampler.initial_points, x01)


def test_sampler_different_seeds_differ():
    problem = StefanProblem()
    t_a, x_a, _ = TrainingSampler(problem, 500, 100, seed=0).sample()
    t_b, x_b, _ = TrainingSampler(problem, 500, 100, seed=1).sample()

    assert not torch.equal(t_a.detach(), t_b.detach())
    assert not torch.equal(x_a.detach(), x_b.detach())


def test_sampler_exposes_points_for_disjointness_checks():
    problem = StefanProblem()
    sampler = TrainingSampler(problem, n_interior=300, n_initial=50, seed=7)
    sampler.sample()

    train_t, train_x = sampler.interior_points
    train_coords = torch.cat([train_t, train_x], dim=-1)

    # a certification-style grid deliberately offset from the random training
    # points so it cannot collide by construction; simulates the check
    # certification/ would run against sampler.interior_points.
    cert_t = torch.linspace(0.0, 1.0, 11) + 0.5 / 11
    cert_x = torch.linspace(0.0, 1.0, 13) + 0.5 / 13
    cert_coords = torch.cartesian_prod(cert_t, cert_x)

    for row in cert_coords:
        assert not (train_coords == row).all(dim=-1).any()


def test_loss_uses_correct_weights_and_does_not_square_complementarity():
    n = 10
    residuals = Residuals(
        R_PDE=torch.full((n, 1), 2.0),
        R_sig=torch.full((n, 1), 3.0),
        R_0=torch.full((n, 1), 4.0),
        R_obs=torch.full((n, 1), 5.0),
        R_com=torch.full((n, 1), 0.5),
        gamma=torch.zeros((n, 1)),
        gamma_minus=torch.zeros((n, 1)),
        w=torch.zeros((n, 1)),
    )

    loss = compute_loss(residuals, w_r=1.0, w_s=2.0, w_0=3.0, w_p=4.0, w_c=5.0)

    assert loss.pde.item() == pytest.approx(4.0)  # mean(2^2)
    assert loss.sigma.item() == pytest.approx(9.0)  # mean(3^2)
    assert loss.initial.item() == pytest.approx(16.0)  # mean(4^2)
    assert loss.penetration.item() == pytest.approx(25.0)  # mean(5^2)
    assert loss.complementarity.item() == pytest.approx(0.5)  # mean, NOT squared

    expected_total = 1.0 * 4.0 + 2.0 * 9.0 + 3.0 * 16.0 + 4.0 * 25.0 + 5.0 * 0.5
    assert loss.total.item() == pytest.approx(expected_total)


def test_trainer_reduces_loss_and_writes_csv(tmp_path):
    problem = StefanProblem()
    torch.manual_seed(0)
    triple = NeuralTriple(problem, width=32, depth=3)

    sampler = TrainingSampler(problem, n_interior=1000, n_initial=100, seed=0)
    t, x, x0 = sampler.sample()

    csv_path = tmp_path / "history.csv"
    history, outcome = train(problem, triple, t, x, x0, n_epochs=50, lr=1e-3, csv_path=str(csv_path))

    assert list(history.columns) == [
        "epoch",
        "phase",
        "loss_total",
        "loss_pde",
        "loss_sigma",
        "loss_initial",
        "loss_penetration",
        "loss_complementarity",
    ]
    assert len(history) == 50
    assert (history["phase"] == "adam").all()
    assert history["loss_total"].iloc[-1] < history["loss_total"].iloc[0]
    assert outcome == LBFGSOutcome(reason="skipped", n_steps_completed=0)

    assert csv_path.exists()
    reloaded = pd.read_csv(csv_path)
    assert len(reloaded) == 50


def test_lbfgs_refines_after_adam_and_logs_both_phases():
    problem = StefanProblem()
    torch.manual_seed(0)
    triple = NeuralTriple(problem, width=32, depth=3)

    sampler = TrainingSampler(problem, n_interior=1000, n_initial=100, seed=0)
    t, x, x0 = sampler.sample()

    # 10 steps stays in the clean, strictly-decreasing regime for this
    # architecture/seed (see test_lbfgs_detects_genuine_line_search_stall for
    # what happens if it runs long enough to hit a real stall).
    n_epochs, n_lbfgs_epochs = 200, 10
    history, outcome = train(problem, triple, t, x, x0, n_epochs=n_epochs, lr=1e-3, n_lbfgs_epochs=n_lbfgs_epochs)

    assert outcome.reason == "completed"
    assert outcome.n_steps_completed == n_lbfgs_epochs

    adam_rows = history[history["phase"] == "adam"]
    lbfgs_rows = history[history["phase"] == "lbfgs"]
    assert len(adam_rows) == n_epochs
    assert len(lbfgs_rows) == n_lbfgs_epochs
    # epoch numbering continues across the phase switch, it does not reset
    assert list(lbfgs_rows["epoch"]) == list(range(n_epochs, n_epochs + n_lbfgs_epochs))

    # L-BFGS should further refine what Adam handed it, not regress on it
    assert lbfgs_rows["loss_total"].iloc[-1] <= adam_rows["loss_total"].iloc[-1] + 1e-9
    assert lbfgs_rows["loss_total"].is_monotonic_decreasing


def test_lbfgs_detects_genuine_line_search_stall():
    """Not a synthetic failure: this exact architecture/seed/point-set
    genuinely runs strong_wolfe L-BFGS into a stall (loss goes bit-identical
    flat while the gradient norm stays >> 0) around step 13. This is the real
    failure mode the abort logic exists for -- it must be caught and reported,
    not silently returned as if training kept progressing.
    """
    problem = StefanProblem()
    torch.manual_seed(0)
    triple = NeuralTriple(problem, width=32, depth=3)

    sampler = TrainingSampler(problem, n_interior=1000, n_initial=100, seed=0)
    t, x, x0 = sampler.sample()

    n_epochs, n_lbfgs_epochs = 200, 30
    history, outcome = train(problem, triple, t, x, x0, n_epochs=n_epochs, lr=1e-3, n_lbfgs_epochs=n_lbfgs_epochs)

    assert outcome.reason == "line_search_stalled"
    assert 0 < outcome.n_steps_completed < n_lbfgs_epochs

    lbfgs_rows = history[history["phase"] == "lbfgs"]
    assert len(lbfgs_rows) == outcome.n_steps_completed  # no corrupted/stalled row appended


def test_lbfgs_aborts_cleanly_on_non_finite_loss():
    problem = StefanProblem()
    torch.manual_seed(0)
    triple = NeuralTriple(problem, width=8, depth=2)

    def nan_forward(t, x):
        return torch.full_like(t, float("nan"))

    triple.multiplier_net.forward = nan_forward  # deterministically poison the graph

    sampler = TrainingSampler(problem, n_interior=200, n_initial=50, seed=0)
    t, x, x0 = sampler.sample()

    history, outcome = train(problem, triple, t, x, x0, n_epochs=0, n_lbfgs_epochs=5)

    assert outcome.reason == "non_finite_loss"
    assert outcome.n_steps_completed == 0
    assert len(history) == 0
