"""Tests for experiments/results_io.py's uniform run-artifact layout."""

import time

import pandas as pd
import pytest
import torch

from certified_obstacle_nn.experiments.results_io import (
    losses_csv_path,
    make_run_dir,
    read_summary,
    write_config,
    write_summary,
    write_weights,
)


class _TinyTriple(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)


def test_make_run_dir_is_fresh_and_timestamped(tmp_path):
    run_dir = make_run_dir("some_experiment", results_dir=tmp_path)
    assert run_dir.exists()
    assert run_dir.parent.name == "some_experiment"
    assert run_dir.parent.parent == tmp_path


def test_make_run_dir_raises_rather_than_silently_reusing_a_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(time, "strftime", lambda *a, **kw: "20260101T000000Z")
    make_run_dir("exp", results_dir=tmp_path)
    with pytest.raises(FileExistsError):
        make_run_dir("exp", results_dir=tmp_path)


def test_write_and_read_config_roundtrip(tmp_path):
    run_dir = make_run_dir("exp", results_dir=tmp_path)
    config = {"seed": 0, "architecture": {"width": 64, "depth": 4}, "N_train": 5000, "grid": 256, "eps_u": None}
    path = write_config(run_dir, config)
    assert path == run_dir / "config.json"
    import json
    assert json.loads(path.read_text()) == config


def test_losses_csv_path_matches_trainer_output(tmp_path):
    from certified_obstacle_nn.models.neural_triple import NeuralTriple
    from certified_obstacle_nn.problems.stefan import StefanProblem
    from certified_obstacle_nn.training.trainer import train

    run_dir = make_run_dir("exp", results_dir=tmp_path)
    problem = StefanProblem()
    torch.manual_seed(0)
    triple = NeuralTriple(problem, width=8, depth=2)
    t = torch.rand(20, 1, requires_grad=True)
    x = torch.rand(20, 1, requires_grad=True)
    x0 = torch.rand(5, 1)

    train(problem, triple, t, x, x0, n_epochs=3, csv_path=str(losses_csv_path(run_dir)))

    path = losses_csv_path(run_dir)
    assert path == run_dir / "losses.csv"
    assert path.exists()
    df = pd.read_csv(path)
    for col in ("loss_total", "loss_pde", "loss_sigma", "loss_initial", "loss_penetration", "loss_complementarity"):
        assert col in df.columns
    assert len(df) == 3


def test_write_summary_is_one_row_with_separate_components(tmp_path):
    run_dir = make_run_dir("exp", results_dir=tmp_path)
    summary = {
        "rho_r": 1.0, "rho_p_tilde": 2.0, "rho_c": 3.0,
        "E_h_lip": 4.0, "E_h_unr": 5.0, "E_h_dat": 6.0, "E_h_0T": 7.0,
        "E_true": 8.0, "E_cert": 9.0, "E_cert_U": 10.0,
    }
    path = write_summary(run_dir, summary)
    assert path == run_dir / "summary.csv"

    loaded = read_summary(run_dir)
    for key, value in summary.items():
        assert loaded[key] == pytest.approx(value)


def test_read_summary_raises_on_multi_row_file(tmp_path):
    run_dir = make_run_dir("exp", results_dir=tmp_path)
    pd.DataFrame([{"a": 1}, {"a": 2}]).to_csv(run_dir / "summary.csv", index=False)
    with pytest.raises(AssertionError):
        read_summary(run_dir)


def test_write_weights_roundtrip(tmp_path):
    run_dir = make_run_dir("exp", results_dir=tmp_path)
    triple = _TinyTriple()
    path = write_weights(run_dir, triple)
    assert path == run_dir / "weights.pt"

    loaded_state = torch.load(path, weights_only=True)
    fresh = _TinyTriple()
    fresh.load_state_dict(loaded_state)
    for p1, p2 in zip(triple.parameters(), fresh.parameters()):
        assert torch.equal(p1, p2)
