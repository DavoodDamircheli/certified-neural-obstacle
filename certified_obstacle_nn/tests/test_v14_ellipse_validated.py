"""v14 Stage 6: the ellipse validated driver (experiments/v14/ellipse_validated.py)
on a tiny frozen random Triple2D: validated status, U's dominate the float64
quadrature, rows carry every bound variant and the provenance fields, the run
directory follows the results_io convention."""

import json

import pandas as pd
import pytest
import torch

from certified_obstacle_nn.errors.diagnostics_2d import diagnostics
from certified_obstacle_nn.experiments.expanding_disk_2d.model import Triple2D
from certified_obstacle_nn.experiments.v14 import ellipse_validated as EV
from certified_obstacle_nn.problems.ellipse import EllipseProblem


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    torch.manual_seed(3)
    torch.set_default_dtype(torch.float64)
    triple = Triple2D(2, 4, "square").double()
    for p in triple.parameters():
        p.requires_grad_(False)
    triple.eval()
    d = tmp_path_factory.mktemp("ell")
    ck = d / "weights.pt"
    torch.save(triple.state_dict(), ck)
    diag = diagnostics(EllipseProblem(0.01), triple, n=4, q=2, n_trace=8)
    diag["rel_change"] = {}
    return triple, ck, diag


def test_ellipse_driver_validated_rows(tiny, tmp_path):
    triple, ck, diag = tiny
    config, rows, part, vols, _ = EV.run(triple, (1, 100), ck, seed=0, budget=600, workers=1, prec=64,
                                         arb_max_cells=10**6, start=8, trace_budget=4096, theta=0.5, diag=diag,
                                         log=lambda m: None)
    assert part.n <= 600
    assert {r["backend"] for r in rows} == {"numpy", "arb"}
    for r in rows:
        assert r["status"] == "validated_upper_bound", r["status_reasons"]
        assert r["problem"] == "ellipse" and r["schedule"] == "main"
        # validated upper sums dominate the (coarse) float64 quadrature of the same integrals
        assert r["U_r"] >= diag["r2"] and r["U_p"] >= diag["p2"] and r["U_c"] >= diag["c2"]
        assert r["U_r_hat"] >= diag["r2_hat"]
        for k in ("E_U_weighted", "E_U_uniform", "E_U_projected", "E_U_original_via_projection", "grad_bound_rel",
                  "partition_hash", "checkpoint_sha256", "config_hash"):
            assert k in r
    nb, ab = rows
    assert abs(nb["U_r"] - ab["U_r"]) <= 1e-6 * ab["U_r"]
    assert config["problem_constants"]["Lambda"] == 2.0
    run_dir = EV.write_run(config, rows, part, vols, ck, results_dir=tmp_path)
    assert run_dir.parent.parent == tmp_path
    assert {p.name for p in run_dir.iterdir()} >= {"config.json", "summary.csv", "cells.npz", "weights.pt"}
    assert len(pd.read_csv(run_dir / "summary.csv")) == 2
    assert json.loads((run_dir / "config.json").read_text())["kappa0"] == [1, 100]


def test_read_summary_keeps_hex_digests_as_text(tmp_path):
    # pandas 2.3.3 segfaults when it infers a type for "5e72945346954026cc..." (float with a huge exponent);
    # read_summary must read every hash column as text.
    from certified_obstacle_nn.experiments.v14.common import read_summary

    h = "5e72945346954026cc6045d2d16ac39aabee982d65113fa35c78812f7a6c4217"
    p = tmp_path / "summary.csv"
    p.write_text(f"U_r,config_hash,partition_hash\n1.5,{h},1e5\n")
    d = read_summary(p)
    assert d["config_hash"][0] == h and d["partition_hash"][0] == "1e5" and d["U_r"][0] == 1.5


def test_find_training_run_requires_a_unique_clean_run(tmp_path, monkeypatch):
    monkeypatch.setattr(EV, "TRAIN_ROOT", tmp_path)
    with pytest.raises(RuntimeError):
        EV.find_training_run(0)
    head = "problem,variant,seed,git_dirty,weights_sha256\n"
    for name, dirty in (("a", False), ("b", True)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "summary.csv").write_text(head + f"ellipse,main,0,{dirty},00\n")
    assert EV.find_training_run(0) == tmp_path / "a"  # the dirty run is ignored
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "summary.csv").write_text(head + "ellipse,main,0,False,00\n")
    with pytest.raises(RuntimeError):
        EV.find_training_run(0)
