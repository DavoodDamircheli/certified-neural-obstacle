"""v17b Task 3: protocol-v2 marking objective and the selection-state import of chunked_certify."""

import json
import time

import numpy as np
import pandas as pd
import pytest

from certified_obstacle_nn.certification.enclose import boxes as Bx
from certified_obstacle_nn.experiments.v17 import chunked_certify as CC
from certified_obstacle_nn.experiments.v17.common import load_stored_run
from certified_obstacle_nn.experiments.v17.stage5_pilot import LONG0_1E5
from certified_obstacle_nn.experiments.v17.stage6 import grad_contrib
from certified_obstacle_nn.tests.stored_data import needs


@pytest.mark.parametrize("D,L", [(0.045, 0.029), (1.0, 1e-3), (1e-3, 1.0)])
def test_grad_weight_is_derivative_ratio_of_G2(D, L):
    rng = np.random.default_rng(1)
    n = 50
    area = rng.uniform(0.5, 1.5, n) / n
    Sh, c, Rh = rng.uniform(0, 1, (3, n))
    cf = 1.0
    scale_d = D / float((area * (Sh + 2 * c)).sum())
    scale_l = L / float((area * Rh).sum())
    vol = {"Sh_term": Sh * scale_d, "c": c * scale_d, "Rh_term": Rh * scale_l}
    mu, w = grad_contrib(vol, area, cf)

    def G2(D_, L_):
        return (np.sqrt(L_) + np.sqrt(D_ + L_)) ** 2

    h = 1e-7
    dL = (G2(D, L + h) - G2(D, L - h)) / (2 * h)
    dD = (G2(D + h, L) - G2(D - h, L)) / (2 * h)
    assert w == pytest.approx(dL / dD, rel=1e-6)
    assert np.allclose(mu, area * (vol["Sh_term"] + 2 * vol["c"]) + w * area * vol["Rh_term"])


@needs('results/v14_disk_validated/20261007T212554Z')
def test_import_selection_equals_chunk_evaluation(tmp_path, monkeypatch):
    from certified_obstacle_nn.experiments import results_io

    monkeypatch.setattr(results_io, "RESULTS_DIR", tmp_path)
    orig = results_io.make_run_dir
    monkeypatch.setattr(results_io, "make_run_dir", lambda exp, results_dir=tmp_path: orig(exp, tmp_path))
    r = load_stored_run(LONG0_1E5)
    part = Bx.uniform(3, 8)
    pd_ = tmp_path / "part"
    pd_.mkdir()
    np.savez_compressed(pd_ / "partition.npz", lo=part.lo, hi=part.hi, lo0=r.tp0.lo, hi0=r.tp0.hi, loT=r.tpT.lo,
                        hiT=r.tpT.hi)
    (pd_ / "partition.json").write_text(json.dumps({"partition_hash": part.hash(),
                                                    "refinement": {"selection_s": 1.0}}))
    np.savez_compressed(pd_ / "selection.npz", **Bx.evaluate_volume(r.encloser, part, "numpy", workers=8))
    j1 = CC.import_selection(LONG0_1E5, str(pd_))
    time.sleep(1.1)  # run dirs are named by the second
    j2 = CC.init(LONG0_1E5, "TM1", 2, str(pd_))
    for i in range(2):
        CC.chunk(j2, i, 8)
    a = pd.read_csv(CC.finalize(j1, 8) / "summary.csv", float_precision="round_trip").iloc[0]
    time.sleep(1.1)
    b = pd.read_csv(CC.finalize(j2, 8) / "summary.csv", float_precision="round_trip").iloc[0]
    for k in ("U_R", "U_S", "U_I", "U_C", "flux_bound_direct_split", "grad_bound_direct_split", "E_dir_split"):
        assert a[k] == b[k], k
    assert json.loads((j1 / "job.json").read_text())["densities_from"]["selection_state"].endswith("selection.npz")
