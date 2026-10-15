"""v19 Task C: the control arm of experiments/v19/multiplier_train.py is experiments/v14/ellipse_train.py (same weights
bit for bit on a short schedule), and the intervention differs from it only by the M_psi output bias at the start."""

import argparse

import torch

from certified_obstacle_nn.experiments.v14 import ellipse_train as ET
from certified_obstacle_nn.experiments.v19 import multiplier_train as MT

SMALL = dict(seed=3, depth=2, width=8, n_int=300, n_init=50, n_adam=4, n_lbfgs=3, w_p=10.0, w_c=10.0,
             kappa0_num=1, kappa0_den=100, threads=1, monitor_every=0, protocol="test", experiment="unused")


def _ellipse_train_reference(a):
    """ellipse_train.main's optimisation, inlined without writing a run directory (same statements, same order)."""
    import math
    import random

    import numpy as np

    from certified_obstacle_nn.experiments.expanding_disk_2d import sampling
    from certified_obstacle_nn.experiments.expanding_disk_2d.losses import compute_loss
    from certified_obstacle_nn.experiments.expanding_disk_2d.model import Triple2D
    from certified_obstacle_nn.problems.ellipse import EllipseProblem

    torch.set_num_threads(a.threads)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    problem = EllipseProblem(kappa0=a.kappa0_num / a.kappa0_den)
    seeds = {"train": 1000 + a.seed, "init": 2000 + a.seed}
    W = dict(w_r=1.0, w_s=1.0, w_0=1.0, w_p=a.w_p, w_c=a.w_c)
    triple = Triple2D(a.depth, a.width, "square").double()
    t, x, y = sampling.interior(a.n_int, seeds["train"])
    x0, y0 = sampling.initial(a.n_init, seeds["init"])
    t.requires_grad_(True), x.requires_grad_(True), y.requires_grad_(True)
    opt = torch.optim.Adam(triple.parameters(), lr=1e-3)
    for _ in range(a.n_adam):
        opt.zero_grad()
        L = compute_loss(ET.residuals(problem, triple, t, x, y, x0, y0), **W)
        L.total.backward()
        opt.step()
    lb = torch.optim.LBFGS(triple.parameters(), lr=1.0, max_iter=1, line_search_fn="strong_wolfe")
    hold = {}

    def closure():
        lb.zero_grad()
        Lc = compute_loss(ET.residuals(problem, triple, t, x, y, x0, y0), **W)
        Lc.total.backward()
        hold["L"] = Lc
        return Lc.total

    prev = None
    for _ in range(a.n_lbfgs):
        lb.step(closure)
        closure()
        tot = hold["L"].total.item()
        gn = math.sqrt(sum(p.grad.detach().pow(2).sum().item() for p in triple.parameters() if p.grad is not None))
        if prev is not None and (tot > prev + 1e-9 or (abs(tot - prev) < 1e-12 and gn > 1e-6)):
            break
        prev = tot
    return triple


def test_control_equals_ellipse_train():
    ref = _ellipse_train_reference(argparse.Namespace(**SMALL, arm="control"))
    o = MT.train(argparse.Namespace(**SMALL, arm="control"), run_dir=None, log=lambda m: None)
    for (k, p), (k2, q) in zip(ref.state_dict().items(), o["triple"].state_dict().items()):
        assert k == k2 and torch.equal(p, q), k
    assert len(o["psi"]) == SMALL["n_adam"] + o["done"]


def test_intervention_changes_only_the_multiplier_output_bias_at_start():
    c = MT.build(argparse.Namespace(**SMALL, arm="control"))
    i = MT.build(argparse.Namespace(**SMALL, arm="intervention"))
    diff = [k for (k, p), (_, q) in zip(c.state_dict().items(), i.state_dict().items()) if not torch.equal(p, q)]
    last = f"mult.layers.{len(c.mult.layers) - 1}.bias"
    assert diff == [last]
    assert torch.all(i.state_dict()[last] == 1.0)
