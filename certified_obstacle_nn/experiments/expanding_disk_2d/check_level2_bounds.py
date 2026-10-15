"""Sanity check of level2_diagnostic's a priori bounds against sampled values on a trained
triple: sup-norm bounds vs sampled maxima, Lipschitz bounds vs two-point secants (exact
inequality at finite separation, first-order autograd only). Exits non-zero on violation."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch  # noqa: E402

from certified_obstacle_nn.experiments.expanding_disk_2d import exact_solution as ex, sampling  # noqa: E402
from certified_obstacle_nn.experiments.expanding_disk_2d import level2_diagnostic as l2  # noqa: E402
from certified_obstacle_nn.experiments.expanding_disk_2d.config import Config  # noqa: E402
from certified_obstacle_nn.experiments.expanding_disk_2d.losses import triple_fields  # noqa: E402
from certified_obstacle_nn.experiments.expanding_disk_2d.model import Triple2D  # noqa: E402

run_dir = Path(sys.argv[1])
cd = json.loads((run_dir / "config.json").read_text())
cfg = Config(**{k: cd[k] for k in Config.__dataclass_fields__ if k in cd})
tr = Triple2D(cfg.depth, cfg.width, cfg.multiplier_head).double()
tr.load_state_dict(torch.load(run_dir / "checkpoint.pt"))
tr.eval()
bd = l2.triple_bounds(tr, cfg, l2.data_bounds(cfg.kappa))


def quantities(t, x, y):
    t, x, y = (v.clone().requires_grad_(True) for v in (t, x, y))
    u, ut, gu, sig, ds, lam = (v.detach() for v in triple_fields(tr, t, x, y, create_graph=False))
    t, x, y = t.detach(), x.detach(), y.detach()
    R = ex.f_exact(t, x, y, cfg.kappa) - ut - ds + lam
    gam = ex.g(t, x, y) - u
    return R, gam, lam


t, x, y = sampling.interior(200_000, 999)
R, gam, lam = quantities(t, x, y)
d = 1e-3
dirs = torch.randn(200_000, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(5))
dirs = d * dirs / dirs.norm(dim=1, keepdim=True)
t2, x2, y2 = ((v + dirs[:, k:k + 1]).clamp(0, 1) for k, v in enumerate((t, x, y)))
dist = torch.sqrt((t2 - t) ** 2 + (x2 - x) ** 2 + (y2 - y) ** 2).clamp(min=1e-12)
R2, gam2, lam2 = quantities(t2, x2, y2)
rows = [("m_R_apriori >= max|R_PDE|", R.abs().max().item(), bd["m_R_apriori"]),
        ("m_gap >= max|gamma|", gam.abs().max().item(), bd["m_gap"]),
        ("m_lam >= max lambda", lam.max().item(), bd["m_lam"]),
        ("L_R >= secant(R_PDE)", ((R2 - R).abs() / dist).max().item(), bd["L_R"]),
        ("L_gamma >= secant(gamma)", ((gam2 - gam).abs() / dist).max().item(), bd["L_gamma"])]
ok = True
for name, s, b in rows:
    ok &= s <= b
    print(f"{name:28s} sampled={s:.4e} bound={b:.4e} ratio={s / b:.2e} {'OK' if s <= b else 'VIOLATED'}")
sys.exit(0 if ok else 1)
