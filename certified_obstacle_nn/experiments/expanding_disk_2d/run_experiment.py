"""CLI: train + evaluate + raw estimator + interface metrics + figures for one configuration.

  python -m certified_obstacle_nn.experiments.expanding_disk_2d.run_experiment \
      --stage stage2 --arch 4x64 --n_int 20000 --seed 0 --kappa 5

Outputs results/expanding_disk_2d/<run_name>/{config.json, metrics.json, loss_history.csv,
monitor_history.csv, train_info.json, checkpoint.pt, figures/}.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch  # noqa: E402

from certified_obstacle_nn.experiments.expanding_disk_2d import evaluate, interface_metrics, plots  # noqa: E402
from certified_obstacle_nn.experiments.expanding_disk_2d.config import Config  # noqa: E402
from certified_obstacle_nn.experiments.expanding_disk_2d.train import make_run_dir, peak_rss_mb, train  # noqa: E402


def run(cfg: Config) -> Path:
    torch.set_num_threads(cfg.extra.get("threads", torch.get_num_threads()))
    run_dir = make_run_dir(cfg)
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    print(f"run dir: {run_dir}", flush=True)
    triple, info = train(cfg, run_dir, log=lambda s: print(s, flush=True))
    for p in triple.parameters():
        p.requires_grad_(False)
    triple.eval()
    t0 = time.time()
    metrics = evaluate.evaluate_all(triple, cfg)
    metrics.update(interface_metrics.compute_interface_metrics(triple, cfg.kappa, cfg.eps_gamma))
    metrics["eval_time_s"] = time.time() - t0
    metrics.update({k: info[k] for k in ("train_time_s", "n_params", "lbfgs_outcome", "lbfgs_steps_completed",
                                         "nan_during_adam")})
    figs = plots.make_run_figures(triple, cfg, run_dir / "figures") + plots.loss_figure(run_dir)
    metrics["peak_rss_mb"] = peak_rss_mb()
    metrics["figures"] = figs
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    keys = ("E_true", "rho_tilde", "I_eff_raw", "rho_r", "rho_p_tilde", "rho_c", "err_u_L2", "err_sigma_L2",
            "err_lambda_L2", "iface_lambda_mean", "iface_lambda_max", "iface_gap_mean", "E_FB", "train_time_s")
    for k in keys:
        print(f"  {k:20s} {metrics[k]}")
    return run_dir


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="dev")
    ap.add_argument("--arch", default="4x64")
    ap.add_argument("--n_int", type=int, default=20_000)
    ap.add_argument("--n_init", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kappa", type=float, default=5.0)
    ap.add_argument("--n_adam", type=int, default=5000)
    ap.add_argument("--n_lbfgs", type=int, default=300)
    ap.add_argument("--w_r", type=float, default=1.0)
    ap.add_argument("--w_s", type=float, default=1.0)
    ap.add_argument("--w_0", type=float, default=1.0)
    ap.add_argument("--w_p", type=float, default=1.0)
    ap.add_argument("--w_c", type=float, default=1.0)
    ap.add_argument("--band_fraction", type=float, default=0.0)
    ap.add_argument("--n_eval", type=int, default=1_000_000)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    extra = {"threads": a.threads} if a.threads else {}
    cfg = Config(stage=a.stage, arch=a.arch, n_int=a.n_int, n_init=a.n_init, seed=a.seed, kappa=a.kappa,
                 n_adam=a.n_adam, n_lbfgs=a.n_lbfgs, w_r=a.w_r, w_s=a.w_s, w_0=a.w_0, w_p=a.w_p, w_c=a.w_c,
                 band_fraction=a.band_fraction, n_eval=a.n_eval, tag=a.tag, extra=extra)
    run(cfg)


if __name__ == "__main__":
    main()
