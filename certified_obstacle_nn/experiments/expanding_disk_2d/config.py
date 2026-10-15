"""Run configuration for the 2D expanding-disk benchmark.

Defaults mirror the 1D Stefan experiment (experiments/stefan_core.py, specification section 3):
loss weights all 1, Adam lr 1e-3 for 5000 steps, then L-BFGS (lr 1, strong_wolfe, max_iter=1
per step) for <= 300 steps, warm-started. Lambda = diam(Omega)^2 = 2 (1D uses diam^2 = 1).
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = REPO_ROOT / "results" / "expanding_disk_2d"

ARCHITECTURES = {"3x32": (3, 32), "4x64": (4, 64), "6x64": (6, 64), "6x128": (6, 128)}


@dataclass
class Config:
    stage: str = "dev"
    arch: str = "4x64"
    seed: int = 0
    kappa: float = 5.0
    multiplier_variant: str = "constant"
    multiplier_head: str = "square"
    # problem constants (A = I, b = c = 0)
    Lambda: float = 2.0
    alpha: float = 1.0
    T: float = 1.0
    # sampling
    n_int: int = 20_000
    n_init: int | None = None  # default: n_int // 10
    band_fraction: float = 0.0  # Stage-2 fallback item 5; 0 = off
    band_halfwidth: float = 0.05
    # optimizer
    n_adam: int = 5000
    adam_lr: float = 1e-3
    n_lbfgs: int = 300
    lbfgs_lr: float = 1.0
    # loss weights (1D Stefan defaults)
    w_r: float = 1.0
    w_s: float = 1.0
    w_0: float = 1.0
    w_p: float = 1.0
    w_c: float = 1.0
    # monitoring / evaluation
    monitor_every_adam: int = 250
    monitor_every_lbfgs: int = 50
    n_monitor: int = 20_000
    n_eval: int = 1_000_000
    n_eval_trace: int = 200_000
    eval_batch: int = 50_000
    eps_gamma: float = 1e-3
    tag: str = ""
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.n_init is None:
            self.n_init = max(self.n_int // 10, 500)
        if self.arch not in ARCHITECTURES:
            raise ValueError(f"unknown arch {self.arch!r}")
        if self.multiplier_variant != "constant":
            raise NotImplementedError("only the default constant-multiplier variant is implemented")

    @property
    def depth(self) -> int:
        return ARCHITECTURES[self.arch][0]

    @property
    def width(self) -> int:
        return ARCHITECTURES[self.arch][1]

    @property
    def run_name(self) -> str:
        name = f"{self.stage}_{self.arch}_N{self.n_int}_s{self.seed}_k{self.kappa:g}"
        return name + (f"_{self.tag}" if self.tag else "")

    # independent Sobol seeds per point set (train / initial / monitor / eval / eval traces)
    @property
    def sobol_seeds(self) -> dict:
        s = self.seed
        return {"train": 1000 + s, "init": 2000 + s, "band": 3000 + s, "monitor": 50_000 + s,
                "eval": 100_000 + s, "eval_t0": 200_000 + s, "eval_T": 300_000 + s}

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(depth=self.depth, width=self.width, run_name=self.run_name, sobol_seeds=self.sobol_seeds,
                 dtype="float64", kappa_p=self.alpha / self.Lambda)
        return d
