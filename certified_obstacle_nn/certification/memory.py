"""Memory observability and OOM-prevention primitives for certification.

Authoritative reference: docs/history/MEMORY_OOM_DEBUGGING_AND_SAFE_RUN_POLICY.md. Every
formula and message format here matches that document section-for-section;
if the two ever disagree, the document wins and this module has a bug.

This module never raises based on virtual address space (RLIMIT_AS) -- see
policy section 3 for why that was tried and abandoned. `MemoryGuard` reads
real resident memory (RSS) via `psutil`, the only signal that actually
predicts whether the OS OOM killer will intervene.

`MemoryGuard.check()` raising is a hard stop, never a silent downgrade
(policy section 4, point 5): no automatic reduction of n_qp, chunk size, grid
resolution, or precision happens inside this module.
"""

import functools
from typing import Optional

import psutil

GIB = 1024 ** 3

# Calibrated against certified_obstacle_nn.certification.raw_estimator's
# shipped (create_graph=True, unchunked) path: forward-pass factor ~1.0 +
# create_graph=True derivative factor ~1.1, rounded up for safety margin (an
# OOM guard should over-, not under-, predict). See policy doc section 2.
CHUNK_BYTES_FACTOR = 2.5


def report_memory(label: str) -> float:
    """Print current RSS in GiB, flush=True. Returns the value in GiB."""
    rss_gib = psutil.Process().memory_info().rss / GIB
    print(f"[{label}] RSS = {rss_gib:.3f} GiB", flush=True)
    return rss_gib


def peak_rss() -> int:
    """This process' high-water-mark RSS in bytes, read from VmHWM in
    /proc/self/status -- lets a process report its own peak without shelling
    out to `/usr/bin/time -v`. VmHWM is monotone non-decreasing by kernel
    definition for the lifetime of the process.
    """
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024  # kB -> bytes
    raise RuntimeError("VmHWM not found in /proc/self/status (not Linux?)")


def gib_for_array(shape: tuple[int, ...], itemsize: int = 8) -> float:
    """Size in GiB of a dense array of the given shape/itemsize (default
    itemsize=8 for float64, specification section 12 invariant 3)."""
    n_elements = 1
    for s in shape:
        n_elements *= s
    return n_elements * itemsize / GIB


def estimate_chunk_bytes(
    n_points: int,
    depth: int,
    width: int,
    n_networks: int,
    bytes_per_element: int = 8,
) -> int:
    """Predicted peak bytes for evaluating `n_points` quadrature points
    through `n_networks` tanh MLPs of the given (depth, width), under the
    CURRENTLY SHIPPED certification path (create_graph=True throughout --
    see policy doc section 1). This is a calibrated approximation, not a
    simulator: log it next to a real measurement (report_memory /
    peak_rss()) so the model's error is visible on every run, per policy
    doc section 2.
    """
    return int(n_points * n_networks * depth * width * bytes_per_element * CHUNK_BYTES_FACTOR)


class CertificationMemoryError(RuntimeError):
    """Raised by MemoryGuard.check() on an RSS breach. Never caught inside
    this module -- the run aborts cleanly, it is never silently downgraded
    (policy doc section 4, point 5)."""


class MemoryGuard:
    """
    guard = MemoryGuard(max_rss_gb=8.0, label_prefix="certify")
    guard.check("before chunk 12")

    `chunk_size`, `n_qp`, `grid` are optional context the guard reports back
    in the error message's knob list (policy doc section 8) if known; when a
    knob's current value wasn't given to the guard, it is still listed (so
    the ordering and the grid-resolution warning are always present
    verbatim), just without a "currently ..." clause.
    """

    def __init__(
        self,
        max_rss_gb: float,
        label_prefix: str = "",
        chunk_size: Optional[int] = None,
        n_qp: Optional[int] = None,
        grid: Optional[tuple[int, int]] = None,
    ):
        self.max_rss_gb = max_rss_gb
        self.label_prefix = label_prefix
        self.chunk_size = chunk_size
        self.n_qp = n_qp
        self.grid = grid

    def check(self, label: str) -> float:
        rss_gib = psutil.Process().memory_info().rss / GIB
        full_label = f"{self.label_prefix}:{label}" if self.label_prefix else label
        if rss_gib > self.max_rss_gb:
            raise CertificationMemoryError(self._format_message(rss_gib, full_label))
        return rss_gib

    def _format_message(self, rss_gib: float, label: str) -> str:
        chunk_line = "chunk size" + (f" (currently {self.chunk_size} pts/chunk)" if self.chunk_size is not None else "")
        nqp_line = "n_qp" + (f" (currently {self.n_qp})" if self.n_qp is not None else "")
        grid_line = "grid resolution" + (f" (currently {self.grid[0]}x{self.grid[1]})" if self.grid is not None else "")
        return (
            f"CertificationMemoryError: RSS={rss_gib:.2f} GiB at '{label}' exceeds "
            f"configured limit max_rss_gb={self.max_rss_gb}. Reduce, in order of effectiveness:\n"
            f"  1. {chunk_line} -- pure cost reduction, no effect on the certified numbers.\n"
            f"  2. {nqp_line} -- pure cost reduction, changes quadrature accuracy of rho_tilde_h, "
            f"not the certificate's validity.\n"
            f"  3. confirm create_graph=False at this call site (see "
            f"docs/history/MEMORY_OOM_DEBUGGING_AND_SAFE_RUN_POLICY.md section 1) -- pure cost reduction if not "
            f"already applied.\n"
            f"  4. {grid_line} -- LAST RESORT: this changes V_h_unr, E_h, and every downstream "
            f"certified number (specification section 7). Never do this silently; report_memory "
            f"anything you change."
        )


class certification_stage:
    """Usable as either a decorator or a context manager. Reports RSS on
    entry and exit and logs the delta -- applied to every top-level
    certification stage (network bounds, raw estimator, cell classification,
    Level-2 remainder, the disjointness check). Never prints inside inner
    loops unless `debug=True` is passed AND the caller explicitly checks
    `.debug` before its own per-iteration prints -- this class itself only
    ever prints twice (entry, exit) regardless of `debug`.

        @certification_stage("raw_estimator")
        def compute_raw_estimator(...): ...

        with certification_stage("certify:raw_estimator") as stage:
            ...
            if stage.debug:
                report_memory("chunk 3/40")
    """

    def __init__(self, name: str, debug: bool = False):
        self.name = name
        self.debug = debug
        self._rss_enter_gib: Optional[float] = None

    def __enter__(self) -> "certification_stage":
        self._rss_enter_gib = report_memory(f"{self.name} enter")
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        rss_exit_gib = report_memory(f"{self.name} exit")
        delta = rss_exit_gib - self._rss_enter_gib
        print(f"[{self.name}] delta = {delta:+.3f} GiB", flush=True)
        return False

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with certification_stage(self.name, debug=self.debug):
                return func(*args, **kwargs)

        return wrapper
