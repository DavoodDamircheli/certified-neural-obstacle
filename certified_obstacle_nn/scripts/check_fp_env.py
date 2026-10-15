"""v18 Task 1b: floating-point environment checks for the validated NumPy backend (enclose/intervals_np.py).

    python3 -m certified_obstacle_nn.scripts.check_fp_env [--out path.txt]

The outward-rounding argument of intervals_np.py needs: IEEE 754 binary64 + - * / correctly rounded to nearest
(ties to even), no flush-to-zero (FTZ) of subnormal results, no denormals-are-zero (DAZ) on subnormal inputs, and no
fused multiply-add contraction of the separate ufunc calls ``a*b`` then ``+c``. Every assertion is run

  * on Python floats and on NumPy arrays of several lengths (scalar loop, short and long SIMD loops),
  * in the main process before and after importing torch / python-flint / the certification engine, and
  * in a worker forked by ProcessPoolExecutor exactly as enclose/boxes.py and enclose/upper_sum.py fork them.

Reference values are exact: ``fractions.Fraction`` products/sums, rounded once to binary64 by ``float(Fraction)``
(correctly rounded in CPython). The script exits with status 1 if any assertion fails and records NumPy, BLAS, CPU
and OS details. It changes no floating-point mode itself.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import platform
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction

import numpy as np

SIZES = (1, 7, 64, 4099)  # scalar path, short tail loop, one SIMD block, long SIMD loop with remainder


def _arr(x, n):
    return np.full(n, x, dtype=np.float64)


def _rn(fr: Fraction) -> float:
    return float(fr)  # correctly rounded (round half to even) in CPython


def check_rounding() -> list[tuple[str, bool, str]]:
    """Round-to-nearest-even sentinels; each distinguishes RN from RZ, RD and RU."""
    out = []
    one = 1.0
    h = 2.0 ** -53  # half an ulp of 1
    hp = 2.0 ** -53 + 2.0 ** -80  # just above half an ulp
    up1 = np.nextafter(1.0, 2.0)
    cases = [
        ("1 + 2^-53 -> 1 (tie to even)", one, h, 1.0),
        ("1 + (2^-53 + 2^-80) -> nextafter(1, +inf)", one, hp, float(up1)),
        ("-1 - (2^-53 + 2^-80) -> nextafter(-1, -inf)", -one, -hp, float(-up1)),
        ("-1 - 2^-53 -> -1 (tie to even)", -one, -h, -1.0),
        ("1 + 3*2^-53 -> 1 + 2^-51 (tie to even, upward)", one, 3 * h, 1.0 + 2.0 ** -51),
    ]
    for name, a, b, want in cases:
        exact = _rn(Fraction(a) + Fraction(b))
        ok = exact == want and (a + b) == want
        for n in SIZES:
            r = _arr(a, n) + _arr(b, n)
            ok &= bool(np.all(r == want))
        out.append((f"rounding: {name}", ok, f"want {want.hex()}"))
    # multiplication and division sentinels against the exact rational
    for name, a, b, op in (("mul (1+2^-30)(1-2^-30)", 1 + 2.0 ** -30, 1 - 2.0 ** -30, "mul"),
                           ("mul (1+2^-27)(1+2^-27)", 1 + 2.0 ** -27, 1 + 2.0 ** -27, "mul"),
                           ("div 1/3", 1.0, 3.0, "div"), ("div -2/3", -2.0, 3.0, "div")):
        ex = Fraction(a) * Fraction(b) if op == "mul" else Fraction(a) / Fraction(b)
        want = _rn(ex)
        ok = True
        for n in SIZES:
            r = _arr(a, n) * _arr(b, n) if op == "mul" else _arr(a, n) / _arr(b, n)
            ok &= bool(np.all(r == want))
        ok &= ((a * b) if op == "mul" else (a / b)) == want
        out.append((f"rounding: {name}", ok, f"want {want.hex()}"))
    return out


def check_subnormals() -> list[tuple[str, bool, str]]:
    out = []
    s = float(np.nextafter(0.0, 1.0))  # 2^-1074
    tiny = 2.0 ** -1022  # smallest normal
    for n in SIZES:
        S = _arr(s, n)
        ok = bool(np.all(S + S == 2 * s)) and 2 * s != 0.0
        out.append((f"no FTZ: subnormal + subnormal survives (n={n})", ok, (S + S)[0].hex()))
        ok = bool(np.all(S * 1.0 == s)) and bool(np.all(S * 3.0 == 3 * s)) and 3 * s != 0.0
        out.append((f"no DAZ: subnormal * 3 survives (n={n})", ok, (S * 3.0)[0].hex()))
        T = _arr(tiny, n)
        ok = bool(np.all(T * 0.5 == 2.0 ** -1023)) and bool(np.all(T * 0.5 != 0.0))
        out.append((f"no FTZ: normal * 0.5 -> subnormal (n={n})", ok, (T * 0.5)[0].hex()))
        ok = bool(np.all(S + 0.0 == s)) and bool(np.all(np.nextafter(_arr(0.0, n), 1.0) == s))
        out.append((f"no DAZ: subnormal + 0 and nextafter(0, 1) (n={n})", ok, np.nextafter(0.0, 1.0).hex()))
    return out


def _adversarial_triples():
    """(a, b, c) with RN(RN(a b) + c) != RN(a b + c): the fused result would differ."""
    rng = np.random.default_rng(20261013)
    tri = [(1 + 2.0 ** -30, 1 - 2.0 ** -30, -1.0),  # a b = 1 - 2^-60: unfused 0, fused -2^-60
           (1 + 2.0 ** -27, 1 + 2.0 ** -27, -(1 + 2.0 ** -26)),  # unfused 0, fused 2^-54
           (0.1, 10.0, -1.0), (1.0 / 3.0, 3.0, -1.0), (2.0 ** 500 * (1 + 2.0 ** -40), 2.0 ** 500, -(2.0 ** 1000))]
    while len(tri) < 64:
        a, b = rng.uniform(0.5, 2.0, 2)
        c = -float(Fraction(a) * Fraction(b))  # c = -RN(a b): unfused 0, fused = -(rounding error) != 0
        if Fraction(a) * Fraction(b) != Fraction(-c):
            tri.append((float(a), float(b), c))
    return tri


def check_fma() -> list[tuple[str, bool, str]]:
    out = []
    tri = _adversarial_triples()
    A = np.array([t[0] for t in tri])
    B = np.array([t[1] for t in tri])
    C = np.array([t[2] for t in tri])
    want_unfused = np.array([_rn(Fraction(_rn(Fraction(a) * Fraction(b))) + Fraction(c)) for a, b, c in tri])
    want_fused = np.array([_rn(Fraction(a) * Fraction(b) + Fraction(c)) for a, b, c in tri])
    n_distinct = int((want_unfused != want_fused).sum())
    for reps in (1, 64):  # short and long (SIMD) loops
        AA, BB, CC = np.tile(A, reps), np.tile(B, reps), np.tile(C, reps)
        r = AA * BB + CC
        ok = bool(np.all(r == np.tile(want_unfused, reps)))
        out.append((f"no FMA contraction in a*b+c (n={AA.size}; {n_distinct} triples where fused differs)", ok,
                    f"mismatches {int((r != np.tile(want_unfused, reps)).sum())}"))
    r = np.array([a * b + c for a, b, c in tri])
    out.append(("no FMA contraction in Python float a*b+c", bool(np.all(r == want_unfused)), ""))
    return out


def check_backend() -> list[tuple[str, bool, str]]:
    """The validated backend's own + and * enclose the exact rational results on the adversarial triples."""
    from certified_obstacle_nn.certification.enclose import intervals_np as N

    tri = _adversarial_triples()
    A = N.thin(np.array([t[0] for t in tri]))
    B = N.thin(np.array([t[1] for t in tri]))
    C = N.thin(np.array([t[2] for t in tri]))
    P = A * B
    Q = P + C
    ok = True
    for i, (a, b, c) in enumerate(tri):
        ex_p = Fraction(a) * Fraction(b)
        ex_q = ex_p + Fraction(c)
        ok &= Fraction(float(P.lo[i])) <= ex_p <= Fraction(float(P.hi[i]))
        ok &= Fraction(float(Q.lo[i])) <= ex_q <= Fraction(float(Q.hi[i]))
    return [("intervals_np: outward a*b and a*b+c contain the exact rationals", bool(ok), f"{len(tri)} triples")]


def run_all() -> list[tuple[str, bool, str]]:
    return check_rounding() + check_subnormals() + check_fma()


def _worker(_):
    res = run_all() + check_backend()
    return [(f"[forked worker] {n}", ok, d) for n, ok, d in res]


def environment() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(),
            "numpy": np.__version__}
    try:
        import flint
        import torch

        info.update({"python-flint": flint.__version__, "torch": torch.__version__,
                     "torch_threads": torch.get_num_threads()})
    except ImportError as e:  # recorded, not ignored: the line is printed
        info["import_error"] = str(e)
    cpu = {}
    with open("/proc/cpuinfo") as fh:
        for line in fh:
            k, _, v = line.partition(":")
            k = k.strip()
            if k in ("model name", "flags", "microcode") and k not in cpu:
                cpu[k] = v.strip()
    flags = cpu.get("flags", "").split()
    info["cpu_model"] = cpu.get("model name")
    info["cpu_microcode"] = cpu.get("microcode")
    info["cpu_fp_flags"] = [f for f in flags if f in ("fma", "sse2", "sse4_1", "sse4_2", "avx", "avx2", "avx512f",
                                                       "f16c", "avx_vnni", "daz")]
    info["cpu_count"] = len([1 for line in open("/proc/cpuinfo") if line.startswith("processor")])
    info["os_release"] = subprocess.run(["sh", "-c", ". /etc/os-release && echo $PRETTY_NAME"], capture_output=True,
                                        text=True).stdout.strip()
    info["kernel"] = platform.release()
    try:
        from numpy.__config__ import CONFIG  # numpy >= 1.25

        info["numpy_blas"] = CONFIG["Build Dependencies"]["blas"]
        info["numpy_simd"] = CONFIG.get("SIMD Extensions")
    except (ImportError, KeyError) as e:
        info["numpy_blas"] = f"unavailable: {e}"
    try:
        from threadpoolctl import threadpool_info

        info["threadpools"] = threadpool_info()
    except ImportError:
        info["threadpools"] = "threadpoolctl not installed"
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    lines = ["# check_fp_env (v18 Task 1b)"]
    results = [(f"[main, before imports] {n}", ok, d) for n, ok, d in run_all()]
    import flint  # noqa: F401
    import torch

    torch.set_default_dtype(torch.float64)
    from certified_obstacle_nn.certification.enclose import engine2d, intervals_np  # noqa: F401

    intervals_np.tables()
    results += [(f"[main, after torch/flint/engine imports] {n}", ok, d) for n, ok, d in run_all() + check_backend()]
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("fork")) as ex:
        for res in ex.map(_worker, range(2)):
            results += res
    n_fail = sum(not ok for _, ok, _ in results)
    lines.append(f"assertions: {len(results)}, failed: {n_fail}")
    for n, ok, d in results:
        lines.append(f"{'PASS' if ok else 'FAIL'}  {n}  {d}")
    lines.append("## environment")
    lines.append(json.dumps(environment(), indent=1, default=str))
    lines.append("RESULT: " + ("PASS" if n_fail == 0 else "FAIL"))
    txt = "\n".join(lines) + "\n"
    print(txt, end="")
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(txt)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
