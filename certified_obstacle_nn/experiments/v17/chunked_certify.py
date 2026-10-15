"""v17 Stages 6-7: chunked, resumable evaluation of one frozen partition with one enclosure candidate, for jobs that
would exceed the 100-minute per-job design limit (prompt Section 1c).

    # 1. split: write the job record (partition hashes, enclosure, chunking) -- seconds
    python3 -m certified_obstacle_nn.experiments.v17.chunked_certify init --source <stored run> \
        --enclosure TM1+AFF64+PROD --chunks 4 [--partition <v17_partitions dir>]
    # 2. one job per chunk (each <= 100 min); a finished chunk is never recomputed
    python3 -m certified_obstacle_nn.experiments.v17.chunked_certify chunk --job <dir> --index i --workers 20
    # 3. traces, arb reduction of all per-cell densities, bounds, diagnostics -> results/v17_split/<ts>/
    python3 -m certified_obstacle_nn.experiments.v17.chunked_certify finalize --job <dir> --workers 20
    # (v17b) a one-chunk job from the selection state of an adaptive partition (intermediate budgets)
    python3 -m certified_obstacle_nn.experiments.v17.chunked_certify import-selection --source <run> --partition <dir>

Each chunk file stores the per-cell densities (float64 outward upper bounds) of a contiguous slice of the frozen
partition, the slice bounds, the partition hash, the enclosure, the checkpoint hash and the wall time; finalize checks
that every slice is present exactly once with matching hashes before it reduces (in arb, as for an unchunked run:
upper_sum.reduce_upper_sums). Cell evaluation is independent per cell, so the chunked result is identical to the
unchunked one (tested).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from certified_obstacle_nn.certification.enclose import boxes as Bx
from certified_obstacle_nn.experiments import results_io
from certified_obstacle_nn.experiments.v14.common import REPO_ROOT, sha256_file
from certified_obstacle_nn.experiments.v17 import split_certify as SC
from certified_obstacle_nn.experiments.v17.common import load_stored_run


def _rel(p: Path) -> str:
    try:
        return str(Path(p).relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def _load(job: Path, workers: int = 1):
    meta = json.loads((job / "job.json").read_text())
    r = load_stored_run(meta["source"], workers=workers)
    if meta.get("partition"):
        z = np.load(REPO_ROOT / meta["partition"] / "partition.npz")
        r = dataclasses.replace(r, part=Bx.BoxPartition(z["lo"], z["hi"]), tp0=Bx.BoxPartition(z["lo0"], z["hi0"]),
                                tpT=Bx.BoxPartition(z["loT"], z["hiT"]))
    for nm, p in (("partition_hash", r.part), ("trace0_hash", r.tp0), ("traceT_hash", r.tpT)):
        if p.hash() != meta[nm]:
            raise RuntimeError(f"{nm} mismatch for job {job}")
    if r.checkpoint_sha256 != meta["checkpoint_sha256"]:
        raise RuntimeError("checkpoint hash mismatch")
    return meta, r


def init(source, enclosure, chunks, partition=None):
    r = load_stored_run(source)
    if partition:
        z = np.load(REPO_ROOT / partition / "partition.npz")
        r = dataclasses.replace(r, part=Bx.BoxPartition(z["lo"], z["hi"]), tp0=Bx.BoxPartition(z["lo0"], z["hi0"]),
                                tpT=Bx.BoxPartition(z["loT"], z["hiT"]))
    d = results_io.make_run_dir("v17_chunks")
    edges = np.linspace(0, r.part.n, chunks + 1).astype(int).tolist()
    meta = {"source": source, "partition": partition, "enclosure": enclosure, "n_cells": int(r.part.n),
            "edges": edges, "partition_hash": r.part.hash(), "trace0_hash": r.tp0.hash(), "traceT_hash": r.tpT.hash(),
            "checkpoint_sha256": r.checkpoint_sha256}
    (d / "job.json").write_text(json.dumps(meta, indent=1))
    print(_rel(d))
    return d


def import_selection(source, partition):
    """A one-chunk job whose densities are the selection state (selection.npz) of a frozen adaptive partition: the
    TM1 numpy densities computed for marking, cell by cell the same deterministic evaluation as a chunk job (bitwise
    equal on the protocol-v1 seed-0 partition, PROGRESS v17b Task 3). Used for the intermediate budgets of protocol v2;
    the job records the provenance ("densities_from") and finalize reduces it like any chunk job."""
    d = init(source, "TM1", 1, partition)
    meta = json.loads((d / "job.json").read_text())
    pj = json.loads((REPO_ROOT / partition / "partition.json").read_text())
    s = np.load(REPO_ROOT / partition / "selection.npz")
    vol = {k: s[k] for k in s.files}
    if vol["r"].size != meta["n_cells"]:
        raise RuntimeError("selection state does not match the partition")
    meta["densities_from"] = {"selection_state": f"{partition}/selection.npz",
                              "selection_sha256": sha256_file(REPO_ROOT / partition / "selection.npz"),
                              "selection_s_cumulative": pj["refinement"]["selection_s"]}
    (d / "job.json").write_text(json.dumps(meta, indent=1))
    tmp = d / "chunk_000.tmp.npz"
    np.savez_compressed(tmp, start=0, end=meta["n_cells"], seconds=0.0, **vol)
    tmp.rename(d / "chunk_000.npz")
    print(_rel(d))
    return d


def chunk(job: Path, index: int, workers: int):
    f = job / f"chunk_{index:03d}.npz"
    if f.exists():
        print("chunk exists, skipped:", f)
        return
    meta, r = _load(job, workers)
    if not hasattr(r.part, "lo"):
        raise NotImplementedError("chunked evaluation is implemented for the 2-D box partitions")
    s, e = meta["edges"][index], meta["edges"][index + 1]
    enc = SC.encloser_for(r, meta["enclosure"])
    tic = time.time()
    vol = Bx.evaluate_volume(enc, r.part.subset(np.arange(s, e)), "numpy", workers=workers)
    dt = time.time() - tic
    tmp = job / f"chunk_{index:03d}.tmp.npz"
    np.savez_compressed(tmp, start=s, end=e, seconds=dt, **vol)
    tmp.rename(f)
    print(time.strftime("%H:%M:%S"), f"chunk {index}: cells {s}-{e} in {dt:.0f}s ({dt / max(e - s, 1) * 1e3:.2f} ms/cell)")


def finalize(job: Path, workers: int, q: int = 2):
    meta, r = _load(job, workers)
    n = len(meta["edges"]) - 1
    parts, total = [], 0.0
    for i in range(n):
        z = np.load(job / f"chunk_{i:03d}.npz")
        if (int(z["start"]), int(z["end"])) != (meta["edges"][i], meta["edges"][i + 1]):
            raise RuntimeError(f"chunk {i} slice mismatch")
        parts.append(z)
        total += float(z["seconds"])
    keys = [k for k in parts[0].files if k not in ("start", "end", "seconds")]
    vol = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    if vol["r"].size != r.part.n:
        raise RuntimeError("chunks do not cover the partition")
    info = {"partition_hash": meta["partition_hash"], "trace0_hash": meta["trace0_hash"],
            "traceT_hash": meta["traceT_hash"], "n_cells": meta["n_cells"],
            "source": meta["partition"] or "stored source partition", "chunked_job": _rel(job),
            "chunk_files_sha256": {f"chunk_{i:03d}.npz": sha256_file(job / f"chunk_{i:03d}.npz") for i in range(n)},
            "volume_eval_s_sum_of_chunks": total}
    if meta.get("densities_from"):
        info["densities_from"] = meta["densities_from"]
    config, rows, cells, r, _ = SC.run(r.run_dir, ["numpy"], workers, q, enclosure=meta["enclosure"], r=r,
                                       partition_info=info, volumes={"numpy": (vol, total)},
                                       log=lambda m: print(time.strftime("%H:%M:%S"), m, flush=True))
    d = SC.write_run(config, rows, cells, r)
    row = rows[0]
    print(f"{meta['enclosure']}: {row['status']} F_dir/N={row['flux_bound_direct_split_rel']:.4f} "
          f"G_dir/N={row['grad_bound_direct_split_rel']:.4f} E_dir={row['E_dir_split']:.4f} -> {_rel(d)}")
    return d


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("init")
    a1.add_argument("--source", required=True)
    a1.add_argument("--enclosure", required=True)
    a1.add_argument("--chunks", type=int, required=True)
    a1.add_argument("--partition", default=None)
    a2 = sub.add_parser("chunk")
    a2.add_argument("--job", required=True)
    a2.add_argument("--index", type=int, nargs="+", required=True)
    a2.add_argument("--workers", type=int, default=20)
    a4 = sub.add_parser("import-selection")
    a4.add_argument("--source", required=True)
    a4.add_argument("--partition", required=True)
    a3 = sub.add_parser("finalize")
    a3.add_argument("--job", required=True)
    a3.add_argument("--workers", type=int, default=20)
    a3.add_argument("--q", type=int, default=2)
    a = ap.parse_args()
    if a.cmd == "init":
        init(a.source, a.enclosure, a.chunks, a.partition)
    elif a.cmd == "import-selection":
        import_selection(a.source, a.partition)
    elif a.cmd == "chunk":
        for i in a.index:
            chunk(REPO_ROOT / a.job, i, a.workers)
    else:
        finalize(REPO_ROOT / a.job, a.workers, a.q)


if __name__ == "__main__":
    main()
