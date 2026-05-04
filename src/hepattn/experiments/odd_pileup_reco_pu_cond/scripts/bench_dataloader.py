"""Benchmark ODDDatasetPileup before vs after the dynamic-augmentation change.

Run protocol:
    1. Stash the current pflow_data.py (`git stash`), run with `--label current`.
    2. Pop the stash. Run with `--label new-val` and `--label new-train`.
    3. Compare init time, RSS, per-item time across the three rows.
"""
from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import psutil

from hepattn.experiments.odd_pileup_reco_pu_cond.pflow_data import ODDDatasetPileup


DATA = "/storage/agrp/barakma/PileupODD/data/ttbar_pu200"
SCALE = (
    "/storage/agrp/barakma/hepattn/src/hepattn/experiments/"
    "odd_pileup_reco_pu_cond/configs/odd_var_transform.yaml"
)


def measure(label: str, num_events: int, n_iter: int, **dataset_kwargs) -> None:
    proc = psutil.Process()
    gc.collect()
    rss0 = proc.memory_info().rss

    files = sorted(Path(DATA).glob("target_particles-*.parquet"))[:1]

    t0 = time.perf_counter()
    ds = ODDDatasetPileup(
        filepath=DATA,
        inputs={"node": []},
        targets={"particle": ["e", "pt", "eta", "sinphi", "cosphi"]},
        scale_dict_path=SCALE,
        num_events=num_events,
        num_objects=400,
        max_nodes=5500,
        window_size=256,
        pu_levels=[200, 120, 50, 0],
        base_pu_level=200,
        pu_sampling_seed=42,
        files_list=files,
        **dataset_kwargs,
    )
    t_init = time.perf_counter() - t0
    rss1 = proc.memory_info().rss

    # Warmup
    for i in range(min(20, len(ds))):
        ds[i % len(ds)]

    t0 = time.perf_counter()
    for i in range(n_iter):
        ds[i % len(ds)]
    t_iter = time.perf_counter() - t0

    print(
        f"[{label:>10s}] "
        f"init={t_init:6.2f}s  "
        f"per-item={1000 * t_iter / n_iter:6.2f}ms  "
        f"dataset_mem={(rss1 - rss0) / 1e6:7.1f}MB  "
        f"len={len(ds)}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True,
                    help="current | new-val | new-train")
    ap.add_argument("--num-events", type=int, default=200)
    ap.add_argument("--n-iter", type=int, default=500)
    args = ap.parse_args()

    if args.label == "current":
        # No is_train kwarg — works against the pre-change codebase.
        measure(args.label, args.num_events, args.n_iter)
    elif args.label == "new-val":
        measure(args.label, args.num_events, args.n_iter, is_train=False)
    elif args.label == "new-train":
        measure(args.label, args.num_events, args.n_iter, is_train=True)
    else:
        raise SystemExit(f"unknown label: {args.label}")
