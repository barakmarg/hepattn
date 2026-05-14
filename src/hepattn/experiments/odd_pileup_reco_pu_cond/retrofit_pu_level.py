"""Retrofit pu_level into an existing prediction-writer H5 — no re-inference needed.

The pu_level for row ``i`` in the H5 is deterministic because:

  * ``ODDDatasetPileup.__len__`` returns ``num_events * len(pu_levels)``
  * ``_event_base_and_pu`` maps ``idx -> pu_levels[idx % len(pu_levels)]``
  * The test dataloader runs with ``shuffle=False``
  * Event filtering happens once at dataset construction (before indexing),
    so no events are dropped mid-iteration.

So row ``i`` → ``pu_levels[i % len(pu_levels)]``.

This script reads ``pu_levels`` from the experiment config and writes a
``pu_level`` dataset alongside the existing ``events`` group. The updated
``load_eval_data_from_h5`` picks it up automatically.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml


DEFAULT_CONFIG = (
    "/storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco_pu_cond/"
    "configs/base.yaml"
)


def load_pu_levels_from_config(config_path: Path) -> list[int]:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    pu_levels = cfg["data"].get("pu_levels")
    if pu_levels is None:
        raise KeyError(f"'data.pu_levels' not found in {config_path}")
    return [int(x) for x in pu_levels]


def retrofit(h5_path: Path, pu_levels: list[int], overwrite: bool) -> None:
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)

    with h5py.File(h5_path, "r+") as f:
        if "events" not in f:
            raise RuntimeError(
                f"{h5_path} has no 'events' dataset — cannot determine row count"
            )
        n_rows = f["events"].shape[0]

        events_has_pu = "pu_level" in f["events"].dtype.names
        toplevel_has_pu = "pu_level" in f

        if events_has_pu and not overwrite:
            print(f"events.pu_level already present in {h5_path.name}; nothing to do.")
            return
        if toplevel_has_pu and not overwrite:
            print(f"top-level pu_level already present in {h5_path.name}; nothing to do.")
            return

        n_levels = len(pu_levels)
        pu_arr = np.asarray(pu_levels, dtype=np.int32)
        pu_per_row = pu_arr[np.arange(n_rows) % n_levels]

        if toplevel_has_pu and overwrite:
            del f["pu_level"]
        f.create_dataset("pu_level", data=pu_per_row)

        unique, counts = np.unique(pu_per_row, return_counts=True)
        print(f"Retrofitted {n_rows} rows in {h5_path.name}:")
        print(f"  pu_levels  = {pu_levels}")
        for pu, c in zip(unique, counts):
            print(f"    PU={int(pu):>4d}  n={int(c):>6d}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("h5", type=str, help="path to prediction-writer H5")
    p.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG,
        help="experiment YAML (used to read data.pu_levels). "
             "Ignored if --pu-levels is given.",
    )
    p.add_argument(
        "--pu-levels", type=int, nargs="+", default=None,
        help="explicit pu_levels (overrides --config). e.g. --pu-levels 200 140 100 60",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="overwrite an existing pu_level dataset if present",
    )
    args = p.parse_args()

    if args.pu_levels is not None:
        pu_levels = [int(x) for x in args.pu_levels]
    else:
        pu_levels = load_pu_levels_from_config(Path(args.config))

    retrofit(Path(args.h5), pu_levels, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
