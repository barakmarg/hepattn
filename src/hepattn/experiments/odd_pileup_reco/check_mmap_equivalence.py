"""Validate that the mmap backend produces IDENTICAL __getitem__ output to the
eager backend, on a few shards. Run this on an interactive node before any
large prep / training run.

It (1) converts the chosen shards with prep_mmap, (2) builds an EagerODDDataset
over the same shard parquet files and a MmapODDDataset over the converted .pt,
and (3) compares every tensor of every event:
  - floats : torch.allclose(atol=1e-6, equal_nan=True)
  - ints/bool : exact torch.equal

Example
-------
python check_mmap_equivalence.py --config configs/base.yaml \
    --parquet-dir /storage/.../ttbar_pu0_overlay_pu200 \
    --out-dir /tmp/mmap_check --shards 6,7 --max-events 200
"""

import argparse
from pathlib import Path

import torch

from hepattn.experiments.odd_pileup_reco.pflow_data import EagerODDDataset, MmapODDDataset
from hepattn.experiments.odd_pileup_reco.prep_mmap import (
    build_index,
    convert_shard,
    data_params_from_config,
    parse_shards,
)


def compare_tensors(name: str, a: torch.Tensor, b: torch.Tensor, atol: float) -> str | None:
    if a.shape != b.shape:
        return f"{name}: shape {tuple(a.shape)} != {tuple(b.shape)}"
    if a.dtype != b.dtype:
        # dtype mismatch is worth flagging but compare values too
        msg_dtype = f"{name}: dtype {a.dtype} != {b.dtype}"
    else:
        msg_dtype = None
    if a.is_floating_point():
        if not torch.allclose(a, b, atol=atol, equal_nan=True):
            diff = (a - b).abs()
            diff = diff[~torch.isnan(diff)]
            mx = float(diff.max()) if diff.numel() else float("nan")
            return f"{name}: float mismatch (max|Δ|={mx:.3e})" + (f"; {msg_dtype}" if msg_dtype else "")
    else:
        if not torch.equal(a, b):
            n_bad = int((a != b).sum())
            return f"{name}: int/bool mismatch ({n_bad} elems)" + (f"; {msg_dtype}" if msg_dtype else "")
    return msg_dtype


def compare_event(idx: int, eager_ds, mmap_ds, atol: float) -> list[str]:
    ein, elab = eager_ds[idx]
    min_, mlab = mmap_ds[idx]
    errs = []
    for tag, ed, md in (("inputs", ein, min_), ("labels", elab, mlab)):
        keys = set(ed) | set(md)
        for k in sorted(keys):
            if k not in ed or k not in md:
                errs.append(f"[evt {idx}] {tag}.{k}: present in only one backend")
                continue
            msg = compare_tensors(f"[evt {idx}] {tag}.{k}", ed[k], md[k], atol)
            if msg:
                errs.append(msg)
    return errs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parquet-dir", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True, help="Temp dir for converted shard .pt files")
    p.add_argument("--shards", type=str, default="6", help="e.g. '6' or '6,7'")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--scale-dict-path", type=str, default=None)
    p.add_argument("--num-objects", type=int, default=None)
    p.add_argument("--max-nodes", type=int, default=None)
    p.add_argument("--hs-energy-threshold", type=float, default=None)
    p.add_argument("--max-events", type=int, default=-1, help="Cap number of events compared (-1 = all)")
    p.add_argument("--atol", type=float, default=1e-6)
    args = p.parse_args()

    parquet_dir = Path(args.parquet_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    indices = parse_shards(args.shards)
    base_kwargs = data_params_from_config(args.config, args)

    # 1. Convert the chosen shards + build the index.
    print("=== converting shards ===")
    for idx in indices:
        convert_shard(parquet_dir, idx, out_dir, base_kwargs)
    build_index(out_dir)

    # 2. Build both datasets over the same shards (same event order).
    print("\n=== building datasets ===")
    shard_files = sorted(f for idx in indices for f in parquet_dir.glob(f"target_particles-{idx:05d}.parquet"))
    eager_ds = EagerODDDataset(
        filepath=str(parquet_dir), files_list=shard_files, num_events=-1,
        compute_deltaR_stats=False, **base_kwargs,
    )
    mmap_ds = MmapODDDataset(
        index_path=str(out_dir / "shards_index.pt"), files_list=None, num_events=-1, **base_kwargs,
    )

    n = min(len(eager_ds), len(mmap_ds))
    if len(eager_ds) != len(mmap_ds):
        print(f"!! length mismatch: eager={len(eager_ds)} mmap={len(mmap_ds)} (comparing first {n})")
    if args.max_events != -1:
        n = min(n, args.max_events)

    # 3. Compare.
    print(f"\n=== comparing {n} events (atol={args.atol}) ===")
    all_errs = []
    for idx in range(n):
        errs = compare_event(idx, eager_ds, mmap_ds, args.atol)
        all_errs.extend(errs)
        if (idx + 1) % 50 == 0:
            print(f"  ...{idx + 1}/{n} compared, {len(all_errs)} mismatches so far")

    print("\n=== result ===")
    if not all_errs:
        print(f"PASS: {n} events identical across eager and mmap backends.")
    else:
        print(f"FAIL: {len(all_errs)} mismatches. First 30:")
        for e in all_errs[:30]:
            print("  " + e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
