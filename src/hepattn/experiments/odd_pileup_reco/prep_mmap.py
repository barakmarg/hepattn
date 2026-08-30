"""Offline converter: parquet shards -> per-shard memory-mappable .pt files.

For each shard index this runs the EXACT eager load path on that single shard
(``EagerODDDataset`` over just that shard's parquet files), then saves the RAW
flat tensors (derived columns dropped) to ``shard_{idx:05d}.pt`` plus a tiny
``shard_{idx:05d}.meta.pt`` holding the per-shard count arrays. A final
``--build-index`` pass merges the metas into one ``shards_index.pt`` that
``MmapODDDataset`` reads at startup (so it never opens every shard file).

Because conversion is per-shard and independent, run it as a PBS job array
(one task per shard range) to beat the single-stream Lustre read limit, then a
single ``--index-only`` pass to build the index.

Examples
--------
# Convert shards 1..50 and (re)build the index afterwards:
python prep_mmap.py --config configs/base.yaml \
    --parquet-dir /storage/.../ttbar_pu0_overlay_pu200 \
    --out-dir     /storage/.../ttbar_pu0_overlay_pu200_mmap \
    --shards 1-50 --build-index

# Job-array task (one shard), no index build:
python prep_mmap.py --config configs/base.yaml \
    --parquet-dir ... --out-dir ... --shards $PBS_ARRAY_INDEX

# Build/refresh the index only (after all array tasks finish):
python prep_mmap.py --out-dir ... --index-only
"""

import argparse
import gc
import re
from pathlib import Path

import torch
import yaml

from hepattn.experiments.odd_pileup_reco.pflow_data import DERIVED_KEYS, EagerODDDataset

COUNT_KEYS = ("n_tracks", "n_clusters", "n_particles", "n_deps", "n_raw_deps")


def parse_shards(spec: str) -> list[int]:
    """Parse '1-50', '3', or '1,4,7-9' into a sorted list of ints."""
    out: set[int] = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def data_params_from_config(config_path: str | None, args) -> dict:
    """Resolve the dataset params that affect what is stored.

    `max_nodes`/`num_objects`/`hard_scatter_energy_threshold` MUST match the
    training config, or the per-shard event filtering diverges from training.
    Values come from --config (if given), overridden by explicit CLI flags.
    """
    cfg: dict = {}
    if config_path is not None:
        with open(config_path) as f:
            data_cfg = yaml.safe_load(f)["data"]
        cfg = {
            "inputs": data_cfg.get("inputs"),
            "targets": data_cfg.get("targets"),
            "scale_dict_path": data_cfg["scale_dict_path"],
            "num_objects": data_cfg.get("num_objects", 400),
            "max_nodes": data_cfg.get("max_nodes", 5500),
            "incidence_cutval": data_cfg.get("incidence_cutval", 0.01),
            "hard_scatter_energy_threshold": data_cfg.get("hard_scatter_energy_threshold", 0.1),
            "window_size": data_cfg.get("window_size", 256),
        }
    # Defaults / overrides
    cfg.setdefault("inputs", {})
    cfg.setdefault("targets", {"particle": ["e", "pt", "eta", "sinphi", "cosphi"]})
    if args.scale_dict_path is not None:
        cfg["scale_dict_path"] = args.scale_dict_path
    if args.num_objects is not None:
        cfg["num_objects"] = args.num_objects
    if args.max_nodes is not None:
        cfg["max_nodes"] = args.max_nodes
    if args.hs_energy_threshold is not None:
        cfg["hard_scatter_energy_threshold"] = args.hs_energy_threshold
    cfg.setdefault("incidence_cutval", 0.01)
    cfg.setdefault("window_size", 256)
    assert "scale_dict_path" in cfg, "scale_dict_path required (via --config or --scale-dict-path)"
    return cfg


def convert_shard(parquet_dir: Path, idx: int, out_dir: Path, base_kwargs: dict, overwrite: bool = False) -> bool:
    """Convert one shard. Returns True if a shard file was written."""
    shard_files = sorted(parquet_dir.glob(f"target_particles-{idx:05d}.parquet"))
    if not shard_files:
        print(f"[skip] shard {idx:05d}: no target_particles parquet")
        return False

    out_path = out_dir / f"shard_{idx:05d}.pt"
    meta_path = out_dir / f"shard_{idx:05d}.meta.pt"
    if not overwrite and out_path.exists() and meta_path.exists():
        print(f"[skip] shard {idx:05d}: already converted (use --overwrite to regenerate)")
        return True

    try:
        ds = EagerODDDataset(
            filepath=str(parquet_dir),
            files_list=shard_files,
            num_events=-1,
            compute_deltaR_stats=False,
            **base_kwargs,
        )
    except Exception as e:  # noqa: BLE001 - per-shard robustness for job arrays
        print(f"[skip] shard {idx:05d}: load failed: {e}")
        return False

    n_events = int(getattr(ds, "num_events", 0))
    if n_events == 0:
        print(f"[skip] shard {idx:05d}: 0 valid events after filtering")
        del ds
        gc.collect()
        return False

    # RAW flat tensors only (derived columns recomputed at read time).
    stored_keys = [k for k in ds.full_data_array if k not in DERIVED_KEYS]
    payload = {k: ds.full_data_array[k].contiguous() for k in stored_keys}
    torch.save(payload, str(out_path))

    meta = {
        "path": out_path.name,  # basename; resolved relative to the index at load
        "n_events": n_events,
        "stored_keys": stored_keys,
        "event_id": ds.event_number,
        "hard_scatter_vz": ds.hard_scatter_vz,
    }
    for k in COUNT_KEYS:
        meta[k] = getattr(ds, k)
    torch.save(meta, str(meta_path))

    print(f"[ok]   shard {idx:05d}: {n_events} events, {len(stored_keys)} keys -> {out_path.name}")
    del ds, payload
    gc.collect()
    return True


def build_index(out_dir: Path) -> None:
    """Merge all shard_*.meta.pt into one shards_index.pt."""
    meta_files = sorted(out_dir.glob("shard_*.meta.pt"), key=lambda p: int(re.search(r"(\d+)", p.name).group(1)))
    if not meta_files:
        raise FileNotFoundError(f"No shard_*.meta.pt found in {out_dir}")
    shards = []
    stored_keys = None
    total_events = 0
    for mf in meta_files:
        meta = torch.load(str(mf), weights_only=False)
        if stored_keys is None:
            stored_keys = meta["stored_keys"]
        shards.append(meta)
        total_events += int(meta["n_events"])
    index = {"stored_keys": stored_keys, "shards": shards}
    index_path = out_dir / "shards_index.pt"
    torch.save(index, str(index_path))
    print(f"[index] {len(shards)} shards, {total_events} events -> {index_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parquet-dir", type=str, default=None, help="Directory with target_particles-*.parquet etc.")
    p.add_argument("--out-dir", type=str, required=True, help="Output directory for shard_*.pt + shards_index.pt")
    p.add_argument("--shards", type=str, default=None, help="Shard indices, e.g. '1-1000' or '3,5,7-9'. Default: all found.")
    p.add_argument("--config", type=str, default=None, help="Training config YAML to pull data params from.")
    p.add_argument("--scale-dict-path", type=str, default=None)
    p.add_argument("--num-objects", type=int, default=None)
    p.add_argument("--max-nodes", type=int, default=None)
    p.add_argument("--hs-energy-threshold", type=float, default=None)
    p.add_argument("--overwrite", action="store_true", help="Regenerate shards even if their .pt already exists.")
    p.add_argument("--build-index", action="store_true", help="Build shards_index.pt after converting.")
    p.add_argument("--index-only", action="store_true", help="Only build the index from existing metas; no conversion.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.index_only:
        build_index(out_dir)
        return

    assert args.parquet_dir is not None, "--parquet-dir required unless --index-only"
    parquet_dir = Path(args.parquet_dir)
    base_kwargs = data_params_from_config(args.config, args)

    if args.shards is not None:
        indices = parse_shards(args.shards)
    else:
        indices = sorted(
            int(re.search(r"target_particles-(\d+)\.parquet", f.name).group(1))
            for f in parquet_dir.glob("target_particles-*.parquet")
        )

    print(f"Converting {len(indices)} shards from {parquet_dir} -> {out_dir}")
    print(f"Data params: max_nodes={base_kwargs['max_nodes']}, num_objects={base_kwargs['num_objects']}, "
          f"hs_energy_threshold={base_kwargs['hard_scatter_energy_threshold']}")

    n_written = 0
    for idx in indices:
        n_written += int(convert_shard(parquet_dir, idx, out_dir, base_kwargs, overwrite=args.overwrite))
    print(f"Done: {n_written}/{len(indices)} shards written.")

    if args.build_index:
        build_index(out_dir)


if __name__ == "__main__":
    main()
