"""Unified evaluation data loading from prediction writer H5 files.

Provides two paths to evaluation data, both producing identical dict formats
consumed by the plotting functions in eval_plots.py:

1. ``load_eval_data_from_h5(h5_path)`` — load an existing prediction writer output
2. ``run_forward_pass(ckpt_path, config_path, ...)`` — run inference, producing an H5,
   then load it

Example (existing H5)::

    from hepattn.experiments.odd_pileup_reco.eval_data import load_eval_data_from_h5
    track, cluster, reco = load_eval_data_from_h5("epoch=042__test.h5")

Example (from scratch)::

    from hepattn.experiments.odd_pileup_reco.eval_data import run_forward_pass, load_eval_data_from_h5
    h5_path = run_forward_pass(ckpt_path="...", config_path="...")
    track, cluster, reco = load_eval_data_from_h5(h5_path)
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

# ── Thresholds (must match eval_plots.py / model config) ──────────────────
RECO_NUM_CLASSES = 6
CALO_PRED_THRESHOLD = 0.2
CALO_HS_FRAC_THRESHOLD = 0.05
CALO_HS_ENERGY_THRESHOLD = 0.15


EventSelector = slice | np.ndarray


def _drop_unit_axes_except_first(arr: np.ndarray) -> np.ndarray:
    """Drop singleton axes except the leading event axis.

    Handles layout variants like (B, N), (B, N, 1), and (B, 1, N)
    without collapsing the batch axis when B == 1.
    """
    axes = tuple(i for i, size in enumerate(arr.shape) if i != 0 and size == 1)
    if axes:
        arr = np.squeeze(arr, axis=axes)
    return arr


def _read_struct_field(ds: h5py.Dataset, name: str, dtype, event_sel: EventSelector) -> np.ndarray:
    """Read a structured-dataset field with robust singleton-axis handling."""
    arr = ds[name][event_sel].astype(dtype)
    return _drop_unit_axes_except_first(arr)


def _infer_n_events(f: h5py.File) -> int:
    """Infer number of events from known top-level datasets."""
    candidate_keys = [
        "events",
        "object_class",
        "regression",
        "node_metadata",
        "pred_incidence",
        "truth_incidence",
    ]
    for key in candidate_keys:
        if key in f and isinstance(f[key], h5py.Dataset):
            return int(f[key].shape[0])
    raise ValueError("Could not infer number of events from H5 file")


def _build_event_selector(
    n_events: int,
    event_start: int | None,
    event_stop: int | None,
    event_indices: list[int] | np.ndarray | None,
) -> EventSelector:
    """Build a validated event selector for H5 slicing."""
    if event_indices is not None and (event_start is not None or event_stop is not None):
        raise ValueError("Use either event_indices or event_start/event_stop, not both")

    if event_indices is not None:
        idx = np.asarray(event_indices, dtype=np.int64).reshape(-1)
        if idx.size == 0:
            return idx

        idx = np.where(idx < 0, idx + n_events, idx)
        if np.any((idx < 0) | (idx >= n_events)):
            raise IndexError(f"event_indices out of range for n_events={n_events}")

        # h5py fancy indexing requires strictly increasing indices.
        if np.any(np.diff(idx) <= 0):
            raise ValueError("event_indices must be strictly increasing for H5 fancy indexing")
        return idx

    start = 0 if event_start is None else int(event_start)
    stop = n_events if event_stop is None else int(event_stop)

    if start < 0:
        start += n_events
    if stop < 0:
        stop += n_events

    start = max(0, min(start, n_events))
    stop = max(0, min(stop, n_events))
    if stop < start:
        stop = start

    return slice(start, stop)


# ---------------------------------------------------------------------------
# H5 loading
# ---------------------------------------------------------------------------

def _h5_n_events(path: str | Path) -> int:
    """Number of events in a single prediction-writer H5 file."""
    with h5py.File(path, "r") as f:
        return _infer_n_events(f)


def map_event_range_to_files(
    paths: list[str | Path],
    event_start: int | None = None,
    event_stop: int | None = None,
) -> list[tuple[Path, int, int]]:
    """Map a global ``[event_start, event_stop)`` range onto an ordered list of
    H5 files (each holding a contiguous block of events).

    Returns a list of ``(path, local_start, local_stop)`` for the files that
    overlap the requested global range; files with no overlap are omitted.
    Negative bounds are interpreted relative to the total event count, and
    ``None`` means "from the beginning"/"to the end".
    """
    paths = [Path(p) for p in paths]
    counts = [_h5_n_events(p) for p in paths]
    total = sum(counts)

    start = 0 if event_start is None else int(event_start)
    stop = total if event_stop is None else int(event_stop)
    if start < 0:
        start += total
    if stop < 0:
        stop += total
    start = max(0, min(start, total))
    stop = max(start, min(stop, total))

    out: list[tuple[Path, int, int]] = []
    base = 0
    for p, n in zip(paths, counts):
        lo = max(start, base) - base
        hi = min(stop, base + n) - base
        if hi > lo:
            out.append((p, lo, hi))
        base += n
    return out


def concat_eval_dicts(dicts: list[dict]) -> dict:
    """Concatenate same-schema eval dicts along the leading (event/flat) axis.

    Non-array values keep the first dict's value. Used to merge per-file
    results when loading predictions split across multiple H5 files.
    """
    dicts = [d for d in dicts if d]
    if not dicts:
        return {}
    if len(dicts) == 1:
        return dicts[0]
    out: dict = {}
    for k in dicts[0]:
        vals = [d[k] for d in dicts if k in d]
        if all(isinstance(v, np.ndarray) for v in vals):
            try:
                out[k] = np.concatenate(vals, axis=0)
            except ValueError:
                out[k] = vals[0]
        else:
            out[k] = vals[0]
    return out


def load_eval_data_from_h5(
    h5_path: str | Path | list[str | Path] | tuple,
    calo_pred_threshold: float = CALO_PRED_THRESHOLD,
    calo_hs_frac_threshold: float = CALO_HS_FRAC_THRESHOLD,
    calo_hs_energy_threshold: float = CALO_HS_ENERGY_THRESHOLD,
    event_start: int | None = None,
    event_stop: int | None = None,
    event_indices: list[int] | np.ndarray | None = None,
    load_reco: bool = True,
) -> tuple[dict, dict, dict]:
    """Load prediction writer H5 and return (track_data, cluster_data, reco_data).

    The returned dicts have the same keys/shapes that ``make_plots()`` in
    ``eval_plots.py`` expects.

    Parameters
    ----------
    h5_path : path to the prediction writer output H5 file, OR a list/tuple of
        paths (e.g. the per-1k-event split files from ``run_forward_pass``).
        When several files are given they are treated as one contiguous event
        stream in the order provided, and the returned dicts are concatenated.
    calo_pred_threshold : binary threshold for calo mask predictions
    calo_hs_frac_threshold : truth HS fraction threshold for mask truth
    calo_hs_energy_threshold : truth HS energy threshold for mask truth
    event_start : optional inclusive start index for contiguous event slicing
        (global across all files when a list is given)
    event_stop : optional exclusive stop index for contiguous event slicing
    event_indices : optional explicit event indices (strictly increasing;
        single-file only)
    load_reco : if True (default), load reco_data (object class/regression and,
        if present, the large truth/pred incidence tensors). Set False to skip
        _load_reco entirely (returns ``{}``) — much faster/lighter when only
        track_data and cluster_data are needed.

    Returns
    -------
    track_data : dict  (empty if node_metadata absent)
    cluster_data : dict  (empty if node_metadata absent)
    reco_data : dict  (always populated if object_class + regression exist)
    """
    # Multiple files: load each (with its share of the global event range) and
    # concatenate. cluster event_idx is offset so it stays globally contiguous.
    if isinstance(h5_path, (list, tuple)):
        paths = [Path(p) for p in h5_path]
        if len(paths) != 1:
            if event_indices is not None:
                raise ValueError(
                    "event_indices is not supported across multiple H5 files; "
                    "use event_start/event_stop instead"
                )
            file_ranges = map_event_range_to_files(paths, event_start, event_stop)
            tracks, clusters, recos = [], [], []
            loaded = 0
            for p, lo, hi in file_ranges:
                t, c, r = load_eval_data_from_h5(
                    p,
                    calo_pred_threshold=calo_pred_threshold,
                    calo_hs_frac_threshold=calo_hs_frac_threshold,
                    calo_hs_energy_threshold=calo_hs_energy_threshold,
                    event_start=lo,
                    event_stop=hi,
                    load_reco=load_reco,
                )
                if isinstance(c.get("event_idx"), np.ndarray) and c["event_idx"].size:
                    c = {**c, "event_idx": c["event_idx"] + loaded}
                tracks.append(t)
                clusters.append(c)
                recos.append(r)
                loaded += hi - lo
            track_data = concat_eval_dicts(tracks)
            cluster_data = concat_eval_dicts(clusters)
            reco_data = concat_eval_dicts(recos)
            reco_data["_source_h5_path"] = ";".join(str(p) for p in paths)
            return track_data, cluster_data, reco_data
        h5_path = paths[0]

    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        n_events = _infer_n_events(f)
        event_sel = _build_event_selector(
            n_events=n_events,
            event_start=event_start,
            event_stop=event_stop,
            event_indices=event_indices,
        )

        reco_data = _load_reco(f, event_sel=event_sel) if load_reco else {}
        track_data = _load_track(f, event_sel=event_sel)
        cluster_data = _load_cluster(
            f,
            calo_pred_threshold=calo_pred_threshold,
            calo_hs_frac_threshold=calo_hs_frac_threshold,
            calo_hs_energy_threshold=calo_hs_energy_threshold,
            event_sel=event_sel,
        )

    # Keep source path for downstream utilities that may lazily load
    # optional tensors not present in older in-memory dicts.
    reco_data["_source_h5_path"] = str(h5_path)

    return track_data, cluster_data, reco_data


def _load_reco(f: h5py.File, event_sel: EventSelector) -> dict:
    """Extract reco (Stream C) evaluation data from H5."""
    reco_data: dict = {}

    if "object_class" not in f or "regression" not in f:
        return reco_data

    oc = f["object_class"]
    reg = f["regression"]

    truth_class = oc["object_class"][event_sel].astype(np.int64)
    pred_class = oc["pflow_class"][event_sel].astype(np.int64)

    reco_data["truth_class"] = truth_class
    reco_data["pred_class"] = pred_class
    reco_data["truth_valid"] = (truth_class < (RECO_NUM_CLASSES - 1)).astype(bool)
    reco_data["pred_valid"] = (pred_class < (RECO_NUM_CLASSES - 1)).astype(bool)

    for prefix in ("truth", "pred"):
        for field in ("pt", "eta", "sinphi", "cosphi"):
            key = f"{prefix}_{field}"
            if key in reg.dtype.names:
                reco_data[key] = reg[key][event_sel].astype(np.float32)

    # Optional incidence tensors (for event-level incidence diagnostics/plots)
    if "pred_incidence" in f:
        pred_inc_ds = f["pred_incidence"]
        if pred_inc_ds.dtype.names:
            pred_field = pred_inc_ds.dtype.names[0]
            reco_data["pred_incidence"] = pred_inc_ds[pred_field][event_sel].astype(np.float32)
        else:
            reco_data["pred_incidence"] = pred_inc_ds[event_sel].astype(np.float32)

    if "truth_incidence" in f:
        truth_inc_ds = f["truth_incidence"]
        if truth_inc_ds.dtype.names:
            truth_field = truth_inc_ds.dtype.names[0]
            reco_data["truth_incidence"] = truth_inc_ds[truth_field][event_sel].astype(np.float32)
        else:
            reco_data["truth_incidence"] = truth_inc_ds[event_sel].astype(np.float32)

    if "reco_node_indices" in f:
        idx_ds = f["reco_node_indices"]
        if idx_ds.dtype.names:
            idx_field = idx_ds.dtype.names[0]
            reco_data["reco_node_indices"] = idx_ds[idx_field][event_sel].astype(np.int64)
        else:
            reco_data["reco_node_indices"] = idx_ds[event_sel].astype(np.int64)

    # Optional reco-node metadata mapped into filtered reco-node space.
    if (
        "node_metadata" in f
        and "reco_node_indices" in reco_data
        and np.asarray(reco_data["reco_node_indices"]).ndim == 2
    ):
        nm = f["node_metadata"]
        reco_idx = np.asarray(reco_data["reco_node_indices"], dtype=np.int64)

        if "node_is_track" in nm.dtype.names:
            node_is_track = _read_struct_field(nm, "node_is_track", bool, event_sel)
            max_node = node_is_track.shape[1] - 1
            safe_idx = np.clip(reco_idx, 0, max_node)
            reco_is_track = np.take_along_axis(node_is_track, safe_idx, axis=1)
            invalid_idx = (reco_idx < 0) | (reco_idx > max_node)
            reco_is_track[invalid_idx] = False
            reco_data["reco_is_track"] = reco_is_track
            reco_data["reco_n_tracks"] = reco_is_track.sum(axis=1).astype(np.int64)

        if "node_valid" in nm.dtype.names:
            node_valid = _read_struct_field(nm, "node_valid", bool, event_sel)
            max_node = node_valid.shape[1] - 1
            safe_idx = np.clip(reco_idx, 0, max_node)
            reco_node_valid = np.take_along_axis(node_valid, safe_idx, axis=1)
            invalid_idx = (reco_idx < 0) | (reco_idx > max_node)
            reco_node_valid[invalid_idx] = False
            reco_data["reco_node_valid"] = reco_node_valid

    # Optional raw node-level fields in full 5500-node space.
    # These are used by reco-analysis diagnostics that cluster jets from
    # calorimeter clusters (valid non-track nodes, including pileup).
    if "node_metadata" in f:
        nm = f["node_metadata"]
        if "node_valid" in nm.dtype.names:
            reco_data["node_valid"] = _read_struct_field(nm, "node_valid", bool, event_sel)
        if "node_is_track" in nm.dtype.names:
            reco_data["node_is_track"] = _read_struct_field(nm, "node_is_track", bool, event_sel)
        if "node_eta" in nm.dtype.names:
            reco_data["node_eta"] = _read_struct_field(nm, "node_eta", np.float32, event_sel)
        if "node_phi" in nm.dtype.names:
            reco_data["node_phi"] = _read_struct_field(nm, "node_phi", np.float32, event_sel)
        if "node_e" in nm.dtype.names:
            reco_data["node_e"] = _read_struct_field(nm, "node_e", np.float32, event_sel)
        if "calo_hs_energy" in nm.dtype.names:
            reco_data["calo_hs_energy"] = _read_struct_field(nm, "calo_hs_energy", np.float32, event_sel)
        if "calo_hs_frac" in nm.dtype.names:
            reco_data["calo_hs_frac"] = _read_struct_field(nm, "calo_hs_frac", np.float32, event_sel)
        if "node_pt" in nm.dtype.names:
            reco_data["node_pt"] = _read_struct_field(nm, "node_pt", np.float32, event_sel)

    # Calo mask probabilities in full node space (Stream B output)
    if "calo_mask" in f:
        calo_prob = _drop_unit_axes_except_first(
            f["calo_mask"]["calo_prob"][event_sel].astype(np.float32)
        )
        if calo_prob.ndim == 2:
            reco_data["calo_prob"] = calo_prob

    return reco_data


def _load_track(f: h5py.File, event_sel: EventSelector) -> dict:
    """Extract track (Stream A) evaluation data from H5.

    Requires ``node_metadata`` and ``track_mask`` groups.
    Returns empty dict if node_metadata is absent (v1 H5 files).
    """
    track_data: dict = {}

    if "node_metadata" not in f:
        print("Warning: H5 file has no node_metadata group — track plots unavailable. "
              "Re-run prediction writer to generate node_metadata.")
        return track_data

    if "track_mask" not in f:
        return track_data

    nm = f["node_metadata"]
    node_valid = _read_struct_field(nm, "node_valid", bool, event_sel)
    is_track = _read_struct_field(nm, "node_is_track", bool, event_sel)
    track_node_mask = node_valid & is_track

    track_prob = _drop_unit_axes_except_first(
        f["track_mask"]["track_prob"][event_sel].astype(np.float32)
    )
    if track_prob.ndim != 2:
        raise ValueError(
            f"Unexpected track_prob shape after squeeze: {track_prob.shape}. "
            "Expected 2D [events, nodes]."
        )

    track_data["probs"] = track_prob[track_node_mask]

    if "tracks_mask" in nm.dtype.names:
        track_data["truth"] = _read_struct_field(nm, "tracks_mask", np.int32, event_sel)[track_node_mask]

    for field, nm_field in [("pt", "node_pt"), ("eta", "node_eta"), ("z0", "node_z0")]:
        if nm_field in nm.dtype.names:
            track_data[field] = _read_struct_field(nm, nm_field, np.float32, event_sel)[track_node_mask]

    return track_data


def _load_cluster(
    f: h5py.File,
    calo_pred_threshold: float,
    calo_hs_frac_threshold: float,
    calo_hs_energy_threshold: float,
    event_sel: EventSelector,
) -> dict:
    """Extract cluster (Stream B) evaluation data from H5.

    Requires ``node_metadata`` and ``calo_mask`` groups.
    Returns empty dict if node_metadata is absent (v1 H5 files).
    """
    cluster_data: dict = {}

    if "node_metadata" not in f:
        print("Warning: H5 file has no node_metadata group — cluster plots unavailable. "
              "Re-run prediction writer to generate node_metadata.")
        return cluster_data

    if "calo_mask" not in f:
        return cluster_data

    nm = f["node_metadata"]
    node_valid = _read_struct_field(nm, "node_valid", bool, event_sel)         # (N, 5500)
    is_track = _read_struct_field(nm, "node_is_track", bool, event_sel)        # (N, 5500)
    cluster_mask = node_valid & (~is_track)                          # (N, 5500)

    calo_prob = _drop_unit_axes_except_first(
        f["calo_mask"]["calo_prob"][event_sel].astype(np.float32)
    )
    node_e = _read_struct_field(nm, "node_e", np.float32, event_sel)           # (N, 5500)
    if calo_prob.ndim != 2:
        raise ValueError(
            f"Unexpected calo_prob shape after squeeze: {calo_prob.shape}. "
            "Expected 2D [events, nodes]."
        )

    # Flat cluster-level arrays
    cluster_data["total_e"] = node_e[cluster_mask]
    cluster_data["calo_mask_probs"] = calo_prob[cluster_mask]

    if "node_eta" in nm.dtype.names:
        cluster_data["eta"] = _read_struct_field(nm, "node_eta", np.float32, event_sel)[cluster_mask]
    if "node_phi" in nm.dtype.names:
        cluster_data["phi"] = _read_struct_field(nm, "node_phi", np.float32, event_sel)[cluster_mask]

    # Truth HS energy and fraction
    if "calo_hs_energy" in nm.dtype.names:
        true_hs_e = _read_struct_field(nm, "calo_hs_energy", np.float32, event_sel)
        cluster_data["true_hs_e"] = true_hs_e[cluster_mask]
    if "calo_hs_frac" in nm.dtype.names:
        true_frac = _read_struct_field(nm, "calo_hs_frac", np.float32, event_sel)
        cluster_data["true_frac"] = true_frac[cluster_mask]

    # Neutral / charged energy
    if "calo_neutral_e" in nm.dtype.names:
        cluster_data["neutral_e"] = _read_struct_field(nm, "calo_neutral_e", np.float32, event_sel)[cluster_mask]
    if "calo_charged_e" in nm.dtype.names:
        cluster_data["charged_e"] = _read_struct_field(nm, "calo_charged_e", np.float32, event_sel)[cluster_mask]

    # Binary mask predictions and truth
    cluster_data["mask_pred"] = (calo_prob > calo_pred_threshold)[cluster_mask]
    if "calo_hs_energy" in nm.dtype.names and "calo_hs_frac" in nm.dtype.names:
        mask_truth_full = (
            (true_frac > calo_hs_frac_threshold)
            & (true_hs_e > calo_hs_energy_threshold)
        )
        cluster_data["mask_truth"] = mask_truth_full[cluster_mask]

    # Calo fraction predictions (if available)
    if "calo_fraction" in f:
        cf = f["calo_fraction"]
        if "calo_frac_pred" in cf.dtype.names:
            cluster_data["pred_frac"] = _drop_unit_axes_except_first(
                cf["calo_frac_pred"][event_sel].astype(np.float32)
            )[cluster_mask]

    # Per-cluster event index
    counts = cluster_mask.sum(axis=-1)  # (N_events,)
    cluster_data["event_idx"] = np.repeat(np.arange(len(counts)), counts)

    # Per-event aggregate energies
    cluster_valid_f = cluster_mask.astype(np.float32)  # (N, 5500)
    pred_mask_f = (calo_prob > calo_pred_threshold).astype(np.float32)  # (N, 5500)

    if "calo_hs_energy" in nm.dtype.names and "calo_hs_frac" in nm.dtype.names:
        truth_mask_f = mask_truth_full.astype(np.float32)

        cluster_data["evt_pred_mask_e"] = (pred_mask_f * cluster_valid_f * node_e).sum(axis=-1)
        cluster_data["evt_truth_mask_e"] = (truth_mask_f * cluster_valid_f * node_e).sum(axis=-1)

        if "calo_neutral_e" in nm.dtype.names:
            neutral_e_full = _read_struct_field(nm, "calo_neutral_e", np.float32, event_sel)
            charged_e_full = _read_struct_field(nm, "calo_charged_e", np.float32, event_sel)

            cluster_data["evt_pred_neutral_e"] = (pred_mask_f * cluster_valid_f * neutral_e_full).sum(axis=-1)
            cluster_data["evt_truth_neutral_e"] = (truth_mask_f * cluster_valid_f * neutral_e_full).sum(axis=-1)
            cluster_data["evt_pred_charged_e"] = (pred_mask_f * cluster_valid_f * charged_e_full).sum(axis=-1)
            cluster_data["evt_truth_charged_e"] = (truth_mask_f * cluster_valid_f * charged_e_full).sum(axis=-1)

    return cluster_data


# ---------------------------------------------------------------------------
# Forward pass runner
# ---------------------------------------------------------------------------

def run_forward_pass(
    ckpt_path: str | Path,
    config_path: str | Path,
    data_dir: str | None = None,
    files: list[str | Path] | None = None,
    num_events: int = -1,
    batch_size: int = 48,
    num_workers: int = 1,
    accelerator: str | None = None,
    devices: int | str | list[int] | None = None,
    precision: str | int | None = None,
    inference_mode: bool | None = None,
    reco_debug_use_truth_masks: bool = False,
    test_suff: str = "",
    events_per_file: int | None = None,
    predict_only: bool = True,
) -> Path | list[Path]:
    """Run Lightning test step with PflowPredictionWriter, return H5 path(s).

    This is the forward-pass path that produces an H5 in the canonical
    prediction writer format, which can then be loaded with
    ``load_eval_data_from_h5()``.

    Parameters
    ----------
    ckpt_path : path to model checkpoint
    config_path : path to YAML config
    data_dir : directory with parquet files (used if files is None)
    files : explicit list of parquet file paths
    num_events : number of events to load (-1 for all)
    batch_size : batch size for test dataloader
    num_workers : dataloader workers
    accelerator : Lightning accelerator override (defaults to config trainer.accelerator)
    devices : Lightning devices override (defaults to config trainer.devices)
    precision : Lightning precision override (defaults to config trainer.precision)
    reco_debug_use_truth_masks : if True, use truth masks (oracle) for Stream C
        reconstruction filtering during inference. Default is False.
    test_suff : suffix appended to output file name
    events_per_file : if set, split the output across multiple H5 files holding
        at most this many events each (the final file holds the remainder).
        Files are named ``..__test{suff}__partNNN.h5``. If None (default), a
        single H5 file is written as before.
    predict_only : if True (default), skip the test-time loss (and its Hungarian
        matching) during the forward pass — much faster, and the writer never
        uses the loss. Set False to also compute/log loss and metrics.

    Returns
    -------
    Path to the generated H5 file, or a list of Paths (one per part) when
    ``events_per_file`` is set. No ROOT files are written.
    """
    import yaml
    from lightning import Trainer

    from hepattn.experiments.odd_pileup_reco.lightning_module import ODDPFlowTwoStream
    from hepattn.experiments.odd_pileup_reco.pflow_data import ODDDataModule
    from hepattn.experiments.odd_pileup_reco.predictionwriter import PflowPredictionWriter

    ckpt_path = Path(ckpt_path)
    config_path = Path(config_path)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    trainer_cfg = cfg.get("trainer", {})

    # Respect config defaults unless explicitly overridden by function args.
    trainer_accelerator = accelerator if accelerator is not None else trainer_cfg.get("accelerator", "auto")
    trainer_devices = devices if devices is not None else trainer_cfg.get("devices", 1)
    trainer_precision = precision if precision is not None else trainer_cfg.get("precision", "32-true")
    trainer_inference_mode = inference_mode if inference_mode is not None else bool(trainer_cfg.get("inference_mode", True))
    trainer_enable_progress_bar = bool(trainer_cfg.get("enable_progress_bar", True))

    # Build dataset
    if files is not None:
        if len(files) == 0:
            raise ValueError("files must be a non-empty list when provided")
        files_list = [Path(f) for f in files]
        filepath = str(files_list[0].parent)
    elif data_dir is not None:
        files_list = None
        filepath = data_dir
    else:
        # Fall back to config's test path
        filepath = data_cfg.get("test_filepath", data_cfg.get("filepath"))
        files_list = None

    # Use fully initialized DataModule so Lightning setup hooks have all attributes.
    datamodule = ODDDataModule(
        train_path=filepath,
        valid_path=filepath,
        test_path=filepath,
        batch_size=batch_size,
        num_workers=num_workers,
        num_test=num_events,
        scale_dict_path=data_cfg["scale_dict_path"],
        inputs=data_cfg["inputs"],
        targets=data_cfg["targets"],
        max_nodes=data_cfg["max_nodes"],
        incidence_cutval=data_cfg.get("incidence_cutval", 0.01),
        hard_scatter_energy_threshold=data_cfg.get("hard_scatter_energy_threshold", 0.03),
        window_size=data_cfg.get("window_size", 512),
        files_list=files_list,
        is_inference=True,
        test_suff=test_suff,
        enable_split=False,
    )

    # Load model
    model = ODDPFlowTwoStream.load_from_checkpoint(str(ckpt_path), map_location="cpu")
    if hasattr(model, "model") and hasattr(model.model, "reco_debug_use_truth_masks"):
        model.model.reco_debug_use_truth_masks = bool(reco_debug_use_truth_masks)
    elif reco_debug_use_truth_masks:
        print("Warning: model does not expose reco_debug_use_truth_masks; running default inference masks")

    # Create prediction writer callback (H5 only — no ROOT output)
    writer = PflowPredictionWriter(
        events_per_file=events_per_file, write_root=False, predict_only=predict_only,
    )

    trainer = Trainer(
        accelerator=trainer_accelerator,
        devices=trainer_devices,
        precision=trainer_precision,
        callbacks=[writer],
        logger=False,
        enable_progress_bar=trainer_enable_progress_bar,
        inference_mode=trainer_inference_mode,
    )

    print("Running test...")
    trainer.test(model, datamodule=datamodule, ckpt_path=str(ckpt_path))

    out_paths = writer.output_paths
    if events_per_file is None:
        h5_path = out_paths[0] if out_paths else writer.output_path
        print(f"Predictions written to {h5_path}")
        return h5_path

    print(f"Predictions written to {len(out_paths)} file(s):")
    for p in out_paths:
        print(f"  {p}")
    return out_paths


# ---------------------------------------------------------------------------
# Teacher-forcing vs prediction-path loss comparison (exposure-bias test)
# ---------------------------------------------------------------------------

def compare_tf_vs_pred_loss(
    ckpt_path: str | Path,
    config_path: str | Path,
    data_dir: str | None = None,
    files: list[str | Path] | None = None,
    num_events: int = 1000,
    batch_size: int = 32,
    num_workers: int = 1,
    device: str | None = None,
    use_amp: bool = True,
    out_dir: str | None = None,
) -> dict:
    """Run the same held-out events through a trained checkpoint twice and compare losses.

    Forces Stream B/C conditioning to the **teacher-forcing** path (truth masks + sampled
    noise, what the model saw during training) and to the **prediction** path (Stream A/B
    predicted masks, what validation/inference uses), on identical events, and reports the
    mean losses. If the TF-path loss is far below the pred-path loss, the train/val gap is
    exposure bias (train/inference mismatch in Stream C), not overfitting.

    Returns a dict with per-stream and total mean losses for both paths.
    """
    import yaml
    import torch

    from hepattn.experiments.odd_pileup_reco.lightning_module import ODDPFlowTwoStream
    from hepattn.experiments.odd_pileup_reco.pflow_data import ODDDataModule

    ckpt_path = Path(ckpt_path)
    config_path = Path(config_path)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg["data"]

    # Resolve data source (mirrors run_forward_pass)
    if files is not None:
        if len(files) == 0:
            raise ValueError("files must be a non-empty list when provided")
        files_list = [Path(f) for f in files]
        filepath = str(files_list[0].parent)
    elif data_dir is not None:
        files_list = None
        filepath = data_dir
    else:
        filepath = data_cfg.get("test_filepath", data_cfg.get("filepath"))
        files_list = None

    datamodule = ODDDataModule(
        train_path=filepath,
        valid_path=filepath,
        test_path=filepath,
        batch_size=batch_size,
        num_workers=num_workers,
        num_test=num_events,
        scale_dict_path=data_cfg["scale_dict_path"],
        inputs=data_cfg["inputs"],
        targets=data_cfg["targets"],
        max_nodes=data_cfg["max_nodes"],
        incidence_cutval=data_cfg.get("incidence_cutval", 0.01),
        hard_scatter_energy_threshold=data_cfg.get("hard_scatter_energy_threshold", 0.03),
        window_size=data_cfg.get("window_size", 512),
        files_list=files_list,
        is_inference=True,
        test_suff="tf_vs_pred",
        enable_split=False,
    )
    datamodule.setup("test")
    loader = datamodule.test_dataloader()

    model = ODDPFlowTwoStream.load_from_checkpoint(str(ckpt_path), map_location="cpu")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    net = model.model  # TwoStreamMaskFormer

    def _to_device(d):
        return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in d.items()}

    def _accumulate(losses: dict, acc: dict, task_acc: dict, n: int) -> float:
        """Sum leaf-tensor losses into per-stream, per-reco-task, and total accumulators."""
        batch_total = 0.0
        for layer_name, layer_losses in losses.items():
            stream = layer_name.split("_", 1)[0]  # track / calo / reco
            for task_name, task_losses in layer_losses.items():
                for loss_value in task_losses.values():
                    if isinstance(loss_value, torch.Tensor):
                        v = float(loss_value.detach().float().cpu())
                        acc[stream] = acc.get(stream, 0.0) + v * n
                        batch_total += v
                        if stream == "reco":  # sum each reco task across all layers
                            task_acc[task_name] = task_acc.get(task_name, 0.0) + v * n
        acc["total"] = acc.get("total", 0.0) + batch_total * n
        return batch_total

    def _squeeze2d(t):
        return t.squeeze(-1) if (t is not None and t.dim() > 2) else t

    def _input_diag(node_is_track, node_e, calo_hs_e, node_valid) -> dict:
        """Per-event Stream-C input characterisation from the model's stored selection.

        Uses net._reco_node_indices / _reco_filtered_valid set during the forward.
        Returns numpy arrays (one value per event):
          occ      : number of valid nodes packed into the reco buffer
          n_calo   : selected calo nodes ; n_track : selected track nodes
          purity   : sum(HS energy) / sum(total energy) of selected calo nodes (FP contamination)
          recall   : sum(selected HS calo energy) / sum(all true HS calo energy) (FN / signal delivered)
        """
        idx = net._reco_node_indices                    # (B, R) long
        valid = net._reco_filtered_valid.bool()          # (B, R)
        it_sel = node_is_track.gather(1, idx).bool()
        e_sel = node_e.gather(1, idx).float()
        hs_sel = calo_hs_e.gather(1, idx).float()
        calo_sel = valid & (~it_sel)
        track_sel = valid & it_sel

        sel_calo_e = (e_sel * calo_sel).sum(1)
        sel_calo_hs = (hs_sel * calo_sel).sum(1)
        calo_node = (~node_is_track.bool()) & node_valid.bool()
        total_hs = (calo_hs_e.float() * calo_node).sum(1)

        return {
            "occ":    valid.sum(1).float().cpu().numpy(),
            "n_calo": calo_sel.sum(1).float().cpu().numpy(),
            "n_track": track_sel.sum(1).float().cpu().numpy(),
            "purity": (sel_calo_hs / sel_calo_e.clamp_min(1e-6)).cpu().numpy(),
            "recall": (sel_calo_hs / total_hs.clamp_min(1e-6)).cpu().numpy(),
        }

    acc_pred: dict = {}
    acc_tf: dict = {}
    treco_pred: dict = {}
    treco_tf: dict = {}
    diag_pred: dict = {k: [] for k in ("occ", "n_calo", "n_track", "purity", "recall")}
    diag_tf: dict = {k: [] for k in ("occ", "n_calo", "n_track", "purity", "recall")}
    n_events = 0

    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if (use_amp and device == "cuda")
        else _nullcontext()
    )

    print(f"Comparing TF vs pred loss on {filepath} (device={device}, num_events={num_events})...")
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            inputs, targets = batch
            inputs = _to_device(inputs)
            targets = _to_device(targets)
            n = next(v.shape[0] for v in targets.values() if isinstance(v, torch.Tensor))

            # Full-node-space arrays for input characterisation. node_is_track /
            # node_valid / node_e live in `inputs`; HS energy lives in `targets`.
            nit = _squeeze2d(inputs.get("node_is_track"))
            nv = _squeeze2d(inputs.get("node_valid"))
            ne = _squeeze2d(inputs.get("node_e"))
            chs = _squeeze2d(targets.get("calo_hard_scatter_energy"))
            can_diag = all(t is not None for t in (nit, nv, ne, chs))

            with autocast:
                out_pred = net(inputs, targets=targets, use_teacher_forcing=False)
                loss_pred = net.loss(out_pred, targets)
                if can_diag:
                    d = _input_diag(nit, ne, chs, nv)
                    for k in diag_pred:
                        diag_pred[k].append(d[k])
                out_tf = net(inputs, targets=targets, use_teacher_forcing=True)
                loss_tf = net.loss(out_tf, targets)
                if can_diag:
                    d = _input_diag(nit, ne, chs, nv)
                    for k in diag_tf:
                        diag_tf[k].append(d[k])

            _accumulate(loss_pred, acc_pred, treco_pred, n)
            _accumulate(loss_tf, acc_tf, treco_tf, n)
            n_events += n
            print(f"  batch {bi:3d}  n={n:3d}  "
                  f"pred_total={acc_pred['total'] / n_events:.4f}  "
                  f"tf_total={acc_tf['total'] / n_events:.4f}")

    pred_means = {k: v / n_events for k, v in acc_pred.items()}
    tf_means = {k: v / n_events for k, v in acc_tf.items()}
    treco_pred = {k: v / n_events for k, v in treco_pred.items()}
    treco_tf = {k: v / n_events for k, v in treco_tf.items()}

    streams = [s for s in ("track", "calo", "reco") if s in pred_means or s in tf_means]

    def _row(label, p, t):
        print(f"{label:<14} {p:>12.4f} {t:>12.4f} {p - t:>12.4f}")

    print("\n" + "=" * 64)
    print(f"TF vs PRED loss comparison   (n_events={n_events})")
    print("=" * 64)
    print(f"{'stream':<14} {'pred path':>12} {'TF path':>12} {'delta':>12}")
    print("-" * 64)
    for s in streams:
        _row(s, pred_means.get(s, float('nan')), tf_means.get(s, float('nan')))
    print("-" * 64)
    _row("TOTAL", pred_means["total"], tf_means["total"])
    print("=" * 64)

    # Reco sub-task breakdown — which objective carries the gap.
    if treco_pred:
        print("\nReco sub-task breakdown (summed over decoder layers)")
        print("-" * 64)
        print(f"{'reco task':<14} {'pred path':>12} {'TF path':>12} {'delta':>12}")
        print("-" * 64)
        for task_name in sorted(set(treco_pred) | set(treco_tf),
                                key=lambda k: -(treco_pred.get(k, 0) - treco_tf.get(k, 0))):
            _row(task_name, treco_pred.get(task_name, float('nan')), treco_tf.get(task_name, float('nan')))
        print("=" * 64)

    # ── Diagnostic plots: Stream-C input distribution shift (TF vs pred) ──
    saved = _plot_tf_vs_pred_diagnostics(
        diag_pred, diag_tf, treco_pred, treco_tf,
        out_dir=out_dir, ckpt_path=ckpt_path, max_reco_nodes=getattr(net, "max_reco_nodes", None),
    )
    for pth in saved:
        print(f"Saved plot: {pth}")
    print()

    return {
        "pred": pred_means, "tf": tf_means,
        "reco_tasks_pred": treco_pred, "reco_tasks_tf": treco_tf,
        "n_events": n_events, "plots": saved,
    }


def _plot_tf_vs_pred_diagnostics(diag_pred, diag_tf, treco_pred, treco_tf,
                                 out_dir=None, ckpt_path=None, max_reco_nodes=None) -> list:
    """Save (1) Stream-C input-distribution histograms and (2) reco sub-task bar chart.

    Returns the list of written file paths.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir) if out_dir is not None else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(ckpt_path).stem if ckpt_path is not None else "tf_vs_pred"
    saved: list = []

    PRED_C, TF_C = "#dd8452", "#55a868"  # match existing reco-analysis palette

    def _cat(d):
        return {k: (np.concatenate(v) if len(v) else np.array([])) for k, v in d.items()}

    dp, dt = _cat(diag_pred), _cat(diag_tf)

    # ── Figure 1: input distribution shift ───────────────────────────────
    if dp.get("occ") is not None and dp["occ"].size:
        title_suffix = f"  (max_reco_nodes={max_reco_nodes})" if max_reco_nodes else ""
        specs = [
            ("occ",    f"reco-buffer occupancy (valid nodes/event){title_suffix}", None),
            ("purity", r"calo purity: $\sum$HS / $\sum$total energy of selected calo  [FP contamination]", (0.0, 1.0)),
            ("recall", r"HS energy recall: $\sum$selected HS / $\sum$all HS calo energy  [FN / signal delivered]", (0.0, 1.2)),
            ("n_calo", "selected calo nodes / event", None),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        for ax, (key, xlabel, rng) in zip(axes.ravel(), specs):
            p, t = dp.get(key), dt.get(key)
            if p is None or t is None or p.size == 0:
                continue
            lo = min(p.min(), t.min()) if rng is None else rng[0]
            hi = max(p.max(), t.max()) if rng is None else rng[1]
            bins = np.linspace(lo, hi, 51)
            ax.hist(t, bins=bins, histtype="step", linewidth=2.2, density=True, color=TF_C,
                    label=f"TF        mean={t.mean():.2f} med={np.median(t):.2f}")
            ax.hist(p, bins=bins, histtype="step", linewidth=2.2, density=True, color=PRED_C,
                    label=f"pred      mean={p.mean():.2f} med={np.median(p):.2f}")
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel("density")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.25)
        fig.suptitle("Stream-C input distribution: teacher-forcing vs prediction path", fontsize=12)
        fig.tight_layout()
        f1 = out_dir / f"tf_vs_pred_input_dist__{stem}.png"
        fig.savefig(f1, dpi=120)
        plt.close(fig)
        saved.append(str(f1))

    # ── Figure 2: reco sub-task loss breakdown ───────────────────────────
    if treco_pred:
        tasks = sorted(set(treco_pred) | set(treco_tf))
        x = np.arange(len(tasks))
        pv = [treco_pred.get(k, 0.0) for k in tasks]
        tv = [treco_tf.get(k, 0.0) for k in tasks]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(x - 0.2, pv, width=0.4, color=PRED_C, label="pred path")
        ax.bar(x + 0.2, tv, width=0.4, color=TF_C, label="TF path")
        for xi, (a, b) in enumerate(zip(pv, tv)):
            ax.text(xi, max(a, b) + 0.02 * max(pv + tv), f"Δ={a - b:+.2f}", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=20)
        ax.set_ylabel("mean loss")
        ax.set_title("Reco sub-task loss: teacher-forcing vs prediction path")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        f2 = out_dir / f"tf_vs_pred_reco_breakdown__{stem}.png"
        fig.savefig(f2, dpi=120)
        plt.close(fig)
        saved.append(str(f2))

    return saved


def analyze_tf_mask_strategies(
    ckpt_path: str | Path,
    config_path: str | Path,
    data_dir: str | None = None,
    files: list[str | Path] | None = None,
    num_events: int = 1000,
    batch_size: int = 32,
    num_workers: int = 1,
    device: str | None = None,
    use_amp: bool = True,
    out_dir: str | None = None,
    drop_frac: float = 0.25,
    track_drop_frac: float = 0.1,
    seed: int = 0,
) -> dict:
    """Compare candidate teacher-forcing node-selection strategies against the
    inference (pred) selection, to decide how the TF mask should be built.

    The premise (Stream B pileup removal is well-optimised) is that the *pred*
    selection is the target distribution; a good TF mask should match it on:
    purity, HS-energy recall, buffer size (occupancy / n_calo / n_track), and the
    per-node selected-energy spectra of clusters and tracks (pooled over events).

    Strategies compared (all replicate the model's topK-by-energy calo cap):
      pred              : sigmoid(calo)>=thr & ~track & valid -> topK ; tracks sigmoid>=0.5  (the target)
      truth             : truth calo (frac/energy thr) -> topK ; truth tracks                (oracle)
      truth+FP(current) : truth + uniform-sampled false-positive PU (current teacher forcing)
      truth+FN_random   : truth with a random fraction of HS nodes dropped
      truth+FN_energy   : truth with the lowest-energy HS fraction dropped (mimics how Stream B misses faint clusters)

    Returns a dict of per-strategy mean metrics and the saved plot paths.
    """
    import yaml
    import torch
    import numpy as np
    from torch.utils.data import DataLoader

    from hepattn.experiments.odd_pileup_reco.lightning_module import ODDPFlowTwoStream
    from hepattn.experiments.odd_pileup_reco.pflow_data import EagerODDDataset

    ckpt_path = Path(ckpt_path)
    config_path = Path(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg["data"]

    if files is not None:
        if len(files) == 0:
            raise ValueError("files must be a non-empty list when provided")
        files_list = [Path(f) for f in files]
        filepath = str(files_list[0].parent)
    elif data_dir is not None:
        files_list, filepath = None, data_dir
    else:
        files_list, filepath = None, data_cfg.get("test_filepath", data_cfg.get("filepath"))

    dataset = EagerODDDataset(
        filepath=filepath, num_events=num_events, files_list=files_list,
        inputs=data_cfg["inputs"], targets=data_cfg["targets"], scale_dict_path=data_cfg["scale_dict_path"],
        num_objects=data_cfg.get("num_objects", 400), max_nodes=data_cfg["max_nodes"],
        incidence_cutval=data_cfg.get("incidence_cutval", 0.01), is_inference=True,
        hard_scatter_energy_threshold=data_cfg.get("hard_scatter_energy_threshold", 0.03),
        window_size=data_cfg.get("window_size", 512), compute_deltaR_stats=False,
    )
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                        shuffle=False, collate_fn=None, pin_memory=True)

    model = ODDPFlowTwoStream.load_from_checkpoint(str(ckpt_path), map_location="cpu")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    net = model.model

    PT = net.calo_pred_threshold              # calo predicted-HS threshold (0.3)
    FT = net.calo_hs_frac_threshold           # truth HS fraction threshold (0.10)
    ET = net.calo_hs_energy_threshold         # truth HS energy threshold (0.15)
    KCAL = net.max_reco_calo_nodes            # calo topK cap (1200)
    MAXN = net.max_reco_nodes                 # buffer size (1400)
    NF = net.reco_calo_noise_mean             # FP calo noise count (250)
    NTF = net.reco_track_noise_mean           # FP track noise count (15)
    gen = torch.Generator(device=device).manual_seed(seed)

    def _sq(t):
        return t.squeeze(-1) if (t is not None and t.dim() > 2) else t

    def _topk_calo(cand, node_e):
        e = torch.where(cand, node_e, torch.zeros_like(node_e))
        kk = min(KCAL, e.shape[-1])
        _, idx = e.topk(kk, dim=-1, sorted=False)
        keep = torch.zeros_like(cand)
        keep.scatter_(1, idx, True)
        return keep & cand

    def _sample(extra, n):
        scores = torch.where(extra, torch.rand(extra.shape, device=device, generator=gen),
                             torch.full(extra.shape, -1.0, device=device))
        kk = min(n, extra.shape[-1])
        _, idx = scores.topk(kk, dim=-1)
        s = torch.zeros_like(extra)
        s.scatter_(1, idx, True)
        return s & extra

    strategies = ["pred", "truth", "truth+FP(current)", "truth+FN_random", "truth+FN_energy"]
    metrics = {s: {k: [] for k in ("purity", "recall", "occ", "n_calo", "n_track")} for s in strategies}
    spectra = {s: {"calo_e": [], "track_pt": []} for s in strategies}

    autocast = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if (use_amp and device == "cuda") else _nullcontext())

    def _record(strat, calo_sel, track_sel, node_e, node_pt, calo_hs_e, calo_node):
        sel_calo_e = (node_e * calo_sel).sum(1)
        sel_calo_hs = (calo_hs_e * calo_sel).sum(1)
        total_hs = (calo_hs_e * calo_node).sum(1)
        m = metrics[strat]
        m["purity"].append((sel_calo_hs / sel_calo_e.clamp_min(1e-6)).cpu().numpy())
        m["recall"].append((sel_calo_hs / total_hs.clamp_min(1e-6)).cpu().numpy())
        m["n_calo"].append(calo_sel.sum(1).float().cpu().numpy())
        m["n_track"].append(track_sel.sum(1).float().cpu().numpy())
        m["occ"].append((calo_sel | track_sel).sum(1).clamp_max(MAXN).float().cpu().numpy())
        spectra[strat]["calo_e"].append(node_e[calo_sel].cpu().numpy())
        spectra[strat]["track_pt"].append(node_pt[track_sel].cpu().numpy())

    print(f"Analyzing TF-mask strategies on {filepath} (device={device}, num_events={num_events})...")
    with torch.no_grad():
        for bi, (inputs, targets) in enumerate(loader):
            inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
            targets = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in targets.items()}

            with autocast:
                outputs = net(inputs, targets=targets, use_teacher_forcing=False)

            # Mask logits are (B, 1, N) — squeeze the middle object dim (NOT trailing).
            calo_prob = outputs["calo_final"]["calo_mask"]["calo_node_logit"].squeeze(1).float().sigmoid()
            track_prob = outputs["track_final"]["mask"]["pflow_node_logit"].squeeze(1).float().sigmoid()
            is_track = _sq(inputs["node_is_track"]).bool()
            valid = _sq(inputs["node_valid"]).bool()
            node_e = _sq(inputs["node_e"]).float()
            node_pt = _sq(inputs["node_pt"]).float()
            calo_hs_e = _sq(targets["calo_hard_scatter_energy"]).float()
            calo_hs_frac = _sq(targets["calo_hard_scatter_energy_frac"]).float()
            truth_tracks = _sq(targets["tracks_mask"]).bool()

            calo_node = (~is_track) & valid
            truth_calo = (calo_hs_frac > FT) & (calo_hs_e > ET) & calo_node
            pred_calo_cand = (calo_prob >= PT) & calo_node
            truth_track = truth_tracks & is_track
            pred_track = (track_prob >= 0.5) & is_track

            rand_keep = torch.rand(node_e.shape, device=device, generator=gen) >= drop_frac
            track_keep = torch.rand(node_e.shape, device=device, generator=gen) >= track_drop_frac

            # energy-biased keep: drop the lowest-energy `drop_frac` of truth HS calo per event
            energy_keep = truth_calo.clone()
            for b in range(node_e.shape[0]):
                te = node_e[b][truth_calo[b]]
                if te.numel() > 0:
                    thr = torch.quantile(te, drop_frac)
                    energy_keep[b] = truth_calo[b] & (node_e[b] >= thr)

            for strat in strategies:
                if strat == "pred":
                    cs, ts = _topk_calo(pred_calo_cand, node_e), pred_track
                elif strat == "truth":
                    cs, ts = _topk_calo(truth_calo, node_e), truth_track
                elif strat == "truth+FP(current)":
                    fp = _sample(pred_calo_cand & ~truth_calo, NF)
                    tfp = _sample(pred_track & ~truth_track, NTF)
                    cs, ts = _topk_calo(truth_calo | fp, node_e), truth_track | tfp
                elif strat == "truth+FN_random":
                    cs, ts = _topk_calo(truth_calo & rand_keep, node_e), truth_track & track_keep
                else:  # truth+FN_energy
                    cs, ts = _topk_calo(energy_keep, node_e), truth_track & track_keep
                _record(strat, cs, ts, node_e, node_pt, calo_hs_e, calo_node)

    # ── Summary table ────────────────────────────────────────────────────
    summary = {}
    print("\n" + "=" * 78)
    print(f"TF-mask strategy comparison vs pred target   (n_events~{num_events})")
    print("=" * 78)
    print(f"{'strategy':<20} {'purity':>9} {'recall':>9} {'occ':>9} {'n_calo':>9} {'n_track':>9}")
    print("-" * 78)
    for s in strategies:
        means = {k: float(np.concatenate(v).mean()) for k, v in metrics[s].items()}
        summary[s] = means
        print(f"{s:<20} {means['purity']:>9.3f} {means['recall']:>9.3f} "
              f"{means['occ']:>9.1f} {means['n_calo']:>9.1f} {means['n_track']:>9.1f}")
    print("=" * 78)
    print("Goal: pick the TF strategy whose row (and distributions) best match 'pred'.\n")

    saved = _plot_strategy_comparison(metrics, spectra, strategies,
                                      out_dir=out_dir, ckpt_path=ckpt_path,
                                      drop_frac=drop_frac, max_reco_nodes=MAXN)
    for p in saved:
        print(f"Saved plot: {p}")
    print()
    return {"summary": summary, "plots": saved}


def _plot_strategy_comparison(metrics, spectra, strategies, out_dir=None, ckpt_path=None,
                              drop_frac=0.25, max_reco_nodes=1400) -> list:
    """Overlay each candidate TF strategy against 'pred' on per-event metrics and
    per-node selected-energy spectra. Returns written file paths."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir) if out_dir is not None else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(ckpt_path).stem if ckpt_path is not None else "tf_strategies"
    saved: list = []

    # pred = bold black reference; others colored.
    colors = {"pred": "#000000", "truth": "#4c72b0", "truth+FP(current)": "#dd8452",
              "truth+FN_random": "#55a868", "truth+FN_energy": "#c44e52"}

    def _cat(strat, key):
        v = metrics[strat][key]
        return np.concatenate(v) if len(v) else np.array([])

    # ── Figure 1: per-event metrics ──────────────────────────────────────
    panels = [
        ("purity", "calo purity  ΣHS/Σtotal (selected calo)", (0.0, 1.0)),
        ("recall", "HS energy recall  Σsel-HS/Σall-HS calo", (0.0, 1.2)),
        ("occ",    f"reco-buffer occupancy (max={max_reco_nodes})", None),
        ("n_calo", "selected calo nodes / event", None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (key, xlabel, rng) in zip(axes.ravel(), panels):
        ref = _cat("pred", key)
        if ref.size == 0:
            continue
        lo = rng[0] if rng else min(_cat(s, key).min() for s in strategies)
        hi = rng[1] if rng else max(_cat(s, key).max() for s in strategies)
        bins = np.linspace(lo, hi, 51)
        for s in strategies:
            arr = _cat(s, key)
            if arr.size == 0:
                continue
            lw = 3.0 if s == "pred" else 1.8
            ls = "--" if s == "pred" else "-"
            ax.hist(arr, bins=bins, histtype="step", linewidth=lw, linestyle=ls, density=True,
                    color=colors.get(s), label=f"{s}  μ={arr.mean():.2f}")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("density")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.25)
    fig.suptitle("TF-mask strategies vs pred target — per-event metrics", fontsize=12)
    fig.tight_layout()
    f1 = out_dir / f"tf_strategies_metrics__{stem}.png"
    fig.savefig(f1, dpi=120)
    plt.close(fig)
    saved.append(str(f1))

    # ── Figure 2: selected-node energy spectra (clusters & tracks) ───────
    def _cat_spec(strat, key):
        v = spectra[strat][key]
        return np.concatenate(v) if len(v) else np.array([])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (key, xlabel) in zip(axes, [("calo_e", "selected calo cluster energy [GeV]"),
                                        ("track_pt", "selected track pT")]):
        ref = _cat_spec("pred", key)
        pos = ref[ref > 0]
        if pos.size == 0:
            continue
        bins = np.logspace(np.log10(max(pos.min(), 1e-3)), np.log10(pos.max()), 60)
        for s in strategies:
            arr = _cat_spec(s, key)
            arr = arr[arr > 0]
            if arr.size == 0:
                continue
            lw = 3.0 if s == "pred" else 1.8
            ls = "--" if s == "pred" else "-"
            ax.hist(arr, bins=bins, histtype="step", linewidth=lw, linestyle=ls, density=True,
                    color=colors.get(s), label=f"{s}  n={arr.size}")
        ax.set_xscale("log")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("density")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.25)
    fig.suptitle("TF-mask strategies vs pred target — selected-node energy spectra (all events)", fontsize=12)
    fig.tight_layout()
    f2 = out_dir / f"tf_strategies_energy_spectra__{stem}.png"
    fig.savefig(f2, dpi=120)
    plt.close(fig)
    saved.append(str(f2))

    return saved


def sweep_tf_mask_schemes(
    ckpt_path: str | Path,
    config_path: str | Path,
    data_dir: str | None = None,
    files: list[str | Path] | None = None,
    num_events: int = 1000,
    batch_size: int = 32,
    num_workers: int = 1,
    device: str | None = None,
    use_amp: bool = True,
    out_dir: str | None = None,
    fn_modes: tuple = ("random",),
    fn_fracs: tuple = (0.10,),
    fp_counts: tuple = (170,),
    fp_count_stds: tuple = (0.0, 75.0, 150.0, 225.0),
    fp_pool_fracs: tuple = (),
    track_fn_frac: float = 0.1,
    track_fp_frac: float = 0.1,
    top_k_plot: int = 6,
    seed: int = 0,
) -> dict:
    """Sweep many combined FP+FN teacher-forcing schemes and rank them by how well
    they reproduce the inference (pred) selection distributions.

    Every scheme = truth HS calo with a fraction of true HS dropped (FN) PLUS sampled
    false-positive pileup (FP), then the model's topK-by-energy cap. The grid varies:
      fn_mode    : how true HS nodes are dropped ("random" Bernoulli | "energy_low" drop lowest-E)
      fn_frac    : fraction of true HS calo dropped
      fp_count   : MEAN number of FP calo nodes added (uniform-random from Stream B's FP pool)
      fp_count_std : per-event Gaussian spread on that count (0 = fixed; >0 widens the size distribution)
      fp_pool_fracs: optional alternative — per-event FP count = frac * |Stream-B FP pool|, which
                     inherits pred's activity-correlated variance (better matches the size *shape*).
    Tracks are NOT swept: FN drop (track_fn_frac, default 0.1) + proportional FP add
    (track_fp_frac, default 0.1, as a fraction of the per-event true-HS-track count).

    Scoring vs pred (lower = closer):
      - Wasserstein distance on per-event purity / recall / n_calo / n_track, each
        standardised by pred's std;
      - Jensen-Shannon divergence on the selected-node energy spectra (calo E, track pT).
    Prints a ranked table and plots the best `top_k_plot` schemes (+ pred + current TF).
    """
    import yaml
    import torch
    import numpy as np
    from torch.utils.data import DataLoader
    from scipy.stats import wasserstein_distance

    from hepattn.experiments.odd_pileup_reco.lightning_module import ODDPFlowTwoStream
    from hepattn.experiments.odd_pileup_reco.pflow_data import EagerODDDataset

    ckpt_path, config_path = Path(ckpt_path), Path(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg["data"]

    if files is not None:
        if len(files) == 0:
            raise ValueError("files must be a non-empty list when provided")
        files_list = [Path(f) for f in files]
        filepath = str(files_list[0].parent)
    elif data_dir is not None:
        files_list, filepath = None, data_dir
    else:
        files_list, filepath = None, data_cfg.get("test_filepath", data_cfg.get("filepath"))

    dataset = EagerODDDataset(
        filepath=filepath, num_events=num_events, files_list=files_list,
        inputs=data_cfg["inputs"], targets=data_cfg["targets"], scale_dict_path=data_cfg["scale_dict_path"],
        num_objects=data_cfg.get("num_objects", 400), max_nodes=data_cfg["max_nodes"],
        incidence_cutval=data_cfg.get("incidence_cutval", 0.01), is_inference=True,
        hard_scatter_energy_threshold=data_cfg.get("hard_scatter_energy_threshold", 0.03),
        window_size=data_cfg.get("window_size", 512), compute_deltaR_stats=False,
    )
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                        shuffle=False, collate_fn=None, pin_memory=True)

    model = ODDPFlowTwoStream.load_from_checkpoint(str(ckpt_path), map_location="cpu")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    net = model.model

    PT, FT, ET = net.calo_pred_threshold, net.calo_hs_frac_threshold, net.calo_hs_energy_threshold
    KCAL, MAXN = net.max_reco_calo_nodes, net.max_reco_nodes
    NF, NTF = net.reco_calo_noise_mean, net.reco_track_noise_mean
    gen = torch.Generator(device=device).manual_seed(seed)

    calo_bins = np.logspace(-2, 3, 61)
    track_bins = np.logspace(-2, 3, 61)

    schemes = []
    for fnm in fn_modes:
        for fnf in fn_fracs:
            for fpc in fp_counts:
                for fps in fp_count_stds:
                    tag = f"fn={fnm}:{fnf:.2f}|fp={fpc}" + (f"±{int(fps)}" if fps > 0 else "")
                    schemes.append(dict(name=tag, kind="count", fn_mode=fnm, fn_frac=fnf,
                                        fp_val=float(fpc), fp_std=float(fps)))
            for frac in fp_pool_fracs:
                schemes.append(dict(name=f"fn={fnm}:{fnf:.2f}|fp=pool:{frac:.2f}", kind="pool",
                                    fn_mode=fnm, fn_frac=fnf, fp_val=float(frac), fp_std=0.0))
    ref_names = ["pred", "truth+FP(current)"]
    all_names = ref_names + [s["name"] for s in schemes]

    metrics = {n: {k: [] for k in ("purity", "recall", "occ", "n_calo", "n_track")} for n in all_names}
    hist = {n: {"calo_e": np.zeros(len(calo_bins) - 1), "track_pt": np.zeros(len(track_bins) - 1)} for n in all_names}

    def _topk_calo(cand, node_e):
        e = torch.where(cand, node_e, torch.zeros_like(node_e))
        _, idx = e.topk(min(KCAL, e.shape[-1]), dim=-1, sorted=False)
        keep = torch.zeros_like(cand)
        keep.scatter_(1, idx, True)
        return keep & cand

    def _topk_mask(scores, k):
        _, idx = scores.topk(min(k, scores.shape[-1]), dim=-1)
        m = torch.zeros(scores.shape, dtype=torch.bool, device=device)
        m.scatter_(1, idx, True)
        return m

    def _rand(shape):
        return torch.rand(shape, device=device, generator=gen)

    def _record(name, calo_sel, track_sel, node_e, node_pt, calo_hs_e, calo_node):
        sce = (node_e * calo_sel).sum(1)
        shs = (calo_hs_e * calo_sel).sum(1)
        ths = (calo_hs_e * calo_node).sum(1)
        m = metrics[name]
        m["purity"].append((shs / sce.clamp_min(1e-6)).cpu().numpy())
        m["recall"].append((shs / ths.clamp_min(1e-6)).cpu().numpy())
        m["n_calo"].append(calo_sel.sum(1).float().cpu().numpy())
        m["n_track"].append(track_sel.sum(1).float().cpu().numpy())
        m["occ"].append((calo_sel | track_sel).sum(1).clamp_max(MAXN).float().cpu().numpy())
        ce = node_e[calo_sel].cpu().numpy()
        tp = node_pt[track_sel].cpu().numpy()
        hist[name]["calo_e"] += np.histogram(ce[ce > 0], bins=calo_bins)[0]
        hist[name]["track_pt"] += np.histogram(tp[tp > 0], bins=track_bins)[0]

    autocast = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if (use_amp and device == "cuda") else _nullcontext())

    print(f"Sweeping {len(schemes)} FP+FN schemes on {filepath} (device={device}, num_events={num_events})...")
    with torch.no_grad():
        for bi, (inputs, targets) in enumerate(loader):
            inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
            targets = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in targets.items()}
            with autocast:
                outputs = net(inputs, targets=targets, use_teacher_forcing=False)

            calo_prob = outputs["calo_final"]["calo_mask"]["calo_node_logit"].squeeze(1).float().sigmoid()
            track_prob = outputs["track_final"]["mask"]["pflow_node_logit"].squeeze(1).float().sigmoid()
            is_track = (inputs["node_is_track"].squeeze(-1) if inputs["node_is_track"].dim() > 2 else inputs["node_is_track"]).bool()
            valid = (inputs["node_valid"].squeeze(-1) if inputs["node_valid"].dim() > 2 else inputs["node_valid"]).bool()
            node_e = (inputs["node_e"].squeeze(-1) if inputs["node_e"].dim() > 2 else inputs["node_e"]).float()
            node_pt = (inputs["node_pt"].squeeze(-1) if inputs["node_pt"].dim() > 2 else inputs["node_pt"]).float()
            calo_hs_e = (targets["calo_hard_scatter_energy"].squeeze(-1) if targets["calo_hard_scatter_energy"].dim() > 2 else targets["calo_hard_scatter_energy"]).float()
            calo_hs_frac = (targets["calo_hard_scatter_energy_frac"].squeeze(-1) if targets["calo_hard_scatter_energy_frac"].dim() > 2 else targets["calo_hard_scatter_energy_frac"]).float()
            truth_tracks = (targets["tracks_mask"].squeeze(-1) if targets["tracks_mask"].dim() > 2 else targets["tracks_mask"]).bool()

            B, N = node_e.shape
            calo_node = (~is_track) & valid
            truth_calo = (calo_hs_frac > FT) & (calo_hs_e > ET) & calo_node
            pred_calo_cand = (calo_prob >= PT) & calo_node
            extra_pred = pred_calo_cand & ~truth_calo
            truth_track = truth_tracks & is_track
            pred_track = (track_prob >= 0.5) & is_track
            textra = pred_track & ~truth_track

            # Per-batch precompute reused across schemes.
            n_truth = truth_calo.sum(1).clamp_min(1).float()                 # (B,)
            e_rank = torch.where(truth_calo, node_e, torch.full_like(node_e, float("-inf")))
            order = e_rank.argsort(dim=1, descending=True)                   # high-E first
            ranks = torch.empty_like(order)
            ranks.scatter_(1, order, torch.arange(N, device=device).unsqueeze(0).expand(B, N))
            rand_calo = _rand((B, N))
            track_keep = _rand((B, N)) >= track_fn_frac
            # FP calo pool ranked by a random score → enables variable per-event FP counts.
            fp_rand = torch.where(extra_pred, _rand((B, N)), torch.full((B, N), -1.0, device=device))
            forder = fp_rand.argsort(dim=1, descending=True)
            fp_ranks = torch.empty_like(forder)
            fp_ranks.scatter_(1, forder, torch.arange(N, device=device).unsqueeze(0).expand(B, N))
            pool_size = extra_pred.sum(1).float()  # (B,) Stream-B FP pool size per event
            tfp_rand = torch.where(textra, _rand((B, N)), torch.full((B, N), -1.0, device=device))

            # tracks (NOT swept): drop track_fn_frac of true HS tracks and add ~track_fp_frac
            # of the per-event true-HS-track count as FP tracks (proportional, vectorized).
            n_truth_trk = truth_track.sum(1).float()
            torder = tfp_rand.argsort(dim=1, descending=True)
            tranks = torch.empty_like(torder)
            tranks.scatter_(1, torder, torch.arange(N, device=device).unsqueeze(0).expand(B, N))
            track_fp = textra & (tranks.float() < (track_fp_frac * n_truth_trk).unsqueeze(1))
            track_sel = (truth_track & track_keep) | track_fp

            # references
            _record("pred", _topk_calo(pred_calo_cand, node_e), pred_track, node_e, node_pt, calo_hs_e, calo_node)
            cur_fp = _topk_mask(fp_rand, NF) & extra_pred
            cur_tfp = _topk_mask(tfp_rand, NTF) & textra
            _record("truth+FP(current)", _topk_calo(truth_calo | cur_fp, node_e),
                    (truth_track | cur_tfp), node_e, node_pt, calo_hs_e, calo_node)

            for s in schemes:
                if s["fn_mode"] == "random":
                    kept = truth_calo & (rand_calo >= s["fn_frac"])
                else:  # energy_low: keep the top (1-frac) by energy, drop lowest-E
                    kept = truth_calo & (ranks.float() < ((1.0 - s["fn_frac"]) * n_truth).unsqueeze(1))
                # Per-event FP count: pool-fraction, Gaussian-spread, or fixed.
                if s["kind"] == "pool":
                    k_ev = (s["fp_val"] * pool_size).round()
                elif s["fp_std"] > 0:
                    k_ev = (torch.randn(B, device=device, generator=gen) * s["fp_std"] + s["fp_val"]).round()
                else:
                    k_ev = torch.full((B,), s["fp_val"], device=device)
                k_ev = k_ev.clamp(min=0.0).minimum(pool_size)
                fp = extra_pred & (fp_ranks.float() < k_ev.unsqueeze(1))
                _record(s["name"], _topk_calo(kept | fp, node_e), track_sel, node_e, node_pt, calo_hs_e, calo_node)

    # ── Score every scheme against pred ──────────────────────────────────
    def _js(p, q):
        p = p / max(p.sum(), 1e-12)
        q = q / max(q.sum(), 1e-12)
        m = 0.5 * (p + q)
        def _kl(a, b):
            mask = a > 0
            return float(np.sum(a[mask] * np.log(a[mask] / np.clip(b[mask], 1e-12, None))))
        return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)

    cat = {n: {k: np.concatenate(v) for k, v in metrics[n].items()} for n in all_names}
    pred_m = cat["pred"]
    pred_hc, pred_ht = hist["pred"]["calo_e"], hist["pred"]["track_pt"]
    LN2 = float(np.log(2))

    rows = []
    for n in all_names:
        m = cat[n]
        wsc = np.mean([
            wasserstein_distance(m[k], pred_m[k]) / (pred_m[k].std() + 1e-6)
            for k in ("purity", "recall", "n_calo", "n_track")
        ])
        js_c = _js(hist[n]["calo_e"], pred_hc) / LN2
        js_t = _js(hist[n]["track_pt"], pred_ht) / LN2
        score = wsc + js_c + js_t
        rows.append(dict(name=n, purity=m["purity"].mean(), recall=m["recall"].mean(),
                         occ=m["occ"].mean(), n_calo=m["n_calo"].mean(), n_track=m["n_track"].mean(),
                         w_scalar=wsc, js_calo=js_c, js_track=js_t, score=score))

    rankable = sorted((r for r in rows if r["name"] != "pred"), key=lambda r: r["score"])
    pred_row = next(r for r in rows if r["name"] == "pred")

    print("\n" + "=" * 118)
    print(f"FP+FN scheme sweep vs pred target   (n_events~{num_events})   [lower score = closer to pred]")
    print("=" * 118)
    hdr = f"{'rank':>4} {'scheme':<26} {'purity':>7} {'recall':>7} {'occ':>7} {'n_calo':>7} {'n_trk':>6} {'Wscal':>7} {'JScalo':>7} {'JStrk':>7} {'SCORE':>7}"
    print(hdr)
    print("-" * 118)
    print(f"{'--':>4} {'pred (TARGET)':<26} {pred_row['purity']:>7.3f} {pred_row['recall']:>7.3f} "
          f"{pred_row['occ']:>7.1f} {pred_row['n_calo']:>7.1f} {pred_row['n_track']:>6.1f} "
          f"{'-':>7} {'-':>7} {'-':>7} {'-':>7}")
    print("-" * 118)
    for i, r in enumerate(rankable):
        print(f"{i:>4} {r['name']:<26} {r['purity']:>7.3f} {r['recall']:>7.3f} {r['occ']:>7.1f} "
              f"{r['n_calo']:>7.1f} {r['n_track']:>6.1f} {r['w_scalar']:>7.3f} {r['js_calo']:>7.3f} "
              f"{r['js_track']:>7.3f} {r['score']:>7.3f}")
    print("=" * 118)
    best = rankable[0]
    print(f"BEST: {best['name']}  (score={best['score']:.3f})\n")

    plot_names = ["pred"] + [r["name"] for r in rankable[:top_k_plot]]
    if "truth+FP(current)" not in plot_names:
        plot_names.append("truth+FP(current)")
    saved = _plot_sweep(cat, hist, plot_names, calo_bins, track_bins,
                        out_dir=out_dir, ckpt_path=ckpt_path, max_reco_nodes=MAXN)
    for p in saved:
        print(f"Saved plot: {p}")
    print()
    return {"ranking": rankable, "pred": pred_row, "best": best, "plots": saved}


def _plot_sweep(cat, hist, plot_names, calo_bins, track_bins, out_dir=None, ckpt_path=None, max_reco_nodes=1400, prefix="tf_sweep") -> list:
    """Overlay the chosen schemes (pred bold black) on per-event metrics + energy spectra.

    Emits: one combined metrics overlay, one metrics figure PER scheme (each vs pred +
    current, with per-event std in the legend — to compare variances), and the spectra.
    """
    import re
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir) if out_dir is not None else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(ckpt_path).stem if ckpt_path is not None else "tf_sweep"
    saved: list = []
    others = [n for n in plot_names if n != "pred"]
    cmap = plt.get_cmap("turbo")
    color = {n: cmap(i / max(1, len(others) - 1)) for i, n in enumerate(others)}
    color["pred"] = "#000000"

    def style(n):
        return (3.0, "--") if n == "pred" else (1.7, "-")

    def _safe(n):
        return re.sub(r"[^0-9a-zA-Z]+", "_", n).strip("_")

    panels = [("purity", "calo purity", (0, 1)), ("recall", "HS energy recall", (0, 1.2)),
              ("occ", f"occupancy (max={max_reco_nodes})", None), ("n_calo", "selected calo nodes", None)]

    def _metric_fig(names, title, fname):
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        for ax, (key, xlabel, rng) in zip(axes.ravel(), panels):
            lo = rng[0] if rng else min(cat[n][key].min() for n in names)
            hi = rng[1] if rng else max(cat[n][key].max() for n in names)
            bins = np.linspace(lo, hi, 51)
            for n in names:
                lw, ls = style(n)
                ax.hist(cat[n][key], bins=bins, histtype="step", linewidth=lw, linestyle=ls,
                        density=True, color=color.get(n, "#888888"),
                        label=f"{n}  μ={cat[n][key].mean():.2f} σ={cat[n][key].std():.1f}")
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel("density")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.25)
        fig.suptitle(title, fontsize=12)
        fig.tight_layout()
        fpath = out_dir / fname
        fig.savefig(fpath, dpi=120)
        plt.close(fig)
        saved.append(str(fpath))

    # Figure 1: combined overlay of all plotted schemes
    _metric_fig(plot_names, "per-event metrics (vs pred)",
                f"{prefix}_metrics__{stem}.png")

    # One metrics figure PER scheme (each vs pred + current) — clean comparison
    for n in others:
        names = ["pred", n] + (["truth+FP(current)"] if "truth+FP(current)" in cat and n != "truth+FP(current)" else [])
        _metric_fig(names, f"{n}  vs pred", f"{prefix}_metrics__{stem}__{_safe(n)}.png")

    # Figure 2: selected-node energy spectra (from accumulated histograms)
    cc = 0.5 * (calo_bins[:-1] + calo_bins[1:])
    tc = 0.5 * (track_bins[:-1] + track_bins[1:])
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, (hkey, centers, xlabel) in zip(axes, [("calo_e", cc, "selected calo cluster energy [GeV]"),
                                                  ("track_pt", tc, "selected track pT")]):
        for n in plot_names:
            h = hist[n][hkey]
            tot = h.sum()
            if tot <= 0:
                continue
            lw, ls = style(n)
            ax.step(centers, h / tot, where="mid", linewidth=lw, linestyle=ls, color=color[n], label=n)
        ax.set_xscale("log")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("fraction")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.25)
    fig.suptitle("selected-node energy spectra (vs pred)", fontsize=12)
    fig.tight_layout()
    f2 = out_dir / f"{prefix}_energy_spectra__{stem}.png"
    fig.savefig(f2, dpi=120)
    plt.close(fig)
    saved.append(str(f2))
    return saved


def diagnose_track_class_attribution(
    ckpt_path: str | Path,
    config_path: str | Path,
    data_dir: str | None = None,
    files: list[str | Path] | None = None,
    num_events: int = 1000,
    batch_size: int = 32,
    num_workers: int = 1,
    device: str | None = None,
    use_amp: bool = True,
) -> dict:
    """Idea 1 — factorial track/calo ablation of the reco classification gap.

    Runs the same events through 4 reco-selection conditionings and reports the
    classification cross-entropy (and total reco loss + mask loss) for each:

        run   calo source   track source
        A     TF (truth)    TF (truth)        -> reproduces the teacher-forcing CE
        B     TF (truth)    pred              -> only tracks switched to inference
        C     pred          TF (truth)        -> only calo switched to inference
        D     pred          pred              -> reproduces the inference CE

    Attribution:
        gap_from_tracks = CE(B) - CE(A)
        gap_from_calo   = CE(C) - CE(A)
        total_gap       = CE(D) - CE(A)   (check additivity: tracks+calo ~ total)

    If CE(B) >> CE(A) (and ~ CE(D)) the classification gap is driven by the track
    selection — supporting the "bad low-energy track prediction" hypothesis.
    """
    import yaml
    import torch
    from torch.utils.data import DataLoader

    from hepattn.experiments.odd_pileup_reco.lightning_module import ODDPFlowTwoStream
    from hepattn.experiments.odd_pileup_reco.pflow_data import EagerODDDataset

    ckpt_path, config_path = Path(ckpt_path), Path(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg["data"]

    if files is not None:
        if len(files) == 0:
            raise ValueError("files must be a non-empty list when provided")
        files_list = [Path(f) for f in files]
        filepath = str(files_list[0].parent)
    elif data_dir is not None:
        files_list, filepath = None, data_dir
    else:
        files_list, filepath = None, data_cfg.get("test_filepath", data_cfg.get("filepath"))

    dataset = EagerODDDataset(
        filepath=filepath, num_events=num_events, files_list=files_list,
        inputs=data_cfg["inputs"], targets=data_cfg["targets"], scale_dict_path=data_cfg["scale_dict_path"],
        num_objects=data_cfg.get("num_objects", 400), max_nodes=data_cfg["max_nodes"],
        incidence_cutval=data_cfg.get("incidence_cutval", 0.01), is_inference=True,
        hard_scatter_energy_threshold=data_cfg.get("hard_scatter_energy_threshold", 0.03),
        window_size=data_cfg.get("window_size", 512), compute_deltaR_stats=False,
    )
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                        shuffle=False, collate_fn=None, pin_memory=True)

    model = ODDPFlowTwoStream.load_from_checkpoint(str(ckpt_path), map_location="cpu")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    net = model.model

    # (name, calo_label, track_label, track_tf, calo_tf)
    combos = [
        ("A", "TF",   "TF",   True,  True),
        ("B", "TF",   "pred", False, True),
        ("C", "pred", "TF",   True,  False),
        ("D", "pred", "pred", False, False),
    ]
    acc = {c[0]: {"cls": 0.0, "mask": 0.0, "reco": 0.0} for c in combos}
    n_events = 0

    def _reco_task_loss(losses, task):
        s = 0.0
        for ln, ll in losses.items():
            if ln.startswith("reco_") and task in ll:
                for v in ll[task].values():
                    if isinstance(v, torch.Tensor):
                        s += float(v.detach().float().cpu())
        return s

    def _reco_total(losses):
        s = 0.0
        for ln, ll in losses.items():
            if ln.startswith("reco_"):
                for tl in ll.values():
                    for v in tl.values():
                        if isinstance(v, torch.Tensor):
                            s += float(v.detach().float().cpu())
        return s

    autocast = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if (use_amp and device == "cuda") else _nullcontext())

    print(f"Factorial track/calo ablation on {filepath} (device={device}, num_events={num_events})...")
    with torch.no_grad():
        for bi, (inputs, targets) in enumerate(loader):
            inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
            targets = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in targets.items()}
            n = next(v.shape[0] for v in targets.values() if isinstance(v, torch.Tensor))
            for name, _cl, _tl, track_tf, calo_tf in combos:
                with autocast:
                    outputs = net(inputs, targets=targets, track_tf=track_tf, calo_tf=calo_tf)
                    losses = net.loss(outputs, targets)
                acc[name]["cls"] += _reco_task_loss(losses, "classification") * n
                acc[name]["mask"] += _reco_task_loss(losses, "mask") * n
                acc[name]["reco"] += _reco_total(losses) * n
            n_events += n
            print(f"  batch {bi:3d}  n={n:3d}  "
                  f"A={acc['A']['cls']/n_events:.3f}  B={acc['B']['cls']/n_events:.3f}  "
                  f"C={acc['C']['cls']/n_events:.3f}  D={acc['D']['cls']/n_events:.3f}")

    means = {k: {m: v / n_events for m, v in d.items()} for k, d in acc.items()}

    print("\n" + "=" * 74)
    print(f"Factorial track/calo ablation   (n_events={n_events})")
    print("=" * 74)
    print(f"{'run':>4} {'calo':>6} {'track':>6} {'cls_CE':>10} {'mask':>10} {'reco_tot':>10}")
    print("-" * 74)
    for name, cl, tl, _t, _c in combos:
        m = means[name]
        print(f"{name:>4} {cl:>6} {tl:>6} {m['cls']:>10.4f} {m['mask']:>10.4f} {m['reco']:>10.4f}")
    print("-" * 74)
    a = means["A"]["cls"]
    gap_tracks = means["B"]["cls"] - a
    gap_calo = means["C"]["cls"] - a
    total_gap = means["D"]["cls"] - a
    print(f"classification CE gap (vs A=TF/TF):")
    print(f"  from tracks (B-A): {gap_tracks:+.4f}")
    print(f"  from calo   (C-A): {gap_calo:+.4f}")
    print(f"  total       (D-A): {total_gap:+.4f}   (additivity: tracks+calo = {gap_tracks + gap_calo:+.4f})")
    if total_gap > 1e-6:
        print(f"  -> tracks explain {100 * gap_tracks / total_gap:.0f}% of the gap, "
              f"calo {100 * gap_calo / total_gap:.0f}%")
    print("=" * 74 + "\n")

    return {"means": means, "gap_tracks": gap_tracks, "gap_calo": gap_calo, "total_gap": total_gap}


def verify_tf_selection(
    ckpt_path: str | Path,
    config_path: str | Path,
    data_dir: str | None = None,
    files: list[str | Path] | None = None,
    num_events: int = 2000,
    batch_size: int = 32,
    num_workers: int = 1,
    device: str | None = None,
    use_amp: bool = True,
    out_dir: str | None = None,
) -> dict:
    """Verify the model's *actual* teacher-forcing selection reproduces the pred distribution.

    Unlike sweep_tf_mask_schemes (which re-implements candidate selections), this runs the real
    model twice — use_teacher_forcing=True (the in-model FN+FP recipe from config) and the pred
    path — and reads the model's stored selection (net._reco_node_indices / _reco_filtered_valid)
    to measure per-event purity / recall / occupancy / n_calo / n_track and the selected-node
    energy spectra (calo E, track pT). Overlays "TF (model)" vs "pred" in the sweep plot style.
    """
    import yaml
    import torch
    import numpy as np
    from torch.utils.data import DataLoader

    from hepattn.experiments.odd_pileup_reco.lightning_module import ODDPFlowTwoStream
    from hepattn.experiments.odd_pileup_reco.pflow_data import EagerODDDataset

    ckpt_path, config_path = Path(ckpt_path), Path(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg["data"]

    if files is not None:
        if len(files) == 0:
            raise ValueError("files must be a non-empty list when provided")
        files_list = [Path(f) for f in files]
        filepath = str(files_list[0].parent)
    elif data_dir is not None:
        files_list, filepath = None, data_dir
    else:
        files_list, filepath = None, data_cfg.get("test_filepath", data_cfg.get("filepath"))

    dataset = EagerODDDataset(
        filepath=filepath, num_events=num_events, files_list=files_list,
        inputs=data_cfg["inputs"], targets=data_cfg["targets"], scale_dict_path=data_cfg["scale_dict_path"],
        num_objects=data_cfg.get("num_objects", 400), max_nodes=data_cfg["max_nodes"],
        incidence_cutval=data_cfg.get("incidence_cutval", 0.01), is_inference=True,
        hard_scatter_energy_threshold=data_cfg.get("hard_scatter_energy_threshold", 0.03),
        window_size=data_cfg.get("window_size", 512), compute_deltaR_stats=False,
    )
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                        shuffle=False, collate_fn=None, pin_memory=True)

    model = ODDPFlowTwoStream.load_from_checkpoint(str(ckpt_path), map_location="cpu")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    net = model.model

    # load_from_checkpoint restores the *checkpoint's* recipe knobs. Override them with the
    # current config so we verify the NEW teacher-forcing recipe (not the old saved one).
    recipe_keys = ["reco_calo_noise_mean", "reco_calo_noise_std", "reco_calo_fn_frac",
                   "reco_track_fn_frac", "reco_track_fp_frac"]
    model_init = cfg.get("model", {}).get("model", {}).get("init_args", {})
    print("Applying TF recipe from config:")
    for k in recipe_keys:
        if k in model_init and hasattr(net, k):
            setattr(net, k, model_init[k])
        print(f"  {k} = {getattr(net, k, None)}")

    calo_bins = np.logspace(-2, 3, 61)
    track_bins = np.logspace(-2, 3, 61)
    names = ["pred", "TF (model)"]
    metrics = {n: {k: [] for k in ("purity", "recall", "occ", "n_calo", "n_track")} for n in names}
    hist = {n: {"calo_e": np.zeros(len(calo_bins) - 1), "track_pt": np.zeros(len(track_bins) - 1)} for n in names}

    def _sq(t):
        return t.squeeze(-1) if (t is not None and t.dim() > 2) else t

    def _record(name, is_track, node_e, node_pt, calo_hs_e, node_valid):
        # Read the model's actual reco selection stored during the last forward.
        idx = net._reco_node_indices                  # (B, R)
        valid = net._reco_filtered_valid.bool()        # (B, R)
        it = is_track.gather(1, idx).bool()
        e = node_e.gather(1, idx).float()
        pt = node_pt.gather(1, idx).float()
        hs = calo_hs_e.gather(1, idx).float()
        calo_sel = valid & (~it)
        track_sel = valid & it
        sce = (e * calo_sel).sum(1)
        shs = (hs * calo_sel).sum(1)
        ths = (calo_hs_e * ((~is_track) & node_valid)).sum(1)
        m = metrics[name]
        m["purity"].append((shs / sce.clamp_min(1e-6)).cpu().numpy())
        m["recall"].append((shs / ths.clamp_min(1e-6)).cpu().numpy())
        m["occ"].append(valid.sum(1).float().cpu().numpy())
        m["n_calo"].append(calo_sel.sum(1).float().cpu().numpy())
        m["n_track"].append(track_sel.sum(1).float().cpu().numpy())
        ce = e[calo_sel].cpu().numpy()
        tp = pt[track_sel].cpu().numpy()
        hist[name]["calo_e"] += np.histogram(ce[ce > 0], bins=calo_bins)[0]
        hist[name]["track_pt"] += np.histogram(tp[tp > 0], bins=track_bins)[0]

    autocast = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if (use_amp and device == "cuda") else _nullcontext())

    print(f"Verifying model TF selection vs pred on {filepath} (device={device}, num_events={num_events})...")
    with torch.no_grad():
        for bi, (inputs, targets) in enumerate(loader):
            inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
            targets = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in targets.items()}
            is_track = _sq(inputs["node_is_track"]).bool()
            node_valid = _sq(inputs["node_valid"]).bool()
            node_e = _sq(inputs["node_e"]).float()
            node_pt = _sq(inputs["node_pt"]).float()
            calo_hs_e = _sq(targets["calo_hard_scatter_energy"]).float()

            with autocast:
                net(inputs, targets=targets, use_teacher_forcing=False)
            _record("pred", is_track, node_e, node_pt, calo_hs_e, node_valid)
            with autocast:
                net(inputs, targets=targets, use_teacher_forcing=True)
            _record("TF (model)", is_track, node_e, node_pt, calo_hs_e, node_valid)

    cat = {n: {k: np.concatenate(v) for k, v in metrics[n].items()} for n in names}

    print("\n" + "=" * 70)
    print(f"Model TF selection vs pred   (n_events~{num_events})")
    print("=" * 70)
    print(f"{'source':<14} {'purity':>9} {'recall':>9} {'occ':>9} {'n_calo':>9} {'n_track':>9}")
    print("-" * 70)
    for n in names:
        c = cat[n]
        print(f"{n:<14} {c['purity'].mean():>9.3f} {c['recall'].mean():>9.3f} "
              f"{c['occ'].mean():>9.1f} {c['n_calo'].mean():>9.1f} {c['n_track'].mean():>9.1f}")
    print("=" * 70)
    print("Goal: 'TF (model)' should sit on 'pred' (the in-model recipe reproduces inference).\n")

    saved = _plot_sweep(cat, hist, names, calo_bins, track_bins,
                        out_dir=out_dir, ckpt_path=ckpt_path, max_reco_nodes=net.max_reco_nodes,
                        prefix="tf_verify")
    for p in saved:
        print(f"Saved plot: {p}")
    print()
    return {"means": {n: {k: float(cat[n][k].mean()) for k in cat[n]} for n in names}, "plots": saved}


class _nullcontext:
    """Minimal no-op context manager (avoids importing contextlib for one use)."""
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
