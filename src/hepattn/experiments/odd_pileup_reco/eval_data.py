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

def load_eval_data_from_h5(
    h5_path: str | Path,
    calo_pred_threshold: float = CALO_PRED_THRESHOLD,
    calo_hs_frac_threshold: float = CALO_HS_FRAC_THRESHOLD,
    calo_hs_energy_threshold: float = CALO_HS_ENERGY_THRESHOLD,
    event_start: int | None = None,
    event_stop: int | None = None,
    event_indices: list[int] | np.ndarray | None = None,
) -> tuple[dict, dict, dict]:
    """Load prediction writer H5 and return (track_data, cluster_data, reco_data).

    The returned dicts have the same keys/shapes that ``make_plots()`` in
    ``eval_plots.py`` expects.

    Parameters
    ----------
    h5_path : path to the prediction writer output H5 file
    calo_pred_threshold : binary threshold for calo mask predictions
    calo_hs_frac_threshold : truth HS fraction threshold for mask truth
    calo_hs_energy_threshold : truth HS energy threshold for mask truth
    event_start : optional inclusive start index for contiguous event slicing
    event_stop : optional exclusive stop index for contiguous event slicing
    event_indices : optional explicit event indices (strictly increasing)

    Returns
    -------
    track_data : dict  (empty if node_metadata absent)
    cluster_data : dict  (empty if node_metadata absent)
    reco_data : dict  (always populated if object_class + regression exist)
    """
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        n_events = _infer_n_events(f)
        event_sel = _build_event_selector(
            n_events=n_events,
            event_start=event_start,
            event_stop=event_stop,
            event_indices=event_indices,
        )

        reco_data = _load_reco(f, event_sel=event_sel)
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
        if "node_pt" in nm.dtype.names:
            reco_data["node_pt"] = _read_struct_field(nm, "node_pt", np.float32, event_sel)

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
) -> Path:
    """Run Lightning test step with PflowPredictionWriter, return H5 path.

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

    Returns
    -------
    Path to the generated H5 file
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

    # Create prediction writer callback
    writer = PflowPredictionWriter()

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

    h5_path = writer.output_path
    print(f"Predictions written to {h5_path}")
    return h5_path
