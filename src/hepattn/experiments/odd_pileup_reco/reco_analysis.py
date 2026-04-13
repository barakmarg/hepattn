"""Particle-level reconstruction analysis from prediction writer H5 files.

Adapted from ``odd_old.py`` (the ODD non-pileup analysis notebook) and extended
to support both H5 formats:

* **ODD (non-pileup)** — single-stream, 650-node space:
  - ``incidence``           : (N, 400, 650) combined truth+pred fields
  - ``object_masks``        : (N, 400, 650) combined truth+pred fields
  - ``regression``          : (N, 400) with truth/pred/proxy kinematic fields

* **ODD pileup-reco** — three-stream, mixed node spaces:
  - ``truth_incidence``     : (N, 400, 5500) — full node space
  - ``pred_incidence``      : (N, 400, 1400) — filtered node space
  - ``reco_node_indices``   : (N, 1400) — filtered→full mapping
  - ``regression``          : (N, 400) same schema

Usage::

    from hepattn.experiments.odd_pileup_reco.reco_analysis import (
        load_pflow_data, run_reco_analysis,
    )

    # From any H5 file (auto-detects format)
    data = load_pflow_data("epoch=042__test.h5")
    figs = run_reco_analysis(data)

    # From eval_data dicts (skip re-loading)
    from hepattn.experiments.odd_pileup_reco.reco_analysis import pflow_data_from_eval_dicts
    data = pflow_data_from_eval_dicts(reco_data)
    figs = run_reco_analysis(data)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, LogNorm

# ── Constants ─────────────────────────────────────────────────────────────
NUM_CLASSES = 6
CLASS_LABELS = ["Ch Had", r"$e$", r"$\mu$", "Neu Had", r"$\gamma$", "null"]
CLASS_COLORS = ["red", "blue", "green", "orange", "purple"]


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_pflow_data(
    h5_path: str | Path,
    eta_cut: float = 4.0,
    event_start: int | None = None,
    event_stop: int | None = None,
    load_incidence: bool = False,
) -> dict:
    """Load prediction writer H5 into a unified PFlow analysis dict.

    Auto-detects ODD (non-pileup, 650-node) vs ODD pileup-reco (5500/1400-node)
    format from the H5 structure.

    Parameters
    ----------
    h5_path : path to the prediction writer H5 file
    eta_cut : |eta| < eta_cut for the valid-particle indicator
    event_start : inclusive start event index (default: 0)
    event_stop  : exclusive stop event index (default: all events)
    load_incidence : whether to load incidence matrices (can be several GB)

    Returns
    -------
    dict with keys:
        pflow_class      (N, P) int64
        truth_class      (N, P) int64
        pflow_ptetaphi   (N, P, 3) float32  — [pt, eta, phi]
        truth_ptetaphi   (N, P, 3) float32
        proxy_ptetaphi   (N, P, 3) float32 or None
        pflow_indicator  (N, P) bool
        truth_indicator  (N, P) bool
        pflow_data       (N, P, 5) float32  — [E, pt, eta, sinphi, cosphi]
        truth_data       (N, P, 5) float32
        proxy_data       (N, P, 5) float32 or None
        pflow_incidence  (N, P, nodes) float32 or None  (if load_incidence)
        truth_incidence  (N, P, nodes) float32 or None  (if load_incidence)
        pflow_incidence_filtered (N, P, 1400) float32 or None (if available)
        reco_node_indices (N, 1400) int64 or None
        reco_is_track     (N, 1400) bool or None
    """
    import h5py

    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        n_events_total = f["object_class"].shape[0]
        start = 0 if event_start is None else max(0, int(event_start))
        stop = n_events_total if event_stop is None else min(n_events_total, int(event_stop))
        sel = slice(start, stop)

        # -- Detect format --
        is_pileup_reco = "truth_incidence" in f and "pred_incidence" in f and "incidence" not in f

        # -- Classes --
        truth_class = f["object_class"]["object_class"][sel].astype(np.int64)
        pflow_class = f["object_class"]["pflow_class"][sel].astype(np.int64)

        # -- Regression --
        truth_data = _read_regression(f["regression"], "truth", sel)
        pflow_data  = _read_regression(f["regression"], "pred",  sel)
        proxy_data  = _read_regression(f["regression"], "proxy", sel)

        # -- ptetaphi --
        truth_ptetaphi = _to_ptetaphi(truth_data)
        pflow_ptetaphi = _to_ptetaphi(pflow_data)
        proxy_ptetaphi = _to_ptetaphi(proxy_data) if proxy_data is not None else None

        # Neutral pt correction: pt = E / cosh(eta)
        _apply_neutral_pt_correction(pflow_ptetaphi, pflow_data, pflow_class)
        if proxy_ptetaphi is not None:
            _apply_neutral_pt_correction(proxy_ptetaphi, proxy_data, pflow_class)

        # -- Indicators --
        pflow_indicator = (pflow_class < (NUM_CLASSES - 1)) & (np.abs(pflow_ptetaphi[..., 1]) < eta_cut)
        truth_indicator = (truth_class < (NUM_CLASSES - 1)) & (np.abs(truth_ptetaphi[..., 1]) < eta_cut)

        # -- Reco-node mapping + metadata --
        reco_node_indices = None
        reco_is_track = None
        if "reco_node_indices" in f:
            idx_ds = f["reco_node_indices"]
            idx_field = idx_ds.dtype.names[0] if idx_ds.dtype.names else None
            reco_node_indices = (
                idx_ds[idx_field][sel] if idx_field is not None else idx_ds[sel]
            ).astype(np.int64)

        if "node_metadata" in f and reco_node_indices is not None:
            nm = f["node_metadata"]
            if "node_is_track" in nm.dtype.names:
                node_is_track = nm["node_is_track"][sel].astype(bool)
                if node_is_track.ndim == 3 and node_is_track.shape[-1] == 1:
                    node_is_track = node_is_track[..., 0]
                max_node = node_is_track.shape[1] - 1
                safe_idx = np.clip(reco_node_indices, 0, max_node)
                reco_is_track = np.take_along_axis(node_is_track, safe_idx, axis=1)
                invalid_idx = (reco_node_indices < 0) | (reco_node_indices > max_node)
                reco_is_track[invalid_idx] = False

        # -- Incidence --
        pflow_incidence = truth_incidence = pflow_incidence_filtered = None
        if load_incidence:
            if is_pileup_reco:
                # Separate datasets in different node spaces; keep both in native space
                truth_incidence = f["truth_incidence"]["truth_incidence"][sel].astype(np.float32)
                pred_inc_filt   = f["pred_incidence"]["pred_incidence"][sel].astype(np.float32)
                pflow_incidence_filtered = pred_inc_filt
                # Expand pred from filtered (1400) to full (5500) node space
                if "reco_node_indices" in f:
                    reco_idx = f["reco_node_indices"]["index"][sel].astype(np.int64)  # (N, 1400)
                    N, P, n_filt = pred_inc_filt.shape
                    n_full = truth_incidence.shape[2]
                    pflow_incidence = np.zeros((N, P, n_full), dtype=np.float32)
                    for evt in range(N):
                        pflow_incidence[evt, :, reco_idx[evt]] = pred_inc_filt[evt]
                else:
                    pflow_incidence = pred_inc_filt
            else:
                # ODD format: single dataset with compound dtype, same node space
                inc_ds = f["incidence"]
                truth_incidence = inc_ds["truth_incidence"][sel].astype(np.float32)
                # mask_logits are raw logits — convert to probabilities via sigmoid
                pflow_incidence = _sigmoid(inc_ds["pred_incidence"][sel].astype(np.float32))
                pflow_incidence_filtered = pflow_incidence

    return {
        "pflow_class":      pflow_class,
        "truth_class":      truth_class,
        "pflow_ptetaphi":   pflow_ptetaphi,
        "truth_ptetaphi":   truth_ptetaphi,
        "proxy_ptetaphi":   proxy_ptetaphi,
        "pflow_indicator":  pflow_indicator,
        "truth_indicator":  truth_indicator,
        "pflow_data":       pflow_data,
        "truth_data":       truth_data,
        "proxy_data":       proxy_data,
        "pflow_incidence":  pflow_incidence,
        "pflow_incidence_filtered": pflow_incidence_filtered,
        "truth_incidence":  truth_incidence,
        "reco_node_indices": reco_node_indices,
        "reco_is_track": reco_is_track,
    }


def pflow_data_from_eval_dicts(reco_data: dict, eta_cut: float = 4.0) -> dict:
    """Convert reco_data dict from ``load_eval_data_from_h5`` to PFlow analysis format.

    Bridges ``eval_data.py`` → ``reco_analysis.py`` so callers that already have
    the eval dicts in memory don't need to re-open the H5 file.
    """
    truth_class = reco_data["truth_class"].astype(np.int64)
    pred_class  = reco_data["pred_class"].astype(np.int64)

    truth_ptetaphi = np.stack([
        reco_data["truth_pt"],
        reco_data["truth_eta"],
        np.arctan2(reco_data["truth_sinphi"], reco_data["truth_cosphi"]),
    ], axis=-1).astype(np.float32)

    pflow_ptetaphi = np.stack([
        reco_data["pred_pt"],
        reco_data["pred_eta"],
        np.arctan2(reco_data["pred_sinphi"], reco_data["pred_cosphi"]),
    ], axis=-1).astype(np.float32)

    pflow_indicator = (pred_class  < (NUM_CLASSES - 1)) & (np.abs(pflow_ptetaphi[..., 1]) < eta_cut)
    truth_indicator = (truth_class < (NUM_CLASSES - 1)) & (np.abs(truth_ptetaphi[..., 1]) < eta_cut)

    # Pull incidence tensors if the updated eval_data stored them
    pflow_incidence = reco_data.get("pred_incidence")
    truth_incidence = reco_data.get("truth_incidence")
    reco_node_indices = reco_data.get("reco_node_indices")
    reco_is_track = reco_data.get("reco_is_track")

    return {
        "pflow_class":     pred_class,
        "truth_class":     truth_class,
        "pflow_ptetaphi":  pflow_ptetaphi,
        "truth_ptetaphi":  truth_ptetaphi,
        "proxy_ptetaphi":  None,
        "pflow_indicator": pflow_indicator,
        "truth_indicator": truth_indicator,
        "pflow_data":      None,
        "truth_data":      None,
        "proxy_data":      None,
        "pflow_incidence": pflow_incidence,
        "pflow_incidence_filtered": pflow_incidence,
        "truth_incidence": truth_incidence,
        "reco_node_indices": reco_node_indices,
        "reco_is_track": reco_is_track,
    }


# ── Private helpers ────────────────────────────────────────────────────────

def _read_regression(reg_ds, prefix: str, sel) -> np.ndarray | None:
    """Read E, pt, eta, sinphi, cosphi fields for a given prefix."""
    keys = [f"{prefix}_{v}" for v in ("e", "pt", "eta", "sinphi", "cosphi")]
    if not all(k in reg_ds.dtype.names for k in keys):
        return None
    return np.stack([reg_ds[k][sel].astype(np.float32) for k in keys], axis=-1)


def _to_ptetaphi(data: np.ndarray) -> np.ndarray:
    """(…, 5)[E, pt, eta, sinphi, cosphi] → (…, 3)[pt, eta, phi]."""
    return np.stack([
        data[..., 1],
        data[..., 2],
        np.arctan2(data[..., 3], data[..., 4]),
    ], axis=-1).astype(np.float32)


def _apply_neutral_pt_correction(ptetaphi: np.ndarray, raw: np.ndarray, pclass: np.ndarray) -> None:
    """For neutral particles (class 3, 4): pt = E / cosh(eta)."""
    mask = (pclass >= 3) & (pclass < (NUM_CLASSES - 1))
    if mask.any():
        eta = ptetaphi[mask, 1]
        ptetaphi[mask, 0] = raw[mask, 0] / np.cosh(np.clip(eta, -10, 10))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64))).astype(np.float32)


import torch
from hepattn.utils.masks import topk_attn

def select_charged_tracks(incidence, is_track):
    """
    Selects the best track for each charged particle based on incidence scores,
    ensuring a one-to-one assignment where possible, following the logic from get_proxy_features.

    Args:
        incidence (torch.Tensor): Incidence matrix of shape (batch, n_particles, n_nodes).
        is_track (torch.Tensor): Boolean mask indicating tracks, shape (batch, n_nodes).

    Returns:
        torch.Tensor: Selected charged incidence matrix of shape (batch, n_particles, n_nodes),
                      with 1s for selected track-particle associations and 0s elsewhere.
    """
    # Multiply incidence by is_track to focus on tracks only
    charged_inc = incidence * is_track.unsqueeze(1)  # Shape: (batch, n_particles, n_nodes)

    # Use the most weighted track as proxy for charged particles
    charged_inc_top2 = (topk_attn(charged_inc, 2, dim=-2) & (charged_inc > 0)).float()
    charged_inc_max = charged_inc.max(-2, keepdim=True)[0]
    charged_inc_new = (charged_inc == charged_inc_max) & (charged_inc > 0)

    # ------------------------
    particle_max_idx = charged_inc.argmax(dim=-1, keepdim=True)
    # Create a mask that is True only at that specific index
    is_first_max_particle = torch.zeros_like(charged_inc, dtype=torch.bool).scatter_(-1, particle_max_idx, True)
    # Apply the filter
    charged_inc_new = charged_inc_new & is_first_max_particle
    # ---------------------
    # TODO: check this
    # charged_inc_new = charged_inc.float()
    zero_track_mask = charged_inc_new.sum(-1, keepdim=True) == 0
    charged_inc = torch.where(zero_track_mask, charged_inc_top2, charged_inc_new)

    # -------------------------
    # --- ADDED: Final Cleanup (Fixes the Top2/Recovery duplicates) ---
    # 1. Look at the incidence scores ONLY for the tracks we have currently selected
    current_scores = incidence * charged_inc
    # 2. Find the single best track among the selected ones
    final_best_idx = current_scores.argmax(dim=-1, keepdim=True)
    # 3. Create a strict mask for that one track
    final_strict_mask = torch.zeros_like(charged_inc, dtype=torch.bool).scatter_(-1, final_best_idx, True)
    # 4. Apply the mask. 
    # Note: If charged_inc was all zeros, intersection with final_strict_mask remains zeros.
    charged_inc = charged_inc * final_strict_mask.float()

    return charged_inc



def _has_proxy(data: dict) -> bool:
    p = data.get("proxy_ptetaphi")
    return p is not None and not np.allclose(p, 0)


def normalize_phi(phi: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(phi), np.cos(phi))


# ═══════════════════════════════════════════════════════════════════════════
# Particle-level plots
# ═══════════════════════════════════════════════════════════════════════════

def plot_class_distribution(data: dict, ind_threshold: float = 0.5) -> plt.Figure:
    """Histogram of particle class index (truth vs PFlow)."""
    truth_flat  = data["truth_class"].ravel()
    pflow_flat  = data["pflow_class"].ravel()
    truth_ind   = data["truth_indicator"].ravel()
    pflow_ind   = data["pflow_indicator"].ravel()

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.arange(-0.5, NUM_CLASSES + 0.5)
    ax.hist(truth_flat[truth_ind > ind_threshold], bins=bins,
            histtype="stepfilled", alpha=0.5, label="Truth", density=True)
    ax.hist(pflow_flat[pflow_ind > ind_threshold], bins=bins,
            histtype="step", label="PFlow", density=True)
    ax.set_xlabel("Class")
    ax.set_xticks(np.arange(NUM_CLASSES - 1))
    ax.set_xticklabels(CLASS_LABELS[:-1])
    ax.legend()
    ax.set_title("Particle class distribution")
    fig.tight_layout()
    return fig


def plot_n_particles(data: dict, ind_threshold: float = 0.5) -> plt.Figure:
    """Per-event particle multiplicity histogram."""
    n_events = len(data["pflow_indicator"])
    n_pflow = (data["pflow_indicator"] > ind_threshold).sum(axis=-1)
    n_truth = (data["truth_indicator"] > ind_threshold).sum(axis=-1)

    top = max(n_truth.max(), n_pflow.max()) * 1.15
    bins = np.linspace(0, top, min(80, int(top) + 2))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(n_truth, bins=bins, histtype="stepfilled", alpha=0.5, label="Truth")
    ax.hist(n_pflow, bins=bins, histtype="step", label="PFlow")
    if _has_proxy(data):
        n_proxy = (data["pflow_indicator"] > ind_threshold).sum(axis=-1)
        ax.hist(n_proxy, bins=bins, histtype="step", linestyle="--", label="Proxy")
    ax.legend()
    ax.set_xlabel("Number of particles per event")
    ax.set_title("Particle multiplicity")
    fig.tight_layout()
    return fig


def plot_n_particles_by_pt_bin(
    data: dict,
    ind_threshold: float = 0.5,
    pt_bins: tuple[float, ...] = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 200.0),
) -> plt.Figure:
    """Per-event particle multiplicity histograms split by particle-pt bins."""
    if len(pt_bins) < 2:
        raise ValueError("pt_bins must contain at least two edges")

    truth_pt = data["truth_ptetaphi"][..., 0]
    pflow_pt = data["pflow_ptetaphi"][..., 0]
    truth_valid = data["truth_indicator"] > ind_threshold
    pflow_valid = data["pflow_indicator"] > ind_threshold

    has_proxy = _has_proxy(data)
    proxy_pt = data["proxy_ptetaphi"][..., 0] if has_proxy else None

    n_panels = len(pt_bins) - 1
    n_cols = 2 if n_panels > 1 else 1
    n_rows = int(np.ceil(n_panels / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 3.8 * n_rows), squeeze=False)
    axes_flat = axes.ravel()

    for i in range(n_panels):
        lo = float(pt_bins[i])
        hi = float(pt_bins[i + 1])
        ax = axes_flat[i]

        truth_counts = (truth_valid & (truth_pt >= lo) & (truth_pt < hi)).sum(axis=-1)
        pflow_counts = (pflow_valid & (pflow_pt >= lo) & (pflow_pt < hi)).sum(axis=-1)

        series_max = [
            int(truth_counts.max()) if truth_counts.size else 0,
            int(pflow_counts.max()) if pflow_counts.size else 0,
        ]
        proxy_counts = None
        if has_proxy and proxy_pt is not None:
            proxy_counts = (pflow_valid & (proxy_pt >= lo) & (proxy_pt < hi)).sum(axis=-1)
            series_max.append(int(proxy_counts.max()) if proxy_counts.size else 0)

        max_count = max(1, *series_max)
        bins = np.arange(-0.5, max_count + 1.5, 1.0)

        ax.hist(truth_counts, bins=bins, histtype="stepfilled", alpha=0.5, label="Truth")
        ax.hist(pflow_counts, bins=bins, histtype="step", label="PFlow")
        if proxy_counts is not None:
            ax.hist(proxy_counts, bins=bins, histtype="step", linestyle="--", label="Proxy")

        ax.set_title(f"{lo:g} <= pt < {hi:g} GeV")
        ax.set_xlabel("Particles / event")
        ax.set_ylabel("Events")
        ax.legend(fontsize=8)

    for ax in axes_flat[n_panels:]:
        ax.axis("off")

    fig.suptitle("Particle multiplicity per event by pt bin", y=1.01)
    fig.tight_layout()
    return fig


def plot_extra_particles_pt_bin_diagnostics(
    data: dict,
    ind_threshold: float = 0.5,
    pt_min: float = 1.0,
    pt_max: float = 2.0,
    incidence_threshold: float = 0.1,
) -> plt.Figure:
    """Diagnostics for extra predicted particles in a pt bin.

    "Extra" means predicted-valid but truth-invalid at the same query slot.
    """

    pflow_pt = data["pflow_ptetaphi"][..., 0]
    pflow_eta = data["pflow_ptetaphi"][..., 1]
    pflow_phi = normalize_phi(data["pflow_ptetaphi"][..., 2])
    pred_valid = data["pflow_indicator"] > ind_threshold
    truth_valid = data["truth_indicator"] > ind_threshold
    pred_class = data["pflow_class"]

    extra_mask = pred_valid & (~truth_valid) & (pflow_pt > pt_min) & (pflow_pt < pt_max)
    event_counts = extra_mask.sum(axis=-1)

    evt_idx, obj_idx = np.where(extra_mask)
    extra_classes = pred_class[extra_mask]
    extra_eta = pflow_eta[extra_mask]
    extra_phi = pflow_phi[extra_mask]

    incidence = data.get("pflow_incidence_filtered")
    if incidence is None:
        incidence = data.get("pflow_incidence")

    inc_rows = None
    if (
        incidence is not None
        and incidence.ndim == 3
        and incidence.shape[0] == extra_mask.shape[0]
        and incidence.shape[1] == extra_mask.shape[1]
        and evt_idx.size > 0
    ):
        inc_rows = incidence[evt_idx, obj_idx, :]

    max_frac = np.array([], dtype=np.float32)
    sum_frac = np.array([], dtype=np.float32)
    n_links = np.array([], dtype=np.int64)
    n_cluster_links = np.array([], dtype=np.int64)
    n_track_links = np.array([], dtype=np.int64)

    if inc_rows is not None and inc_rows.size > 0:
        max_frac = np.max(inc_rows, axis=1)
        sum_frac = np.sum(inc_rows, axis=1)
        n_links = np.sum(inc_rows > incidence_threshold, axis=1).astype(np.int64)

        reco_is_track = data.get("reco_is_track")
        if (
            reco_is_track is not None
            and reco_is_track.ndim == 2
            and reco_is_track.shape[0] == incidence.shape[0]
            and reco_is_track.shape[1] == incidence.shape[2]
        ):
            track_mask_rows = reco_is_track[evt_idx].astype(bool)
            track_hits = (inc_rows > incidence_threshold) & track_mask_rows
            cluster_hits = (inc_rows > incidence_threshold) & (~track_mask_rows)
            n_track_links = np.sum(track_hits, axis=1).astype(np.int64)
            n_cluster_links = np.sum(cluster_hits, axis=1).astype(np.int64)

    fig, axes = plt.subplots(3, 3, figsize=(18, 13))
    ax = axes.ravel()

    bins_cls = np.arange(-0.5, NUM_CLASSES + 0.5, 1)
    ax[0].hist(extra_classes, bins=bins_cls, histtype="stepfilled", alpha=0.7, color="tab:blue")
    ax[0].set_xticks(np.arange(NUM_CLASSES - 1))
    ax[0].set_xticklabels(CLASS_LABELS[:-1])
    ax[0].set_title(f"Extra particle classes (n={extra_classes.size})")
    ax[0].set_ylabel("Count")

    max_evt = max(1, int(event_counts.max()) if event_counts.size else 1)
    ax[1].hist(event_counts, bins=np.arange(-0.5, max_evt + 1.5, 1.0), histtype="stepfilled", alpha=0.7, color="tab:orange")
    ax[1].set_title("Extra particles per event")
    ax[1].set_xlabel("Count / event")
    ax[1].set_ylabel("Events")

    if n_cluster_links.size > 0:
        max_c = max(1, int(n_cluster_links.max()))
        ax[2].hist(n_cluster_links, bins=np.arange(-0.5, max_c + 1.5, 1.0), histtype="stepfilled", alpha=0.7, color="tab:green")
        ax[2].set_title("Cluster links per extra particle")
        ax[2].set_xlabel(f"# links (> {incidence_threshold:g})")
        ax[2].set_ylabel("Particles")
    elif n_links.size > 0:
        max_l = max(1, int(n_links.max()))
        ax[2].hist(n_links, bins=np.arange(-0.5, max_l + 1.5, 1.0), histtype="stepfilled", alpha=0.7, color="tab:green")
        ax[2].set_title("Incidence links per extra particle")
        ax[2].set_xlabel(f"# links (> {incidence_threshold:g})")
        ax[2].set_ylabel("Particles")
    else:
        ax[2].text(0.5, 0.5, "No incidence data", ha="center", va="center", transform=ax[2].transAxes)
        ax[2].set_title("Incidence occupancy")

    if max_frac.size > 0:
        ax[3].hist(max_frac, bins=50, histtype="stepfilled", alpha=0.7, color="tab:red")
        ax[3].set_title(f"Max incidence fraction (mean={max_frac.mean():.3g})")
        ax[3].set_xlabel("max incidence value / particle")
        ax[3].set_ylabel("Particles")
    else:
        ax[3].text(0.5, 0.5, "No incidence data", ha="center", va="center", transform=ax[3].transAxes)
        ax[3].set_title("Max incidence fraction")

    if sum_frac.size > 0:
        ax[4].hist(sum_frac, bins=50, histtype="stepfilled", alpha=0.7, color="tab:purple")
        ax[4].set_title(f"Sum incidence fraction (mean={sum_frac.mean():.3g})")
        ax[4].set_xlabel("sum incidence values / particle")
        ax[4].set_ylabel("Particles")
    else:
        ax[4].text(0.5, 0.5, "No incidence data", ha="center", va="center", transform=ax[4].transAxes)
        ax[4].set_title("Sum incidence fraction")

    if n_track_links.size > 0:
        max_t = max(1, int(n_track_links.max()))
        ax[5].hist(n_track_links, bins=np.arange(-0.5, max_t + 1.5, 1.0), histtype="stepfilled", alpha=0.7, color="tab:brown")
        ax[5].set_title("Track links per extra particle")
        ax[5].set_xlabel(f"# links (> {incidence_threshold:g})")
        ax[5].set_ylabel("Particles")
    else:
        ax[5].text(0.5, 0.5, "Track mask unavailable", ha="center", va="center", transform=ax[5].transAxes)
        ax[5].set_title("Track-link occupancy")

    finite_eta = extra_eta[np.isfinite(extra_eta)]
    if finite_eta.size > 0:
        ax[6].hist(finite_eta, bins=np.linspace(-4.0, 4.0, 60), histtype="stepfilled", alpha=0.7, color="tab:cyan")
        ax[6].set_title("Extra particle eta distribution")
        ax[6].set_xlabel("eta")
        ax[6].set_ylabel("Particles")
    else:
        ax[6].text(0.5, 0.5, "No finite eta values", ha="center", va="center", transform=ax[6].transAxes)
        ax[6].set_title("Extra particle eta distribution")

    finite_phi = extra_phi[np.isfinite(extra_phi)]
    if finite_phi.size > 0:
        ax[7].hist(finite_phi, bins=np.linspace(-np.pi, np.pi, 60), histtype="stepfilled", alpha=0.7, color="tab:pink")
        ax[7].set_title("Extra particle phi distribution")
        ax[7].set_xlabel("phi [rad]")
        ax[7].set_ylabel("Particles")
    else:
        ax[7].text(0.5, 0.5, "No finite phi values", ha="center", va="center", transform=ax[7].transAxes)
        ax[7].set_title("Extra particle phi distribution")

    ax[8].axis("off")

    fig.suptitle(
        (
            f"Extra predicted particles diagnostics | "
            f"{pt_min:g} < pt < {pt_max:g} GeV | "
            f"events={extra_mask.shape[0]} | particles={extra_classes.size}"
        ),
        y=1.02,
    )
    fig.tight_layout()
    return fig


def plot_feature_distributions(data: dict, ind_threshold: float = 0.5) -> plt.Figure:
    """3×3 grid: [pt, eta, phi] × [All, Charged, Neutral] distributions."""
    truth_f = data["truth_ptetaphi"].reshape(-1, 3)
    pflow_f = data["pflow_ptetaphi"].reshape(-1, 3)
    tc      = data["truth_class"].ravel()
    pc      = data["pflow_class"].ravel()
    ti      = data["truth_indicator"].ravel()
    pi      = data["pflow_indicator"].ravel()
    proxy_f = data["proxy_ptetaphi"].reshape(-1, 3) if _has_proxy(data) else None

    bins = [np.linspace(0, 200, 80), np.linspace(-4, 4, 50), np.linspace(-np.pi, np.pi, 50)]
    row_labels = ["All", "Charged", "Neutral"]
    feat_labels = ["pt [GeV]", "eta", "phi [rad]"]

    truth_masks = [
        (tc < (NUM_CLASSES - 1)) & (ti > ind_threshold),
        (tc < 3) & (ti > ind_threshold),
        (tc >= 3) & (tc < (NUM_CLASSES - 1)) & (ti > ind_threshold),
    ]
    pflow_masks = [
        (pc < (NUM_CLASSES - 1)) & (pi > ind_threshold),
        (pc < 3) & (pi > ind_threshold),
        (pc >= 3) & (pc < (NUM_CLASSES - 1)) & (pi > ind_threshold),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for i in range(3):   # feature
        for j in range(3):  # particle type
            ax = axes[j, i]
            tm, pm = truth_masks[j], pflow_masks[j]
            ax.hist(truth_f[:, i][tm], bins=bins[i], histtype="stepfilled",
                    alpha=0.5, label="Truth", density=True)
            if proxy_f is not None:
                ax.hist(proxy_f[:, i][pm], bins=bins[i], histtype="step",
                        label="Proxy", density=True)
            ax.hist(pflow_f[:, i][pm], bins=bins[i], histtype="step",
                    label="PFlow", density=True)
            ax.set_xlabel(feat_labels[i])
            if i == 0:
                ax.set_yscale("log")
            ax.set_title(row_labels[j])
            ax.legend(fontsize=8)
    fig.suptitle("Particle feature distributions", y=1.01)
    fig.tight_layout()
    return fig


def plot_feature_scatter(data: dict, ind_threshold: float = 0.5) -> plt.Figure:
    """3×3 density maps: truth vs PFlow for [pt, eta, phi] × [All, Charged, Neutral]."""
    truth_f = data["truth_ptetaphi"].reshape(-1, 3)
    pflow_f = data["pflow_ptetaphi"].reshape(-1, 3)
    tc      = data["truth_class"].ravel()
    pc      = data["pflow_class"].ravel()
    ti      = data["truth_indicator"].ravel()
    pi      = data["pflow_indicator"].ravel()
    proxy_f = data["proxy_ptetaphi"].reshape(-1, 3) if _has_proxy(data) else None

    feat_labels = ["pt [GeV]", "eta", "phi [rad]"]
    row_labels  = ["All", "Charged", "Neutral"]
    feat_bins = [
        np.linspace(0.0, 200.0, 70),
        np.linspace(-4.0, 4.0, 70),
        np.linspace(-np.pi, np.pi, 70),
    ]
    masks = [
        (pi > ind_threshold) & (ti > ind_threshold),
        (pi > ind_threshold) & (ti > ind_threshold) & (pc < 3) & (tc < 3),
        (pi > ind_threshold) & (ti > ind_threshold)
            & (pc >= 3) & (pc < (NUM_CLASSES - 1))
            & (tc >= 3) & (tc < (NUM_CLASSES - 1)),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for i in range(3):
        for j in range(3):
            ax = axes[j, i]
            m = masks[j]
            t = truth_f[:, i][m]
            p = pflow_f[:, i][m]
            finite = np.isfinite(t) & np.isfinite(p)

            if finite.any():
                bins = feat_bins[i]
                ax.hist2d(
                    t[finite],
                    p[finite],
                    bins=[bins, bins],
                    norm=LogNorm(),
                    cmap="viridis",
                )
                lo, hi = bins[0], bins[-1]
                ax.plot([lo, hi], [lo, hi], linestyle="--", color="red", linewidth=1.0)
                ax.text(
                    0.03,
                    0.97,
                    f"n={int(finite.sum())}",
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=8,
                    color="white",
                    bbox={"facecolor": "black", "alpha": 0.35, "pad": 2},
                )

                if proxy_f is not None:
                    pr = proxy_f[:, i][m]
                    proxy_finite = np.isfinite(t) & np.isfinite(pr)
                    if proxy_finite.any():
                        h_proxy, xedges, yedges = np.histogram2d(
                            t[proxy_finite], pr[proxy_finite], bins=[bins, bins]
                        )
                        positive = h_proxy[h_proxy > 0]
                        if positive.size > 2:
                            levels = np.unique(np.percentile(positive, [70, 90]))
                            if levels.size > 0:
                                xc = 0.5 * (xedges[:-1] + xedges[1:])
                                yc = 0.5 * (yedges[:-1] + yedges[1:])
                                ax.contour(xc, yc, h_proxy.T, levels=levels, colors="white", linewidths=0.9)
            else:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)

            ax.set_xlabel(f"Truth {feat_labels[i]}")
            ax.set_ylabel(f"Pred {feat_labels[i]}")
            ax.set_title(row_labels[j])
    fig.suptitle("Feature density (hist2d): truth vs prediction", y=1.01)
    fig.tight_layout()
    return fig


def plot_ptetaphi_residuals(data: dict, ind_threshold: float = 0.5) -> plt.Figure:
    """3×3 grid of (pred−truth) residual histograms for [pt, eta, phi] × [All, Charged, Neutral]."""
    def _stats_label(name: str, values: np.ndarray) -> str:
        vals = values[np.isfinite(values)]
        if vals.size == 0:
            return f"{name} (mu=nan, std=nan, IQR=nan)"
        q1, q3 = np.percentile(vals, [25, 75])
        return (
            f"{name} (mu={np.mean(vals):.3g}, std={np.std(vals):.3g}, "
            f"IQR={(q3 - q1):.3g})"
        )

    def _percentile_bins(values: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(values, [0.5, 99.5])
        if not np.isfinite(lo) or not np.isfinite(hi):
            lo, hi = np.min(values), np.max(values)
        if hi <= lo:
            hi = lo + 1e-6
        return np.linspace(lo, hi, 60)

    truth_f = data["truth_ptetaphi"].reshape(-1, 3)
    pflow_f = data["pflow_ptetaphi"].reshape(-1, 3)
    tc      = data["truth_class"].ravel()
    pc      = data["pflow_class"].ravel()
    ti      = data["truth_indicator"].ravel()
    pi      = data["pflow_indicator"].ravel()
    proxy_f = data["proxy_ptetaphi"].reshape(-1, 3) if _has_proxy(data) else None

    row_labels  = ["All", "Charged", "Neutral"]
    feat_labels = ["Δpt/pt", "Δeta", "Δphi"]
    masks = [
        (pi > ind_threshold) & (ti > ind_threshold),
        (pi > ind_threshold) & (ti > ind_threshold) & (pc < 3) & (tc < 3),
        (pi > ind_threshold) & (ti > ind_threshold)
            & (pc >= 3) & (pc < (NUM_CLASSES - 1))
            & (tc >= 3) & (tc < (NUM_CLASSES - 1)),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for i in range(3):
        for j in range(3):
            ax = axes[j, i]
            m = masks[j]
            t = truth_f[:, i][m]
            p = pflow_f[:, i][m]
            t_safe = np.clip(np.abs(t), 1e-8, None) if i == 0 else 1.0
            res_p = (p - t) / t_safe

            if proxy_f is not None:
                pr = proxy_f[:, i][m]
                res_pr = (pr - t) / t_safe
                finite = np.isfinite(res_pr)
                if finite.any():
                    proxy_vals = res_pr[finite]
                    bins = _percentile_bins(proxy_vals)
                    ax.hist(
                        proxy_vals,
                        bins=bins,
                        histtype="step",
                        label=_stats_label("Proxy", proxy_vals),
                        density=True,
                    )

            finite = np.isfinite(res_p)
            if finite.any():
                pflow_vals = res_p[finite]
                bins = _percentile_bins(pflow_vals)
                ax.hist(
                    pflow_vals,
                    bins=bins,
                    histtype="stepfilled",
                    alpha=0.5,
                    label=_stats_label("PFlow", pflow_vals),
                    density=True,
                )
            ax.set_xlabel(feat_labels[i])
            ax.set_title(row_labels[j])
            ax.set_ylabel("Density")
            ax.set_yscale("log")
            ax.legend(fontsize=7)
    fig.suptitle("Kinematic residuals (pred − truth) / truth", y=1.01)
    fig.tight_layout()
    return fig


def plot_pt_residual_absolute(data: dict, ind_threshold: float = 0.5) -> plt.Figure:
    """3×1 grid of absolute pt residual histograms: Δpt = pred_pt − truth_pt."""
    def _stats_label(name: str, values: np.ndarray) -> str:
        vals = values[np.isfinite(values)]
        if vals.size == 0:
            return f"{name} (mu=nan, std=nan, IQR=nan)"
        q1, q3 = np.percentile(vals, [25, 75])
        return (
            f"{name} (mu={np.mean(vals):.3g}, std={np.std(vals):.3g}, "
            f"IQR={(q3 - q1):.3g})"
        )

    def _percentile_bins(values: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(values, [0.5, 99.5])
        if not np.isfinite(lo) or not np.isfinite(hi):
            lo, hi = np.min(values), np.max(values)
        if hi <= lo:
            hi = lo + 1e-6
        return np.linspace(lo, hi, 60)

    truth_pt = data["truth_ptetaphi"][..., 0].ravel()
    pflow_pt = data["pflow_ptetaphi"][..., 0].ravel()
    tc       = data["truth_class"].ravel()
    pc       = data["pflow_class"].ravel()
    ti       = data["truth_indicator"].ravel()
    pi       = data["pflow_indicator"].ravel()
    proxy_pt = data["proxy_ptetaphi"][..., 0].ravel() if _has_proxy(data) else None

    row_labels = ["All", "Charged", "Neutral"]
    masks = [
        (pi > ind_threshold) & (ti > ind_threshold),
        (pi > ind_threshold) & (ti > ind_threshold) & (pc < 3) & (tc < 3),
        (pi > ind_threshold) & (ti > ind_threshold)
            & (pc >= 3) & (pc < (NUM_CLASSES - 1))
            & (tc >= 3) & (tc < (NUM_CLASSES - 1)),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(9, 11))
    for j in range(3):
        ax = axes[j]
        m = masks[j]
        t = truth_pt[m]
        p = pflow_pt[m]
        res_p = p - t

        if proxy_pt is not None:
            pr = proxy_pt[m]
            res_pr = pr - t
            finite = np.isfinite(res_pr)
            if finite.any():
                proxy_vals = res_pr[finite]
                bins = _percentile_bins(proxy_vals)
                ax.hist(
                    proxy_vals,
                    bins=bins,
                    histtype="step",
                    label=_stats_label("Proxy", proxy_vals),
                    density=True,
                )

        finite = np.isfinite(res_p)
        if finite.any():
            pflow_vals = res_p[finite]
            bins = _percentile_bins(pflow_vals)
            ax.hist(
                pflow_vals,
                bins=bins,
                histtype="stepfilled",
                alpha=0.5,
                label=_stats_label("PFlow", pflow_vals),
                density=True,
            )

        ax.set_xlabel("Δpt [GeV]")
        ax.set_title(row_labels[j])
        ax.set_ylabel("Density")
        ax.set_yscale("log")
        ax.legend(fontsize=7)

    fig.suptitle("Absolute pt residuals: pred − truth", y=1.01)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Jet analysis
# ═══════════════════════════════════════════════════════════════════════════

def _cluster_jets_single(
    ptetaphi: np.ndarray,
    indicator: np.ndarray,
    ind_threshold: float,
    jet_R: float,
    min_const: int,
    min_pt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cluster one event's particles into jets with FastJet."""
    import fastjet as fj

    sel = indicator > ind_threshold
    if sel.sum() < min_const:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    pt, eta, phi = ptetaphi[sel, 0], ptetaphi[sel, 1], ptetaphi[sel, 2]
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    e  = np.sqrt(px**2 + py**2 + pz**2)

    pj = [fj.PseudoJet(float(px[k]), float(py[k]), float(pz[k]), float(e[k])) for k in range(len(px))]
    cs   = fj.ClusterSequence(pj, fj.JetDefinition(fj.kt_algorithm, jet_R))
    jets = [j for j in fj.sorted_by_pt(cs.inclusive_jets())
            if len(j.constituents()) >= min_const and j.pt() > min_pt]

    if not jets:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    return (
        np.array([j.pt()    for j in jets], dtype=np.float32),
        np.array([j.eta()   for j in jets], dtype=np.float32),
        np.array([normalize_phi(j.phi()) for j in jets], dtype=np.float32),
        np.array([j.m()     for j in jets], dtype=np.float32),
        np.array([len(j.constituents()) for j in jets], dtype=np.int32),
    )


def cluster_jets(
    data: dict,
    ind_threshold: float = 0.5,
    jet_R: float = 0.7,
    min_constituents: int = 3,
    min_pt: float = 10.0,
) -> dict:
    """Cluster particles into jets for all events.

    Returns a dict with keys ``{pflow,truth,proxy}_jet_{pt,eta,phi,mass,nconst}``
    (object arrays, one numpy array per event).
    """
    try:
        import fastjet  # noqa: F401
    except ImportError as e:
        raise ImportError("fastjet is required for jet clustering") from e

    from tqdm import tqdm

    n_events = len(data["pflow_ptetaphi"])
    jets: dict = {}

    sources = [
        ("pflow", "pflow_ptetaphi", "pflow_indicator"),
        ("truth", "truth_ptetaphi", "truth_indicator"),
    ]
    if _has_proxy(data):
        sources.append(("proxy", "proxy_ptetaphi", "pflow_indicator"))

    for prefix, ptetaphi_key, ind_key in sources:
        pt_l, eta_l, phi_l, m_l, nc_l = [], [], [], [], []
        for i in tqdm(range(n_events), desc=f"Jets ({prefix})"):
            out = _cluster_jets_single(
                data[ptetaphi_key][i], data[ind_key][i],
                ind_threshold, jet_R, min_constituents, min_pt,
            )
            pt_l.append(out[0]); eta_l.append(out[1]); phi_l.append(out[2])
            m_l.append(out[3]);  nc_l.append(out[4])
        for field, arr in zip(("pt", "eta", "phi", "mass", "nconst"),
                               (pt_l, eta_l, phi_l, m_l, nc_l)):
            jets[f"{prefix}_jet_{field}"] = np.array(arr, dtype=object)

    return jets


def match_jets(
    pred_pt: np.ndarray,
    pred_eta: np.ndarray,
    pred_phi: np.ndarray,
    truth_pt: np.ndarray,
    truth_eta: np.ndarray,
    truth_phi: np.ndarray,
    dr_cut: float = 0.4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hungarian-match reco jets to truth jets per event with a ΔR cut.

    Returns (truth_indices, pred_indices, mean_dr_per_event).
    """
    from scipy.optimize import linear_sum_assignment
    from tqdm import tqdm

    n = len(pred_pt)
    truth_ix = np.empty(n, dtype=object)
    pred_ix  = np.empty(n, dtype=object)
    hung_dr  = np.full(n, 1e3)

    for i in tqdm(range(n), desc="Matching jets"):
        pm = pred_pt[i] > 0
        tm = truth_pt[i] > 0
        pe, pp = pred_eta[i][pm], pred_phi[i][pm]
        te, tp = truth_eta[i][tm], truth_phi[i][tm]

        if len(pe) == 0 or len(te) == 0:
            truth_ix[i] = np.array([], dtype=int)
            pred_ix[i]  = np.array([], dtype=int)
            continue

        deta = te[:, None] - pe[None, :]
        dphi = normalize_phi(tp[:, None] - pp[None, :])
        dr   = np.sqrt(deta**2 + dphi**2)

        cost = dr.copy()
        cost[dr > dr_cut] = 1e3
        cost[~np.isfinite(cost)] = 1e3

        ti_arr, pi_arr = linear_sum_assignment(cost)
        valid = dr[ti_arr, pi_arr] <= dr_cut
        ti_arr, pi_arr = ti_arr[valid], pi_arr[valid]

        truth_ix[i] = ti_arr
        pred_ix[i]  = pi_arr
        hung_dr[i]  = dr[ti_arr, pi_arr].mean() if len(ti_arr) else 1e3

    return truth_ix, pred_ix, hung_dr


def get_jet_residuals(
    truth_ix: np.ndarray,
    pred_ix: np.ndarray,
    truth_pt: np.ndarray,
    truth_eta: np.ndarray,
    truth_phi: np.ndarray,
    pred_pt: np.ndarray,
    pred_eta: np.ndarray,
    pred_phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Δpt/pt, Δeta, Δphi residuals for matched jet pairs."""
    res_pt, res_eta, res_phi = [], [], []
    for i in range(len(truth_ix)):
        ti, pi = truth_ix[i], pred_ix[i]
        if len(ti) == 0:
            continue
        tp = truth_pt[i][ti]
        res_pt.append((pred_pt[i][pi] - tp) / np.clip(tp, 1e-8, None))
        res_eta.append(pred_eta[i][pi] - truth_eta[i][ti])
        res_phi.append(normalize_phi(pred_phi[i][pi] - truth_phi[i][ti]))
    if not res_pt:
        return np.array([]), np.array([]), np.array([])
    return np.concatenate(res_pt), np.concatenate(res_eta), np.concatenate(res_phi)


def plot_jet_resolution(
    jets: dict,
    data: dict,
    dr_cut: float = 0.4,
) -> plt.Figure:
    """2×2 resolution grid: Δeta, Δphi, N_constituents, Δpt/pt."""
    from scipy.stats import iqr

    n_pflow = np.array([len(e) for e in jets["pflow_jet_pt"]])
    n_truth = np.array([len(e) for e in jets["truth_jet_pt"]])
    mask    = (n_pflow > 0) & (n_truth > 0)

    tr_pf, pf_ix, _ = match_jets(
        jets["pflow_jet_pt"][mask], jets["pflow_jet_eta"][mask], jets["pflow_jet_phi"][mask],
        jets["truth_jet_pt"][mask], jets["truth_jet_eta"][mask], jets["truth_jet_phi"][mask],
        dr_cut=dr_cut,
    )
    pf_res = get_jet_residuals(
        tr_pf, pf_ix,
        jets["truth_jet_pt"][mask], jets["truth_jet_eta"][mask], jets["truth_jet_phi"][mask],
        jets["pflow_jet_pt"][mask], jets["pflow_jet_eta"][mask], jets["pflow_jet_phi"][mask],
    )
    pf_nc  = np.concatenate([e for e in jets["pflow_jet_nconst"] if len(e) > 0])
    tr_nc  = np.concatenate([e for e in jets["truth_jet_nconst"] if len(e) > 0])

    has_proxy = _has_proxy(data) and "proxy_jet_pt" in jets
    if has_proxy:
        n_proxy = np.array([len(e) for e in jets["proxy_jet_pt"]])
        mask_pr = mask & (n_proxy > 0)
        tr_pr, pr_ix, _ = match_jets(
            jets["proxy_jet_pt"][mask_pr], jets["proxy_jet_eta"][mask_pr], jets["proxy_jet_phi"][mask_pr],
            jets["truth_jet_pt"][mask_pr], jets["truth_jet_eta"][mask_pr], jets["truth_jet_phi"][mask_pr],
            dr_cut=dr_cut,
        )
        pr_res = get_jet_residuals(
            tr_pr, pr_ix,
            jets["truth_jet_pt"][mask_pr], jets["truth_jet_eta"][mask_pr], jets["truth_jet_phi"][mask_pr],
            jets["proxy_jet_pt"][mask_pr], jets["proxy_jet_eta"][mask_pr], jets["proxy_jet_phi"][mask_pr],
        )
        pr_nc = np.concatenate([e for e in jets["proxy_jet_nconst"] if len(e) > 0])

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    configs = [
        (pf_res[1], r"Jet $\Delta\eta$",          np.linspace(-0.2, 0.2, 50)),
        (pf_res[2], r"Jet $\Delta\phi$",           np.linspace(-0.2, 0.2, 50)),
        (pf_nc,     "Jet # Constituents",           None),
        (pf_res[0], r"Jet $p_T$ residual",         np.linspace(-1.0, 4.0, 110)),
    ]

    for idx, (d, xlabel, bins) in enumerate(configs):
        ax = axes[idx // 2, idx % 2]
        if len(d) == 0:
            ax.text(0.5, 0.5, "no matched jets", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_xlabel(xlabel)
            continue

        b = bins if bins is not None else np.arange(d.min() - 0.5, d.max() + 1.5)
        hist_density = idx != 3
        ax.hist(d, bins=b, histtype="stepfilled", alpha=0.5, density=hist_density,
                label=rf"PFlow  $\mu$={np.nanmean(d):.3f}, IQR={iqr(d):.3f}")

        if has_proxy:
            pr_d = [pr_res[1], pr_res[2], pr_nc, pr_res[0]][idx]
            if len(pr_d) > 0:
                ax.hist(pr_d, bins=b, histtype="step", density=hist_density,
                        label=rf"Proxy  $\mu$={np.nanmean(pr_d):.3f}, IQR={iqr(pr_d):.3f}")

        if idx == 2 and len(tr_nc) > 0:
            ax.hist(tr_nc, bins=b, histtype="stepfilled", alpha=0.4, color="orange", density=True,
                    label=rf"Truth  $\mu$={np.nanmean(tr_nc):.3f}, IQR={iqr(tr_nc):.3f}")

        if idx == 3:
            ax.set_yscale("log")
        ylo, yhi = ax.get_ylim()
        if idx == 3:
            ylo = max(ylo, 1e-4)
        ax.set_ylim(ylo, yhi * 1.35)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count" if idx == 3 else "Density")
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Jet resolution")
    fig.tight_layout()
    return fig


def plot_jet_multiplicity(jets: dict, data: dict) -> plt.Figure:
    """Jet multiplicity histogram per event."""
    n_truth = np.array([len(e) for e in jets["truth_jet_pt"]])
    n_pflow = np.array([len(e) for e in jets["pflow_jet_pt"]])
    mx   = max(n_truth.max(), n_pflow.max(), 1) + 1
    bins = np.arange(-0.5, mx + 0.5)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(n_truth, bins=bins, histtype="stepfilled", alpha=0.5, label="Truth")
    ax.hist(n_pflow, bins=bins, histtype="step", label="PFlow")
    if _has_proxy(data) and "proxy_jet_pt" in jets:
        ax.hist(np.array([len(e) for e in jets["proxy_jet_pt"]]), bins=bins,
                histtype="step", linestyle="--", label="Proxy")
    ax.legend()
    ax.set_xlabel("Number of jets per event")
    ax.set_title("Jet multiplicity")
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Event display
# ═══════════════════════════════════════════════════════════════════════════

def _draw_jet_circles(ax, jet_phi, jet_eta, color, R=0.4):
    for phi, eta in zip(jet_phi, jet_eta):
        ax.add_artist(plt.Circle((float(phi), float(eta)), R, color=color, fill=False, lw=1.2))


def plot_event(
    data: dict,
    idx: int = 0,
    ind_threshold: float = 0.5,
    jets: dict | None = None,
    jet_R_display: float = 0.4,
) -> plt.Figure:
    """Eta-phi scatter of one event, color-coded by class.

    Truth = filled circles, PFlow = ×, Proxy = open squares.
    Jet cones drawn as circles if jets dict is provided.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    pf_mask = data["pflow_indicator"][idx] > ind_threshold
    tr_mask = data["truth_indicator"][idx] > ind_threshold

    for cls_i, (color, label) in enumerate(zip(CLASS_COLORS, CLASS_LABELS[:-1])):
        pf_c = data["pflow_class"][idx] == cls_i
        tr_c = data["truth_class"][idx]  == cls_i

        pf_pt  = data["pflow_ptetaphi"][idx, pf_mask & pf_c, 0]
        pf_phi = normalize_phi(data["pflow_ptetaphi"][idx, pf_mask & pf_c, 2])
        pf_eta = data["pflow_ptetaphi"][idx, pf_mask & pf_c, 1]

        tr_pt  = data["truth_ptetaphi"][idx, tr_mask & tr_c, 0]
        tr_phi = normalize_phi(data["truth_ptetaphi"][idx, tr_mask & tr_c, 2])
        tr_eta = data["truth_ptetaphi"][idx, tr_mask & tr_c, 1]

        ax.scatter(tr_phi, tr_eta, s=np.clip(tr_pt * 8, 5, 300), alpha=0.5, color=color)
        ax.scatter(pf_phi, pf_eta, s=np.clip(pf_pt * 8, 5, 300), marker="x",
                   color=color, linewidths=1.2)

        if _has_proxy(data):
            pr_pt  = data["proxy_ptetaphi"][idx, pf_mask & pf_c, 0]
            pr_phi = normalize_phi(data["proxy_ptetaphi"][idx, pf_mask & pf_c, 2])
            pr_eta = data["proxy_ptetaphi"][idx, pf_mask & pf_c, 1]
            ax.scatter(pr_phi, pr_eta, s=np.clip(pr_pt * 8, 5, 300),
                       marker="s", facecolors="none", color=color, linewidths=0.8)

    if jets is not None:
        for key, color in [("truth_jet", "black"), ("pflow_jet", "red")]:
            phi_arr = jets.get(f"{key}_phi", [np.array([])])[idx]
            eta_arr = jets.get(f"{key}_eta", [np.array([])])[idx]
            if len(phi_arr):
                _draw_jet_circles(ax, phi_arr, eta_arr, color=color, R=jet_R_display)

    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-4, 4)
    ax.set_xlabel("phi")
    ax.set_ylabel("eta")
    ax.set_title(f"Event {idx}")
    handles = [plt.Line2D([0], [0], marker="o", color=c, linestyle="None", label=l)
               for c, l in zip(CLASS_COLORS, CLASS_LABELS[:-1])]
    ax.legend(handles=handles, fontsize=8)
    fig.tight_layout()
    return fig


def plot_event_split(
    data: dict,
    idx: int = 0,
    ind_threshold: float = 0.5,
    jets: dict | None = None,
    jet_R_display: float = 0.4,
) -> plt.Figure:
    """Two-panel event display: (phi-eta) and (pt-eta)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), dpi=150)
    pf_mask = data["pflow_indicator"][idx] > ind_threshold
    tr_mask = data["truth_indicator"][idx] > ind_threshold

    for cls_i, (color, label) in enumerate(zip(CLASS_COLORS, CLASS_LABELS[:-1])):
        pf_c = data["pflow_class"][idx] == cls_i
        tr_c = data["truth_class"][idx]  == cls_i

        pf = data["pflow_ptetaphi"][idx][pf_mask & pf_c]
        tr = data["truth_ptetaphi"][idx][tr_mask & tr_c]

        # left: phi-eta
        axes[0].scatter(normalize_phi(tr[:, 2]), tr[:, 1], alpha=0.3, color=color)
        axes[0].scatter(normalize_phi(pf[:, 2]), pf[:, 1], marker="x", color=color)
        # right: pt-eta
        axes[1].scatter(tr[:, 0], tr[:, 1], alpha=0.3, color=color)
        axes[1].scatter(pf[:, 0], pf[:, 1], marker="x", color=color)

        if _has_proxy(data):
            pr = data["proxy_ptetaphi"][idx][pf_mask & pf_c]
            axes[0].scatter(normalize_phi(pr[:, 2]), pr[:, 1],
                            marker="s", facecolors="none", color=color, linewidths=0.8)
            axes[1].scatter(pr[:, 0], pr[:, 1],
                            marker="s", facecolors="none", color=color, linewidths=0.8)

    if jets is not None:
        for key, color in [("truth_jet", "black"), ("pflow_jet", "red")]:
            phi_arr = jets.get(f"{key}_phi", [np.array([])])[idx]
            eta_arr = jets.get(f"{key}_eta", [np.array([])])[idx]
            if len(phi_arr):
                _draw_jet_circles(axes[0], phi_arr, eta_arr, color=color, R=jet_R_display)

    axes[0].set_xlim(-np.pi, np.pi)
    axes[0].set_ylim(-4, 4)
    axes[0].set_xlabel("phi")
    axes[0].set_ylabel("eta")
    axes[1].set_ylim(-4, 4)
    axes[1].set_xlabel("pt [GeV]")
    axes[1].set_ylabel("eta")

    handles = [plt.Line2D([0], [0], marker="o", color=c, linestyle="None", label=l)
               for c, l in zip(CLASS_COLORS, CLASS_LABELS[:-1])]
    axes[0].legend(handles=handles, fontsize=8)
    fig.suptitle(f"Event {idx}")
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Incidence matrix visualization
# ═══════════════════════════════════════════════════════════════════════════

def plot_incidences(
    data: dict,
    idx: int = 0,
    ind_threshold: float = 0.5,
    max_nodes: int = 100,
) -> plt.Figure | None:
    """Side-by-side heatmaps of PFlow vs truth incidence for one event."""
    if data.get("pflow_incidence") is None or data.get("truth_incidence") is None:
        return None

    pf_sel = data["pflow_incidence"][idx][data["pflow_indicator"][idx] > ind_threshold]
    tr_sel = data["truth_incidence"][idx][data["truth_indicator"][idx] > ind_threshold]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].imshow(pf_sel[:, :max_nodes], aspect="auto", vmin=0, vmax=1)
    axes[0].set_title("PFlow")
    axes[0].set_xlabel("Node index")
    axes[0].set_ylabel("Particle")
    axes[1].imshow(tr_sel[:, :max_nodes], aspect="auto", vmin=0, vmax=1)
    axes[1].set_title("Truth")
    axes[1].set_xlabel("Node index")
    fig.suptitle(f"Incidence matrix — event {idx}")
    fig.tight_layout()
    return fig


def plot_incidence_diff(
    data: dict,
    idx: int = 0,
    ind_threshold: float = 0.5,
    max_nodes: int = 100,
) -> plt.Figure | None:
    """Heatmap of (PFlow − Truth) incidence difference for one event."""
    if data.get("pflow_incidence") is None or data.get("truth_incidence") is None:
        return None

    pf_sel = data["pflow_incidence"][idx][data["pflow_indicator"][idx] > ind_threshold]
    tr_sel = data["truth_incidence"][idx][data["truth_indicator"][idx] > ind_threshold]
    n = min(len(pf_sel), len(tr_sel))

    fig, ax = plt.subplots(figsize=(10, 8))
    if n == 0:
        ax.text(0.5, 0.5, "no valid particles", ha="center", va="center",
                transform=ax.transAxes)
    else:
        im = ax.imshow(pf_sel[:n, :max_nodes] - tr_sel[:n, :max_nodes],
                       aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
        fig.colorbar(im, ax=ax)
    ax.set_xlabel("Node index")
    ax.set_ylabel("Particle")
    ax.set_title(f"Incidence diff (PFlow − Truth) — event {idx}")
    fig.tight_layout()
    return fig


def plot_pileup_token_incidence_value_hist(
    data: dict,
    pileup_row: int = 0,
    bins: int = 100,
) -> plt.Figure | None:
    """Histogram of incidence values for pileup token row: prediction vs truth."""
    pred_all = data.get("pflow_incidence_filtered")
    if pred_all is None:
        pred_all = data.get("pflow_incidence")
    truth_all = data.get("truth_incidence")

    if pred_all is None or truth_all is None:
        return None

    pred_arr = np.asarray(pred_all, dtype=np.float32)
    truth_arr = np.asarray(truth_all, dtype=np.float32)
    if pred_arr.ndim != 3 or truth_arr.ndim != 3:
        return None
    if pileup_row < 0 or pileup_row >= pred_arr.shape[1] or pileup_row >= truth_arr.shape[1]:
        return None

    pred_row = pred_arr[:, pileup_row, :]
    truth_row = truth_arr[:, pileup_row, :]

    # Project truth into pred node space when needed (5500 -> 1400).
    if truth_row.shape[1] != pred_row.shape[1]:
        reco_idx = data.get("reco_node_indices")
        if reco_idx is not None:
            reco_idx = np.asarray(reco_idx, dtype=np.int64)
            if reco_idx.ndim == 2 and reco_idx.shape[0] == truth_row.shape[0] and reco_idx.shape[1] == pred_row.shape[1]:
                max_node = truth_row.shape[1] - 1
                safe_idx = np.clip(reco_idx, 0, max_node)
                truth_row = np.take_along_axis(truth_row, safe_idx, axis=1)
                invalid_idx = (reco_idx < 0) | (reco_idx > max_node)
                truth_row[invalid_idx] = 0.0
            else:
                min_nodes = min(truth_row.shape[1], pred_row.shape[1])
                truth_row = truth_row[:, :min_nodes]
                pred_row = pred_row[:, :min_nodes]
        else:
            min_nodes = min(truth_row.shape[1], pred_row.shape[1])
            truth_row = truth_row[:, :min_nodes]
            pred_row = pred_row[:, :min_nodes]

    pred_vals = pred_row.reshape(-1)
    truth_vals = truth_row.reshape(-1)
    pred_vals = pred_vals[np.isfinite(pred_vals)]
    truth_vals = truth_vals[np.isfinite(truth_vals)]
    if pred_vals.size == 0 and truth_vals.size == 0:
        return None

    max_val = 1.0
    if pred_vals.size:
        max_val = max(max_val, float(np.percentile(pred_vals, 99.9)))
    if truth_vals.size:
        max_val = max(max_val, float(np.percentile(truth_vals, 99.9)))
    max_val = max(1e-6, max_val)

    hist_bins = np.linspace(0.0, max_val, int(bins) + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    if truth_vals.size:
        ax.hist(
            truth_vals,
            bins=hist_bins,
            histtype="step",
            linewidth=1.8,
            density=True,
            label=f"Truth (mean={truth_vals.mean():.3g})",
        )
    if pred_vals.size:
        ax.hist(
            pred_vals,
            bins=hist_bins,
            histtype="step",
            linewidth=1.8,
            density=True,
            label=f"Pred (mean={pred_vals.mean():.3g})",
        )

    ax.set_yscale("log")
    ax.set_xlabel("Incidence value")
    ax.set_ylabel("Density")
    ax.set_title(f"Pileup token incidence distribution (row={pileup_row})")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_incidence_track_match(
    data: dict,
    idx: int = 0,
    slice_index: int = 130,
    pred_threshold: float = 0.0,
    truth_threshold: float = 0.0,
    numerical_eps: float = 1e-4,
    pileup_pred_floor: float = 1e-3,
    include_pileup_row: bool = True,
) -> plt.Figure | None:
    """Track-only binary incidence match view in filtered reco-node space.

    Truth incidence is projected from full node space to filtered reco-node space
    with ``reco_node_indices`` when needed, then compared against prediction on
    track columns only.
    """
    pred_all = data.get("pflow_incidence_filtered")
    if pred_all is None:
        pred_all = data.get("pflow_incidence")
    truth_all = data.get("truth_incidence")

    if pred_all is None or truth_all is None:
        return None

    if idx < 0 or idx >= len(pred_all) or idx >= len(truth_all):
        return None

    pred_evt = np.asarray(pred_all[idx], dtype=np.float32)
    truth_evt = np.asarray(truth_all[idx], dtype=np.float32)
    if pred_evt.ndim != 2 or truth_evt.ndim != 2:
        return None

    if truth_evt.shape[1] != pred_evt.shape[1]:
        reco_idx_all = data.get("reco_node_indices")
        if reco_idx_all is None:
            print("  incidence_track_match skipped: reco_node_indices missing for truth->1400 mapping")
            return None
        reco_idx = np.asarray(reco_idx_all[idx], dtype=np.int64)
        if reco_idx.ndim != 1:
            return None
        max_node = truth_evt.shape[1] - 1
        safe_idx = np.clip(reco_idx, 0, max_node)
        truth_evt = np.take(truth_evt, safe_idx, axis=1)
        invalid_idx = (reco_idx < 0) | (reco_idx > max_node)
        if invalid_idx.any():
            truth_evt[:, invalid_idx] = 0.0

    reco_is_track_all = data.get("reco_is_track")
    if reco_is_track_all is None:
        print("  incidence_track_match skipped: reco_is_track missing for track-only selection")
        return None
    track_mask = np.asarray(reco_is_track_all[idx]).astype(bool)
    if track_mask.ndim == 2 and track_mask.shape[-1] == 1:
        track_mask = track_mask[..., 0]
    if track_mask.ndim != 1 or track_mask.shape[0] != pred_evt.shape[1]:
        print("  incidence_track_match skipped: reco_is_track shape mismatch")
        return None

    pred_thr_eff = max(float(pred_threshold), float(numerical_eps))
    truth_thr_eff = max(float(truth_threshold), float(numerical_eps))
    pu_thr = max(pred_thr_eff, float(pileup_pred_floor))

    # Suppress tiny scores before proxy-track selection so numerical noise does
    # not produce artificial track hits.
    pred_for_select = np.array(pred_evt, copy=True)
    pred_for_select[pred_for_select <= pred_thr_eff] = 0.0
    if pred_for_select.shape[0] > 0:
        pred_for_select[0, pred_for_select[0] <= pu_thr] = 0.0

    import torch

    pred_binary_full = select_charged_tracks(
        torch.from_numpy(pred_for_select).unsqueeze(0),
        torch.from_numpy(track_mask.astype(bool)).unsqueeze(0),
    )
    pred_binary_full = pred_binary_full.squeeze(0).cpu().numpy() > 0

    pred_binary_tracks = pred_binary_full[:, track_mask]
    truth_binary_tracks = truth_evt[:, track_mask] > truth_thr_eff

    if not include_pileup_row and pred_binary_tracks.shape[0] > 1:
        pred_binary_tracks = pred_binary_tracks[1:]
        truth_binary_tracks = truth_binary_tracks[1:]

    n_rows = min(int(slice_index), pred_binary_tracks.shape[0], truth_binary_tracks.shape[0])
    if n_rows <= 0 or pred_binary_tracks.shape[1] == 0:
        return None

    pred_binary = pred_binary_tracks[:n_rows]
    truth_binary = truth_binary_tracks[:n_rows]

    matches = truth_binary & pred_binary
    pflow_only = pred_binary & (~truth_binary)
    truth_only = truth_binary & (~pred_binary)

    n_matches = int(np.count_nonzero(matches))
    n_pflow_only = int(np.count_nonzero(pflow_only))
    n_truth_only = int(np.count_nonzero(truth_only))
    n_total_truth = int(np.count_nonzero(truth_binary))
    n_total_pflow = int(np.count_nonzero(pred_binary))

    def _safe_pct(num: int, den: int) -> float:
        return (100.0 * float(num) / float(den)) if den > 0 else 0.0

    match_eff = _safe_pct(n_matches, n_total_truth)
    fp_rate = _safe_pct(n_pflow_only, n_total_pflow)
    fn_rate = _safe_pct(n_truth_only, n_total_truth)

    display_pflow = np.zeros_like(pred_binary, dtype=np.int8)
    display_truth = np.zeros_like(truth_binary, dtype=np.int8)
    display_pflow[matches] = 1
    display_pflow[pflow_only] = 2
    display_truth[matches] = 1
    display_truth[truth_only] = 3

    cmap = ListedColormap(["white", "lime", "gold", "red"])

    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    for axis, view, title_prefix in [
        (ax[0], display_pflow, "PFlow View (Predicted)"),
        (ax[1], display_truth, "Truth View (Ground Truth)"),
    ]:
        axis.imshow(view, interpolation="nearest", aspect="auto", cmap=cmap, vmin=0, vmax=3)
        axis.set_xlabel("Track index in filtered reco space")
        axis.set_ylabel("Particle index")
        axis.set_title(title_prefix)

    summary = (
        f"Track-only incidence match | event={idx} | rows={n_rows} | tracks={int(track_mask.sum())}\n"
        f"thresholds: pred>{pred_thr_eff:.1e}, truth>{truth_thr_eff:.1e}, pileup_pred>{max(pred_thr_eff, float(pileup_pred_floor)):.1e}\n"
        f"Matched={n_matches} ({match_eff:.1f}%)  "
        f"FalsePos={n_pflow_only} ({fp_rate:.1f}%)  "
        f"Missed={n_truth_only} ({fn_rate:.1f}%)"
    )
    fig.suptitle(summary)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Top-level orchestrator
# ═══════════════════════════════════════════════════════════════════════════

def run_reco_analysis(
    data: dict,
    ind_threshold: float = 0.5,
    do_jets: bool = True,
    jet_R: float = 0.7,
    dr_cut: float = 0.4,
    event_display_indices: list[int] | None = None,
) -> dict[str, plt.Figure]:
    """Run all reco analysis plots; return a dict of named Figures.

    Parameters
    ----------
    data : output of ``load_pflow_data()`` or ``pflow_data_from_eval_dicts()``
    ind_threshold : particle indicator threshold (default 0.5)
    do_jets : run jet clustering + resolution (requires fastjet)
    jet_R : jet-clustering radius
    dr_cut : ΔR threshold for jet matching
    event_display_indices : list of event indices for event displays (default: [0])

    Returns
    -------
    dict mapping plot name → matplotlib Figure
    """
    import time

    figs: dict[str, plt.Figure] = {}

    def _plot(name: str, fn, *args, **kw):
        t0 = time.perf_counter()
        result = fn(*args, **kw)
        if result is not None:
            figs[name] = result
            print(f"  {name:<50s} {time.perf_counter() - t0:.2f}s")
        else:
            print(f"  {name:<50s} skipped (data unavailable)")

    print("Particle-level plots…")
    _plot("reco_analysis/class_distribution",   plot_class_distribution,   data, ind_threshold)
    _plot("reco_analysis/n_particles",           plot_n_particles,          data, ind_threshold)
    _plot("reco_analysis/n_particles_by_pt_bin", plot_n_particles_by_pt_bin, data, ind_threshold)
    _plot("reco_analysis/feature_distributions", plot_feature_distributions, data, ind_threshold)
    _plot("reco_analysis/feature_scatter",       plot_feature_scatter,      data, ind_threshold)
    _plot("reco_analysis/ptetaphi_residuals",    plot_ptetaphi_residuals,   data, ind_threshold)
    _plot("reco_analysis/pt_residual_absolute",  plot_pt_residual_absolute, data, ind_threshold)
    _plot("reco_analysis/extra_particles_pt1to2", plot_extra_particles_pt_bin_diagnostics, data, ind_threshold, 1.0, 2.0)
    _plot("reco_analysis/pileup_token_incidence_value_hist", plot_pileup_token_incidence_value_hist, data)
    _plot("reco_analysis/incidence_heatmap",     plot_incidences,           data, 0, ind_threshold)
    _plot("reco_analysis/incidence_diff",        plot_incidence_diff,       data, 0, ind_threshold)
    _plot("reco_analysis/incidence_track_match", plot_incidence_track_match, data, 0)

    jets = None
    if do_jets:
        try:
            print("Clustering jets…")
            jets = cluster_jets(data, ind_threshold=ind_threshold, jet_R=jet_R)
            _plot("reco_analysis/jet_multiplicity", plot_jet_multiplicity, jets, data)
            print("Jet resolution…")
            _plot("reco_analysis/jet_resolution",   plot_jet_resolution,   jets, data, dr_cut)
        except ImportError:
            print("  fastjet not available — skipping jet analysis")

    if event_display_indices is None:
        event_display_indices = [0]
    n_events = len(data["pflow_ptetaphi"])
    for idx in event_display_indices:
        if idx >= n_events:
            continue
        _plot(f"reco_analysis/event_display_{idx}", plot_event,       data, idx, ind_threshold, jets)
        _plot(f"reco_analysis/event_split_{idx}",   plot_event_split, data, idx, ind_threshold, jets)

    print(f"Done — {len(figs)} plots generated.")
    return figs
