"""PUPPI with perfect PFlow linking and charged pileup subtraction.

This variant extends puppi_perfect by subtracting charged pileup energy from
clusters in addition to hard-scatter charged energy. It still uses truth
tracks_mask for CHS and PUPPI alpha-shape weights for neutrals.

Required truth inputs:
- calo_charged_e: per-cluster hard-scatter charged energy deposit
- calo_pu_charged_e: per-cluster pileup charged energy deposit

Public API:
- compute_puppi_weights_perfect_pu_charged(data, ...) -> (n_events, n_nodes) float32 | None
- cluster_puppi_perfect_pu_charged_jets(data, ...) -> dict | None
"""
from __future__ import annotations

import numpy as np

from hepattn.experiments.odd_pileup_reco.puppi import compute_puppi_weights


def _node_pt_neutral_remainder_pu_charged(data: dict) -> np.ndarray:
    """Per-node pT with charged PU subtraction for clusters.

    Tracks use node_pt. Clusters use (node_e - calo_charged_e - calo_pu_charged_e)
    divided by cosh(eta), floored at 0 to avoid negative energies.
    """
    node_is_track = np.asarray(data["node_is_track"]).astype(bool)
    node_eta = np.asarray(data["node_eta"]).astype(np.float32)
    node_e = np.asarray(data["node_e"]).astype(np.float32)
    track_pt = np.asarray(data["node_pt"]).astype(np.float32)
    calo_charged_e = np.asarray(data["calo_charged_e"]).astype(np.float32)
    calo_pu_charged_e = np.asarray(data["calo_pu_charged_e"]).astype(np.float32)

    charged_total = calo_charged_e + calo_pu_charged_e
    cluster_e_neutral = np.maximum(node_e - charged_total, 0.0)
    cluster_pt = cluster_e_neutral / np.cosh(np.clip(node_eta, -10.0, 10.0))
    return np.where(node_is_track, track_pt, cluster_pt).astype(np.float32)


def compute_puppi_weights_perfect_pu_charged(
    data: dict,
    *,
    R0: float = 0.139,
    rms_pt_min: float = 0.080,
    min_neutral_pt: float = 0.504,
    min_neutral_pt_slope: float = 1.837,
    min_weight: float = 0.051,
    eta_max_extrap: float = 2.543,
    apply_lv_adjust: bool = False,
    **kwargs,
) -> np.ndarray | None:
    """PUPPI weights with truth HS and PU charged subtraction for clusters."""
    required = ("calo_charged_e", "calo_pu_charged_e", "node_e", "node_is_track")
    if any(data.get(k) is None for k in required):
        return None

    node_is_track = np.asarray(data["node_is_track"]).astype(bool)
    node_e = np.asarray(data["node_e"]).astype(np.float32)
    calo_charged_e = np.asarray(data["calo_charged_e"]).astype(np.float32)
    calo_pu_charged_e = np.asarray(data["calo_pu_charged_e"]).astype(np.float32)

    charged_total = calo_charged_e + calo_pu_charged_e

    modified = dict(data)
    modified["node_e"] = np.where(
        node_is_track, node_e, np.maximum(node_e - charged_total, 0.0),
    ).astype(np.float32)
    return compute_puppi_weights(
        modified,
        R0=R0,
        rms_pt_min=rms_pt_min,
        min_neutral_pt=min_neutral_pt,
        min_neutral_pt_slope=min_neutral_pt_slope,
        min_weight=min_weight,
        eta_max_extrap=eta_max_extrap,
        apply_lv_adjust=apply_lv_adjust,
        **kwargs,
    )


def cluster_puppi_perfect_pu_charged_jets(
    data: dict,
    jet_R: float = 0.7,
    min_constituents: int = 3,
    min_pt: float = 10.0,
    *,
    weights: np.ndarray | None = None,
) -> dict | None:
    """Cluster jets using LV tracks and charged-subtracted clusters."""
    try:
        import fastjet  # noqa: F401
    except ImportError as e:
        raise ImportError("fastjet is required for jet clustering") from e

    if weights is None:
        weights = compute_puppi_weights_perfect_pu_charged(data)
        if weights is None:
            return None

    required = (
        "node_valid",
        "node_pt",
        "node_eta",
        "node_phi",
        "node_e",
        "node_is_track",
        "calo_charged_e",
        "calo_pu_charged_e",
    )
    if any(data.get(k) is None for k in required):
        return None

    node_valid = np.asarray(data["node_valid"]).astype(bool)
    node_eta = np.asarray(data["node_eta"]).astype(np.float32)
    node_phi = np.asarray(data["node_phi"]).astype(np.float32)
    node_pt = _node_pt_neutral_remainder_pu_charged(data)

    weighted_pt = (node_pt * weights).astype(np.float32)
    selectable = (
        node_valid
        & np.isfinite(weighted_pt)
        & np.isfinite(node_eta)
        & np.isfinite(node_phi)
        & (weighted_pt > 0)
    )

    from hepattn.experiments.odd_pileup_reco.reco_analysis import _cluster_jets_single

    try:
        from tqdm import tqdm
        iterator = tqdm(range(node_valid.shape[0]), desc="Jets (puppi-perfect-pu)")
    except ImportError:
        iterator = range(node_valid.shape[0])

    pt_l, eta_l, phi_l, m_l, nc_l = [], [], [], [], []
    for i in iterator:
        ptetaphi = np.stack([weighted_pt[i], node_eta[i], node_phi[i]], axis=-1)
        out = _cluster_jets_single(ptetaphi, selectable[i], 0.5, jet_R, min_constituents, min_pt)
        pt_l.append(out[0])
        eta_l.append(out[1])
        phi_l.append(out[2])
        m_l.append(out[3])
        nc_l.append(out[4])

    return {
        "puppi_jet_pt": np.array(pt_l, dtype=object),
        "puppi_jet_eta": np.array(eta_l, dtype=object),
        "puppi_jet_phi": np.array(phi_l, dtype=object),
        "puppi_jet_mass": np.array(m_l, dtype=object),
        "puppi_jet_nconst": np.array(nc_l, dtype=object),
    }
