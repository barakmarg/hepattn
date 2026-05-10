"""PUPPI with **perfect PFlow linking** — strongest possible PUPPI baseline.

This variant uses MORE truth than ``puppi.py``'s truth-aware version:

- ``tracks_mask``        — per-track HS-vs-PU label (same as ``compute_puppi_weights``)
- ``calo_charged_e``     — per-cluster HS-charged energy contribution (NEW)

``calo_charged_e`` tells us, per cluster, exactly how much energy came from
HS-charged particles. Subtracting it gives the cluster's "neutral remainder"
energy — the part not already represented by an HS track. We can then add
LV tracks back into the jet inputs without double-counting their calo deposits.

Why this is unfair: real PFlow has to *guess* which track contributed to which
cluster (track-cluster matching) and *estimate* the deposit; we use the truth
attribution. So this is essentially "perfect PFlow + α-shape PU mitigation" —
a ceiling for what PUPPI could achieve in principle.

Pipeline:

      tracks (with truth tracks_mask)         clusters (node_e − calo_charged_e)
                  │                                          │
                  ▼                                          ▼
     LV: weight 1, full pT  /  PU: weight 0    PUPPI α-shape weight × neutral pT
                  │                                          │
                  └──────────────┬───────────────────────────┘
                                 ▼
                            jet clustering

Compared to the calo-only PUPPI in ``puppi.py``:

- Adds LV tracks to the jet inputs (recovers HS-charged jet pT).
- No double-count because cluster pT is HS-charged-subtracted.
- PUPPI weights computed exactly the same way (uses ``compute_puppi_weights``
  internally, but with cluster ``node_e`` overridden to the neutral remainder).

Public API:
- ``compute_puppi_weights_perfect(data, ...) -> (n_events, n_nodes) float32 | None``
- ``cluster_puppi_perfect_jets(data, ...) -> dict | None``
"""
from __future__ import annotations

import numpy as np

from hepattn.experiments.odd_pileup_reco.puppi import compute_puppi_weights


def _node_pt_neutral_remainder(data: dict) -> np.ndarray:
    """Per-node pT: ``node_pt`` for tracks; ``(node_e − calo_charged_e)/cosh(η)``
    for clusters. Subtracting ``calo_charged_e`` (truth HS-charged-deposit
    energy per cluster) prevents double-counting once LV tracks are added to
    the jet inputs.
    """
    node_is_track = np.asarray(data["node_is_track"]).astype(bool)
    node_eta = np.asarray(data["node_eta"]).astype(np.float32)
    node_e = np.asarray(data["node_e"]).astype(np.float32)
    track_pt = np.asarray(data["node_pt"]).astype(np.float32)
    calo_charged_e = np.asarray(data["calo_charged_e"]).astype(np.float32)
    cluster_e_neutral = np.maximum(node_e - calo_charged_e, 0.0)
    cluster_pt = cluster_e_neutral / np.cosh(np.clip(node_eta, -10.0, 10.0))
    return np.where(node_is_track, track_pt, cluster_pt).astype(np.float32)


def compute_puppi_weights_perfect(
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
    """PUPPI weights with truth-aware ``tracks_mask`` AND truth ``calo_charged_e``.

    Reuses ``compute_puppi_weights`` but supplies a modified data dict where
    cluster ``node_e`` is replaced by the neutral remainder
    (``node_e − calo_charged_e``). This propagates into:

    - ``MinNeutralPt`` cut (which uses cluster pT) → applied to neutral remainder
    - ``_node_pt`` inside compute_puppi_weights → cluster pT for that cut

    The α-shape calibration itself is unaffected (α uses only LV-track pT).

    Defaults are the Optuna-tuned operating point (v2: 200 trials × 100 events,
    objective=|bias|+IQR+0.5|nc_rel| on dpt_over_truth + nconst,
    verified on held-out 1900 events: bias=+0.017, IQR=0.301, nc_rel=−0.22).
    Key differences vs the cluster-only baseline:

    - ``apply_lv_adjust=False`` — CMS's low-PU shift hurts in ODD cluster space.
    - ``min_neutral_pt + min_neutral_pt_slope · N_PU`` (~0.5 + 1.84·N_PU GeV) —
      a PU-aware threshold tracks per-event PU density, eliminating the need for
      a fixed hard cutoff.
    - ``R0=0.139`` — tighter cone exploits the dense HS-jet core in cluster
      space (~30× more clusters per PU vertex than CMS PFlow candidates).
    - ``eta_max_extrap=2.543`` — extend calibration region into forward η.
    """
    required = ("calo_charged_e", "node_e", "node_is_track")
    if any(data.get(k) is None for k in required):
        return None

    node_is_track = np.asarray(data["node_is_track"]).astype(bool)
    node_e = np.asarray(data["node_e"]).astype(np.float32)
    calo_charged_e = np.asarray(data["calo_charged_e"]).astype(np.float32)

    # Override cluster node_e with the neutral remainder so downstream
    # cluster-pT consumers see the post-PFlow energy. Tracks unchanged.
    modified = dict(data)
    modified["node_e"] = np.where(
        node_is_track, node_e, np.maximum(node_e - calo_charged_e, 0.0),
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


def cluster_puppi_perfect_jets(
    data: dict,
    jet_R: float = 0.7,
    min_constituents: int = 3,
    min_pt: float = 10.0,
    *,
    weights: np.ndarray | None = None,
) -> dict | None:
    """Cluster jets using BOTH LV tracks (weight 1, full track pT) AND clusters
    (PUPPI-weighted, neutral-remainder pT). PU tracks are excluded by their
    weight=0; cluster double-counting is prevented by HS-charged subtraction.

    Returns a dict keyed ``puppi_jet_*`` so it drops in wherever
    ``cluster_puppi_jets`` is consumed (e.g. via the ``puppi_jets=`` override
    on ``plot_jet_resolution_with_calo``).
    """
    try:
        import fastjet  # noqa: F401
    except ImportError as e:
        raise ImportError("fastjet is required for jet clustering") from e

    if weights is None:
        weights = compute_puppi_weights_perfect(data)
        if weights is None:
            return None

    required = ("node_valid", "node_pt", "node_eta", "node_phi", "node_e",
                "node_is_track", "calo_charged_e")
    if any(data.get(k) is None for k in required):
        return None

    node_valid = np.asarray(data["node_valid"]).astype(bool)
    node_eta = np.asarray(data["node_eta"]).astype(np.float32)
    node_phi = np.asarray(data["node_phi"]).astype(np.float32)
    node_pt = _node_pt_neutral_remainder(data)

    weighted_pt = (node_pt * weights).astype(np.float32)
    # Include both tracks AND clusters — no ``& (~node_is_track)`` filter.
    # PU tracks fall out automatically (weight=0 → weighted_pt=0).
    selectable = (
        node_valid
        & np.isfinite(weighted_pt)
        & np.isfinite(node_eta)
        & np.isfinite(node_phi)
        & (weighted_pt > 0)
    )

    # Lazy import to avoid circular import.
    from hepattn.experiments.odd_pileup_reco.reco_analysis import _cluster_jets_single

    try:
        from tqdm import tqdm
        iterator = tqdm(range(node_valid.shape[0]), desc="Jets (puppi-perfect)")
    except ImportError:
        iterator = range(node_valid.shape[0])

    pt_l, eta_l, phi_l, m_l, nc_l = [], [], [], [], []
    for i in iterator:
        ptetaphi = np.stack([weighted_pt[i], node_eta[i], node_phi[i]], axis=-1)
        out = _cluster_jets_single(ptetaphi, selectable[i], 0.5, jet_R, min_constituents, min_pt)
        pt_l.append(out[0]); eta_l.append(out[1]); phi_l.append(out[2])
        m_l.append(out[3]);  nc_l.append(out[4])

    return {
        "puppi_jet_pt": np.array(pt_l, dtype=object),
        "puppi_jet_eta": np.array(eta_l, dtype=object),
        "puppi_jet_phi": np.array(phi_l, dtype=object),
        "puppi_jet_mass": np.array(m_l, dtype=object),
        "puppi_jet_nconst": np.array(nc_l, dtype=object),
    }
