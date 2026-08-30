"""Standalone PUPPI (PileUp Per Particle Identification) for ODD pileup-reco.

Mirrors the CMS PUPPI algorithm
(`/storage/agrp/barakma/cmssw/CommonTools/PileupAlgos`) on the node-level
inputs of the ODD pileup-reco H5 dataset. Uses only raw inputs and truth
vertex labels at the **track** level (``tracks_mask``); calo clusters get
no per-cluster truth. No ML predictions consumed.

Fairness contract:
  - Allowed truth: ``tracks_mask`` (HS vs PU vertex association for tracks).
    This is the analog of CMS's CHS, which uses Δz to the primary vertex.
  - Disallowed truth: ``calo_hs_energy``, ``calo_hs_frac``, ``calo_neutral_e``,
    ``calo_charged_e``, ``truth_incidence`` — anything that says, per cluster,
    how much energy is HS vs PU.

Per node, the algorithm yields a weight in [0, 1]:
  - HS-LV tracks (``node_is_track=1`` & ``tracks_mask=1``) → weight = 1
  - PU tracks    (``node_is_track=1`` & ``tracks_mask=0``) → weight = 0
  - Neutrals     (``node_is_track=0``)                     → weight from α-shape

α (CMS algoId=5):
    α_i = log Σ_{j: HS-LV, ΔR_ij ∈ (0, R0)} pT_j² / ΔR_ij²

Calibration (α_med, α_rms) is fit on PU tracks (the only nodes truth-labeled
as pileup), then signed χ²(df=1) → p-value gives the neutral weight.

ODD-specific tuning notes:
  - Cone ``R0=0.2`` (vs CMS 0.4): ODD operates in cluster space, where each
    PU vertex deposits many more clusters per vertex than CMS PFlow has
    per-vertex candidates. A tighter cone exploits the fact that HS clusters
    sit at the dense core of HS jets (lots of LV tracks within ΔR<0.2),
    while PU clusters near LV tracks are accidental and don't have that local
    density.
  - The PUPPI baseline here is **calo-only**: jets are clustered from the
    weighted clusters alone, with tracks only used as α-seeds. Adding tracks
    back into the jet inputs would double-count charged HS energy that's
    already in the clusters, since we don't have CMS-style PFlow linking.

Public API:
  - ``compute_puppi_weights(data, ...) -> (n_events, n_nodes) float32 | None``
  - ``cluster_puppi_jets(data, ...) -> dict | None``
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _phi_diff(phi_a: np.ndarray, phi_b: np.ndarray) -> np.ndarray:
    d = phi_a - phi_b
    return (d + np.pi) % (2.0 * np.pi) - np.pi


def _alpha_for_event(
    pt: np.ndarray,
    eta: np.ndarray,
    phi: np.ndarray,
    is_lv_track: np.ndarray,
    is_target: np.ndarray,
    R0: float,
) -> np.ndarray:
    """α = log Σ pT_j² / ΔR_ij² over LV-track neighbors j with ΔR_ij ∈ (0, R0).

    Mirrors CMS ``PuppiContainer::var_within_R`` (algoId=5): pairs with
    ΔR² < 1e-4 are skipped (line 89 in PuppiContainer.cc), eliminating
    self-pair / near-overlap blowups.
    """
    n_lv = int(is_lv_track.sum())
    if n_lv == 0:
        return np.zeros(eta.shape, dtype=np.float32)

    lv_idx = np.where(is_lv_track)[0]
    lv_pt = pt[lv_idx]
    lv_eta = eta[lv_idx]
    lv_phi = phi[lv_idx]

    tgt_idx = np.where(is_target)[0]
    if tgt_idx.size == 0:
        return np.zeros(eta.shape, dtype=np.float32)

    deta = eta[tgt_idx, None] - lv_eta[None, :]
    dphi = _phi_diff(phi[tgt_idx, None], lv_phi[None, :])
    dr2 = deta * deta + dphi * dphi

    within = (dr2 < R0 * R0) & (dr2 > 1e-4)
    contrib = np.where(within, (lv_pt[None, :] ** 2) / dr2, 0.0)
    sum_contrib = contrib.sum(axis=1)

    alpha = np.zeros(eta.shape, dtype=np.float32)
    valid = sum_contrib > 0.0
    alpha_target = np.where(valid, np.log(sum_contrib), 0.0).astype(np.float32)
    alpha[tgt_idx] = alpha_target
    return alpha


def _node_pt(data: dict) -> np.ndarray:
    """Per-node pT: track pT for tracks, E/cosh(η) for clusters.

    The H5 stores ``node_pt`` only for tracks; calo clusters use ``node_e``.
    Mirrors what ``cluster_calo_jets`` does for the cluster branch.
    """
    node_is_track = np.asarray(data["node_is_track"]).astype(bool)
    node_eta = np.asarray(data["node_eta"]).astype(np.float32)
    node_e = np.asarray(data["node_e"]).astype(np.float32)
    track_pt = np.asarray(data["node_pt"]).astype(np.float32)
    cluster_pt = node_e / np.cosh(np.clip(node_eta, -10.0, 10.0))
    return np.where(node_is_track, track_pt, cluster_pt).astype(np.float32)


def compute_puppi_weights(
    data: dict,
    *,
    R0: float = 0.2,
    rms_pt_min: float = 0.1,
    min_neutral_pt: float = 0.3,
    min_neutral_pt_slope: float = 0.0,
    n_pu_proxy: float | np.ndarray | None = 0.0,
    min_weight: float = 0.01,
    eta_max_extrap: float = 2.0,
    apply_lv_adjust: bool = True,
) -> np.ndarray | None:
    """Per-node PUPPI weight in [0, 1]. Returns None if required fields missing.

    Output shape matches ``node_pt``: ``(n_events, n_nodes)``.

    Parameters
    ----------
    R0
        Cone size for α neighbor sum. CMS phase2 default 0.4.
    rms_pt_min
        Minimum track pT for entries to enter the α calibration sample. Phase2 0.1 GeV.
    min_neutral_pt, min_neutral_pt_slope
        ``MinNeutralPt + MinNeutralPtSlope · n_pu_proxy`` is the threshold below which
        a neutral's weighted-pT is hard-zeroed (CMS PuppiAlgo::neutralPt). CMS phase2
        central defaults are (0.2, 0.015) calibrated against PFlow candidates per PU
        vertex. ODD operates in cluster space, where each PU vertex deposits ~30× more
        clusters than CMS PFlow has candidates per vertex, so ``min_neutral_pt_slope``
        defaults to 1.0 (≈ CMS · 30×) to compensate. Tune for your pileup level.
    n_pu_proxy
        CMS ``iPUProxy``. If None, estimated per-event from the PU-track count
        (``tracks_mask == 0`` & ``|η| < eta_max_extrap`` & ``pt > rms_pt_min``).
    eta_max_extrap
        Central |η| limit for the α calibration sample (CMS EtaMaxExtrap, default 2.0).
    apply_lv_adjust
        Apply CMS's PuppiAlgo low-PU correction: shift ``α_med`` and ``α_rms`` down
        based on the fraction of LV-track αs lying below the PU median (PuppiAlgo.cc:159).
    """
    required = ("node_valid", "node_is_track", "tracks_mask", "node_pt", "node_eta", "node_phi", "node_e")
    if any(data.get(k) is None for k in required):
        return None

    node_valid = np.asarray(data["node_valid"]).astype(bool)
    node_is_track = np.asarray(data["node_is_track"]).astype(bool)
    tracks_mask = np.asarray(data["tracks_mask"]).astype(np.int8)
    node_eta = np.asarray(data["node_eta"]).astype(np.float32)
    node_phi = np.asarray(data["node_phi"]).astype(np.float32)
    node_pt = _node_pt(data)

    if any(a.ndim != 2 for a in (node_valid, node_is_track, tracks_mask, node_pt, node_eta, node_phi)):
        return None
    if not (node_valid.shape == node_is_track.shape == tracks_mask.shape == node_pt.shape == node_eta.shape == node_phi.shape):
        return None

    n_events, _ = node_valid.shape
    weights = np.zeros_like(node_pt, dtype=np.float32)

    from scipy.stats import chi2 as _chi2

    # Per-event PU proxy. CMS treats this as ~ # PU vertices (NPU); the
    # MinNeutralPtSlope (default 0.015) is calibrated against NPU. We have
    # PU-track counts, not vertex counts, so we estimate NPU as
    # ``n_pu_tracks_central / tracks_per_pu_vertex`` (typical PU vertex yields
    # ~25–35 reco-able tracks). Override by passing ``n_pu_proxy`` explicitly.
    tracks_per_pu_vertex = 30.0
    if n_pu_proxy is None:
        pu_central = (
            (tracks_mask == 0) & node_is_track & node_valid
            & (np.abs(node_eta) < eta_max_extrap) & (node_pt > rms_pt_min)
        )
        n_pu_arr = (pu_central.sum(axis=1).astype(np.float32) / tracks_per_pu_vertex)
    else:
        n_pu_arr = (
            np.full(n_events, float(n_pu_proxy), dtype=np.float32)
            if np.ndim(n_pu_proxy) == 0
            else np.asarray(n_pu_proxy, dtype=np.float32)
        )

    try:
        from tqdm import tqdm
        iterator = tqdm(range(n_events), desc="PUPPI weights")
    except ImportError:
        iterator = range(n_events)

    for i in iterator:
        valid = node_valid[i] & np.isfinite(node_pt[i]) & np.isfinite(node_eta[i]) & np.isfinite(node_phi[i])
        if not valid.any():
            continue

        is_track = node_is_track[i] & valid
        is_lv = is_track & (tracks_mask[i] == 1)
        is_pu = is_track & (tracks_mask[i] == 0)
        is_neutral = (~node_is_track[i]) & valid

        # Targets that need an α: PU tracks (calibration), neutrals (scoring),
        # and LV tracks within EtaMaxExtrap (used for the LV-adjust correction).
        is_target = is_pu | is_neutral | is_lv

        alpha = _alpha_for_event(
            node_pt[i], node_eta[i], node_phi[i],
            is_lv_track=is_lv, is_target=is_target, R0=R0,
        )

        # Calibration sample: PU tracks within EtaMaxExtrap with pT > rms_pt_min.
        cal_mask = is_pu & (np.abs(node_eta[i]) < eta_max_extrap) & (node_pt[i] > rms_pt_min) & (alpha != 0.0)
        if cal_mask.sum() < 2:
            cal_mask = is_target & (np.abs(node_eta[i]) < eta_max_extrap) & (alpha != 0.0)
        if cal_mask.sum() < 2:
            alpha_med = 0.0
            alpha_rms = 1.0
        else:
            cal_alpha = alpha[cal_mask]
            alpha_med = float(np.median(cal_alpha))
            # CMS applyLowPUCorr=True: RMS only over αs ≤ median (low-PU symmetric estimate)
            below = cal_alpha[cal_alpha <= alpha_med]
            if below.size >= 2:
                alpha_rms = float(np.sqrt(np.mean((below - alpha_med) ** 2)))
            else:
                alpha_rms = float(np.sqrt(np.mean((cal_alpha - alpha_med) ** 2)))
            if alpha_rms < 1e-6:
                alpha_rms = 1e-6

            # CMS LV-adjust (PuppiAlgo.cc:159–169): shift med, rms down by
            # sqrt(chi2_quantile(lAdjust, 1) * rms), where lAdjust is the
            # fraction of LV αs falling below the PU median.
            if apply_lv_adjust:
                pv_mask = is_lv & (np.abs(node_eta[i]) < eta_max_extrap) & (alpha != 0.0)
                n_pv = int(pv_mask.sum())
                if n_pv > 0:
                    n_pv_below = int((alpha[pv_mask] <= alpha_med).sum())
                    n_pu_cal = int(cal_mask.sum())
                    l_adjust = n_pv_below / (n_pv_below + 0.5 * n_pu_cal) if (n_pv_below + 0.5 * n_pu_cal) > 0 else 0.0
                    if 0.0 < l_adjust < 1.0:
                        shift = float(np.sqrt(_chi2.ppf(l_adjust, df=1) * alpha_rms))
                        alpha_med -= shift
                        alpha_rms = max(alpha_rms - shift, 1e-6)

        # Neutral weight: signed χ²(df=1) p-value, matching PuppiAlgo::compute (line 206):
        #   lVal = (pVal - cur_Med) * |pVal - cur_Med| / cur_RMS²
        # Negative lVal → χ²-CDF(0,1) = 0.
        diff = (alpha - alpha_med).astype(np.float32)
        lval = (diff * np.abs(diff) / (alpha_rms * alpha_rms)).astype(np.float32)
        p_vals = np.where(lval > 0.0, _chi2.cdf(np.maximum(lval, 0.0), df=1), 0.0).astype(np.float32)
        w = p_vals.copy()

        # Category overrides (CHS veto)
        w[is_lv] = 1.0
        w[is_pu] = 0.0
        w[~valid] = 0.0
        w[~(is_lv | is_pu | is_neutral)] = 0.0

        # Min-weight cutoff
        w[w < min_weight] = 0.0
        # Pileup-aware neutral pT threshold: weighted pT must clear neutral_pt_min.
        neutral_pt_thresh = min_neutral_pt + min_neutral_pt_slope * float(n_pu_arr[i])
        neutral_below = is_neutral & (w * node_pt[i] < neutral_pt_thresh)
        w[neutral_below] = 0.0

        weights[i] = w

    return weights


def _estimate_pv_z(z0: np.ndarray, pt: np.ndarray, is_track_valid: np.ndarray, pt_min: float = 1.0) -> float:
    """pT²-weighted median z0 of high-pT tracks — proxy for the hard-scatter primary vertex.

    The classic PUPPI/CHS PV estimator: hard-scatter has the highest-pT charged
    activity, so weighting track z0s by pT² and taking the weighted median
    gives a bias-resistant pointer to the LV.
    """
    sel = is_track_valid & np.isfinite(z0) & np.isfinite(pt) & (pt > pt_min)
    if not sel.any():
        return 0.0
    z0_sel = z0[sel]
    w_sel = pt[sel] ** 2
    order = np.argsort(z0_sel)
    z0_sorted = z0_sel[order]
    w_sorted = w_sel[order]
    cumw = np.cumsum(w_sorted)
    half = cumw[-1] / 2.0
    idx = int(np.searchsorted(cumw, half))
    idx = min(idx, len(z0_sorted) - 1)
    return float(z0_sorted[idx])


def synthesize_tracks_mask(
    data: dict,
    *,
    dz_cut: float = 0.7,
    pv_pt_min: float = 1.0,
) -> np.ndarray | None:
    """Build a truth-free tracks_mask via PV-estimation + Δz CHS-style cut.

    For each event:
      1. Estimate PV_z = pT²-weighted median z0 of tracks with pT > ``pv_pt_min``.
      2. Tag each track as LV (mask=1) if ``|z0 - PV_z| < dz_cut``, else PU (mask=0).

    Returns ``(n_events, n_nodes)`` int8 array, or ``None`` if ``node_z0`` is missing.
    Non-track nodes get mask=0 (irrelevant; PUPPI only consults the mask for tracks).
    """
    if data.get("node_z0") is None:
        return None
    node_valid = np.asarray(data["node_valid"]).astype(bool)
    node_is_track = np.asarray(data["node_is_track"]).astype(bool)
    node_z0 = np.asarray(data["node_z0"]).astype(np.float32)
    node_pt = np.asarray(data["node_pt"]).astype(np.float32)
    n_events, n_nodes = node_valid.shape
    out = np.zeros((n_events, n_nodes), dtype=np.int8)
    for i in range(n_events):
        is_track_v = node_is_track[i] & node_valid[i]
        if not is_track_v.any():
            continue
        pv_z = _estimate_pv_z(node_z0[i], node_pt[i], is_track_v, pt_min=pv_pt_min)
        dz = np.abs(node_z0[i] - pv_z)
        out[i] = (is_track_v & np.isfinite(node_z0[i]) & (dz < dz_cut)).astype(np.int8)
    return out


def compute_puppi_weights_no_truth(
    data: dict,
    *,
    dz_cut: float = 0.7,
    pv_pt_min: float = 1.0,
    **puppi_kwargs,
) -> np.ndarray | None:
    """PUPPI weights using a truth-free tracks_mask synthesized from ``node_z0``.

    Equivalent to ``compute_puppi_weights`` but builds the LV/PU track classification
    by Δz to a pT²-weighted PV estimate, instead of consuming the truth
    ``tracks_mask``. This is the analog of how real CMS/ATLAS reco classifies
    tracks (CHS Δz cut), so it's the "fully fair, no truth" PUPPI.

    Extra parameters (``dz_cut``, ``pv_pt_min``) control the CHS step;
    everything else is forwarded to ``compute_puppi_weights``.
    """
    if data.get("node_z0") is None or data.get("node_pt") is None:
        return None
    synth_mask = synthesize_tracks_mask(data, dz_cut=dz_cut, pv_pt_min=pv_pt_min)
    if synth_mask is None:
        return None
    synth_data = dict(data)
    synth_data["tracks_mask"] = synth_mask
    return compute_puppi_weights(synth_data, **puppi_kwargs)


def compute_puppi_weights_event(
    node_pt: np.ndarray,
    node_eta: np.ndarray,
    node_phi: np.ndarray,
    node_z0: np.ndarray,
    node_is_track: np.ndarray,
    node_valid: np.ndarray,
    *,
    R0: float = 0.2,
    rms_pt_min: float = 0.1,
    min_neutral_pt: float = 0.3,
    min_neutral_pt_slope: float = 0.0,
    n_pu_proxy: float = 0.0,
    min_weight: float = 0.01,
    eta_max_extrap: float = 2.0,
    apply_lv_adjust: bool = True,
    dz_cut: float = 0.7,
    pv_pt_min: float = 1.0,
) -> np.ndarray:
    """Single-event truth-free PUPPI weights, optimized for use in
    ``pflow_data.py::ODDDatasetPileup.__getitem__``.

    All inputs are 1-D arrays of length ``n_nodes`` (the per-event padded shape
    that pflow_data.py uses). Accepts numpy arrays or anything ``np.asarray``
    accepts (torch tensors auto-convert). Returns ``(n_nodes,)`` float32 weights.

    Numerically equivalent to the per-event branch inside
    ``compute_puppi_weights_no_truth``, but ~2× faster thanks to:

    1. **η-prefilter** — only nodes within ``R0`` of *some* LV track in η can
       have non-zero α anyway (their α-neighbor sum is empty otherwise), so we
       compute pairs only for those targets.
    2. **Inline df=1 χ² formulas** via ``scipy.special.ndtr`` / ``ndtri`` — the
       ``scipy.stats.chi2`` wrappers add ~0.5 ms of Python overhead per call.
    """
    from scipy.special import ndtr, ndtri

    node_pt = np.asarray(node_pt, dtype=np.float32)
    node_eta = np.asarray(node_eta, dtype=np.float32)
    node_phi = np.asarray(node_phi, dtype=np.float32)
    node_z0 = np.asarray(node_z0, dtype=np.float32)
    node_is_track = np.asarray(node_is_track).astype(bool)
    node_valid = np.asarray(node_valid).astype(bool)

    valid = node_valid & np.isfinite(node_pt) & np.isfinite(node_eta) & np.isfinite(node_phi)
    is_track_v = node_is_track & valid

    # ---- Step 1: PV_z = pT²-weighted median z0 of high-pT tracks ----
    sel = is_track_v & np.isfinite(node_z0) & (node_pt > pv_pt_min)
    if not sel.any():
        pv_z = 0.0
    else:
        z0_sel = node_z0[sel]
        w_sel = (node_pt[sel] ** 2).astype(np.float64)
        order = np.argsort(z0_sel)
        cumw = np.cumsum(w_sel[order])
        pv_z = float(z0_sel[order][int(np.searchsorted(cumw, cumw[-1] / 2.0))])

    # ---- Step 2: synth tracks_mask via Δz CHS cut ----
    is_lv = is_track_v & np.isfinite(node_z0) & (np.abs(node_z0 - pv_z) < dz_cut)
    is_pu = is_track_v & ~is_lv
    is_neutral = (~node_is_track) & valid

    n_nodes = node_pt.shape[0]
    weights = np.zeros(n_nodes, dtype=np.float32)
    if not is_lv.any():
        return weights  # no LV tracks → α undefined → all neutral weights = 0

    lv_idx = np.where(is_lv)[0]
    lv_pt = node_pt[lv_idx]
    lv_eta = node_eta[lv_idx]
    lv_phi = node_phi[lv_idx]

    # Targets that need α: PU (calibration), neutrals (scoring), LV (LV-adjust input).
    is_target = is_pu | is_neutral | is_lv

    # ---- Step 3: η-prefilter targets to those near any LV track ----
    # A target outside [min(lv_eta) - R0, max(lv_eta) + R0] cannot have any
    # LV neighbor inside R0 → α stays 0 → weight stays 0. Skip them.
    eta_lo = lv_eta.min() - R0
    eta_hi = lv_eta.max() + R0
    in_eta_band = (node_eta >= eta_lo) & (node_eta <= eta_hi)
    tgt_mask = is_target & in_eta_band
    tgt_idx = np.where(tgt_mask)[0]
    alpha = np.zeros(n_nodes, dtype=np.float32)

    if tgt_idx.size > 0:
        deta = node_eta[tgt_idx, None] - lv_eta[None, :]
        dphi = node_phi[tgt_idx, None] - lv_phi[None, :]
        dphi = (dphi + np.pi) % (2.0 * np.pi) - np.pi
        dr2 = deta * deta + dphi * dphi
        within = (dr2 < R0 * R0) & (dr2 > 1e-4)
        contrib = np.where(within, (lv_pt[None, :] ** 2) / np.where(dr2 > 0, dr2, 1.0), 0.0)
        sum_contrib = contrib.sum(axis=1)
        alpha_tgt = np.where(sum_contrib > 0, np.log(np.where(sum_contrib > 0, sum_contrib, 1.0)), 0.0).astype(np.float32)
        alpha[tgt_idx] = alpha_tgt

    # ---- Step 4: calibrate α_med, α_rms from PU tracks ----
    cal_mask = is_pu & (np.abs(node_eta) < eta_max_extrap) & (node_pt > rms_pt_min) & (alpha != 0.0)
    if cal_mask.sum() < 2:
        cal_mask = is_target & (np.abs(node_eta) < eta_max_extrap) & (alpha != 0.0)
    if cal_mask.sum() < 2:
        alpha_med = np.float32(0.0)
        alpha_rms = np.float32(1.0)
    else:
        cal_alpha = alpha[cal_mask]
        alpha_med = np.float32(np.median(cal_alpha))
        below = cal_alpha[cal_alpha <= alpha_med]
        if below.size >= 2:
            alpha_rms = float(np.sqrt(np.mean((below - alpha_med) ** 2)))
        else:
            alpha_rms = float(np.sqrt(np.mean((cal_alpha - alpha_med) ** 2)))
        alpha_rms = np.float32(max(alpha_rms, 1e-6))

        # CMS LV-adjust: shift med, rms by sqrt(chi2_quantile(l, 1) * rms),
        # where l = N_LV_below / (N_LV_below + 0.5 N_PU_cal).
        # chi2_quantile(p, 1) = ndtri((p+1)/2)**2.
        if apply_lv_adjust:
            pv_mask = is_lv & (np.abs(node_eta) < eta_max_extrap) & (alpha != 0.0)
            n_pv = int(pv_mask.sum())
            if n_pv > 0:
                n_pv_below = int((alpha[pv_mask] <= alpha_med).sum())
                n_pu_cal = int(cal_mask.sum())
                denom = n_pv_below + 0.5 * n_pu_cal
                l_adjust = n_pv_below / denom if denom > 0 else 0.0
                if 0.0 < l_adjust < 1.0:
                    chi2_q = float(ndtri((l_adjust + 1.0) / 2.0)) ** 2
                    shift = np.float32(np.sqrt(chi2_q * alpha_rms))
                    alpha_med = np.float32(alpha_med - shift)
                    alpha_rms = np.float32(max(alpha_rms - shift, 1e-6))

    # ---- Step 5: per-particle weight via signed χ²(df=1) CDF ----
    # χ²-CDF(x, 1) = 2·Φ(√x) − 1 for x ≥ 0; 0 elsewhere.
    diff = (alpha - alpha_med).astype(np.float32)
    lval = (diff * np.abs(diff) / (alpha_rms * alpha_rms)).astype(np.float32)
    sqrt_lval = np.sqrt(np.maximum(lval, 0.0))
    p_vals = (2.0 * ndtr(sqrt_lval) - 1.0).astype(np.float32)
    w = np.where(lval > 0.0, p_vals, np.float32(0.0))

    # Category overrides + cuts
    w = np.where(is_lv, np.float32(1.0), w)
    w = np.where(is_pu, np.float32(0.0), w)
    w = np.where(~node_valid, np.float32(0.0), w)
    valid_cat = is_lv | is_pu | is_neutral
    w = np.where(valid_cat, w, np.float32(0.0))
    w = np.where(w < min_weight, np.float32(0.0), w)

    # Pileup-aware neutral pT cut. ``n_pu_proxy=None`` triggers an auto-estimate
    # from this event's PU-track count; otherwise treat as a scalar.
    if n_pu_proxy is None:
        pu_central = is_pu & (np.abs(node_eta) < eta_max_extrap) & (node_pt > rms_pt_min)
        n_pu_eff = float(pu_central.sum()) / 30.0
    else:
        n_pu_eff = float(n_pu_proxy)
    thr = min_neutral_pt + min_neutral_pt_slope * n_pu_eff
    neutral_below = is_neutral & ((w * node_pt) < thr)
    w = np.where(neutral_below, np.float32(0.0), w)

    return w.astype(np.float32)


def cluster_puppi_no_truth_jets(
    data: dict,
    jet_R: float = 0.7,
    min_constituents: int = 3,
    min_pt: float = 10.0,
    *,
    dz_cut: float = 0.7,
    pv_pt_min: float = 1.0,
    **puppi_kwargs,
) -> dict | None:
    """Cluster PUPPI jets using the truth-free vertex association. Returns a dict
    keyed ``puppi_nt_jet_*`` so it can coexist with the truth-aware version
    in the same plot dict.
    """
    weights = compute_puppi_weights_no_truth(
        data, dz_cut=dz_cut, pv_pt_min=pv_pt_min, **puppi_kwargs,
    )
    if weights is None:
        return None
    j = cluster_puppi_jets(data, jet_R=jet_R, min_constituents=min_constituents, min_pt=min_pt, weights=weights)
    if j is None:
        return None
    return {f"puppi_nt_jet_{k.split('_jet_')[1]}": v for k, v in j.items()}


def cluster_puppi_jets(
    data: dict,
    jet_R: float = 0.7,
    min_constituents: int = 3,
    min_pt: float = 10.0,
    *,
    weights: np.ndarray | None = None,
) -> dict | None:
    """Cluster anti-/kt jets from PUPPI-weighted **clusters only**.

    Mirrors ``cluster_calo_jets`` (calo-only inputs) with PUPPI weights applied.
    Tracks are NOT added to the jet inputs because they would double-count the
    charged-particle energy that is already deposited in the calorimeter
    clusters — CMS PUPPI avoids this by consuming PFlow particles, which we
    don't have here. So PUPPI here is "calo-only with α-shape PU mitigation",
    directly comparable to the existing calo / calo-HS baselines.
    """
    try:
        import fastjet  # noqa: F401
    except ImportError as e:
        raise ImportError("fastjet is required for jet clustering") from e

    if weights is None:
        weights = compute_puppi_weights(data)
        if weights is None:
            return None

    required = ("node_valid", "node_pt", "node_eta", "node_phi", "node_e", "node_is_track")
    if any(data.get(k) is None for k in required):
        return None

    node_valid = np.asarray(data["node_valid"]).astype(bool)
    node_is_track = np.asarray(data["node_is_track"]).astype(bool)
    node_eta = np.asarray(data["node_eta"]).astype(np.float32)
    node_phi = np.asarray(data["node_phi"]).astype(np.float32)
    node_pt = _node_pt(data)

    weighted_pt = (node_pt * weights).astype(np.float32)
    selectable = (
        node_valid
        & (~node_is_track)
        & np.isfinite(weighted_pt)
        & np.isfinite(node_eta)
        & np.isfinite(node_phi)
        & (weighted_pt > 0)
    )

    # Lazy import to avoid circular import (reco_analysis → puppi → reco_analysis).
    from hepattn.experiments.odd_pileup_reco.reco_analysis import _cluster_jets_single

    try:
        from tqdm import tqdm
        iterator = tqdm(range(node_valid.shape[0]), desc="Jets (puppi)")
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


def _summarize(data: dict, weights: np.ndarray) -> None:
    node_valid = np.asarray(data["node_valid"]).astype(bool)
    node_is_track = np.asarray(data["node_is_track"]).astype(bool)
    tracks_mask = np.asarray(data["tracks_mask"]).astype(np.int8)
    node_eta = np.asarray(data["node_eta"]).astype(np.float32)
    node_phi = np.asarray(data["node_phi"]).astype(np.float32)
    node_pt = _node_pt(data)

    is_lv = node_valid & node_is_track & (tracks_mask == 1)
    is_pu = node_valid & node_is_track & (tracks_mask == 0)
    is_neutral = node_valid & (~node_is_track)

    print(f"events: {node_valid.shape[0]}, nodes/event: {node_valid.shape[1]}")
    print(f"  LV tracks: mean weight = {weights[is_lv].mean():.4f} (expect 1.0), count = {is_lv.sum()}")
    print(f"  PU tracks: mean weight = {weights[is_pu].mean():.4f} (expect 0.0), count = {is_pu.sum()}")
    print(f"  Neutrals : mean weight = {weights[is_neutral].mean():.4f}, count = {is_neutral.sum()}")
    print(f"  Neutrals  w>0.5: {(weights[is_neutral] > 0.5).sum()}/{is_neutral.sum()} "
          f"({100.0 * (weights[is_neutral] > 0.5).mean():.2f}%)")

    # Neutrals near vs. far from any LV track (per event, on first N events for speed).
    n_show = min(50, node_valid.shape[0])
    near_w, far_w = [], []
    for i in range(n_show):
        lv_idx = np.where(is_lv[i])[0]
        nu_idx = np.where(is_neutral[i])[0]
        if lv_idx.size == 0 or nu_idx.size == 0:
            continue
        deta = node_eta[i, nu_idx, None] - node_eta[i, lv_idx][None, :]
        dphi = _phi_diff(node_phi[i, nu_idx, None], node_phi[i, lv_idx][None, :])
        dr2_min = (deta * deta + dphi * dphi).min(axis=1)
        near = dr2_min < 0.4 * 0.4
        near_w.append(weights[i, nu_idx][near])
        far_w.append(weights[i, nu_idx][~near])
    if near_w and far_w:
        nw = np.concatenate(near_w) if near_w else np.array([])
        fw = np.concatenate(far_w) if far_w else np.array([])
        print(f"  Neutrals near LV (ΔR<0.4): mean weight = {nw.mean():.4f} (n={len(nw)})")
        print(f"  Neutrals far  LV (ΔR≥0.4): mean weight = {fw.mean():.4f} (n={len(fw)})")


def _main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Standalone PUPPI sanity check on an ODD pileup-reco H5.")
    p.add_argument("h5_path", type=str)
    p.add_argument("--max-events", type=int, default=20, help="Limit events for speed.")
    args = p.parse_args()

    from hepattn.experiments.odd_pileup_reco.reco_analysis import load_pflow_data

    data = load_pflow_data(Path(args.h5_path), event_start=0, event_stop=args.max_events)
    print(f"Loaded {Path(args.h5_path).name}: keys = {sorted(data)}")

    weights = compute_puppi_weights(data)
    if weights is None:
        print("compute_puppi_weights returned None — required fields missing.")
        raise SystemExit(1)

    _summarize(data, weights)


if __name__ == "__main__":
    _main()
