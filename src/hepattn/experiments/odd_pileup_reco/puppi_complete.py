"""Complete CMS-faithful PUPPI on the full truth particle list.

Closest analog to the production CMS implementation in
``/storage/agrp/barakma/cmssw/CommonTools/PileupAlgos``: operates on the
**per-particle** list (HS + PU together) with full reconstructed-like info
(pt, η, φ, vz, pdg_id) but without knowing per-particle which are HS vs PU.

Pipeline (mirrors ``PuppiContainer::getRMSAvg`` + ``PuppiAlgo::compute``):

  1. **CHS step.** Estimate PV_z = pT²-weighted z median of charged particles
     with track info. Classify each charged particle as LV (|vz − PV_z| < dz_cut)
     or PU. This is the analog of CMS's vertex-association loop.

  2. **α-shape.** For every particle (charged + neutral), compute
        α_i = log Σ_j pT_j² / ΔR_ij²  over all valid neighbors j within R0.
     This matches CMS default ``useCharged=False`` (PuppiAlgo.cc:221):
     the α sum is over the full particle list, not just LV-charged.

  3. **Calibration.** (α_med, α_rms) fit on the PU-charged sample within
     ``EtaMaxExtrap``. RMS is computed only over αs below the median
     (CMS ``applyLowPUCorr=True``).

  4. **LV-adjust** (PuppiAlgo.cc:159). Shift med/rms by
     ``sqrt(chi2_quantile(l, 1) * rms)`` where ``l`` is the fraction of LV
     αs below the PU median.

  5. **Per-particle weight.** Signed χ²(df=1) CDF on
     ``(α − α_med)|α − α_med| / α_rms²``. LV-charged → 1, PU-charged → 0,
     neutrals → χ² CDF weight in (0, 1).

  6. **MinNeutralPt cut.** Drop neutrals (or "trackless") whose weighted pT
     falls below ``min_neutral_pt + slope · N_PU``.

The "complete" naming reflects that this is the strongest possible PUPPI we
can build on this dataset: PFlow-quality candidates instead of raw clusters,
and α summed over the full particle list as in the production CMS algo.

Public API:
  - ``compute_puppi_weights_complete(particles) -> (n_events, n_particles) float32``
  - ``cluster_puppi_complete_jets(particles, ...) -> dict``
  - ``cluster_truth_hs_jets(particles, ...) -> dict`` (reference, HS only)
  - ``load_truth_particles(parquet_dir, event_start, event_stop) -> dict``
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


# PDG codes of charged particles (|charge| > 0). Covers everything in the
# ODD ttbar-PU200 sample plus common rare codes. Anything else → treated as
# neutral, which is the safe default for unknown nuclear codes.
_CHARGED_PDG_ABS = frozenset({
    11, 13, 15,                                   # leptons (e, μ, τ)
    211, 213, 321, 411, 413, 421, 431, 521,       # mesons
    2212, 2214, 3112, 3114, 3222, 3224,           # baryons
    3312, 3314, 3334, 4122, 4222, 4232,           # heavy / hyperons
})


def _is_charged(pdg_id: np.ndarray) -> np.ndarray:
    apdg = np.abs(pdg_id).astype(np.int64)
    out = np.zeros(apdg.shape, dtype=bool)
    for code in _CHARGED_PDG_ABS:
        out |= apdg == code
    # Nuclear codes (1000..) are mostly charged ions; treat as charged.
    out |= apdg >= 1000000000
    return out


def _phi_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a - b
    return (d + np.pi) % (2.0 * np.pi) - np.pi


def _pt2_weighted_median_z(z: np.ndarray, pt: np.ndarray) -> float:
    """pT²-weighted median z — the standard PUPPI/CHS PV estimator."""
    if z.size == 0:
        return 0.0
    order = np.argsort(z)
    z_s = z[order]
    w_s = (pt[order] ** 2).astype(np.float64)
    cumw = np.cumsum(w_s)
    half = cumw[-1] / 2.0
    idx = int(np.searchsorted(cumw, half))
    return float(z_s[min(idx, len(z_s) - 1)])


def compute_puppi_weights_complete_event(
    pt: np.ndarray,
    eta: np.ndarray,
    phi: np.ndarray,
    vz: np.ndarray,
    has_track: np.ndarray,
    pdg_id: np.ndarray,
    valid: np.ndarray,
    *,
    R0: float = 0.354,
    dz_cut: float = 1.997,
    pv_pt_min: float = 1.0,
    rms_pt_min: float = 0.066,
    min_neutral_pt: float = 0.918,
    min_neutral_pt_slope: float = 0.0085,
    n_pu_proxy: float | None = None,
    eta_max_extrap: float = 1.637,
    min_weight: float = 0.293,
    apply_lv_adjust: bool = True,
) -> np.ndarray:
    """Single-event PUPPI weight for every particle.

    Inputs are 1-D arrays of length N (particle count). Returns ``(N,)`` float32.
    """
    from scipy.special import ndtr, ndtri

    pt = np.asarray(pt, dtype=np.float32)
    eta = np.asarray(eta, dtype=np.float32)
    phi = np.asarray(phi, dtype=np.float32)
    vz = np.asarray(vz, dtype=np.float32)
    has_track = np.asarray(has_track).astype(bool)
    pdg_id = np.asarray(pdg_id, dtype=np.int64)
    valid = np.asarray(valid).astype(bool)

    valid = valid & np.isfinite(pt) & np.isfinite(eta) & np.isfinite(phi) & (pt > 0)
    charged = _is_charged(pdg_id) & has_track & valid
    neutral = valid & ~charged

    n = pt.shape[0]
    weights = np.zeros(n, dtype=np.float32)
    if not valid.any():
        return weights

    # --- 1. CHS: PV_z from pT²-weighted z median of charged-with-track ---
    pv_sel = charged & (pt > pv_pt_min) & np.isfinite(vz)
    pv_z = _pt2_weighted_median_z(vz[pv_sel], pt[pv_sel]) if pv_sel.any() else 0.0
    is_lv = charged & np.isfinite(vz) & (np.abs(vz - pv_z) < dz_cut)
    is_pu = charged & ~is_lv

    # --- 2. α-shape: sum over ALL valid particles within R0 ---
    # Match CMS default useCharged=False (PuppiAlgo.cc:221).
    # Implementation trick: sort all particles by η, then for each seed only
    # consider neighbors within ±R0 in η (binary-search bracket). This drops
    # the cost from O(N²) to ~O(N·k) where k = neighbors-in-η-strip ≪ N.
    src_idx = np.where(valid)[0]
    if src_idx.size == 0:
        return weights
    src_pt = pt[src_idx]
    src_eta = eta[src_idx]
    src_phi = phi[src_idx]

    order = np.argsort(src_eta)
    s_eta = src_eta[order]
    s_phi = src_phi[order]
    s_pt2 = (src_pt[order] ** 2).astype(np.float64)
    eta_lo = np.searchsorted(s_eta, src_eta - R0, side="left")
    eta_hi = np.searchsorted(s_eta, src_eta + R0, side="right")

    alpha = np.zeros(n, dtype=np.float32)
    for k, gi in enumerate(src_idx):
        i_lo = eta_lo[k]
        i_hi = eta_hi[k]
        if i_hi <= i_lo:
            continue
        d_eta = s_eta[i_lo:i_hi] - eta[gi]
        d_phi = _phi_diff(s_phi[i_lo:i_hi], np.float32(phi[gi]))
        dr2 = d_eta * d_eta + d_phi * d_phi
        mask = (dr2 < R0 * R0) & (dr2 > 1e-4)
        if not mask.any():
            continue
        s = (s_pt2[i_lo:i_hi][mask] / dr2[mask]).sum()
        if s > 0.0:
            alpha[gi] = np.float32(np.log(s))

    # --- 3. Calibration from PU-charged within eta_max_extrap ---
    cal_mask = is_pu & (np.abs(eta) < eta_max_extrap) & (pt > rms_pt_min) & (alpha != 0.0)
    if cal_mask.sum() < 2:
        cal_mask = valid & (np.abs(eta) < eta_max_extrap) & (alpha != 0.0)
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

        # --- 4. LV-adjust (CMS PuppiAlgo.cc:159) ---
        if apply_lv_adjust:
            lv_mask = is_lv & (np.abs(eta) < eta_max_extrap) & (alpha != 0.0)
            n_lv = int(lv_mask.sum())
            if n_lv > 0:
                n_lv_below = int((alpha[lv_mask] <= alpha_med).sum())
                n_pu_cal = int(cal_mask.sum())
                denom = n_lv_below + 0.5 * n_pu_cal
                l_adj = n_lv_below / denom if denom > 0 else 0.0
                if 0.0 < l_adj < 1.0:
                    # chi2_q(p, df=1) = (ndtri((p+1)/2))^2
                    chi2_q = float(ndtri((l_adj + 1.0) / 2.0)) ** 2
                    shift = np.float32(np.sqrt(chi2_q * alpha_rms))
                    alpha_med = np.float32(alpha_med - shift)
                    alpha_rms = np.float32(max(alpha_rms - shift, 1e-6))

    # --- 5. Per-particle weight via signed χ²(df=1) CDF ---
    diff = (alpha - alpha_med).astype(np.float32)
    lval = (diff * np.abs(diff) / (alpha_rms * alpha_rms)).astype(np.float32)
    sqrt_lval = np.sqrt(np.maximum(lval, 0.0))
    p_vals = (2.0 * ndtr(sqrt_lval) - 1.0).astype(np.float32)
    w = np.where(lval > 0.0, p_vals, np.float32(0.0))

    w = np.where(is_lv, np.float32(1.0), w)
    w = np.where(is_pu, np.float32(0.0), w)
    w = np.where(~valid, np.float32(0.0), w)
    w = np.where(w < min_weight, np.float32(0.0), w)

    # --- 6. MinNeutralPt cut (pileup-aware) ---
    if n_pu_proxy is None:
        # Auto-estimate ~N_PU vertices from PU-charged count (≈30 tracks/PU vertex).
        pu_cnt = is_pu & (np.abs(eta) < eta_max_extrap) & (pt > rms_pt_min)
        n_pu_eff = float(pu_cnt.sum()) / 30.0
    else:
        n_pu_eff = float(n_pu_proxy)
    thr = min_neutral_pt + min_neutral_pt_slope * n_pu_eff
    below_thr = neutral & (w * pt < thr)
    w = np.where(below_thr, np.float32(0.0), w)

    return w.astype(np.float32)


def compute_puppi_weights_complete(particles: dict, **kwargs) -> np.ndarray:
    """Per-event PUPPI weights for the full particle list.

    Expected keys in ``particles``: ``pt``, ``eta``, ``phi``, ``vz``,
    ``has_track``, ``pdg_id``, ``valid`` — each an object array of length
    ``n_events`` whose entries are 1-D arrays of particles (variable length
    per event).

    Returns an object array of length ``n_events`` of 1-D per-particle weights.
    """
    n_events = len(particles["pt"])
    weights_out = np.empty(n_events, dtype=object)
    try:
        from tqdm import tqdm
        it = tqdm(range(n_events), desc="PUPPI weights (complete)")
    except ImportError:
        it = range(n_events)
    for i in it:
        weights_out[i] = compute_puppi_weights_complete_event(
            particles["pt"][i],
            particles["eta"][i],
            particles["phi"][i],
            particles["vz"][i],
            particles["has_track"][i],
            particles["pdg_id"][i],
            particles["valid"][i],
            **kwargs,
        )
    return weights_out


def cluster_puppi_complete_jets(
    particles: dict,
    jet_R: float = 0.7,
    min_constituents: int = 3,
    min_pt: float = 10.0,
    *,
    weights: np.ndarray | None = None,
    **puppi_kwargs,
) -> dict:
    """Cluster jets from PUPPI-weighted particles. Returns ``puppi_jet_*`` dict."""
    try:
        import fastjet  # noqa: F401
    except ImportError as e:
        raise ImportError("fastjet is required for jet clustering") from e

    from hepattn.experiments.odd_pileup_reco.reco_analysis import _cluster_jets_single

    if weights is None:
        weights = compute_puppi_weights_complete(particles, **puppi_kwargs)

    n_events = len(particles["pt"])
    pt_l, eta_l, phi_l, m_l, nc_l = [], [], [], [], []
    try:
        from tqdm import tqdm
        it = tqdm(range(n_events), desc="Jets (puppi-complete)")
    except ImportError:
        it = range(n_events)
    for i in it:
        pt = np.asarray(particles["pt"][i], dtype=np.float32)
        eta = np.asarray(particles["eta"][i], dtype=np.float32)
        phi = np.asarray(particles["phi"][i], dtype=np.float32)
        valid = np.asarray(particles["valid"][i]).astype(bool)
        w = np.asarray(weights[i], dtype=np.float32)
        wpt = w * pt
        selectable = valid & np.isfinite(wpt) & np.isfinite(eta) & np.isfinite(phi) & (wpt > 0)
        ptetaphi = np.stack([wpt, eta, phi], axis=-1)
        out = _cluster_jets_single(ptetaphi, selectable, 0.5, jet_R, min_constituents, min_pt)
        pt_l.append(out[0]); eta_l.append(out[1]); phi_l.append(out[2])
        m_l.append(out[3]);  nc_l.append(out[4])

    return {
        "puppi_jet_pt":     np.array(pt_l, dtype=object),
        "puppi_jet_eta":    np.array(eta_l, dtype=object),
        "puppi_jet_phi":    np.array(phi_l, dtype=object),
        "puppi_jet_mass":   np.array(m_l, dtype=object),
        "puppi_jet_nconst": np.array(nc_l, dtype=object),
    }


def cluster_truth_hs_jets(
    particles: dict,
    jet_R: float = 0.7,
    min_constituents: int = 3,
    min_pt: float = 10.0,
) -> dict:
    """Cluster jets from truth HS particles only (``vertex_primary == 1``).
    Returns ``truth_jet_*`` dict for downstream matching.
    """
    try:
        import fastjet  # noqa: F401
    except ImportError as e:
        raise ImportError("fastjet is required for jet clustering") from e

    from hepattn.experiments.odd_pileup_reco.reco_analysis import _cluster_jets_single

    n_events = len(particles["pt"])
    pt_l, eta_l, phi_l, m_l, nc_l = [], [], [], [], []
    try:
        from tqdm import tqdm
        it = tqdm(range(n_events), desc="Jets (truth-hs)")
    except ImportError:
        it = range(n_events)
    for i in it:
        pt = np.asarray(particles["pt"][i], dtype=np.float32)
        eta = np.asarray(particles["eta"][i], dtype=np.float32)
        phi = np.asarray(particles["phi"][i], dtype=np.float32)
        valid = np.asarray(particles["valid"][i]).astype(bool)
        is_hs = np.asarray(particles["vertex_primary"][i]) == 1
        sel = valid & is_hs & np.isfinite(pt) & np.isfinite(eta) & np.isfinite(phi) & (pt > 0)
        ptetaphi = np.stack([pt, eta, phi], axis=-1)
        out = _cluster_jets_single(ptetaphi, sel, 0.5, jet_R, min_constituents, min_pt)
        pt_l.append(out[0]); eta_l.append(out[1]); phi_l.append(out[2])
        m_l.append(out[3]);  nc_l.append(out[4])

    return {
        "truth_jet_pt":     np.array(pt_l, dtype=object),
        "truth_jet_eta":    np.array(eta_l, dtype=object),
        "truth_jet_phi":    np.array(phi_l, dtype=object),
        "truth_jet_mass":   np.array(m_l, dtype=object),
        "truth_jet_nconst": np.array(nc_l, dtype=object),
    }


def load_truth_particles(
    parquet_dir: str | Path,
    event_start: int = 0,
    event_stop: int | None = None,
) -> dict:
    """Load truth particles from chunked ``target_particles-*.parquet`` files.

    Each event becomes a 1-D entry in object arrays for: ``pt``, ``eta``,
    ``phi``, ``e``, ``vz``, ``has_track``, ``pdg_id``, ``vertex_primary``,
    ``valid``.
    """
    import polars as pl

    parqs = sorted(Path(parquet_dir).glob("target_particles-*.parquet"))
    if not parqs:
        raise FileNotFoundError(f"No target_particles-*.parquet in {parquet_dir}")

    pt_l, eta_l, phi_l, e_l, vz_l, ht_l, pdg_l, vp_l, vld_l = [], [], [], [], [], [], [], [], []
    n_loaded = 0
    n_skipped = 0
    for p in parqs:
        df = pl.read_parquet(p)
        for row in df.iter_rows(named=True):
            if n_skipped < event_start:
                n_skipped += 1
                continue
            if event_stop is not None and n_loaded >= (event_stop - event_start):
                break
            pt_l.append(np.asarray(row["pt"], dtype=np.float32))
            eta_l.append(np.asarray(row["eta"], dtype=np.float32))
            phi_l.append(np.asarray(row["phi"], dtype=np.float32))
            e_l.append(np.asarray(row["energy"], dtype=np.float32))
            vz_l.append(np.asarray(row["vz"], dtype=np.float32))
            ht_l.append(np.asarray(row["has_track"], dtype=bool))
            pdg_l.append(np.asarray(row["pdg_id"], dtype=np.int64))
            vp_l.append(np.asarray(row["vertex_primary"], dtype=np.int64))
            vld_l.append(np.ones(len(row["pt"]), dtype=bool))
            n_loaded += 1
        if event_stop is not None and n_loaded >= (event_stop - event_start):
            break

    return {
        "pt":              np.array(pt_l,  dtype=object),
        "eta":             np.array(eta_l, dtype=object),
        "phi":             np.array(phi_l, dtype=object),
        "e":               np.array(e_l,   dtype=object),
        "vz":              np.array(vz_l,  dtype=object),
        "has_track":       np.array(ht_l,  dtype=object),
        "pdg_id":          np.array(pdg_l, dtype=object),
        "vertex_primary":  np.array(vp_l,  dtype=object),
        "valid":           np.array(vld_l, dtype=object),
    }
