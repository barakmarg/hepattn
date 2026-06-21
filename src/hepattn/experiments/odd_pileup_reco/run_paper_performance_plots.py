"""Paper performance plots — parallel map-reduce over H5 shards.

Produces the paper figure set from one or more prediction-writer H5 shards
(e.g. the per-1k-event split files from ``run_forward_pass``):

  1. jet_resolution            — 3x2 jet residual panels (deta, dphi, #const,
                                 abs dpT, rel dpT/pT [log], energy [log]),
                                 each curve labelled mu + IQR; GLOW-UP / PUPPI / Target
  2. jet_resolution_binned     — mean & IQR vs Target jet pT, GLOW-UP vs PUPPI
  3. track_f1_vs_pt            — track-classifier F1 vs pT
  4. n_particles_by_pt_bin     — per-event multiplicity by pT bin (GLOW-UP/Target)
  5. feature_scatter           — pT/eta/phi density, GLOW-UP vs Target
  6. calo_recall_vs_pt         — HS cluster recall vs cluster energy
  7. calo_pred_vs_truth_e_dist — per-cluster energy, pred-HS vs truth-HS

Design (fast + low memory + resumable):
  * The PUPPI parquet is read ONCE (single scan) for all not-yet-cached shards,
    then handed to workers via fork copy-on-write (no per-shard re-scan, no
    pickling of big arrays).
  * Each shard is analysed in its own worker process (one shard's H5 in RAM at a
    time, incidence tensors never read) and reduced to small mergeable
    aggregates cached to ``<out>/state/<shard>.pkl`` (resume: cached shards skip).
  * REDUCE concatenates the small residual arrays and sums the histogram/count
    arrays; PLOT renders the figures (PDF + PNG).

PUPPI is REQUIRED: if the parquet dir does not contain a shard's events (or PUPPI
fails), the run raises and stops — it never silently drops the PUPPI curve.

Naming: the ML model (formerly "pflow") is **GLOW-UP**; truth is **Target**.
"""
from __future__ import annotations

# Pin BLAS/polars to one thread *before* numpy is imported so forked workers
# inherit single-threaded math libs (N workers x 1 thread = clean utilisation).
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("POLARS_MAX_THREADS", "1")

import argparse
import glob
import json
import multiprocessing as mp
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm

from hepattn.experiments.odd_pileup_reco.eval_data import load_eval_data_from_h5
from hepattn.experiments.odd_pileup_reco.puppi_charged_subtract import (
    cluster_puppi_charged_subtract_jets,
    compute_puppi_weights_charged_subtract,
    load_charged_subtract_events,
)
from hepattn.experiments.odd_pileup_reco.reco_analysis import cluster_jets, load_pflow_data
from hepattn.experiments.odd_pileup_reco.run_puppi_jet_resolution_charged_subtract_full import (
    DEFAULT_H5,
    _make_binned_plots,
    _match_and_residuals,
)

# PUPPI parquet source: the all-vertices "paper" sample the H5 was generated from
# (the chunked dir does NOT contain these events).
DEFAULT_PARQUET_DIR = "/storage/agrp/barakma/PileupODD/data/ttbar_pu200_all_vertices_paper"
# PUPPI optuna params (v2, anti-kT R=0.4 tuning).
DEFAULT_BEST_JSON = (
    "/storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco/"
    "optuna_puppi_charged_subtract/puppi_charged_subtract_v2_antikt_R04_best.json"
)

# ── Paper labels & palette ────────────────────────────────────────────────
GLOWUP = "GLOW-UP"
TARGET = "Target"
PUPPI_LABEL = "PUPPI"
GLOWUP_COLOR = "tomato"
TARGET_COLOR = "darkorange"
PUPPI_COLOR = "seagreen"

# ── Fixed bin edges (kept constant so per-shard partials are mergeable) ─────
TRACK_PT_BINS = np.logspace(np.log10(0.3), np.log10(200), 25)            # 24 bins
FEAT_BINS = [
    np.linspace(0.0, 200.0, 70),       # pt
    np.linspace(-4.0, 4.0, 70),        # eta
    np.linspace(-np.pi, np.pi, 70),    # phi
]
FEAT_LABELS = ["pt [GeV]", "eta", "phi [rad]"]
FEAT_ROWS = ["All", "Charged", "Neutral"]
NPART_PT_BINS = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 200.0)
CALO_E_BINS = np.logspace(np.log10(0.05), np.log10(2000.0), 60)
RESIDUAL_KEYS = ("dpt", "dpt_over_truth", "deta", "dphi", "truth_pt")
IND_THRESHOLD = 0.5
PARQUET_EVENTS_PER_FILE = 100   # event_ids per parquet shard file (NNNNN = eid // this)


# ---------------------------------------------------------------------------
# Per-shard aggregate builders (small, mergeable)
# ---------------------------------------------------------------------------

def _feature_hist(data: dict) -> np.ndarray:
    truth_f = data["truth_ptetaphi"].reshape(-1, 3)
    pflow_f = data["pflow_ptetaphi"].reshape(-1, 3)
    tc = np.asarray(data["truth_class"]).ravel()
    pc = np.asarray(data["pflow_class"]).ravel()
    ti = np.asarray(data["truth_indicator"]).ravel() > IND_THRESHOLD
    pi = np.asarray(data["pflow_indicator"]).ravel() > IND_THRESHOLD
    base = pi & ti
    masks = [base, base & (pc < 3) & (tc < 3), base & (pc >= 3) & (pc < 5) & (tc >= 3) & (tc < 5)]
    nb = len(FEAT_BINS[0]) - 1
    out = np.zeros((3, 3, nb, nb), dtype=np.float64)
    for j, m in enumerate(masks):
        for i in range(3):
            t = truth_f[:, i][m]
            p = pflow_f[:, i][m]
            fin = np.isfinite(t) & np.isfinite(p)
            if fin.any():
                h, _, _ = np.histogram2d(t[fin], p[fin], bins=[FEAT_BINS[i], FEAT_BINS[i]])
                out[j, i] = h
    return out


def _npart_counts(data: dict) -> tuple[np.ndarray, np.ndarray]:
    truth_pt = data["truth_ptetaphi"][..., 0]
    pflow_pt = data["pflow_ptetaphi"][..., 0]
    tv = np.asarray(data["truth_indicator"]) > IND_THRESHOLD
    pv = np.asarray(data["pflow_indicator"]) > IND_THRESHOLD
    nbin = len(NPART_PT_BINS) - 1
    nt = np.zeros((truth_pt.shape[0], nbin), dtype=np.int64)
    npr = np.zeros((pflow_pt.shape[0], nbin), dtype=np.int64)
    for i in range(nbin):
        lo, hi = NPART_PT_BINS[i], NPART_PT_BINS[i + 1]
        nt[:, i] = (tv & (truth_pt >= lo) & (truth_pt < hi)).sum(axis=-1)
        npr[:, i] = (pv & (pflow_pt >= lo) & (pflow_pt < hi)).sum(axis=-1)
    return nt, npr


def _track_counts(track_data: dict) -> dict:
    nb = len(TRACK_PT_BINS) - 1
    out = {k: np.zeros(nb, dtype=np.int64) for k in ("tp", "fp", "fn", "total")}
    probs = track_data.get("probs")
    if probs is None or len(probs) == 0:
        return out
    is_pred = np.asarray(probs) > 0.5
    is_true = np.asarray(track_data["truth"]) == 1
    pt = np.asarray(track_data["pt"])
    idx = np.digitize(pt, TRACK_PT_BINS) - 1
    inb = (idx >= 0) & (idx < nb)
    out["tp"] = np.bincount(idx[inb & is_pred & is_true], minlength=nb)[:nb]
    out["fp"] = np.bincount(idx[inb & is_pred & ~is_true], minlength=nb)[:nb]
    out["fn"] = np.bincount(idx[inb & ~is_pred & is_true], minlength=nb)[:nb]
    out["total"] = np.bincount(idx[inb], minlength=nb)[:nb]
    return out


def _calo_counts(cluster_data: dict) -> dict:
    nb = len(CALO_E_BINS) - 1
    out = {k: np.zeros(nb, dtype=np.int64) for k in ("truth", "recovered", "pred")}
    e = np.asarray(cluster_data.get("total_e", []), dtype=float)
    mp_ = cluster_data.get("mask_pred")
    mt = cluster_data.get("mask_truth")
    if e.size == 0 or mp_ is None or mt is None:
        return out
    mp_ = np.asarray(mp_).astype(bool)
    mt = np.asarray(mt).astype(bool)
    idx = np.digitize(e, CALO_E_BINS) - 1
    inb = (idx >= 0) & (idx < nb) & np.isfinite(e) & (e > 0)
    out["truth"] = np.bincount(idx[inb & mt], minlength=nb)[:nb]
    out["recovered"] = np.bincount(idx[inb & mp_ & mt], minlength=nb)[:nb]
    out["pred"] = np.bincount(idx[inb & mp_], minlength=nb)[:nb]
    return out


def _jet_nconst(jets: dict, prefix: str) -> np.ndarray:
    chunks = [np.asarray(a) for a in jets[f"{prefix}_jet_nconst"] if len(a) > 0]
    return np.concatenate(chunks).astype(np.float32) if chunks else np.array([], dtype=np.float32)


def _jet_energy(jets: dict, prefix: str) -> np.ndarray:
    chunks = []
    for pt, eta, mass in zip(jets[f"{prefix}_jet_pt"], jets[f"{prefix}_jet_eta"], jets[f"{prefix}_jet_mass"]):
        if len(pt) == 0:
            continue
        e = np.sqrt(np.clip((pt * np.cosh(eta)) ** 2 + np.maximum(mass, 0.0) ** 2, 0.0, None))
        e = e[np.isfinite(e) & (e > 0)]
        if len(e) > 0:
            chunks.append(e)
    return np.concatenate(chunks).astype(np.float32) if chunks else np.array([], dtype=np.float32)


# ---------------------------------------------------------------------------
# MAP — analyse one shard in a worker process (PUPPI events inherited via fork)
# ---------------------------------------------------------------------------

def _puppi_jets_for_shard(parquet_dir: str, eids: list[int], events_per_file: int,
                          subtract_pu: bool, puppi_params: dict, jet_cfg: dict) -> dict:
    """PUPPI jets for a shard, computed ONE parquet file at a time.

    Loading all ~1000 events at once peaks at ~12 GB (the loader materialises
    every event's per-cluster deposits as Python objects). Processing per file
    (<=events_per_file events) keeps the peak at ~one file (~1 GB) and reads each
    file exactly once; only the tiny per-event jet arrays are kept. Returned as
    per-event object arrays in ``eids`` order. Raises if any event is missing.
    """
    import collections

    keys = ("pt", "eta", "phi", "mass", "nconst")
    by_eid: dict[int, tuple] = {}
    groups: dict[int, list[int]] = collections.defaultdict(list)
    for e in eids:
        groups[e // events_per_file].append(e)        # one group == one parquet file

    for gids in groups.values():
        ev = load_charged_subtract_events(parquet_dir, event_ids=gids, events_per_file=events_per_file)
        w = compute_puppi_weights_charged_subtract(ev, subtract_pu_charged=subtract_pu, **puppi_params)
        pj = cluster_puppi_charged_subtract_jets(ev, weights=w, subtract_pu_charged=subtract_pu, **jet_cfg)
        for i, e in enumerate(gids):
            by_eid[e] = tuple(pj[f"puppi_jet_{k}"][i] for k in keys)

    out: dict = {}
    for j, k in enumerate(keys):
        arr = np.empty(len(eids), dtype=object)
        for i, e in enumerate(eids):
            arr[i] = by_eid[e][j]
        out[f"puppi_jet_{k}"] = arr
    return out


def analyze_shard(shard_path: str, parquet_dir: str, events_per_file: int,
                  jet_cfg: dict, subtract_pu: bool, puppi_params: dict) -> dict:
    shard = Path(shard_path)
    data = load_pflow_data(shard_path, load_incidence=False)
    track_data, cluster_data, _ = load_eval_data_from_h5(shard, load_reco=False)
    n_events = int(data["pflow_ptetaphi"].shape[0])

    with h5py.File(shard, "r") as f:
        eids = [int(x) for x in np.asarray(f["events"]["event_number"][:]).reshape(-1)]

    # GLOW-UP + Target jets + matched residuals.
    jets = cluster_jets(data, **jet_cfg)
    glow_res = _match_and_residuals(jets, jets, "pflow")

    # PUPPI — only this shard's files, one file at a time (memory ~1 file).
    # Mandatory: a missing event raises (KeyError) and aborts the run.
    puppi_jets = _puppi_jets_for_shard(parquet_dir, eids, events_per_file,
                                       subtract_pu, puppi_params, jet_cfg)
    puppi_res = _match_and_residuals(puppi_jets, jets, "puppi")

    npart_truth, npart_pred = _npart_counts(data)
    return {
        "n_events": n_events,
        "glow_res": {k: np.asarray(glow_res.get(k, []), dtype=np.float32) for k in RESIDUAL_KEYS},
        "puppi_res": {k: np.asarray(puppi_res.get(k, []), dtype=np.float32) for k in RESIDUAL_KEYS},
        "nconst": {
            "pflow": _jet_nconst(jets, "pflow"),
            "puppi": _jet_nconst(puppi_jets, "puppi"),
            "truth": _jet_nconst(jets, "truth"),
        },
        "energy": {
            "pflow": _jet_energy(jets, "pflow"),
            "puppi": _jet_energy(puppi_jets, "puppi"),
            "truth": _jet_energy(jets, "truth"),
        },
        "npart_truth": npart_truth,
        "npart_pred": npart_pred,
        "feat_hist": _feature_hist(data),
        "track": _track_counts(track_data),
        "calo": _calo_counts(cluster_data),
    }


# ---------------------------------------------------------------------------
# REDUCE
# ---------------------------------------------------------------------------

def _concat_res(reslist: list[dict]) -> dict:
    return {k: np.concatenate([r[k] for r in reslist]) for k in RESIDUAL_KEYS}


def merge_aggregates(aggs: list[dict]) -> dict:
    if not aggs:
        raise ValueError("no shard aggregates to merge")
    m: dict = {"n_events": sum(a["n_events"] for a in aggs)}
    m["glow_res"] = _concat_res([a["glow_res"] for a in aggs])
    m["puppi_res"] = _concat_res([a["puppi_res"] for a in aggs])
    m["nconst"] = {src: np.concatenate([a["nconst"][src] for a in aggs]) for src in ("pflow", "puppi", "truth")}
    m["energy"] = {src: np.concatenate([a["energy"][src] for a in aggs]) for src in ("pflow", "puppi", "truth")}
    m["npart_truth"] = np.vstack([a["npart_truth"] for a in aggs])
    m["npart_pred"] = np.vstack([a["npart_pred"] for a in aggs])
    m["feat_hist"] = np.sum([a["feat_hist"] for a in aggs], axis=0)
    m["track"] = {k: np.sum([a["track"][k] for a in aggs], axis=0) for k in ("tp", "fp", "fn", "total")}
    m["calo"] = {k: np.sum([a["calo"][k] for a in aggs], axis=0) for k in ("truth", "recovered", "pred")}
    return m


# ---------------------------------------------------------------------------
# PLOT
# ---------------------------------------------------------------------------

def _pct_bins(d: np.ndarray, n: int, positive: bool = False) -> np.ndarray:
    d = d[np.isfinite(d)]
    if positive:
        d = d[d > 0]
    if d.size < 2:
        return np.linspace(0.0, 1.0, n)
    lo, hi = np.percentile(d, [0.5, 99.5])
    if positive:
        lo = max(float(lo), 1e-3)
    if hi <= lo:
        hi = lo + 1e-6
    return np.linspace(lo, hi, n)


def _plot_jet_resolution(merged: dict):
    """Reference 3x2 jet-resolution panels, mu + IQR labels; GLOW-UP/PUPPI/Target."""
    from scipy.stats import iqr

    g, pu = merged["glow_res"], merged["puppi_res"]
    nc, en = merged["nconst"], merged["energy"]

    def series(key: str, which: str) -> np.ndarray:
        if key == "nconst":
            return nc.get(which, np.array([]))
        if key == "energy":
            return en.get(which, np.array([]))
        if which == "pflow":
            return g.get(key, np.array([]))
        if which == "puppi":
            return pu.get(key, np.array([]))
        return np.array([])     # truth has no matched residual

    def lbl(name, d):
        d = d[np.isfinite(d)]
        if d.size == 0:
            return name
        return rf"{name}  $\mu$={np.mean(d):.3f}, IQR={iqr(d):.3f}"

    panels = [
        ("deta", r"Jet $\Delta\eta$", np.linspace(-0.2, 0.2, 50), True, False),
        ("dphi", r"Jet $\Delta\phi$", np.linspace(-0.2, 0.2, 50), True, False),
        ("nconst", "Jet # Constituents", None, True, False),
        ("dpt", r"Jet absolute $\Delta p_T$ [GeV]", None, True, False),
        ("dpt_over_truth", r"Jet relative $\Delta p_T / p_T^{\mathrm{Target}}$", np.linspace(-1.0, 4.0, 110), False, True),
        ("energy", "Jet energy [GeV]", None, False, True),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    for idx, (key, xlabel, bins, density, logy) in enumerate(panels):
        ax = axes[idx // 2, idx % 2]
        glow = np.asarray(series(key, "pflow"), dtype=float)
        if glow.size == 0:
            ax.text(0.5, 0.5, "no matched jets", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel(xlabel)
            continue

        if bins is not None:
            b = bins
        elif key == "nconst":
            allv = np.concatenate([series(key, w) for w in ("pflow", "puppi", "truth")])
            b = np.arange(allv.min() - 0.5, allv.max() + 1.5) if allv.size else 20
        elif key == "energy":
            b = _pct_bins(glow, 80, positive=True)
        else:   # dpt
            b = _pct_bins(glow, 90)

        ax.hist(glow, bins=b, histtype="stepfilled", alpha=0.5, density=density,
                color=GLOWUP_COLOR, label=lbl(GLOWUP, glow))
        pud = np.asarray(series(key, "puppi"), dtype=float)
        if pud.size:
            ax.hist(pud, bins=b, histtype="step", linestyle="-.", linewidth=1.8, density=density,
                    color=PUPPI_COLOR, label=lbl(PUPPI_LABEL, pud))
        if key in ("nconst", "energy"):
            td = np.asarray(series(key, "truth"), dtype=float)
            if td.size:
                ax.hist(td, bins=b, histtype="stepfilled", alpha=0.4, density=density,
                        color=TARGET_COLOR, label=lbl(TARGET, td))

        if logy:
            ax.set_yscale("log")
            ylo, yhi = ax.get_ylim()
            ax.set_ylim(max(ylo, 1e-4), yhi * 1.35)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density" if density else "Count")
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Jet resolution: {GLOWUP} vs {PUPPI_LABEL} (vs {TARGET})")
    fig.tight_layout()
    return fig


def _plot_track_f1_from_counts(track: dict):
    edges = TRACK_PT_BINS
    centers = np.sqrt(edges[:-1] * edges[1:])
    tp, fp, fn, n = (track[k].astype(float) for k in ("tp", "fp", "fn", "total"))
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
        err = np.sqrt(np.clip(f1 * (1 - f1), 0, None) / np.where(n > 0, n, 1))
    f1 = np.where(n > 0, f1, np.nan)
    err = np.where(n > 0, err, np.nan)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(centers, f1, yerr=err, fmt="o-", capsize=3, color="steelblue")
    ax.axhline(1.0, color="gray", lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("Track pT [GeV]")
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.1)
    ax.set_title("Track F1 vs pT")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_n_particles_from_counts(npart_truth: np.ndarray, npart_pred: np.ndarray):
    n_panels = len(NPART_PT_BINS) - 1
    n_cols = 2
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 3.8 * n_rows), squeeze=False)
    axes_flat = axes.ravel()
    for i in range(n_panels):
        ax = axes_flat[i]
        tcol, pcol = npart_truth[:, i], npart_pred[:, i]
        max_count = max(1, int(tcol.max(initial=0)), int(pcol.max(initial=0)))
        bins = np.arange(-0.5, max_count + 1.5, 1.0)
        ax.hist(tcol, bins=bins, histtype="stepfilled", alpha=0.5, label=TARGET, color=TARGET_COLOR)
        ax.hist(pcol, bins=bins, histtype="step", label=GLOWUP, color=GLOWUP_COLOR)
        ax.set_title(f"{NPART_PT_BINS[i]:g} <= pt < {NPART_PT_BINS[i + 1]:g} GeV")
        ax.set_xlabel("Particles / event")
        ax.set_ylabel("Events")
        ax.legend(fontsize=8)
    for ax in axes_flat[n_panels:]:
        ax.axis("off")
    fig.suptitle(f"Particle multiplicity per event by pt bin ({GLOWUP} vs {TARGET})", y=1.01)
    fig.tight_layout()
    return fig


def _plot_feature_scatter_from_hist(feat_hist: np.ndarray):
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for i in range(3):
        for j in range(3):
            ax = axes[j, i]
            h = feat_hist[j, i]
            edges = FEAT_BINS[i]
            if h.sum() > 0:
                ax.pcolormesh(edges, edges, np.ma.masked_where(h.T == 0, h.T),
                              norm=LogNorm(vmin=1), cmap="viridis")
                lo, hi = edges[0], edges[-1]
                ax.plot([lo, hi], [lo, hi], ls="--", color="red", lw=1.0)
                ax.text(0.03, 0.97, f"n={int(h.sum())}", transform=ax.transAxes,
                        va="top", ha="left", fontsize=8, color="white",
                        bbox={"facecolor": "black", "alpha": 0.35, "pad": 2})
            else:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel(f"{TARGET} {FEAT_LABELS[i]}")
            ax.set_ylabel(f"{GLOWUP} {FEAT_LABELS[i]}")
            ax.set_title(FEAT_ROWS[j])
    fig.suptitle(f"Feature density (hist2d): {TARGET} vs {GLOWUP}", y=1.01)
    fig.tight_layout()
    return fig


def _plot_calo_recall_from_counts(calo: dict):
    edges = CALO_E_BINS
    centers = np.sqrt(edges[:-1] * edges[1:])
    truth, rec = calo["truth"].astype(float), calo["recovered"].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = np.where(truth > 0, rec / truth, np.nan)
        err = np.where(truth > 0, np.sqrt(np.clip(recall * (1 - recall), 0, None) / np.where(truth > 0, truth, 1)), np.nan)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(centers, recall, yerr=err, fmt="o-", capsize=3, color=GLOWUP_COLOR, label=GLOWUP)
    ax.axhline(1.0, color="gray", lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("calorimeter cluster energy [GeV]")
    ax.set_ylabel("HS cluster recall")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"{GLOWUP} pileup-removal recall vs cluster energy")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def _plot_calo_e_dist_from_hist(calo: dict):
    edges = CALO_E_BINS
    truth, pred = calo["truth"].astype(float), calo["pred"].astype(float)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.stairs(truth, edges, color=TARGET_COLOR, linewidth=2.0, label=f"{TARGET} (truth HS)  n={int(truth.sum())}")
    ax.stairs(pred, edges, color=GLOWUP_COLOR, linewidth=2.0, label=f"{GLOWUP} (pred HS)  n={int(pred.sum())}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("calorimeter cluster energy [GeV]")
    ax.set_ylabel("clusters")
    ax.set_title("Calorimeter cluster energy in pileup removal: pred vs truth HS")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def render_all(merged: dict) -> dict:
    methods = [(GLOWUP, merged["glow_res"]), (PUPPI_LABEL, merged["puppi_res"])]
    return {
        "jet_resolution": _plot_jet_resolution(merged),
        "jet_resolution_binned": _make_binned_plots(methods),
        "track_f1_vs_pt": _plot_track_f1_from_counts(merged["track"]),
        "n_particles_by_pt_bin": _plot_n_particles_from_counts(merged["npart_truth"], merged["npart_pred"]),
        "feature_scatter": _plot_feature_scatter_from_hist(merged["feat_hist"]),
        "calo_recall_vs_pt": _plot_calo_recall_from_counts(merged["calo"]),
        "calo_pred_vs_truth_e_dist": _plot_calo_e_dist_from_hist(merged["calo"]),
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_fig(fig, out_dir: Path, name: str, dpi: int) -> None:
    if fig is None:
        print(f"  [skip] {name}: figure is None")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"  saved {path}")
    plt.close(fig)


def _resolve_h5_inputs(h5_args: list[str]) -> list[Path]:
    files: list[Path] = []
    for a in h5_args:
        if any(ch in a for ch in "*?["):
            files.extend(sorted(Path(x) for x in glob.glob(a)))
            continue
        p = Path(a)
        if p.is_dir():
            parts = sorted(p.glob("*__test*part*.h5"))
            files.extend(parts if parts else sorted(p.glob("*__test*.h5")))
        else:
            files.append(p)
    seen: set[Path] = set()
    out: list[Path] = []
    for f in files:
        f = f.resolve()
        if f in seen:
            continue
        seen.add(f)
        if not f.is_file():
            raise FileNotFoundError(f"H5 file not found: {f}")
        out.append(f)
    if not out:
        raise ValueError("No H5 files resolved from --h5")
    return out


def _shard_event_numbers(path: Path) -> list[int]:
    with h5py.File(path, "r") as h:
        return [int(x) for x in np.asarray(h["events"]["event_number"][:]).reshape(-1)]


def _load_pickle(path: Path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _save_pickle(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def _init_worker() -> None:
    """Silence per-worker stderr (inner tqdm + numeric warnings); pin threads.
    Exceptions still propagate to the parent via the future, so failures surface.
    """
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS"):
        os.environ[var] = "1"
    sys.stderr = open(os.devnull, "w")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5", type=str, nargs="+", default=[DEFAULT_H5],
                   help="One or more H5 shards, a directory, or a glob.")
    p.add_argument("--parquet-dir", type=str, default=DEFAULT_PARQUET_DIR)
    p.add_argument("--events-per-file", type=int, default=PARQUET_EVENTS_PER_FILE,
                   help="event_ids per parquet shard file; lets PUPPI read only the "
                        "~N/this files a shard needs instead of the whole dir.")
    p.add_argument("--best-json", type=str, default=DEFAULT_BEST_JSON)
    p.add_argument("--out-dir", type=str, default=None,
                   help="Output directory (default: <first-h5-dir>/paper_plots).")
    p.add_argument("--state-dir", type=str, default=None,
                   help="Per-shard aggregate cache (default: <out-dir>/state).")
    p.add_argument("--workers", type=int, default=None,
                   help="Worker processes (default: min(cpu_count, n_shards)).")
    p.add_argument("--force", action="store_true", help="Recompute all shards (ignore cache).")
    p.add_argument("--jet-R", type=float, default=0.4)
    p.add_argument("--min-constituents", type=int, default=3)
    p.add_argument("--min-pt", type=float, default=10.0)
    p.add_argument("--jet-algorithm", type=str, default="antikt", choices=["antikt", "kt"])
    p.add_argument("--no-subtract-pu", action="store_true", default=False)
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    files = _resolve_h5_inputs(args.h5)
    out_dir = Path(args.out_dir) if args.out_dir else (files[0].parent / "paper_plots")
    state_dir = Path(args.state_dir) if args.state_dir else (out_dir / "state")
    state_dir.mkdir(parents=True, exist_ok=True)
    print(f"Resolved {len(files)} shard(s); out_dir={out_dir}", flush=True)

    shard_events_n = {f: _shard_event_numbers(f) for f in files}
    shard_n = {f: len(v) for f, v in shard_events_n.items()}
    total_events = sum(shard_n.values())

    puppi_params: dict = {}
    if args.best_json and Path(args.best_json).is_file():
        puppi_params = json.loads(Path(args.best_json).read_text()).get("best_params", {})
        print(f"Loaded PUPPI best params: {puppi_params}", flush=True)
    jet_cfg = dict(jet_R=args.jet_R, min_constituents=args.min_constituents,
                   min_pt=args.min_pt, jet_algorithm=args.jet_algorithm)
    subtract_pu = not args.no_subtract_pu

    def state_path(f: Path) -> Path:
        return state_dir / f"{f.stem}.pkl"

    aggregates: dict[Path, dict] = {}
    todo: list[Path] = []
    for f in files:
        sp = state_path(f)
        if sp.is_file() and not args.force:
            try:
                aggregates[f] = _load_pickle(sp)
                continue
            except Exception as exc:  # noqa: BLE001 - corrupt cache -> recompute
                print(f"  cache unreadable for {f.name} ({exc}); recomputing", file=sys.stderr)
        todo.append(f)

    workers = args.workers if args.workers else min(os.cpu_count() or 1, max(1, len(todo)))
    print(f"{len(aggregates)} cached, {len(todo)} to analyse on {workers} worker(s)", flush=True)

    bar = tqdm(total=total_events, unit="event", desc="Analyzing", smoothing=0.1)
    for f in files:
        if f in aggregates:
            bar.update(shard_n[f])
    try:
        if todo:
            ctx = mp.get_context("fork")
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                                     initializer=_init_worker) as ex:
                fut2f = {ex.submit(analyze_shard, str(f), args.parquet_dir,
                                   args.events_per_file, jet_cfg, subtract_pu, puppi_params): f
                         for f in todo}
                for fut in as_completed(fut2f):
                    f = fut2f[fut]
                    agg = fut.result()           # propagates worker exceptions -> abort
                    _save_pickle(state_path(f), agg)
                    aggregates[f] = agg
                    bar.update(shard_n[f])
    finally:
        bar.close()

    ordered = [aggregates[f] for f in files if f in aggregates]
    merged = merge_aggregates(ordered)
    _save_pickle(out_dir / "merged_aggregate.pkl", merged)
    print(f"Merged {len(ordered)} shard(s), {merged['n_events']} events. Rendering...", flush=True)

    for name, fig in render_all(merged).items():
        save_fig(fig, out_dir, name, args.dpi)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
