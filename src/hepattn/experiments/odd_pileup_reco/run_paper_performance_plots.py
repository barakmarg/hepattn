"""Paper performance plots — parallel map-reduce over H5 shards.

Produces the paper figure set from one or more prediction-writer H5 shards
(e.g. the per-1k-event split files from ``run_forward_pass``):

  1. jet_resolution            — 3x2 jet residual panels (deta, dphi, #const,
                                 abs dpT, rel dpT/pT [log], energy [log]),
                                 each curve labelled mu + IQR; GLOW-UP / PUPPI / Target
  2. jet_resolution_binned     — mean & IQR vs Target jet pT, GLOW-UP vs PUPPI
  3. track_f1_vs_pt            — track-classifier F1 vs pT
  4. n_particles_by_pt_bin     — per-event multiplicity by pT bin (GLOW-UP/Target)
  4b. class_counts_*           — per-class particle counts (bar chart, log y),
                                 GLOW-UP vs Target; nocut / pt0p5 / pt1 / pt2
  4c. class_pt_spectrum        — per-class particle pT spectrum (count vs pT, log-log),
                                 GLOW-UP (solid) vs Target (dashed), 5 classes
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

# ── Figure style ──────────────────────────────────────────────────────────
# Adapted from cajohare/HowToMakeAPlot. Its three .mplstyle sheets are a single
# self-consistent set tuned for `figure.figsize : 13, 12` -- ticks 15/10 pt,
# tick widths 2/1, axes.linewidth 2.5, axis labels 45 pt, ticks 30 pt. Our
# figures are 7.5-8 in wide, a little under 0.6x their linear size, so the
# ABSOLUTE numbers cannot be copied: 15 pt ticks on an 8x5 panel are enormous.
# What transfers is the ratios, scaled by ~0.6, plus the scale-free choices
# (tick direction, round caps, frameless legend, the colour cycle).
#
# Already matching the reference before this: inward ticks on all four sides,
# minor ticks visible, and a major-only grid.

# Text. Legends mostly set an explicit fontsize per plot and ignore the default.
matplotlib.rcParams["axes.labelsize"] = 17.0    # x/y axis labels (bottom/side)
matplotlib.rcParams["axes.titlesize"] = 18.0    # per-axes title (top)
matplotlib.rcParams["figure.titlesize"] = 19.0  # fig.suptitle
matplotlib.rcParams["xtick.labelsize"] = 15.0
matplotlib.rcParams["ytick.labelsize"] = 15.0
matplotlib.rcParams["legend.fontsize"] = 13.0   # only legends without an explicit size

# Serif text and matching maths, so the figures sit next to LaTeX body text.
# The reference sets `text.usetex : True` with Palatino / Computer Modern; this
# cluster has no LaTeX binary and neither font, so the equivalent is STIX --
# a Times-metric serif whose mathtext companion covers every glyph these
# figures use, including the minus sign that cmr10 is missing.
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["STIXGeneral", "DejaVu Serif"]
matplotlib.rcParams["mathtext.fontset"] = "stix"

# Grid on every figure, but only at the MAJOR ticks (minor ticks stay bare).
# Set here rather than per-plot: there are no explicit ax.grid() calls any more,
# so this is the single switch for all figures.
matplotlib.rcParams["axes.grid"] = True
matplotlib.rcParams["axes.grid.which"] = "major"
matplotlib.rcParams["axes.grid.axis"] = "both"
matplotlib.rcParams["grid.alpha"] = 0.3
matplotlib.rcParams["grid.linestyle"] = "-"
matplotlib.rcParams["grid.linewidth"] = 0.8

# HEP publication tick style: inward ticks on all four sides, plus minor ticks.
# Sizes and widths are the reference's, scaled ~0.6 for our smaller figures;
# note the reference makes minor ticks proportionally longer than we had them
# (2/3 of major, not 1/2), which is what makes a log axis readable between
# labelled decades.
matplotlib.rcParams["xtick.direction"] = "in"
matplotlib.rcParams["ytick.direction"] = "in"
matplotlib.rcParams["xtick.top"] = True
matplotlib.rcParams["ytick.right"] = True
matplotlib.rcParams["xtick.minor.visible"] = True
matplotlib.rcParams["ytick.minor.visible"] = True
matplotlib.rcParams["xtick.major.size"] = 9.0
matplotlib.rcParams["ytick.major.size"] = 9.0
matplotlib.rcParams["xtick.minor.size"] = 6.0
matplotlib.rcParams["ytick.minor.size"] = 6.0
matplotlib.rcParams["xtick.major.width"] = 1.3
matplotlib.rcParams["ytick.major.width"] = 1.3
matplotlib.rcParams["xtick.minor.width"] = 0.9
matplotlib.rcParams["ytick.minor.width"] = 0.9
matplotlib.rcParams["xtick.major.pad"] = 5.0

# Heavier frame and lines. matplotlib's default 0.8 pt spine is thin enough to
# disappear in print next to the data; the reference uses 2.5 at its scale.
matplotlib.rcParams["axes.linewidth"] = 1.5
matplotlib.rcParams["lines.linewidth"] = 1.8
matplotlib.rcParams["lines.solid_capstyle"] = "round"

# Frameless legends: one less box competing with the data. Safe here because
# every legend either has headroom reserved for it or is fitted to the data
# beneath it.
matplotlib.rcParams["legend.frameon"] = False

# The reference's colour cycle, for any artist that does not name a colour.
matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler(
    color=["#0C5DA5", "#00B945", "#FF9500", "#FF2C00", "#845B97", "#474747", "#9e9e9e"])

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm

from hepattn.experiments.odd_pileup_reco.eval_data import (CALO_HS_ENERGY_THRESHOLD, CALO_HS_FRAC_THRESHOLD,
                                                     load_eval_data_from_h5)
from hepattn.experiments.odd_pileup_reco.puppi_charged_subtract import (
    cluster_puppi_calo_hs_charged_jets,
    cluster_puppi_charged_subtract_jets,
    compute_puppi_weights_charged_subtract,
    load_charged_subtract_events,
)
from hepattn.experiments.odd_pileup_reco.reco_analysis import (
    CLASS_COLORS,
    CLASS_LABELS,
    _cluster_jets_single,
    cluster_jets,
    match_jets,
    normalize_phi,
    load_pflow_data,
)
from hepattn.experiments.odd_pileup_reco.run_puppi_jet_resolution_charged_subtract_full import (
    DEFAULT_H5,
    PT_BIN_EDGES,
    _binned_mean_iqr,
    _make_binned_plots,
    _match_and_residuals,
)

# PUPPI parquet source: the all-vertices "paper" sample the H5 was generated from
# (the chunked dir does NOT contain these events).
DEFAULT_PARQUET_DIR = "/storage/agrp/barakma/PileupODD/data/ttbar_pu200_all_vertices_paper"
#DEFAULT_PARQUET_DIR = "/storage/agrp/barakma/PileupODD/data/dihiggs_pu200_all_vertices_paper"
# PUPPI optuna params (v2, anti-kT R=0.4 tuning).
DEFAULT_BEST_JSON = (
    "/storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco/"
    "optuna_puppi_charged_subtract/puppi_charged_subtract_v2_antikt_R04_best.json"
)
# PUPPI tuned for the CALORIMETER-jet comparison (jet_relative_pt_calo_vs_hscalo
# and friends): same algorithm, data and loss as the tuning above, but optimised
# with PUPPI calo jets against truth-HS calo jets (optuna_puppi_charged_subtract
# --target calo, trial #135). Used ONLY for the "PUPPI calo" jets; particle-level
# PUPPI keeps DEFAULT_BEST_JSON. Archived in odd_paper/puppi/.
DEFAULT_BEST_JSON_CALO = (
    "/storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco/"
    "optuna_puppi_charged_subtract/puppi_calo_v3setup_best.json"
)

# ── Paper labels & palette ────────────────────────────────────────────────
GLOWUP = "GLOW-UP"
TARGET = "Target"
PUPPI_LABEL = "Ideal PUPPI with truth information"
GLOWUP_COLOR = "tomato"
TARGET_COLOR = "cornflowerblue"
PUPPI_COLOR = "seagreen"

# Per-class palette, from the reference's colour cycle. It replaces the
# saturated red/blue/green/orange/purple imported from reco_analysis -- exactly
# the primaries its clutter chapter argues against -- while keeping each class
# on the same hue it already had, so a reader comparing against an older figure
# is not sent looking for a recolouring that did not happen.
CLASS_COLORS = [
    "#FF2C00",   # Ch Had   (was red)
    "#0C5DA5",   # e        (was blue)
    "#00B945",   # mu       (was green)
    "#FF9500",   # Neu Had  (was orange)
    "#845B97",   # gamma    (was purple)
]

# ── Fixed bin edges (kept constant so per-shard partials are mergeable) ─────
TRACK_PT_BINS = np.logspace(np.log10(0.3), np.log10(200), 25)            # 24 bins
PR_SCORE_BINS = np.linspace(0.0, 1.0, 1001)   # 1000 bins for the threshold-free track PR/AP curve
FEAT_BINS = [
    np.linspace(0.0, 200.0, 70),       # pt
    np.linspace(-4.0, 4.0, 70),        # eta
    np.linspace(-np.pi, np.pi, 70),    # phi
]
FEAT_LABELS = [r"$p_T$ [GeV]", r"$\eta$", r"$\phi$ [rad]"]
FEAT_ROWS = ["All", "Charged", "Neutral"]
NPART_PT_BINS = (0.0, 2.0, 5.0, 10.0, 20.0, 200.0)   # 0-1/1-2 merged into 0-2; 20-50/50-200 merged into 20-200
CALO_E_BINS = np.logspace(np.log10(0.05), np.log10(2000.0), 60)
RESIDUAL_KEYS = ("dpt", "dpt_over_truth", "deta", "dphi", "truth_pt")
# The calo-vs-truth-HS-calo comparison also carries the jet energy residual.
CALO_HS_RES_KEYS = RESIDUAL_KEYS + ("de_over_truth", "truth_e")
# Truth-HS calo jet energy bins for the energy box plot (E >= pT >= 10 GeV).
JET_RES_E_EDGES = (10.0, 30.0, 60.0, 100.0, 200.0, 400.0, float("inf"))
IND_THRESHOLD = 0.5
PARQUET_EVENTS_PER_FILE = 100   # event_ids per parquet shard file (NNNNN = eid // this)

# ── Per-particle-class performance (assumes MATCHED objects: predict_only=False) ─
N_CLASSES = 5                                            # real classes 0..4 (5 = null)
PERF_PT_BINS = np.logspace(np.log10(1.0), np.log10(500.0), 16)   # 15 truth-pT bins
CLASS_PT_DIST_BINS = np.logspace(np.log10(0.1), np.log10(500.0), 61)   # per-class truth-pT spectrum
# Track-stream and per-event distributions. Fixed rather than percentile-derived
# because each shard is histogrammed independently and the counts are summed:
# per-shard bin edges could not be merged. Ranges cover the observed data --
# track pT reaches ~425 GeV but p99.9 is ~115, |z0| maxes near 300 mm, and the
# per-event HS energy ratio sits below ~1.1.
TRACK_PT_BINS = np.logspace(np.log10(0.3), np.log10(200.0), 61)
TRACK_Z0_BINS = np.linspace(-300.0, 300.0, 101)
# d0 range measured from the source parquet (p0.1 = -3.6 mm, p99.9 = +3.7 mm).
TRACK_D0_BINS = np.linspace(-5.0, 5.0, 101)
ETA_DIST_BINS = np.linspace(-3.5, 3.5, 71)
PHI_DIST_BINS = np.linspace(-np.pi, np.pi, 73)
# Particle eta reaches ~3.6 for targets and ~3.9 for predictions, past the
# |eta| < 3.5 the track/cluster figures use, so particles get their own range.
PART_ETA_BINS = np.linspace(-4.0, 4.0, 81)
# Particle pT: targets reach down to ~0.15 GeV (p0.1) and predictions up to
# ~1.1 TeV, so log bins over 0.05-1000 GeV keep everything in range.
PART_PT_BINS = np.logspace(np.log10(0.05), np.log10(1000.0), 71)
# Non-null particles per event: at most the 400 query slots.
NPART_EVT_BINS = np.arange(-0.5, 401.5, 5.0)
# pT threshold applied to BOTH Target and GLOW-UP in every particle-level
# GLOW-UP-vs-Target comparison, so the two populations are compared over the
# same phase space (the model rarely predicts the softest targets).
PART_CMP_PT_CUT = 1.0
HS_RATIO_BINS = np.linspace(0.0, 2.0, 81)
# Per-class particle counts (bar chart, GLOW-UP vs Target), one figure per pT cut.
CLASS_COUNT_PT_CUTS = (0.0, 0.5, 1.0, 2.0)   # >0 GeV (no cut), >0.5 GeV, >1 GeV, >2 GeV
RESP_BINS = np.linspace(0.0, 4.0, 201)                   # pred_pT / truth_pT ratio
JET_PT_BINS = np.logspace(np.log10(10.0), np.log10(600.0), 13)   # jet matching eff/fake bins
DPTREL_BINS = np.linspace(-1.0, 2.0, 151)    # (reco-truth)/truth, per-class residual figure
DANG_BINS = np.linspace(-0.25, 0.25, 151)    # delta-eta / delta-phi, per-class residual figure
# Jet-vs-pT bin edges for the binned/IQR/box jet figures (300-500-inf merged into 300-inf).
JET_RES_PT_EDGES = (10.0, 20.0, 50.0, 90.0, 150.0, 300.0, float("inf"))


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


def _class_count_by_cut(data: dict) -> dict:
    """Per-class particle counts at each pT cut, Target (truth) vs GLOW-UP (pflow).

    A slot is a real particle of class c when class == c (< N_CLASSES; null = 5 is
    dropped), matching the ``tv``/``pv`` convention in ``_per_class_counts``. Returns
    (N_CLASSES, n_cuts) count arrays: entry [c, k] = # class-c particles with
    pT > CLASS_COUNT_PT_CUTS[k]. Directly summable across shards.
    """
    tc = np.asarray(data["truth_class"]).ravel()
    pc = np.asarray(data["pflow_class"]).ravel()
    tpt = np.asarray(data["truth_ptetaphi"][..., 0]).ravel()
    ppt = np.asarray(data["pflow_ptetaphi"][..., 0]).ravel()
    ncut = len(CLASS_COUNT_PT_CUTS)
    truth = np.zeros((N_CLASSES, ncut), dtype=np.int64)
    pflow = np.zeros((N_CLASSES, ncut), dtype=np.int64)
    for c in range(N_CLASSES):
        tm = (tc == c) & np.isfinite(tpt)
        pm = (pc == c) & np.isfinite(ppt)
        for k, cut in enumerate(CLASS_COUNT_PT_CUTS):
            truth[c, k] = int((tm & (tpt > cut)).sum())
            pflow[c, k] = int((pm & (ppt > cut)).sum())
    return {"truth": truth, "pflow": pflow}


def _class_pt_spectrum(data: dict) -> dict:
    """Per-class particle pT spectrum (count vs pT), Target (truth) vs GLOW-UP (pflow).

    Returns (N_CLASSES, nbins) count arrays over CLASS_PT_DIST_BINS for each source
    (real classes only; null = 5 dropped). Directly summable across shards.
    """
    tc = np.asarray(data["truth_class"]).ravel()
    pc = np.asarray(data["pflow_class"]).ravel()
    tpt = np.asarray(data["truth_ptetaphi"][..., 0]).ravel()
    ppt = np.asarray(data["pflow_ptetaphi"][..., 0]).ravel()
    nb = len(CLASS_PT_DIST_BINS) - 1
    truth = np.zeros((N_CLASSES, nb), dtype=np.int64)
    pflow = np.zeros((N_CLASSES, nb), dtype=np.int64)
    for c in range(N_CLASSES):
        tm = (tc == c) & np.isfinite(tpt)
        pm = (pc == c) & np.isfinite(ppt)
        truth[c] = np.histogram(tpt[tm], bins=CLASS_PT_DIST_BINS)[0]
        pflow[c] = np.histogram(ppt[pm], bins=CLASS_PT_DIST_BINS)[0]
    return {"truth": truth, "pflow": pflow}


def _track_counts(track_data: dict) -> dict:
    nb = len(TRACK_PT_BINS) - 1
    nps = len(PR_SCORE_BINS) - 1
    out = {k: np.zeros(nb, dtype=np.int64) for k in ("tp", "fp", "fn", "total")}
    # Score histograms split by truth label feed the threshold-free PR/AP curve.
    out["score_pos"] = np.zeros(nps, dtype=np.int64)
    out["score_neg"] = np.zeros(nps, dtype=np.int64)
    probs = track_data.get("probs")
    if probs is None or len(probs) == 0:
        return out
    probs = np.asarray(probs, dtype=float)
    is_pred = probs > 0.5
    is_true = np.asarray(track_data["truth"]) == 1
    pt = np.asarray(track_data["pt"])
    idx = np.digitize(pt, TRACK_PT_BINS) - 1
    inb = (idx >= 0) & (idx < nb)
    out["tp"] = np.bincount(idx[inb & is_pred & is_true], minlength=nb)[:nb]
    out["fp"] = np.bincount(idx[inb & is_pred & ~is_true], minlength=nb)[:nb]
    out["fn"] = np.bincount(idx[inb & ~is_pred & is_true], minlength=nb)[:nb]
    out["total"] = np.bincount(idx[inb], minlength=nb)[:nb]
    out["score_pos"] = np.histogram(probs[is_true], bins=PR_SCORE_BINS)[0]
    out["score_neg"] = np.histogram(probs[~is_true], bins=PR_SCORE_BINS)[0]
    return out


def _calo_counts(cluster_data: dict) -> dict:
    nb = len(CALO_E_BINS) - 1
    out = {k: np.zeros(nb, dtype=np.int64)
           for k in ("truth", "recovered", "pred", "hs_truth", "hs_recovered", "hs_pred")}
    e = np.asarray(cluster_data.get("total_e", []), dtype=float)
    mp_ = cluster_data.get("mask_pred")
    mt = cluster_data.get("mask_truth")
    if e.size == 0 or mp_ is None or mt is None:
        return out
    mp_ = np.asarray(mp_).astype(bool)
    mt = np.asarray(mt).astype(bool)
    # Binned by TOTAL cluster energy (feeds the pred-vs-truth energy distribution).
    idx = np.digitize(e, CALO_E_BINS) - 1
    inb = (idx >= 0) & (idx < nb) & np.isfinite(e) & (e > 0)
    out["truth"] = np.bincount(idx[inb & mt], minlength=nb)[:nb]
    out["recovered"] = np.bincount(idx[inb & mp_ & mt], minlength=nb)[:nb]
    out["pred"] = np.bincount(idx[inb & mp_], minlength=nb)[:nb]
    # Binned by the cluster's HS (hard-scatter) energy (feeds the recall plot).
    hse = np.asarray(cluster_data.get("true_hs_e", []), dtype=float)
    if hse.size == e.size:
        idxh = np.digitize(hse, CALO_E_BINS) - 1
        inbh = (idxh >= 0) & (idxh < nb) & np.isfinite(hse) & (hse > 0)
        out["hs_truth"] = np.bincount(idxh[inbh & mt], minlength=nb)[:nb]          # recall denom (truth-HS)
        out["hs_recovered"] = np.bincount(idxh[inbh & mp_ & mt], minlength=nb)[:nb]  # TP
        out["hs_pred"] = np.bincount(idxh[inbh & mp_], minlength=nb)[:nb]          # precision denom (pred-HS)
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

def _flat_jet_pt(jets: dict, prefix: str) -> np.ndarray:
    chunks = [np.asarray(a) for a in jets[f"{prefix}_jet_pt"] if len(a) > 0]
    p = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
    return p[np.isfinite(p) & (p > 0)]


def _jet_match_counts(jets: dict, puppi_jets: dict, glow_res: dict, puppi_res: dict) -> dict:
    """Per-jet-pT-bin counts for jet matching efficiency and fake rate.

    Efficiency = matched / all truth jets (binned by truth-jet pT).
    Fake rate   = (all reco - matched) / all reco jets (binned by reco-jet pT).
    Matched truth/reco pT come straight from the residual dicts (no re-matching):
    matched-truth = res["truth_pt"], matched-reco = res["truth_pt"] + res["dpt"].
    """
    def H(x):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x) & (x > 0)]
        return np.histogram(x, bins=JET_PT_BINS)[0].astype(np.int64)

    return {
        "truth_total": H(_flat_jet_pt(jets, "truth")),
        "glow_eff_num": H(glow_res["truth_pt"]),
        "puppi_eff_num": H(puppi_res["truth_pt"]),
        "glow_total": H(_flat_jet_pt(jets, "pflow")),
        "puppi_total": H(_flat_jet_pt(puppi_jets, "puppi")),
        "glow_match_pred": H(glow_res["truth_pt"] + glow_res["dpt"]),
        "puppi_match_pred": H(puppi_res["truth_pt"] + puppi_res["dpt"]),
    }


def _per_class_counts(data: dict) -> dict:
    """Matched per-truth-class counts for the per-class performance figures.

    Uses slot-aligned (matched) pred/truth objects. Returns:
      conf       (5, 6) int : conf[truth_class, pred_class] over truth-valid slots
                              (pred col 5 = predicted-null = missed)
      eff_denom  (5, NPT)   : # truth objects of class c per truth-pT bin
      eff_reco   (5, NPT)   : ... that got a valid (non-null) prediction
      eff_correct(5, NPT)   : ... predicted with the correct class
      resp       (5, NPT, NR): pred_pT/truth_pT histogram per (class, pT bin)
    """
    tc = np.asarray(data["truth_class"]).ravel()
    pc = np.asarray(data["pflow_class"]).ravel()
    tpt = np.asarray(data["truth_ptetaphi"][..., 0]).ravel()
    ppt = np.asarray(data["pflow_ptetaphi"][..., 0]).ravel()

    tv = tc < N_CLASSES        # truth is a real particle (0..4)
    pv = pc < N_CLASSES        # pred is a real particle (not null)

    npt = len(PERF_PT_BINS) - 1
    nr = len(RESP_BINS) - 1
    conf = np.zeros((N_CLASSES, N_CLASSES + 1), dtype=np.int64)
    conf_pt = np.zeros((N_CLASSES, N_CLASSES + 1, npt), dtype=np.int64)   # truth-pT-binned, for per-class F1
    eff_denom = np.zeros((N_CLASSES, npt), dtype=np.int64)
    eff_reco = np.zeros((N_CLASSES, npt), dtype=np.int64)
    eff_correct = np.zeros((N_CLASSES, npt), dtype=np.int64)
    resp = np.zeros((N_CLASSES, npt, nr), dtype=np.int64)
    fake_pred_n = np.zeros((N_CLASSES, npt), dtype=np.int64)    # predicted class c, by PRED pT
    fake_wrong_n = np.zeros((N_CLASSES, npt), dtype=np.int64)   # ... whose truth class != c (fake/misID)
    nptd = len(CLASS_PT_DIST_BINS) - 1
    class_pt = np.zeros((N_CLASSES, nptd), dtype=np.int64)      # truth-pT spectrum per class

    # Confusion (rows = truth class, cols = pred class with null mapped to col 5).
    sel = tv
    np.add.at(conf, (tc[sel], np.clip(pc[sel], 0, N_CLASSES)), 1)

    ptbin = np.digitize(tpt, PERF_PT_BINS) - 1
    inpt = (ptbin >= 0) & (ptbin < npt)
    sel_pt = tv & inpt
    np.add.at(conf_pt, (tc[sel_pt], np.clip(pc[sel_pt], 0, N_CLASSES), ptbin[sel_pt]), 1)

    # Fake rate is over PREDICTED objects, binned by PREDICTED pT.
    pred_ptbin = np.digitize(ppt, PERF_PT_BINS) - 1
    pinpt = (pred_ptbin >= 0) & (pred_ptbin < npt)
    for c in range(N_CLASSES):
        cm = tv & (tc == c) & inpt
        eff_denom[c] = np.bincount(ptbin[cm], minlength=npt)[:npt]
        eff_reco[c] = np.bincount(ptbin[cm & pv], minlength=npt)[:npt]
        eff_correct[c] = np.bincount(ptbin[cm & (pc == c)], minlength=npt)[:npt]
        rsel = cm & pv & (tpt > 0) & np.isfinite(ppt)
        if rsel.any():
            rb = np.digitize(ppt[rsel] / tpt[rsel], RESP_BINS) - 1
            pb = ptbin[rsel]
            ok = (rb >= 0) & (rb < nr)
            np.add.at(resp[c], (pb[ok], rb[ok]), 1)
        psel = pv & (pc == c) & pinpt
        fake_pred_n[c] = np.bincount(pred_ptbin[psel], minlength=npt)[:npt]
        fake_wrong_n[c] = np.bincount(pred_ptbin[psel & (tc != c)], minlength=npt)[:npt]
        class_pt[c] = np.histogram(tpt[tv & (tc == c)], bins=CLASS_PT_DIST_BINS)[0]

    return {"conf": conf, "conf_pt": conf_pt, "eff_denom": eff_denom,
            "eff_reco": eff_reco, "eff_correct": eff_correct, "resp": resp,
            "fake_pred_n": fake_pred_n, "fake_wrong_n": fake_wrong_n, "class_pt": class_pt}


def _class_residual_counts(data: dict) -> dict:
    """Per-class GLOW-UP kinematic residual histograms (reco - truth), matched objects.

    dptrel = pT_reco/pT_truth - 1 ; deta = eta_reco - eta_truth ; dphi = wrapped.
    res_truth_n = truth count per class; res_n = matched count per class (f = res_n/res_truth_n).
    """
    nd, na = len(DPTREL_BINS) - 1, len(DANG_BINS) - 1
    out = {
        "res_truth_n": np.zeros(N_CLASSES, dtype=np.int64),
        "res_n": np.zeros(N_CLASSES, dtype=np.int64),
        "res_dptrel": np.zeros((N_CLASSES, nd), dtype=np.int64),
        "res_deta": np.zeros((N_CLASSES, na), dtype=np.int64),
        "res_dphi": np.zeros((N_CLASSES, na), dtype=np.int64),
    }
    tc = np.asarray(data["truth_class"]).ravel()
    pc = np.asarray(data["pflow_class"]).ravel()
    tv = tc < N_CLASSES
    pv = pc < N_CLASSES
    tru = data["truth_ptetaphi"].reshape(-1, 3)
    mm = data["pflow_ptetaphi"].reshape(-1, 3)
    tpt, teta, tphi = tru[:, 0], tru[:, 1], tru[:, 2]
    dptrel = mm[:, 0] / np.clip(tpt, 1e-8, None) - 1.0
    deta = mm[:, 1] - teta
    dphi = ((mm[:, 2] - tphi + np.pi) % (2 * np.pi)) - np.pi
    base = tv & pv & (tpt > 0) & np.isfinite(mm[:, 0]) & np.isfinite(mm[:, 1]) & np.isfinite(mm[:, 2])
    for c in range(N_CLASSES):
        out["res_truth_n"][c] = int((tv & (tc == c)).sum())
        cm = base & (tc == c)
        out["res_n"][c] = int(cm.sum())
        out["res_dptrel"][c] = np.histogram(dptrel[cm], bins=DPTREL_BINS)[0]
        out["res_deta"][c] = np.histogram(deta[cm], bins=DANG_BINS)[0]
        out["res_dphi"][c] = np.histogram(dphi[cm], bins=DANG_BINS)[0]
    return out


def _hist_percentile(counts: np.ndarray, edges: np.ndarray, q: float) -> float:
    """Percentile q (0..100) of a distribution given as histogram counts."""
    tot = counts.sum()
    if tot <= 0:
        return np.nan
    centers = 0.5 * (edges[:-1] + edges[1:])
    cdf = np.cumsum(counts) / tot
    return float(np.interp(q / 100.0, cdf, centers))


def _puppi_jets_for_shard(parquet_dir: str, eids: list[int], events_per_file: int,
                          subtract_pu: bool, puppi_params: dict, jet_cfg: dict,
                          puppi_params_calo: dict) -> dict:
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
        # Calo-only PUPPI jets (clusters only, truth HS charged energy added back),
        # with their OWN weights from the calo-tuned parameters: each PUPPI
        # variant uses the tuning made for the comparison it appears in.
        w_calo = compute_puppi_weights_charged_subtract(ev, subtract_pu_charged=subtract_pu,
                                                        **puppi_params_calo)
        pc = cluster_puppi_calo_hs_charged_jets(ev, w_calo, subtract_pu_charged=subtract_pu, **jet_cfg)
        for i, e in enumerate(gids):
            by_eid[e] = (tuple(pj[f"puppi_jet_{k}"][i] for k in keys)
                         + tuple(pc[f"puppicalo_jet_{k}"][i] for k in keys))

    out: dict = {}
    for j, k in enumerate(keys):
        for off, prefix in ((0, "puppi"), (len(keys), "puppicalo")):
            arr = np.empty(len(eids), dtype=object)
            for i, e in enumerate(eids):
                arr[i] = by_eid[e][off + j]
            out[f"{prefix}_jet_{k}"] = arr
    return out


def _f32_key(phi: np.ndarray, z0: np.ndarray) -> np.ndarray:
    """Exact (phi, z0) identity of a track as one uint64: the two float32 bit
    patterns side by side. Exact, not approximate -- see _track_d0_from_parquet."""
    hi = np.ascontiguousarray(phi, dtype=np.float32).view(np.uint32).astype(np.uint64)
    lo = np.ascontiguousarray(z0, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return (hi << np.uint64(32)) | lo


def _track_d0_from_parquet(shard: Path, parquet_dir: str, events_per_file: int) -> np.ndarray:
    """Track d0 for every track node of an H5 shard, aligned with track_data.

    The prediction H5s predate node_d0, so d0 is recovered from the source
    parquet the shard was made from -- the same parquet the PUPPI baseline reads.

    Events join on event_id, which IS the H5 event_number (file =
    event_id // events_per_file). Within an event the H5 carries no track id, so
    tracks join on their exact (phi, z0) float32 bit patterns. Both sides hold
    the very same float32 values, and on the paper sample every H5 track node
    matches exactly one parquet track -- no tolerance is involved, so a match
    cannot silently pair the wrong track. Anything unmatched or ambiguous raises
    rather than being dropped.

    The returned array is in eval_data's track order -- event-major, node order,
    over node_valid & node_is_track -- so it lines up element for element.
    """
    with h5py.File(shard, "r") as f:
        eids = np.asarray(f["events"]["event_number"][:]).reshape(-1).astype(np.int64)
        nm = f["node_metadata"][:]
    mask = nm["node_valid"].astype(bool) & nm["node_is_track"].astype(bool)
    ev = np.broadcast_to(eids[:, None], mask.shape)[mask]
    h5_tab = pl.DataFrame({
        "event_id": ev,
        "key": _f32_key(nm["node_phi"][mask], nm["node_z0"][mask]),
    }).with_row_index("pos")

    pdir = Path(parquet_dir)
    parts = []
    for fidx in sorted({int(e) // events_per_file for e in eids}):
        tr = (pl.read_parquet(pdir / f"tracks-{fidx:05d}.parquet",
                              columns=["event_id", "phi", "z0", "d0"])
              .filter(pl.col("event_id").is_in(eids.tolist()))
              .explode(["phi", "z0", "d0"]))
        parts.append(pl.DataFrame({
            "event_id": tr["event_id"].to_numpy().astype(np.int64),
            "key": _f32_key(tr["phi"].to_numpy(), tr["z0"].to_numpy()),
            "d0": tr["d0"].to_numpy().astype(np.float32),
        }))
    pq_tab = pl.concat(parts)

    dup = pq_tab.group_by(["event_id", "key"]).len().filter(pl.col("len") > 1)
    if dup.height:
        raise RuntimeError(f"{shard.name}: {dup.height} (event, phi, z0) keys are not "
                           "unique in the parquet; d0 cannot be assigned unambiguously")
    joined = h5_tab.join(pq_tab, on=["event_id", "key"], how="left").sort("pos")
    missing = int(joined["d0"].null_count())
    if missing:
        raise RuntimeError(f"{shard.name}: {missing} of {joined.height} track nodes have no "
                           f"parquet match in {parquet_dir}; is it the sample this H5 was made from?")
    return joined["d0"].to_numpy()


def _track_hs_dists(track_data: dict, threshold: float = 0.5) -> dict:
    """Truth-HS vs predicted-HS track distributions, as fixed-bin histograms.

    The track stream scores every track for "came from the hard scatter". This
    compares the kinematics of the tracks it selects against the kinematics of
    the tracks that really are hard-scatter -- a shape check, not an
    efficiency: a selection can have the right spectrum and still pick the
    wrong individual tracks.
    """
    probs = np.asarray(track_data["probs"])
    truth = np.asarray(track_data["truth"])
    pt, z0 = np.asarray(track_data["pt"]), np.asarray(track_data["z0"])
    eta, phi = np.asarray(track_data["eta"]), np.asarray(track_data["phi"])
    is_true, is_pred = truth == 1, probs > threshold
    h = lambda v, m, b: np.histogram(v[m], bins=b)[0].astype(np.int64)
    out = {
        "pt_truth": h(pt, is_true, TRACK_PT_BINS),
        "pt_pred": h(pt, is_pred, TRACK_PT_BINS),
        "z0_truth": h(z0, is_true, TRACK_Z0_BINS),
        "z0_pred": h(z0, is_pred, TRACK_Z0_BINS),
        "eta_truth": h(eta, is_true, ETA_DIST_BINS),
        "eta_pred": h(eta, is_pred, ETA_DIST_BINS),
        "phi_truth": h(phi, is_true, PHI_DIST_BINS),
        "phi_pred": h(phi, is_pred, PHI_DIST_BINS),
        "n_truth": np.array([is_true.sum()], dtype=np.int64),
        "n_pred": np.array([is_pred.sum()], dtype=np.int64),
    }
    # d0 is absent from H5s written before node_d0 was added to the prediction
    # writer. Aggregating it conditionally means a re-run of the forward pass
    # turns the figure on with no further code change -- and an old shard still
    # analyses, just without it.
    if "d0" in track_data:
        d0 = np.asarray(track_data["d0"])
        out["d0_truth"] = h(d0, is_true, TRACK_D0_BINS)
        out["d0_pred"] = h(d0, is_pred, TRACK_D0_BINS)
    return out



def _calo_hs_dists(cluster_data: dict) -> dict:
    """Truth-HS vs predicted-HS calorimeter cluster angular distributions.

    The calorimeter counterpart of _track_hs_dists. "Truth HS" is a cluster
    whose truth hard-scatter fraction and energy clear the thresholds in
    eval_data; "predicted HS" is one the calo mask selects. As with the tracks
    this is a shape check on where in the detector the selection lands, not a
    per-cluster efficiency.
    """
    if "mask_truth" not in cluster_data or "mask_pred" not in cluster_data:
        return None
    eta = np.asarray(cluster_data["eta"])
    phi = np.asarray(cluster_data["phi"])
    is_true = np.asarray(cluster_data["mask_truth"], dtype=bool)
    is_pred = np.asarray(cluster_data["mask_pred"], dtype=bool)
    h = lambda v, m, b: np.histogram(v[m], bins=b)[0].astype(np.int64)
    return {
        "eta_truth": h(eta, is_true, ETA_DIST_BINS),
        "eta_pred": h(eta, is_pred, ETA_DIST_BINS),
        "phi_truth": h(phi, is_true, PHI_DIST_BINS),
        "phi_pred": h(phi, is_pred, PHI_DIST_BINS),
        "n_truth": np.array([is_true.sum()], dtype=np.int64),
        "n_pred": np.array([is_pred.sum()], dtype=np.int64),
    }


def _particle_angular_dists(data: dict) -> dict:
    """Target vs GLOW-UP particle eta and phi, over every real (non-null) object.

    Unlike the per-class figures this needs no matching: it compares the two
    populations as wholes -- where the model puts particles against where the
    targets are. Counts, not shapes, are kept so a regional excess or deficit
    of reconstructed particles shows up as one.
    """
    out = {}
    for tag, key, cls in (("truth", "truth_ptetaphi", "truth_class"),
                          ("pred", "pflow_ptetaphi", "pflow_class")):
        a = np.asarray(data[key])
        real = np.asarray(data[cls]) < N_CLASSES
        # Two selections: every non-null object (target-only figures), and
        # pT > PART_CMP_PT_CUT (suffix _c1, the GLOW-UP-vs-Target comparisons).
        for suffix, sel in (("", real), ("_c1", real & (a[..., 0] > PART_CMP_PT_CUT))):
            # Per-event multiplicity, histogrammed (so shards merge by summing)
            # and with exact sums for the mean.
            n_evt = sel.reshape(sel.shape[0], -1).sum(axis=1)
            out[f"nevt{suffix}_{tag}"] = np.histogram(n_evt, bins=NPART_EVT_BINS)[0].astype(np.int64)
            out[f"nevt_sum{suffix}_{tag}"] = np.array([n_evt.sum(), (n_evt.astype(np.float64) ** 2).sum()])
            pt, eta, phi = a[..., 0][sel], a[..., 1][sel], a[..., 2][sel]
            out[f"pt{suffix}_{tag}"] = np.histogram(pt, bins=PART_PT_BINS)[0].astype(np.int64)
            out[f"eta{suffix}_{tag}"] = np.histogram(eta, bins=PART_ETA_BINS)[0].astype(np.int64)
            out[f"phi{suffix}_{tag}"] = np.histogram(phi, bins=PHI_DIST_BINS)[0].astype(np.int64)
            out[f"n{suffix}_{tag}"] = np.array([sel.sum()], dtype=np.int64)
    return out


def _class_totals(data: dict) -> dict:
    """Truth and predicted counts for all six classes, null included.

    Distinct from ``conf``, which is restricted to truth-valid slots and so
    cannot say how many slots the model left empty.
    """
    tc = np.asarray(data["truth_class"]).ravel()
    pc = np.asarray(data["pflow_class"]).ravel()
    n = N_CLASSES + 1
    return {"truth": np.bincount(np.clip(tc, 0, N_CLASSES), minlength=n)[:n].astype(np.int64),
            "pred": np.bincount(np.clip(pc, 0, N_CLASSES), minlength=n)[:n].astype(np.int64)}


def _hs_energy_ratio(cluster_data: dict) -> dict:
    """Per-event (predicted mask HS energy / truth mask HS energy), by type.

    Histogrammed on fixed bins so shards merge by summing. The moments are
    accumulated separately as n/sum/sumsq rather than read off the histogram,
    so the mu and sigma in the legend are exact and not bin-resolution limited.
    """
    out = {}
    for tag in ("neutral", "charged"):
        pred = np.asarray(cluster_data[f"evt_pred_{tag}_e"], dtype=np.float64)
        truth = np.asarray(cluster_data[f"evt_truth_{tag}_e"], dtype=np.float64)
        ok = truth > 0                      # events with no truth HS energy say nothing
        r = pred[ok] / truth[ok]
        out[f"{tag}_hist"] = np.histogram(r, bins=HS_RATIO_BINS)[0].astype(np.int64)
        out[f"{tag}_moments"] = np.array([r.size, r.sum(), (r ** 2).sum()], dtype=np.float64)
    return out


CALO_MASK_THRESHOLD = 0.3   # the checkpoint's calo_pred_threshold: the stage-B mask fed to reco
_INC_CHUNK = 100            # events per pred_incidence read: (100, 400, 1400) float32 ~ 220 MB


def _calo_jets_from_h5(shard: Path, jet_cfg: dict) -> tuple[dict, dict]:
    """GLOW-UP calorimeter jets, and truth-HS calorimeter jets, from one H5 shard.

    Three cluster-only jet collections, all from the same clusters' energies:

      glowcalo     clusters the pileup-removal stage keeps (calo_prob >= 0.3, the
                   checkpoint's own threshold), at their FULL energy -- the
                   boolean mask alone.
      glowcaloinc  the same clusters, energy rescaled by the HS share the
                   reconstruction stage assigns them: sum over non-null query
                   slots of pred_incidence. Each cluster's incidence column is a
                   softmax over all 400 slots, and in the truth incidence the
                   non-null share is exactly the HS energy fraction (calo_hs_frac),
                   so the null share is the model's own pileup estimate.
      hscalo       every cluster at its TRUTH hard-scatter energy -- perfect
                   pileup removal at calorimeter level. The ideal calo jet.

    Cluster pT is E / cosh(eta), as for PUPPI. Returns (jets, diagnostics).
    """
    keys = ("pt", "eta", "phi", "mass", "nconst")
    jets = {p: {k: [] for k in keys} for p in ("glowcalo", "glowcaloinc", "hscalo")}
    n_mask = n_mask_no_inc = 0
    # Per-cluster energy spectra on CALO_E_BINS, each cluster at E * (HS fraction):
    # Target uses the truth fraction (calo_hs_frac) on the truth-HS clusters
    # (eval_data's mask_truth cuts); GLOW-UP uses the predicted fraction -- the
    # non-null incidence share, i.e. 1 - predicted PU fraction -- on its kept
    # clusters.
    nb_e = len(CALO_E_BINS) - 1
    e_hist = {k: np.zeros(nb_e, dtype=np.int64) for k in ("e_hist_truth_hs", "e_hist_pred_inc")}
    with h5py.File(shard, "r") as f:
        nm = f["node_metadata"][:]
        calo_prob = np.asarray(f["calo_mask"]["calo_prob"][:], dtype=np.float32).reshape(nm.shape)
        pflow_cls = np.asarray(f["object_class"]["pflow_class"][:])
        reco_idx = np.asarray(f["reco_node_indices"]["index"][:]).reshape(nm.shape[0], -1)
        n_events = nm.shape[0]
        for s0 in range(0, n_events, _INC_CHUNK):
            inc = np.asarray(f["pred_incidence"]["pred_incidence"][s0:s0 + _INC_CHUNK], dtype=np.float32)
            inc = inc.reshape(inc.shape[0], inc.shape[1], -1)                  # (b, 400, 1400)
            real = (pflow_cls[s0:s0 + inc.shape[0]] < N_CLASSES)[..., None]    # non-null slots
            hs_share_reco = (inc * real).sum(axis=1)                           # (b, 1400)
            col_sum_reco = inc.sum(axis=1)
            for j in range(inc.shape[0]):
                i = s0 + j
                share = np.zeros(nm.shape[1], dtype=np.float32)
                colsum = np.zeros(nm.shape[1], dtype=np.float32)
                share[reco_idx[i]] = hs_share_reco[j]
                colsum[reco_idx[i]] = col_sum_reco[j]

                eta = nm["node_eta"][i].astype(np.float32)
                phi = nm["node_phi"][i].astype(np.float32)
                cosh = np.cosh(np.clip(eta, -10.0, 10.0))
                clu = (nm["node_valid"][i] == 1) & (nm["node_is_track"][i] == 0)
                mask = clu & (calo_prob[i] >= CALO_MASK_THRESHOLD)
                n_mask += int(mask.sum())
                # A kept cluster the reco stage gave no incidence column to keeps
                # zero energy in glowcaloinc; counted so it cannot pass unnoticed.
                n_mask_no_inc += int((mask & (colsum < 0.5)).sum())

                e_full = nm["node_e"][i].astype(np.float32)
                e_hs = nm["calo_hs_energy"][i].astype(np.float32)
                f_hs = nm["calo_hs_frac"][i].astype(np.float32)
                truth_hs = clu & (f_hs > CALO_HS_FRAC_THRESHOLD) & (e_hs > CALO_HS_ENERGY_THRESHOLD)
                for k, e_sel in (("e_hist_truth_hs", (e_full * f_hs)[truth_hs]),
                                 ("e_hist_pred_inc", (e_full * share)[mask])):
                    e_hist[k] += np.histogram(e_sel[np.isfinite(e_sel)], bins=CALO_E_BINS)[0]

                pt_full = e_full / cosh
                pt_hs = e_hs / cosh
                for prefix, pt, sel in (("glowcalo", pt_full, mask),
                                        ("glowcaloinc", pt_full * share, mask),
                                        ("hscalo", pt_hs, clu)):
                    sel = sel & np.isfinite(pt) & (pt > 0)
                    res = _cluster_jets_single(np.stack([pt, eta, phi], axis=-1),
                                               sel.astype(np.float32), 0.5,
                                               jet_cfg["jet_R"], jet_cfg["min_constituents"],
                                               jet_cfg["min_pt"], jet_algorithm=jet_cfg["jet_algorithm"])
                    for k, v in zip(keys, res):
                        jets[prefix][k].append(v)
    out = {f"{p}_jet_{k}": np.array(v, dtype=object) for p, d in jets.items() for k, v in d.items()}
    # Jet totals per collection: n_jets_hscalo is the denominator of the match
    # fraction when the truth-HS calo jet is the reference.
    n_jets = {f"n_jets_{p}": np.array([sum(len(e) for e in d["pt"])], dtype=np.int64)
              for p, d in jets.items()}
    return out, {"n_mask_clusters": np.array([n_mask], dtype=np.int64),
                 "n_mask_no_incidence": np.array([n_mask_no_inc], dtype=np.int64),
                 **n_jets, **e_hist}


def _jet_e(pt: np.ndarray, eta: np.ndarray, mass: np.ndarray) -> np.ndarray:
    return np.sqrt((pt * np.cosh(eta)) ** 2 + np.maximum(mass, 0.0) ** 2)


def _match_and_residuals_with_e(pred_jets: dict, ref_jets: dict, prefix: str,
                                dr_cut: float = 0.4) -> dict:
    """_match_and_residuals plus the jet energy residual (E_reco - E_ref) / E_ref.

    Same per-event Hungarian Delta-R matching (match_jets, Delta-R <= dr_cut) and
    the same residual definitions, so the pT/eta/phi entries are identical to
    _match_and_residuals; the energy E = sqrt((pT cosh eta)^2 + m^2) is computed
    for the very same matched pairs. ``ref_jets`` uses truth_jet_* keys and must
    carry mass.
    """
    out = {k: [] for k in CALO_HS_RES_KEYS}
    pk = f"{prefix}_jet"
    n_pred = np.array([len(e) for e in pred_jets[f"{pk}_pt"]])
    n_ref = np.array([len(e) for e in ref_jets["truth_jet_pt"]])
    sel = (n_pred > 0) & (n_ref > 0)
    if sel.any():
        P = {k: pred_jets[f"{pk}_{k}"][sel] for k in ("pt", "eta", "phi", "mass")}
        T = {k: ref_jets[f"truth_jet_{k}"][sel] for k in ("pt", "eta", "phi", "mass")}
        tr_ix, pr_ix, _ = match_jets(P["pt"], P["eta"], P["phi"], T["pt"], T["eta"], T["phi"],
                                     dr_cut=dr_cut)
        for i in range(len(tr_ix)):
            ti, pi = tr_ix[i], pr_ix[i]
            if len(ti) == 0:
                continue
            tpt, ppt = T["pt"][i][ti], P["pt"][i][pi]
            te = _jet_e(tpt, T["eta"][i][ti], T["mass"][i][ti])
            pe = _jet_e(ppt, P["eta"][i][pi], P["mass"][i][pi])
            out["dpt"].append(ppt - tpt)
            out["dpt_over_truth"].append((ppt - tpt) / np.clip(tpt, 1e-8, None))
            out["deta"].append(P["eta"][i][pi] - T["eta"][i][ti])
            out["dphi"].append(normalize_phi(P["phi"][i][pi] - T["phi"][i][ti]))
            out["truth_pt"].append(tpt)
            out["de_over_truth"].append((pe - te) / np.clip(te, 1e-8, None))
            out["truth_e"].append(te)
    return {k: (np.concatenate(v) if v else np.array([])).astype(np.float32) for k, v in out.items()}


def analyze_shard(shard_path: str, parquet_dir: str, events_per_file: int,
                  jet_cfg: dict, subtract_pu: bool, puppi_params: dict,
                  per_class: bool = True, puppi_params_calo: dict | None = None) -> dict:
    if not puppi_params_calo:
        raise ValueError("puppi_params_calo is required (the calo-tuned PUPPI parameters)")
    shard = Path(shard_path)
    data = load_pflow_data(shard_path, load_incidence=False)
    track_data, cluster_data, _ = load_eval_data_from_h5(shard, load_reco=False)
    # H5s written before node_d0 existed carry no d0; recover it from the parquet.
    if "d0" not in track_data:
        track_data["d0"] = _track_d0_from_parquet(shard, parquet_dir, events_per_file)
    n_events = int(data["pflow_ptetaphi"].shape[0])

    with h5py.File(shard, "r") as f:
        eids = [int(x) for x in np.asarray(f["events"]["event_number"][:]).reshape(-1)]

    # GLOW-UP + Target jets + matched residuals.
    jets = cluster_jets(data, **jet_cfg)
    glow_res = _match_and_residuals(jets, jets, "pflow")

    # PUPPI — only this shard's files, one file at a time (memory ~1 file).
    # Mandatory: a missing event raises (KeyError) and aborts the run.
    puppi_jets = _puppi_jets_for_shard(parquet_dir, eids, events_per_file,
                                       subtract_pu, puppi_params, jet_cfg, puppi_params_calo)
    puppi_res = _match_and_residuals(puppi_jets, jets, "puppi")

    # Calorimeter-level jets, for a like-for-like comparison of pileup removal.
    # Residuals are taken against the Target particle jets (the reference every
    # other jet figure uses) AND against the truth-HS calorimeter jets, which
    # removes calorimeter response from the comparison and isolates pileup
    # removal alone.
    calo_jets, calo_diag = _calo_jets_from_h5(shard, jet_cfg)
    calo_jets.update({k: v for k, v in puppi_jets.items() if k.startswith("puppicalo_")})
    hscalo_ref = {f"truth_jet_{k}": calo_jets[f"hscalo_jet_{k}"] for k in ("pt", "eta", "phi", "mass")}
    calo_res = {m: _match_and_residuals(calo_jets, jets, m)
                for m in ("glowcalo", "glowcaloinc", "puppicalo", "hscalo")}
    calo_res_hs = {m: _match_and_residuals_with_e(calo_jets, hscalo_ref, m)
                   for m in ("glowcalo", "glowcaloinc", "puppicalo")}
    as_f32 = lambda r: {k: np.asarray(r.get(k, []), dtype=np.float32) for k in RESIDUAL_KEYS}

    npart_truth, npart_pred = _npart_counts(data)

    # Per-class performance aggregates (assume matched objects; predict_only=False).
    per_class_agg = _per_class_counts(data) if per_class else None
    if per_class_agg is not None:
        per_class_agg.update(_class_residual_counts(data))

    return {
        "n_events": n_events,
        "per_class": per_class_agg,
        "calo_res": {m: as_f32(r) for m, r in calo_res.items()},
        "calo_res_hs": calo_res_hs,
        "calo_diag": calo_diag,
        "track_dist": _track_hs_dists(track_data),
        "calo_dist": _calo_hs_dists(cluster_data),
        "class_totals": _class_totals(data),
        "particle_dist": _particle_angular_dists(data),
        "hs_ratio": _hs_energy_ratio(cluster_data),
        "jet_match": _jet_match_counts(jets, puppi_jets, glow_res, puppi_res),
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
        "class_count": _class_count_by_cut(data),
        "class_pt_spec": _class_pt_spectrum(data),
        "feat_hist": _feature_hist(data),
        "track": _track_counts(track_data),
        "calo": _calo_counts(cluster_data),
    }


# ---------------------------------------------------------------------------
# REDUCE
# ---------------------------------------------------------------------------

def _concat_res(reslist: list[dict], keys=RESIDUAL_KEYS) -> dict:
    return {k: np.concatenate([r[k] for r in reslist]) for k in keys if k in reslist[0]}


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
    if aggs[0].get("class_count") is not None:
        m["class_count"] = {k: np.sum([a["class_count"][k] for a in aggs], axis=0)
                            for k in ("truth", "pflow")}
    if aggs[0].get("class_pt_spec") is not None:
        m["class_pt_spec"] = {k: np.sum([a["class_pt_spec"][k] for a in aggs], axis=0)
                              for k in ("truth", "pflow")}
    m["feat_hist"] = np.sum([a["feat_hist"] for a in aggs], axis=0)
    nps = len(PR_SCORE_BINS) - 1
    m["track"] = {k: np.sum([a["track"][k] for a in aggs], axis=0) for k in ("tp", "fp", "fn", "total")}
    for k in ("score_pos", "score_neg"):
        m["track"][k] = np.sum([a["track"].get(k, np.zeros(nps, dtype=np.int64)) for a in aggs], axis=0)
    nb_calo = len(CALO_E_BINS) - 1
    m["calo"] = {k: np.sum([a["calo"].get(k, np.zeros(nb_calo, dtype=np.int64)) for a in aggs], axis=0)
                 for k in ("truth", "recovered", "pred", "hs_truth", "hs_recovered", "hs_pred")}
    if aggs[0].get("per_class") is not None:
        pcs = [a["per_class"] for a in aggs]
        m["per_class"] = {k: np.sum([pc[k] for pc in pcs], axis=0) for k in pcs[0]}
    if aggs[0].get("jet_match") is not None:
        jms = [a["jet_match"] for a in aggs]
        m["jet_match"] = {k: np.sum([jm[k] for jm in jms], axis=0) for k in jms[0]}
    # Every entry of these three is a plain count or a (n, sum, sumsq) triple,
    # so summing over shards is exact -- including the moments, which is why
    # they are carried as sums rather than as a per-shard mean and sigma.
    for key in ("calo_res", "calo_res_hs"):
        if aggs[0].get(key) is not None:
            m[key] = {meth: _concat_res([a[key][meth] for a in aggs], CALO_HS_RES_KEYS) for meth in aggs[0][key]}
    for key in ("track_dist", "calo_dist", "class_totals", "hs_ratio", "particle_dist", "calo_diag"):
        if aggs[0].get(key) is not None:
            parts = [a[key] for a in aggs]
            m[key] = {k: np.sum([pt[k] for pt in parts], axis=0) for k in parts[0]}
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


# Jet-resolution panels: key -> (xlabel, fixed bins | None, density?, log-y?)
JET_PANELS = {
    "deta": (r"Jet $\Delta\eta$", np.linspace(-0.15, 0.15, 50), True, False),
    "dphi": (r"Jet $\Delta\phi$", np.linspace(-0.15, 0.15, 50), True, False),
    "nconst": ("Jet # Constituents", None, True, False),
    "dpt": (r"Jet absolute $\Delta p_T$ [GeV]", None, True, False),
    "dpt_over_truth": (r"Jet $(p_T^{\mathrm{reco}} - p_T^{\mathrm{target}}) / p_T^{\mathrm{target}}$",
                       np.linspace(-1.0, 4.0, 110), False, True),
    "energy": ("Jet energy [GeV]", None, False, True),
}


def _draw_jet_panel(ax, merged: dict, key: str, legend_fontsize: float = 11,
                    legend_loc: str = "upper right", logy_headroom: float = 1.6,
                    liny_headroom: float = 1.5, tick_labelsize: float | None = None,
                    axis_label_fontsize: float | None = None,
                    logy: bool | None = None,
                    xlim: tuple[float, float] | None = None,
                    show_puppi: bool = True) -> None:
    """Draw one jet-resolution panel onto ``ax`` (shared by the 3x2 grid and the
    standalone single-panel figures so they stay identical).

    ``logy`` overrides the y-scale from :data:`JET_PANELS`; ``None`` keeps it.
    The override exists so one distribution can be rendered both ways -- log to
    show the tails, linear to show the shape of the core -- without duplicating
    the panel or perturbing the log-scaled figure the paper uses.

    ``xlim`` narrows the view window only: binning, and the M/IQR in the legend,
    are unchanged and still computed over the whole sample. It exists because a
    range chosen to show a long tail on a log axis leaves the core crammed into
    a corner once the axis is linear.
    """
    from scipy.stats import iqr

    g, pu = merged["glow_res"], merged["puppi_res"]
    nc, en = merged["nconst"], merged["energy"]
    xlabel, bins, density, panel_logy = JET_PANELS[key]
    logy = panel_logy if logy is None else logy

    # f = jet matching fraction (matched truth jets / all truth jets) per method.
    frac: dict = {}
    jm = merged.get("jet_match")
    if jm is not None and jm["truth_total"].sum() > 0:
        tot = float(jm["truth_total"].sum())
        frac[GLOWUP] = float(jm["glow_eff_num"].sum()) / tot
        frac[PUPPI_LABEL] = float(jm["puppi_eff_num"].sum()) / tot

    def series(which: str) -> np.ndarray:
        if key == "nconst":
            return nc.get(which, np.array([]))
        if key == "energy":
            return en.get(which, np.array([]))
        if which == "pflow":
            return g.get(key, np.array([]))
        if which == "puppi":
            return pu.get(key, np.array([]))
        return np.array([])     # truth has no matched residual

    def stats(name, d):
        d = d[np.isfinite(d)]
        if d.size == 0:
            return None
        # Median (not mean) so the centre is as outlier-robust as the IQR beside it,
        # and consistent with the M= labels on the per-class residual figures.
        s = rf"M={np.median(d):.3f}, IQR={iqr(d):.3f}"
        f = frac.get(name)
        if f is not None and np.isfinite(f):
            s += f", f={f:.3f}"
        return s

    glow = np.asarray(series("pflow"), dtype=float)
    if glow.size == 0:
        ax.text(0.5, 0.5, "no matched jets", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel(xlabel)
        return

    if bins is not None:
        b = bins
    elif key == "nconst":
        allv = np.concatenate([series(w) for w in ("pflow", "puppi", "truth")])
        b = np.arange(allv.min() - 0.5, allv.max() + 1.5) if allv.size else 20
    elif key == "energy":
        b = _pct_bins(glow, 80, positive=True)
    else:   # dpt
        b = _pct_bins(glow, 90)

    # Legend handles/labels are built manually (rather than via each hist's `label=`)
    # so PUPPI's caveat line can be its own row with an invisible handle: that keeps
    # the "PUPPI  M=..." row's own handle - the green box - aligned with that row's
    # text, instead of matplotlib centering a 2-line label against a single marker.
    handles, labels = [], []

    # Only GLOW-UP is filled; PUPPI and Target are distinct outlines.
    ax.hist(glow, bins=b, histtype="stepfilled", alpha=0.45, linewidth=1.6,
            density=density, color=GLOWUP_COLOR, edgecolor=GLOWUP_COLOR)
    glow_stats = stats(GLOWUP, glow)
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=GLOWUP_COLOR, edgecolor=GLOWUP_COLOR, alpha=0.45))
    labels.append(f"{GLOWUP}  {glow_stats}" if glow_stats else GLOWUP)

    # show_puppi=False drops PUPPI entirely -- histogram and both legend rows.
    pud = np.asarray(series("puppi"), dtype=float) if show_puppi else np.array([])
    if pud.size:
        ax.hist(pud, bins=b, histtype="step", linestyle="-", linewidth=2.2,
                density=density, color=PUPPI_COLOR)

    if key in ("nconst", "energy"):
        td = np.asarray(series("truth"), dtype=float)
        if td.size:
            ax.hist(td, bins=b, histtype="step", linestyle="--", linewidth=1.8,
                    density=density, color=TARGET_COLOR)
            td_stats = stats(TARGET, td)
            handles.append(plt.Line2D([], [], color=TARGET_COLOR, linestyle="--", linewidth=1.8))
            labels.append(f"{TARGET}  {td_stats}" if td_stats else TARGET)

    # PUPPI last, per request.
    if pud.size:
        puppi_stats = stats(PUPPI_LABEL, pud)
        handles.append(plt.Line2D([], [], color=PUPPI_COLOR, linestyle="-", linewidth=2.2))
        labels.append(f"PUPPI  {puppi_stats}" if puppi_stats else "PUPPI")
        handles.append(plt.Line2D([], [], color="none"))   # invisible: own row, no marker
        labels.append("(Ideal PUPPI with truth information)")

    # Top headroom so the upper-right legend never hides the central peak
    # (this was missing for the linear deta/dphi panels).
    if logy:
        ax.set_yscale("log")
        ylo, yhi = ax.get_ylim()
        ax.set_ylim(max(ylo, 1e-4), yhi * logy_headroom)
    else:
        ylo, yhi = ax.get_ylim()
        ax.set_ylim(ylo, yhi * liny_headroom)
    if xlim is not None:
        ax.set_xlim(*xlim)
    elif key in ("deta", "dphi"):
        ax.set_xlim(-0.15, 0.15)
    ax.set_xlabel(xlabel, fontsize=axis_label_fontsize)
    ax.set_ylabel("Density" if density else "Count", fontsize=axis_label_fontsize)
    if tick_labelsize is not None:
        ax.tick_params(axis="both", which="major", labelsize=tick_labelsize)
    ax.legend(handles=handles, labels=labels, fontsize=legend_fontsize, loc=legend_loc,
              framealpha=0.9, markerscale=1.6, handlelength=2.2)


# Methods in the multi-method jet figure: (key, label, colour, linestyle).
# Particle-level jets solid, calorimeter-level jets dashed; GLOW-UP warm, PUPPI
# green, so each like-for-like pair reads at a glance.
JET_METHODS = [
    ("glow", f"{GLOWUP} (particles)", GLOWUP_COLOR, "fill"),
    ("puppi", "Ideal PUPPI (tracks + clusters)", PUPPI_COLOR, "-"),
    ("glowcalo", f"{GLOWUP} calo: HS mask", "#E66100", "--"),
    ("glowcaloinc", f"{GLOWUP} calo: HS mask x incidence", "#8B0000", "-."),
    ("puppicalo", "PUPPI calo + truth HS charged", "#1B7837", "--"),
    ("hscalo", "Truth-HS calo (ideal calo)", "0.30", ":"),
]


def _jet_method_residuals(merged: dict, vs_hscalo: bool) -> list:
    """(key, label, colour, style, dpt_over_truth array) for each available method."""
    out = []
    for key, label, colour, style in JET_METHODS:
        if vs_hscalo:
            src = merged.get("calo_res_hs", {}).get(key)
        elif key == "glow":
            src = merged["glow_res"]
        elif key == "puppi":
            src = merged["puppi_res"]
        else:
            src = merged.get("calo_res", {}).get(key)
        if src is not None and len(src.get("dpt_over_truth", [])):
            out.append((key, label, colour, style, np.asarray(src["dpt_over_truth"], dtype=float)))
    return out


def _plot_jet_relative_pt_methods(merged: dict, vs_hscalo: bool = False):
    """Jet relative-pT residual for every method on one linear axis.

    Default reference is the Target particle jet -- the same as
    jet_relative_pt_linear -- so the two existing methods appear unchanged and
    the calorimeter jets are judged on the same footing. With ``vs_hscalo`` the
    calorimeter methods are instead referred to the truth-HS calorimeter jet:
    that removes the calorimeter's own response, which every calo jet shares,
    and leaves only how well each method removed pileup.

    f is the fraction of reference jets (Target, or truth-HS calo) with a match.
    """
    from scipy.stats import iqr

    bins = np.linspace(-1.0, 4.0, 201)
    methods = _jet_method_residuals(merged, vs_hscalo)
    # f = matched jets / reference jets: Target jets by default, truth-HS calo
    # jets when those are the reference.
    n_truth = None
    jm = merged.get("jet_match")
    if vs_hscalo:
        diag = merged.get("calo_diag", {})
        if "n_jets_hscalo" in diag and diag["n_jets_hscalo"].sum() > 0:
            n_truth = float(diag["n_jets_hscalo"].sum())
    elif jm is not None and jm["truth_total"].sum() > 0:
        n_truth = float(jm["truth_total"].sum())

    fig, ax = plt.subplots(figsize=(10, 6.5))
    handles, labels = [], []
    for key, label, colour, style, d in methods:
        d = d[np.isfinite(d)]
        if style == "fill":
            ax.hist(d, bins=bins, histtype="stepfilled", alpha=0.35, color=colour,
                    edgecolor=colour, linewidth=1.6)
            handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=colour, edgecolor=colour, alpha=0.35))
        else:
            ax.hist(d, bins=bins, histtype="step", linestyle=style, linewidth=2.2, color=colour)
            handles.append(plt.Line2D([], [], color=colour, linestyle=style, linewidth=2.2))
        txt = rf"M={np.median(d):+.3f}, IQR={iqr(d):.3f}"
        if n_truth:
            txt += f", f={d.size / n_truth:.3f}"
        labels.append(f"{label}\n    {txt}")

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.05)
    ref = "truth-HS calo jet" if vs_hscalo else "Target jet"
    ax.set_xlabel(rf"Jet $(p_T^{{\mathrm{{reco}}}} - p_T^{{\mathrm{{ref}}}}) / p_T^{{\mathrm{{ref}}}}$   (ref = {ref})",
                  fontsize=17.0)
    ax.set_ylabel("Count", fontsize=20.4)
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.legend(handles=handles, labels=labels, fontsize=10.5, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, labelspacing=0.9)
    fig.tight_layout()
    return fig


CALO_GLOW_LABEL = "GLOW-UP calo"
CALO_GLOW_NOTE = "(HS mask x predicted incidence)"
CALO_PUPPI_LABEL = "PUPPI calo"
CALO_PUPPI_NOTE = "(with truth HS charged energy; calo-tuned)"


def _plot_calo_vs_hscalo(merged: dict, key: str = "dpt_over_truth"):
    """Calo-level jet relative-pT residual vs the truth-HS calo jet: GLOW-UP calo
    (HS mask x predicted incidence) against PUPPI calo + truth HS charged.

    Drawn exactly like jet_relative_pt_linear -- same panel size, bins, x window,
    fonts and legend layout, GLOW-UP filled and PUPPI outlined -- so the two
    figures read as a pair: particle level there, calorimeter level here.

    ``key`` picks the residual: ``dpt_over_truth`` (relative pT), ``deta`` or
    ``dphi``; bins, range and normalisation follow the matching standalone
    GLOW-UP panel (jet_relative_pt_linear, jet_delta_eta, jet_delta_phi).
    """
    from scipy.stats import iqr

    res = merged["calo_res_hs"]
    n_ref = float(merged.get("calo_diag", {}).get("n_jets_hscalo", np.array([0])).sum())
    _, bins, density, _ = JET_PANELS[key]
    xlim = (-1.0, 1.0) if key == "dpt_over_truth" else (-0.15, 0.15)
    xlabel = {
        "dpt_over_truth": r"Jet $(p_T^{\mathrm{reco}} - p_T^{\mathrm{HS\,calo}}) / p_T^{\mathrm{HS\,calo}}$",
        "deta": r"Jet $\eta^{\mathrm{reco}} - \eta^{\mathrm{HS\,calo}}$",
        "dphi": r"Jet $\phi^{\mathrm{reco}} - \phi^{\mathrm{HS\,calo}}$",
    }[key]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    handles, labels = [], []
    for mkey, label, note, colour, filled in (
            ("glowcaloinc", CALO_GLOW_LABEL, CALO_GLOW_NOTE, GLOWUP_COLOR, True),
            ("puppicalo", CALO_PUPPI_LABEL, CALO_PUPPI_NOTE, PUPPI_COLOR, False)):
        d = np.asarray(res[mkey][key], dtype=float)
        d = d[np.isfinite(d)]
        if filled:
            ax.hist(d, bins=bins, histtype="stepfilled", alpha=0.45, linewidth=1.6,
                    density=density, color=colour, edgecolor=colour)
            handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=colour, edgecolor=colour, alpha=0.45))
        else:
            ax.hist(d, bins=bins, histtype="step", linestyle="-", linewidth=2.2,
                    density=density, color=colour)
            handles.append(plt.Line2D([], [], color=colour, linestyle="-", linewidth=2.2))
        txt = rf"M={round(float(np.median(d)), 3) + 0.0:.3f}, IQR={iqr(d):.3f}"
        if n_ref > 0:
            txt += f", f={d.size / n_ref:.3f}"
        labels.append(f"{label}  {txt}")
        handles.append(plt.Line2D([], [], color="none"))   # own row, no marker
        labels.append(note)

    ax.set_xlim(*xlim)
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi * 1.9)
    ax.set_xlabel(xlabel, fontsize=20.4)
    ax.set_ylabel("Density" if density else "Count", fontsize=20.4)
    ax.tick_params(axis="both", which="major", labelsize=22)
    ax.legend(handles=handles, labels=labels, fontsize=13.5, loc="upper right",
              framealpha=0.9, markerscale=1.6, handlelength=2.2)
    fig.tight_layout()
    return fig


def _plot_jet_resolution(merged: dict):
    """Reference 3x2 jet-resolution panels, mu + IQR + f labels; GLOW-UP/PUPPI/Target."""
    keys = ["deta", "dphi", "nconst", "dpt", "dpt_over_truth", "energy"]
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    for idx, key in enumerate(keys):
        _draw_jet_panel(axes[idx // 2, idx % 2], merged, key)
    fig.suptitle(f"Jet resolution: {GLOWUP} vs {PUPPI_LABEL} (vs {TARGET})")
    fig.tight_layout()
    return fig


def _plot_jet_single(merged: dict, key: str, logy: bool | None = None,
                     liny_headroom: float = 1.5,
                     xlim: tuple[float, float] | None = None,
                     legend_fontsize: float = 16.8,
                     show_puppi: bool = False):
    """Standalone single-panel version of one jet-resolution distribution.

    ``logy``/``xlim`` override the panel's y-scale and view window (see
    :func:`_draw_jet_panel`); ``liny_headroom`` is raised for linear-scale
    panels, whose legend needs more room above a peak that is not compressed by
    a log axis.
    """
    if not show_puppi:
        # Without PUPPI the legend loses two of its rows, so the room reserved
        # above the data for it shrinks with it -- otherwise the panel is mostly
        # empty sky.
        liny_headroom = 1.0 + (liny_headroom - 1.0) * 0.45
        logy_headroom = 12.0
    else:
        logy_headroom = 80.0
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    # The full-width 3-row legend can't dodge a central peak horizontally, so give
    # generous top headroom (log scale: 5x more than before) to seat it clear of the bars.
    _draw_jet_panel(ax, merged, key, legend_fontsize=legend_fontsize, legend_loc="upper right",
                    logy_headroom=logy_headroom, liny_headroom=liny_headroom, tick_labelsize=22,
                    axis_label_fontsize=20.4,   # 20% over the inherited 17
                    logy=logy, xlim=xlim, show_puppi=show_puppi)
    fig.tight_layout()
    return fig


def _plot_jet_iqr_binned(methods):
    """IQR vs truth-jet-pT for the three residuals (the IQR row of the binned
    figure on its own), GLOW-UP vs PUPPI."""
    edges = JET_RES_PT_EDGES
    centers = np.arange(len(edges) - 1)
    labels_x = [
        f"[{int(edges[i])},{('∞' if np.isinf(edges[i + 1]) else int(edges[i + 1]))})"
        for i in range(len(edges) - 1)
    ]
    residual_keys = [
        ("dpt_over_truth", r"$\Delta p_T / p_T^{\mathrm{Target}}$"),
        ("deta", r"$\Delta\eta$"),
        ("dphi", r"$\Delta\phi$"),
    ]
    colors = {GLOWUP: GLOWUP_COLOR, PUPPI_LABEL: PUPPI_COLOR}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharex=True)
    for col, (key, latex) in enumerate(residual_keys):
        ax = axes[col]
        for label, res in methods:
            if len(res.get(key, [])) == 0:
                continue
            _, iqrs, _ = _binned_mean_iqr(res[key], res["truth_pt"], edges=edges)
            ax.plot(centers, iqrs, marker="o", color=colors.get(label), label=label)
        ax.set_title(f"IQR {latex} vs truth $p_T$")
        ax.set_xlabel(r"truth jet $p_T$ bin [GeV]")
        ax.set_xticks(centers)
        ax.set_xticklabels(labels_x, rotation=30, ha="right")
    axes[0].set_ylabel("IQR of residual")
    axes[0].legend(fontsize=9, loc="best")
    fig.tight_layout()
    return fig


def _plot_jet_residual_box_single(methods, key, ylab, ylim, colors=None,
                                  xlabel=r"truth jet $p_T$ bin [GeV]", legend_fontsize=15.6,
                                  bin_key="truth_pt", edges=JET_RES_PT_EDGES):
    """One standalone box-per-pT-bin residual figure (GLOW-UP & PUPPI) for a single
    variable. Kept as three separate figures (rather than one multi-panel grid) and
    laid out with minimal padding, so they can be packed tightly side by side in the
    paper at whatever size/arrangement fits."""
    nbin = len(edges) - 1
    labels_x = [
        f"[{int(edges[i])},{('∞' if np.isinf(edges[i + 1]) else int(edges[i + 1]))})"
        for i in range(nbin)
    ]
    method_colors = {GLOWUP: GLOWUP_COLOR, PUPPI_LABEL: PUPPI_COLOR, **(colors or {})}
    # One method sits on the bin centre; two straddle it.
    offsets = {0: 0.0} if len(methods) == 1 else {0: -0.2, 1: 0.2}
    width = 0.32
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    for mi, (label, res) in enumerate(methods):
        vals = np.asarray(res.get(key, []), dtype=float)
        tpt = np.asarray(res.get(bin_key, []), dtype=float)
        if vals.size == 0:
            continue
        idx = np.digitize(tpt, edges) - 1
        pb = [vals[(idx == b) & np.isfinite(vals)] for b in range(nbin)]
        data = [v if v.size else np.array([np.nan]) for v in pb]
        off = offsets.get(mi, 0.0)
        bx = ax.boxplot(data, positions=np.arange(nbin) + off, widths=width,
                        patch_artist=True, showfliers=False, medianprops={"color": "black"})
        for b in bx["boxes"]:
            b.set_facecolor(method_colors.get(label, "gray"))
            b.set_alpha(0.65)
    if ylim:
        # Keep the tuned data window, but extend the top by 8% of the span so the
        # tallest whisker tips get some clear air below the axis edge.
        lo, hi = ylim
        ax.set_ylim(lo, hi + 0.08 * (hi - lo))
    ax.axhline(0.0, color="gray", lw=1, ls=":")
    ax.set_ylabel(ylab, fontsize=20.4)
    ax.set_xlabel(xlabel, fontsize=20.4)
    ax.set_xticks(np.arange(nbin))
    ax.set_xticklabels(labels_x, rotation=30, ha="right")
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)   # categorical x
    ax.tick_params(axis="both", which="major", labelsize=18)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=method_colors[m], alpha=0.65, label=m)
               for m, _ in methods]
    # Legend sits just above the frame, not inside the plot area.
    ax.legend(handles=handles, fontsize=legend_fontsize, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), ncol=1 if any(len(m) > 20 for m, _ in methods) else 2)
    # Small pad: minimal whitespace border so several of these tile compactly.
    fig.tight_layout(pad=0.3)
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
    ax.errorbar(centers, f1, yerr=err, fmt="o-", capsize=3, color=GLOWUP_COLOR)
    ax.axhline(1.0, color="gray", lw=1, ls="--")
    ax.set_xscale("log")
    # Matched to calo_pred_vs_truth_e_dist, since the two are shown side by side.
    ax.set_xlabel(r"Track $p_T$ [GeV]", fontsize=21.25)
    ax.set_ylabel("F1 Score", fontsize=21.25)
    ax.tick_params(axis="both", which="major", labelsize=22)
    ax.set_ylim(0, 1.1)
    ax.set_title("Track F1 vs pT", fontsize=22.5)
    fig.tight_layout()
    return fig


def _plot_track_pr_curve(track: dict):
    """Threshold-free track-classifier PR curve + Average Precision (AP / PR-AUC),
    reconstructed from the per-score positive/negative count histograms.

    AP is the standard interpolated-free weighted-mean-of-precisions estimator,
    AP = sum_k (R_k - R_{k-1}) * P_k, the most widely reported threshold-free
    metric under extreme class imbalance."""
    pos = track.get("score_pos")
    neg = track.get("score_neg")
    if pos is None or neg is None or (pos.sum() + neg.sum()) == 0:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.text(0.5, 0.5, "no track score data\n(re-run with --force)", ha="center", va="center",
                transform=ax.transAxes)
        return fig
    pos = pos.astype(float)
    neg = neg.astype(float)
    P = pos.sum()           # total real (positive) tracks
    # Sweep the threshold from high score to low. tp[k]/fp[k] are the positives /
    # negatives passing the k-th cut, ordered so recall is ASCENDING in k: the
    # tightest cut (top score bin only) first, the loosest (everything) last.
    tp = np.cumsum(pos[::-1])           # tp[k] = positives with score >= (N-k)-th edge
    fp = np.cumsum(neg[::-1])           # fp[k] = negatives with score >= same edge
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 1.0)
        recall = tp / P if P > 0 else np.zeros_like(tp)
    # AP = sum_k (R_k - R_{k-1}) * P_k with recall ascending; prepend the R=0 origin.
    rec = np.concatenate([[0.0], recall])
    ap = np.sum(np.diff(rec) * precision)
    # Prevalence = baseline AP of a random classifier under this imbalance.
    baseline = P / (P + neg.sum()) if (P + neg.sum()) > 0 else 0.0

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, "-", color=GLOWUP_COLOR, lw=2.0,
            label=f"{GLOWUP} (AP = {ap:.4f})")
    ax.axhline(baseline, color="gray", lw=1, ls="--",
               label=f"random baseline (prevalence = {baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Track classifier Precision-Recall ({int(P):,} real / {int(neg.sum()):,} fake)")
    ax.legend(fontsize=12, loc="lower left")
    fig.tight_layout()
    return fig


def _plot_n_particles_from_counts(npart_truth: np.ndarray, npart_pred: np.ndarray):
    """Per-event particle multiplicity per pT bin as grouped box plots
    (Target vs GLOW-UP), one pair of boxes per pT bin."""
    n_bins = len(NPART_PT_BINS) - 1
    bin_labels = [
        f"{NPART_PT_BINS[i]:g}-{NPART_PT_BINS[i + 1]:g}" for i in range(n_bins)
    ]
    fig, ax = plt.subplots(figsize=(1.6 * n_bins + 3, 7.15))   # height 30% taller (5.5 -> 7.15)
    width = 0.32
    pos = np.arange(n_bins)
    # GLOW-UP is always the LEFT box of each pair, matching the jet_residual_box_* figures.
    box_p = ax.boxplot([npart_pred[:, i] for i in range(n_bins)],
                       positions=pos - width / 2 - 0.02, widths=width, patch_artist=True,
                       showfliers=False, medianprops={"color": "black"})
    box_t = ax.boxplot([npart_truth[:, i] for i in range(n_bins)],
                       positions=pos + width / 2 + 0.02, widths=width, patch_artist=True,
                       showfliers=False, medianprops={"color": "black"})
    for b in box_t["boxes"]:
        b.set_facecolor(TARGET_COLOR)
        b.set_alpha(0.7)
    for b in box_p["boxes"]:
        b.set_facecolor(GLOWUP_COLOR)
        b.set_alpha(0.7)
    ax.set_xticks(pos)
    ax.set_xticklabels(bin_labels, rotation=30, ha="right")
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)   # categorical x
    # Matched to calo_pred_vs_truth_e_dist / track_f1_vs_pt sizing.
    ax.tick_params(axis="both", which="major", labelsize=22)
    ax.set_xlabel(r"Particle $p_T$ bin [GeV]", fontsize=21.25)
    ax.set_ylabel("Particles / event", fontsize=21.25)
    
    ax.set_title(f"Particle multiplicity per event by $p_T$ bin ({GLOWUP} vs {TARGET})",
                 fontsize=22.5)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=GLOWUP_COLOR, alpha=0.7, label=GLOWUP),
        plt.Rectangle((0, 0), 1, 1, facecolor=TARGET_COLOR, alpha=0.7, label=TARGET),
    ]
    ax.legend(handles=handles, fontsize=16.25)
    fig.tight_layout()
    return fig


def _plot_feature_scatter_from_hist(feat_hist: np.ndarray):
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    for i in range(3):
        for j in range(3):
            ax = axes[j, i]
            h = feat_hist[j, i]
            edges = FEAT_BINS[i]
            if h.sum() > 0:
                im = ax.pcolormesh(edges, edges, np.ma.masked_where(h.T == 0, h.T),
                                   norm=LogNorm(vmin=1), cmap="viridis")
                lo, hi = edges[0], edges[-1]
                ax.plot([lo, hi], [lo, hi], ls="--", color="red", lw=1.0)
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=25)
            else:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes,
                        fontsize=22.5)
            ax.set_xlabel(f"{TARGET} {FEAT_LABELS[i]}", fontsize=27.5)
            ax.set_ylabel(f"{GLOWUP} {FEAT_LABELS[i]}", fontsize=27.5)
            ax.set_title(FEAT_ROWS[j], fontsize=30)
            ax.tick_params(axis="both", which="major", labelsize=25)
    fig.suptitle(f"Feature density: {TARGET} vs {GLOWUP}",
                 fontsize=32.5, y=1.02)
    fig.tight_layout()
    return fig


def _plot_calo_recall_from_counts(calo: dict):
    edges = CALO_E_BINS
    centers = np.sqrt(edges[:-1] * edges[1:])
    # Prefer recall vs the cluster's HS energy; fall back to total cluster energy
    # for caches that predate the HS-energy binning.
    use_hs = "hs_truth" in calo and calo["hs_truth"].sum() > 0
    truth = (calo["hs_truth"] if use_hs else calo["truth"]).astype(float)
    rec = (calo["hs_recovered"] if use_hs else calo["recovered"]).astype(float)
    xlabel = "HS energy in cluster [GeV]" if use_hs else "Calorimeter cluster energy [GeV]"

    def _ratio_err(num, den):
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(den > 0, num / den, np.nan)
            e = np.where(den > 0, np.sqrt(np.clip(r * (1 - r), 0, None) / np.where(den > 0, den, 1)), np.nan)
        return r, e

    recall, recall_err = _ratio_err(rec, truth)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(centers, recall, yerr=recall_err, fmt="o-", capsize=3, color=GLOWUP_COLOR,
                label="HS recall")
    # Precision = TP / all predicted-HS clusters, per HS-energy bin.
    if use_hs and calo.get("hs_pred") is not None and calo["hs_pred"].sum() > 0:
        prec, prec_err = _ratio_err(calo["hs_recovered"].astype(float), calo["hs_pred"].astype(float))
        ax.errorbar(centers, prec, yerr=prec_err, fmt="s--", capsize=3, color="black",
                    label="HS precision")
    ax.axhline(1.0, color="gray", lw=1, ls="--")
    ax.set_xscale("log")
    if use_hs:
        ax.set_xlim(left=0.15)   # clip sub-threshold (degenerate) HS-energy bins
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"{GLOWUP} pileup removal: recall & precision vs cluster HS energy")
    ax.legend()
    fig.tight_layout()
    return fig


def _plot_calo_e_dist_from_hist(calo: dict):
    edges = CALO_E_BINS
    truth, pred = calo["truth"].astype(float), calo["pred"].astype(float)
    fig, ax = plt.subplots(figsize=(7, 5))
    h_truth = ax.stairs(truth, edges, color=TARGET_COLOR, linewidth=2.0,
                        label=f"{TARGET} (truth HS)  n={int(truth.sum()):,}")
    h_pred = ax.stairs(pred, edges, color=GLOWUP_COLOR, linewidth=2.0,
                       label=f"{GLOWUP} (pred HS)  n={int(pred.sum()):,}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    # 25% over the global sizes (labels 17, title 18, ticks 15) for this figure.
    ax.set_xlabel("Calorimeter cluster energy [GeV]", fontsize=21.25)
    ax.set_ylabel("Clusters", fontsize=21.25)
    ax.set_title("Calorimeter cluster energy in\npileup removal: pred vs truth HS",
                 fontsize=22.5)
    ax.tick_params(axis="both", which="major", labelsize=22)   # the 10^x tick text
    # Stretch the y-axis one full decade upward (log scale) so the legend clears
    # the distribution.
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi * 100.0)
    # GLOW-UP listed first (draw order keeps Target behind it).
    ax.legend(handles=[h_pred, h_truth], loc="upper right", framealpha=0.9,
              fontsize=16.25)   # 25% over the 13pt default
    fig.tight_layout()
    return fig


def _plot_calo_e_dist_incidence(diag: dict):
    """Cluster-energy spectrum with the incidence pileup correction.

    Same layout as calo_pred_vs_truth_e_dist, but every cluster enters at
    E * (HS fraction): Target with the truth fraction, on the truth-HS clusters;
    GLOW-UP with the predicted fraction -- sum of pred_incidence over non-null
    slots = 1 - predicted PU fraction -- on the clusters it keeps
    (calo_prob >= 0.3).
    """
    edges = CALO_E_BINS
    truth = diag["e_hist_truth_hs"].astype(float)
    pred = diag["e_hist_pred_inc"].astype(float)
    fig, ax = plt.subplots(figsize=(7, 5))
    h_truth = ax.stairs(truth, edges, color=TARGET_COLOR, linewidth=2.0,
                        label=rf"{TARGET} ($E\times f_{{\mathrm{{HS}}}}^{{\mathrm{{truth}}}}$)  n={int(truth.sum()):,}")
    h_pred = ax.stairs(pred, edges, color=GLOWUP_COLOR, linewidth=2.0,
                       label=rf"{GLOWUP} ($E\times f_{{\mathrm{{HS}}}}^{{\mathrm{{pred}}}}$)  n={int(pred.sum()):,}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Calorimeter cluster HS energy [GeV]", fontsize=21.25)
    ax.set_ylabel("Clusters", fontsize=21.25)
    ax.set_title("Calorimeter cluster energy, incidence-\ncorrected: pred vs truth HS",
                 fontsize=22.5)
    ax.tick_params(axis="both", which="major", labelsize=22)
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi * 100.0)
    ax.legend(handles=[h_pred, h_truth], loc="upper right", framealpha=0.9,
              fontsize=16.25)
    fig.tight_layout()
    return fig


# ── Per-class performance figures (matched objects) ───────────────────────

def _perf_pt_centers() -> np.ndarray:
    return np.sqrt(PERF_PT_BINS[:-1] * PERF_PT_BINS[1:])


def _plot_class_confusion(pc: dict):
    conf = pc["conf"].astype(float)
    rowsum = conf.sum(axis=1, keepdims=True)
    norm = np.divide(conf, rowsum, out=np.zeros_like(conf), where=rowsum > 0)
    col_labels = CLASS_LABELS[:N_CLASSES] + ["null/miss"]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(N_CLASSES + 1))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticks(range(N_CLASSES))
    ax.set_yticklabels(CLASS_LABELS[:N_CLASSES])
    ax.tick_params(which="minor", bottom=False, top=False, left=False, right=False)
    ax.set_xlabel(f"{GLOWUP} predicted class")
    ax.set_ylabel(f"{TARGET} class")
    for i in range(N_CLASSES):
        if rowsum[i, 0] == 0:
            continue
        for j in range(N_CLASSES + 1):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="row-normalized fraction")
    ax.set_title(f"Class confusion: {GLOWUP} vs {TARGET} (row-normalized)")
    fig.tight_layout()
    return fig


def _plot_class_pt_response(pc: dict):
    resp = pc["resp"]
    centers = _perf_pt_centers()
    fig, ax = plt.subplots(figsize=(8, 5))
    for c in range(N_CLASSES):
        med = np.array([_hist_percentile(resp[c, b], RESP_BINS, 50) for b in range(len(centers))])
        ax.plot(centers, med, marker="o", color=CLASS_COLORS[c], label=CLASS_LABELS[c])
    ax.axhline(1.0, color="gray", lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel(r"truth $p_T$ [GeV]")
    ax.set_ylabel(r"median $p_T^{\mathrm{pred}} / p_T^{\mathrm{truth}}$")
    ax.set_title(f"{GLOWUP} $p_T$ response per class")
    ax.legend()
    fig.tight_layout()
    return fig


def _plot_class_pt_resolution(pc: dict):
    resp = pc["resp"]
    centers = _perf_pt_centers()
    fig, ax = plt.subplots(figsize=(8, 5))
    for c in range(N_CLASSES):
        iqr = np.array([
            _hist_percentile(resp[c, b], RESP_BINS, 75) - _hist_percentile(resp[c, b], RESP_BINS, 25)
            for b in range(len(centers))
        ])
        ax.plot(centers, iqr, marker="o", color=CLASS_COLORS[c], label=CLASS_LABELS[c])
    ax.set_xscale("log")
    ax.set_xlabel(r"truth $p_T$ [GeV]")
    ax.set_ylabel(r"IQR of $(p_T^{\mathrm{pred}} - p_T^{\mathrm{truth}}) / p_T^{\mathrm{truth}}$")
    ax.set_title(f"{GLOWUP} $p_T$ resolution per class")
    ax.legend(fontsize=18.59)
    fig.tight_layout()
    return fig


def _plot_class_pt_distribution(pc: dict):
    """Truth particle pT spectrum, one step histogram per class (counts, log-log)."""
    cpt = pc.get("class_pt")
    bins = CLASS_PT_DIST_BINS
    if cpt is None:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.text(0.5, 0.5, "no class_pt data\n(re-run with --force)", ha="center", va="center",
                transform=ax.transAxes)
        return fig
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for c in range(N_CLASSES):
        h = cpt[c].astype(float)
        if h.sum() <= 0:
            continue
        ax.stairs(h, bins, color=CLASS_COLORS[c], linewidth=1.8, label=CLASS_LABELS[c])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.0, 200.0)
    # Stretch the log y-axis 5x upward to give the legend clear air above the curves.
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi * 5.0)
    # All text 30% over the global sizes (labels 17, title 18, ticks 15) and the
    # previous 11pt legend.
    ax.set_xlabel(r"Particle $p_T$ [GeV]", fontsize=22.1)
    ax.set_ylabel("Particles", fontsize=22.1)
    ax.set_title(f"{TARGET} particle $p_T$ spectrum per class", fontsize=23.4)
    ax.tick_params(axis="both", which="major", labelsize=19.5)
    ax.legend(fontsize=18.59)
    fig.tight_layout()
    return fig


def _plot_class_counts(cc: dict, cut_idx: int):
    """Per-class particle count bar chart (log y), Target vs GLOW-UP, for one pT cut.

    x-axis = the N_CLASSES real particle classes (Ch Had / e / mu / Neu Had / gamma);
    two bars per class (Target, GLOW-UP); the count above each cut CLASS_COUNT_PT_CUTS[cut_idx].
    """
    cut = CLASS_COUNT_PT_CUTS[cut_idx]
    truth = cc["truth"][:, cut_idx].astype(float)
    pflow = cc["pflow"][:, cut_idx].astype(float)
    x = np.arange(N_CLASSES)
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.bar(x - width / 2, truth, width, color=TARGET_COLOR, alpha=0.85,
           label=f"{TARGET} (N={int(truth.sum()):,})")
    ax.bar(x + width / 2, pflow, width, color=GLOWUP_COLOR, alpha=0.85,
           label=f"{GLOWUP} (N={int(pflow.sum()):,})")
    # Annotate each bar with its count (skip empties so log-scale gaps stay clean).
    for xpos, val in [*zip(x - width / 2, truth), *zip(x + width / 2, pflow)]:
        if val > 0:
            ax.text(xpos, val, f"{int(val):,}", ha="center", va="bottom", fontsize=8, rotation=90)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_LABELS[:N_CLASSES])
    ax.set_xlabel("Particle class")
    ax.set_ylabel("Particles")
    title = f"Particle count per class: {GLOWUP} vs {TARGET}"
    if cut > 0:
        title += rf" ($p_T > {cut:g}$ GeV)"
    ax.set_title(title)
    ax.legend(fontsize=12)
    fig.tight_layout()
    return fig


def _plot_class_pt_spectrum(sp: dict):
    """Per-class particle pT spectrum (count vs pT), log-log; one color per class,
    Target dashed vs GLOW-UP solid."""
    bins = CLASS_PT_DIST_BINS
    truth = sp["truth"].astype(float)     # (N_CLASSES, nbins)
    pflow = sp["pflow"].astype(float)
    fig, ax = plt.subplots(figsize=(9, 6))
    for c in range(N_CLASSES):
        if truth[c].sum() > 0:
            ax.stairs(truth[c], bins, color=CLASS_COLORS[c], linestyle="--", linewidth=1.6)
        if pflow[c].sum() > 0:
            ax.stairs(pflow[c], bins, color=CLASS_COLORS[c], linestyle="-", linewidth=1.9,
                      label=f"{CLASS_LABELS[c]} (N={int(pflow[c].sum()):,})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.15, 200.0)
    ax.set_xlabel(r"Particle $p_T$ [GeV]")
    ax.set_ylabel("Particles")
    ax.set_title(f"Particle $p_T$ spectrum per class: {GLOWUP} vs {TARGET}")
    # Class colors (solid = GLOW-UP) + a line-style key for the source.
    leg1 = ax.legend(fontsize=11, loc="upper right", title="class")
    ax.add_artist(leg1)
    style_handles = [
        plt.Line2D([], [], color="black", linestyle="-", linewidth=1.9, label=GLOWUP),
        plt.Line2D([], [], color="black", linestyle="--", linewidth=1.6, label=TARGET),
    ]
    ax.legend(handles=style_handles, fontsize=11, loc="lower left")
    fig.tight_layout()
    return fig


def _plot_class_f1_vs_pt(pc: dict):
    """Per-class classification F1 vs truth pT (over matched real objects)."""
    conf_pt = pc["conf_pt"].astype(float)   # (truth 0..4, pred 0..5, pT bin)
    centers = _perf_pt_centers()
    fig, ax = plt.subplots(figsize=(8, 5))
    for c in range(N_CLASSES):
        tp = conf_pt[c, c, :]
        fn = conf_pt[c, :, :].sum(axis=0) - tp      # truth c predicted as something else (incl null)
        fp = conf_pt[:, c, :].sum(axis=0) - tp      # other truth predicted as c
        with np.errstate(invalid="ignore", divide="ignore"):
            prec = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
            rec = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
            f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), np.nan)
        ax.plot(centers, f1, marker="o", color=CLASS_COLORS[c], label=CLASS_LABELS[c])
    ax.axhline(1.0, color="gray", lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(r"truth $p_T$ [GeV]")
    ax.set_ylabel("classification F1")
    ax.set_title(f"{GLOWUP} per-class classification F1 vs pT")
    ax.legend()
    fig.tight_layout()
    return fig


def _plot_class_efficiency(pc: dict):
    den = pc["eff_denom"].astype(float)
    num = pc["eff_reco"].astype(float)
    centers = _perf_pt_centers()
    fig, ax = plt.subplots(figsize=(8, 5))
    for c in range(N_CLASSES):
        with np.errstate(invalid="ignore", divide="ignore"):
            e = np.where(den[c] > 0, num[c] / den[c], np.nan)
        ax.plot(centers, e, marker="o", color=CLASS_COLORS[c], label=CLASS_LABELS[c])
    ax.axhline(1.0, color="gray", lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xlim(None, 200.0)
    ax.set_ylim(0, 1.1)
    # Text sizes matched to class_pt_distribution.
    ax.set_xlabel(r"truth $p_T$ [GeV]", fontsize=22.1)
    ax.set_ylabel("reconstruction efficiency", fontsize=22.1)
    ax.set_title(f"{GLOWUP} reconstruction efficiency per class", fontsize=23.4)
    ax.tick_params(axis="both", which="major", labelsize=19.5)
    ax.legend(fontsize=14.3)
    fig.tight_layout()
    return fig


def _plot_class_fake_rate_vs_pt(pc: dict):
    """Per-class fake rate vs predicted pT: predicted-class-c objects whose truth
    class != c (fakes / mis-ID) divided by all predicted class c, per pred-pT bin."""
    den = pc["fake_pred_n"].astype(float)
    num = pc["fake_wrong_n"].astype(float)
    centers = _perf_pt_centers()
    fig, ax = plt.subplots(figsize=(8, 5))
    for c in range(N_CLASSES):
        with np.errstate(invalid="ignore", divide="ignore"):
            fr = np.where(den[c] > 0, num[c] / den[c], np.nan)
        ax.plot(centers, fr, marker="o", color=CLASS_COLORS[c], label=CLASS_LABELS[c])
    ax.set_xscale("log")
    ax.set_xlim(None, 200.0)
    ax.set_ylim(0, 1.05)
    # Text sizes matched to class_pt_distribution.
    ax.set_xlabel(r"Pred $p_T$ [GeV]", fontsize=22.1)
    ax.set_ylabel("Fake rate", fontsize=22.1)
    ax.set_title(f"{GLOWUP} per-class fake rate vs $p_T$", fontsize=23.4)
    ax.tick_params(axis="both", which="major", labelsize=19.5)
    ax.legend(fontsize=18.59)
    fig.tight_layout()
    return fig


def _plot_class_match_accuracy_vs_pt(pc: dict):
    """Per-class classification accuracy among *matched* particles, vs truth pT.

    For truth particles that were reconstructed at all -- i.e. matched to a
    predicted particle -- what fraction were given the right class?

    ``p(class match | match) = eff_correct / eff_reco``

    The conditioning is the whole point: dividing by the matched count rather
    than by all truth particles takes reconstruction efficiency out of the
    picture, so this isolates how well the model *identifies* what it found from
    how much it found. A particle that was never reconstructed is absent from
    both numerator and denominator, and so cannot drag the curve down.

    Complements the two neighbouring figures, which cut the same data the other
    two ways: class_efficiency_vs_pt is ``eff_reco / eff_denom`` (did we find
    it), and class_fake_rate_vs_pt is predicted-side, binned by predicted pT
    (of what we called class c, how much was not). This one is truth-side and
    binned by truth pT -- the row-normalised diagonal of the confusion matrix,
    restricted to matched, per pT bin.
    """
    num = pc["eff_correct"].astype(float)
    den = pc["eff_reco"].astype(float)
    centers = _perf_pt_centers()
    fig, ax = plt.subplots(figsize=(8, 5))
    for c in range(N_CLASSES):
        with np.errstate(invalid="ignore", divide="ignore"):
            # Empty bins become gaps rather than zeros: a pT bin holding no
            # matched particles of this class says nothing about its accuracy.
            acc = np.where(den[c] > 0, num[c] / den[c], np.nan)
        ax.plot(centers, acc, marker="o", color=CLASS_COLORS[c], label=CLASS_LABELS[c])
    ax.axhline(1.0, color="gray", lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xlim(None, 200.0)
    ax.set_ylim(0, 1.05)
    # Text sizes matched to class_fake_rate_vs_pt.
    ax.set_xlabel(r"Truth $p_T$ [GeV]", fontsize=22.1)
    ax.set_ylabel(r"$p(\mathrm{class\ match}\,|\,\mathrm{match})$", fontsize=22.1)
    ax.set_title(f"{GLOWUP} per-class ID accuracy vs $p_T$", fontsize=23.4)
    ax.tick_params(axis="both", which="major", labelsize=19.5)
    ax.legend(fontsize=14.3)
    fig.tight_layout()
    return fig


def _plot_hs_dist(td: dict, which: str, stream: str = "track"):
    """Truth-HS vs predicted-HS track kinematics: the pT or z0 spectrum.

    A shape comparison of the track stream's selection against the truth
    hard-scatter tracks. Agreement here means the selection is unbiased in that
    variable; it is NOT an efficiency, and a perfectly-shaped selection can
    still be picking the wrong individual tracks. track_f1_vs_pt is the
    per-track figure.
    """
    noun = "Track" if stream == "track" else "Cluster"
    bins, xlabel, logx = {
        "pt": (TRACK_PT_BINS, r"Track $p_T$ [GeV]", True),
        "z0": (TRACK_Z0_BINS, r"Track $z_0$ [mm]", False),
        "d0": (TRACK_D0_BINS, r"Track $d_0$ [mm]", False),
        "eta": (ETA_DIST_BINS, rf"{noun} $\eta$", False),
        "phi": (PHI_DIST_BINS, rf"{noun} $\phi$", False),
    }[which]
    truth = td[f"{which}_truth"].astype(float)
    pred = td[f"{which}_pred"].astype(float)
    widths = np.diff(bins)
    dens = lambda h: h / h.sum() / widths if h.sum() > 0 else h

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.stairs(dens(truth), bins, fill=True, alpha=0.45,
              color=TARGET_COLOR,
              label=f"Truth HS {noun.lower()}s  ({int(td['n_truth'][0]):,})")
    ax.stairs(dens(pred), bins, color=GLOWUP_COLOR, linewidth=2.2,
              label=f"Predicted HS {noun.lower()}s  ({int(td['n_pred'][0]):,})")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=22.1)
    ax.set_ylabel("Normalised density", fontsize=22.1)
    ax.set_title(f"{GLOWUP} HS {noun.lower()} selection", fontsize=23.4)
    ax.tick_params(axis="both", which="major", labelsize=19.5)
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(0, yhi * 1.35)
    ax.legend(fontsize=15.0)
    fig.tight_layout()
    return fig


def _plot_particle_dist(pd: dict, which: str, n_events: int):
    """Target particle multiplicity per event in eta or phi.

    Target only, by choice: this documents what the model is asked to
    reconstruct. The GLOW-UP counts are still aggregated (pred_*), so a
    comparison can be drawn again without re-analysing the H5s.
    """
    bins, xlabel = {"eta": (PART_ETA_BINS, r"Target particle $\eta$"),
                    "phi": (PHI_DIST_BINS, r"Target particle $\phi$")}[which]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.stairs(pd[f"{which}_truth"].astype(float) / max(n_events, 1), bins, fill=True,
              alpha=0.55, color=TARGET_COLOR, edgecolor=TARGET_COLOR, linewidth=1.6,
              label=f"{TARGET} particles  ({int(pd['n_truth'][0]):,} in {n_events:,} events)")
    ax.set_xlabel(xlabel, fontsize=22.1)
    ax.set_ylabel("Particles / event / bin", fontsize=22.1)
    sym = r"$\eta$" if which == "eta" else r"$\phi$"
    ax.set_title(f"Target particle {sym} distribution", fontsize=23.4)
    ax.tick_params(axis="both", which="major", labelsize=19.5)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    ax.legend(fontsize=15.0)
    fig.tight_layout()
    return fig


def _plot_particle_dist_vs_pred(pd: dict, which: str, n_events: int):
    """Target vs GLOW-UP particles per event in eta or phi, all non-null objects.

    The companion of the target-only particle_{eta,phi}_dist. No matching is
    needed: it compares the two populations as wholes, as counts per event, so a
    regional excess or deficit of reconstructed particles shows up as one.
    """
    bins, xlabel = {"eta": (PART_ETA_BINS, r"Particle $\eta$"),
                    "phi": (PHI_DIST_BINS, r"Particle $\phi$"),
                    "pt": (PART_PT_BINS, r"Particle $p_T$ [GeV]")}[which]
    per_evt = lambda h: h.astype(float) / max(n_events, 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    c = "_c1"   # both populations cut at pT > PART_CMP_PT_CUT
    ax.stairs(per_evt(pd[f"{which}{c}_truth"]), bins, fill=True, alpha=0.45, color=TARGET_COLOR,
              label=f"{TARGET}  ({int(pd[f'n{c}_truth'][0]):,})")
    ax.stairs(per_evt(pd[f"{which}{c}_pred"]), bins, color=GLOWUP_COLOR, linewidth=2.2,
              label=f"{GLOWUP}  ({int(pd[f'n{c}_pred'][0]):,})")
    ax.plot([], [], " ", label=rf"$p_T > {PART_CMP_PT_CUT:g}$ GeV, both")
    ax.set_xlabel(xlabel, fontsize=22.1)
    ax.set_ylabel("Particles / event / bin", fontsize=22.1)
    sym = {"eta": r"$\eta$", "phi": r"$\phi$", "pt": r"$p_T$"}[which]
    ax.set_title(f"Particle {sym}: {GLOWUP} vs {TARGET}", fontsize=23.4)
    ax.tick_params(axis="both", which="major", labelsize=19.5)
    if which == "pt":
        # A steeply falling spectrum over four decades: log-log, with headroom
        # above the peak for the legend.
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(PART_CMP_PT_CUT * 0.8, PART_PT_BINS[-1])   # nothing below the cut
        lo, hi = ax.get_ylim()
        ax.set_ylim(max(lo, 1e-4), hi * 30)
    else:
        # Room for a 3-row legend: phi is flat across the whole width, so the
        # legend cannot sit beside the data and must sit above it.
        ax.set_ylim(0, ax.get_ylim()[1] * 1.6)
    ax.legend(fontsize=15.0)
    fig.tight_layout()
    return fig


def _plot_particles_per_event(pd: dict, n_events: int):
    """Distribution of the number of particles per event, Target vs GLOW-UP."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for tag, label, colour, filled in (("truth", TARGET, TARGET_COLOR, True),
                                       ("pred", GLOWUP, GLOWUP_COLOR, False)):
        h = pd[f"nevt_c1_{tag}"].astype(float)          # pT > PART_CMP_PT_CUT, both
        s1, s2 = pd[f"nevt_sum_c1_{tag}"]
        mu = s1 / max(n_events, 1)
        sd = np.sqrt(max(s2 / max(n_events, 1) - mu ** 2, 0.0))
        lab = rf"{label}   $\mu$={mu:.1f}, $\sigma$={sd:.1f}"
        if filled:
            ax.stairs(h, NPART_EVT_BINS, fill=True, alpha=0.45, color=colour, label=lab)
        else:
            ax.stairs(h, NPART_EVT_BINS, color=colour, linewidth=2.2, label=lab)
    ax.plot([], [], " ", label=rf"$p_T > {PART_CMP_PT_CUT:g}$ GeV, both")
    nz = np.nonzero(pd["nevt_c1_truth"] + pd["nevt_c1_pred"])[0]
    if nz.size:
        ax.set_xlim(NPART_EVT_BINS[max(nz[0] - 2, 0)], NPART_EVT_BINS[min(nz[-1] + 3, len(NPART_EVT_BINS) - 1)])
    ax.set_xlabel(rf"Particles per event ($p_T > {PART_CMP_PT_CUT:g}$ GeV)", fontsize=22.1)
    ax.set_ylabel("Events", fontsize=22.1)
    ax.set_title(f"Particle multiplicity: {GLOWUP} vs {TARGET}", fontsize=23.4)
    ax.tick_params(axis="both", which="major", labelsize=19.5)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
    ax.legend(fontsize=15.0)
    fig.tight_layout()
    return fig


def _plot_class_totals(ct: dict):
    """Truth vs predicted particle counts across all six classes, null included.

    Null dominates -- most of the 400 query slots are empty in any event -- so
    the bars are normalised and the raw counts annotated. This is a budget
    check: does the model produce about the right number of each species? It
    says nothing about whether the individual assignments are right, which is
    what class_match_accuracy_vs_pt covers.
    """
    truth = ct["truth"].astype(float)
    pred = ct["pred"].astype(float)
    labels = CLASS_LABELS                      # 5 real classes + "null"
    x = np.arange(len(labels))
    tn, pn = truth / truth.sum(), pred / pred.sum()

    fig, ax = plt.subplots(figsize=(11, 5.5))
    w = 0.4
    ax.bar(x - w / 2, tn, w, color=TARGET_COLOR, alpha=0.75, label="Target")
    ax.bar(x + w / 2, pn, w, color=GLOWUP_COLOR, alpha=0.9, label=GLOWUP)
    for i in range(len(labels)):
        ax.annotate(f"T:{int(truth[i]):,}\nP:{int(pred[i]):,}",
                    (x[i], max(tn[i], pn[i])), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=12.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=19.5)
    ax.set_xlabel("Class", fontsize=22.1)
    ax.set_ylabel("Fraction of objects", fontsize=22.1)
    ax.set_title(f"Particle class budget: {GLOWUP} vs {TARGET}", fontsize=23.4)
    ax.tick_params(axis="y", which="major", labelsize=19.5)
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(0, yhi * 1.28)
    ax.legend(fontsize=16.0)
    fig.tight_layout()
    return fig


def _plot_hs_energy_ratio(hr: dict):
    """Per-event predicted/truth hard-scatter mask energy, neutral vs charged.

    Charged hard-scatter energy is anchored by a track, so the model recovers
    it almost exactly. Neutral energy has no track to anchor it and must be
    inferred from the calorimeter alone, which is where the loss sits -- the
    gap between the two peaks is the cost of being trackless.
    """
    bins = HS_RATIO_BINS
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for tag, colour, label in (("neutral", TARGET_COLOR, "Neutral (trackless)"),
                               ("charged", GLOWUP_COLOR, "Charged (tracked)")):
        h = hr[f"{tag}_hist"].astype(float)
        n, ssum, sqsum = hr[f"{tag}_moments"]
        mu = ssum / n if n else np.nan
        sd = np.sqrt(max(sqsum / n - mu ** 2, 0.0)) if n else np.nan
        ax.stairs(h, bins, fill=True, alpha=0.5, color=colour,
                  label=rf"{label}   $\mu$={mu:.3f}, $\sigma$={sd:.3f}")
    ax.axvline(1.0, color="0.35", linestyle="--", linewidth=1.6, label="Ideal (1.0)")
    ax.set_yscale("log")
    ax.set_xlim(0.0, 1.5)
    ax.set_xlabel(r"$\sum E^{\mathrm{HS}}_{\mathrm{pred}} / \sum E^{\mathrm{HS}}_{\mathrm{truth}}$  (per event)",
                  fontsize=20.0)
    ax.set_ylabel("Events", fontsize=22.1)
    ax.set_title("Per-event HS energy recovery", fontsize=23.4)
    ax.tick_params(axis="both", which="major", labelsize=19.5)
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(max(ylo, 0.7), yhi * 12.0)
    ax.legend(fontsize=14.5, loc="upper left")
    fig.tight_layout()
    return fig


def _plot_class_residuals(pc: dict):
    """GLOW-UP kinematic residuals grouped Charged / Neutral, overlaying the
    classes as normalized (density) step histograms. Rows = (dpT/pT, deta, dphi),
    cols = Charged / Neutral (2-wide, so it fits the paper column)."""
    rows = [
        ("res_dptrel", DPTREL_BINS, r"$(p_T^{\mathrm{reco}} - p_T^{\mathrm{target}}) / p_T^{\mathrm{target}}$", (-0.5, 0.5)),
        ("res_deta", DANG_BINS, r"$\eta^{\mathrm{reco}} - \eta^{\mathrm{target}}$", (-0.1, 0.1)),
        ("res_dphi", DANG_BINS, r"$\phi^{\mathrm{reco}} - \phi^{\mathrm{target}}$", (-0.1, 0.1)),
    ]
    groups = [("Charged", [0, 1, 2]), ("Neutral", [3, 4])]   # class indices
    truth_n = pc["res_truth_n"]
    fig, axes = plt.subplots(3, 2, figsize=(11, 13), squeeze=False)
    for gi, (gname, classes) in enumerate(groups):
        for ri, (key, bins, xlab, xlim) in enumerate(rows):
            ax = axes[ri][gi]
            for c in classes:
                h = pc[key][c].astype(float)
                if h.sum() <= 0:
                    continue
                med = _hist_percentile(h, bins, 50)
                iqrv = _hist_percentile(h, bins, 75) - _hist_percentile(h, bins, 25)
                f = pc["res_n"][c] / truth_n[c] if truth_n[c] > 0 else np.nan
                widths = np.diff(bins)
                dens = h / (h.sum() * widths)     # normalize to unit area
                ax.stairs(dens, bins, color=CLASS_COLORS[c], linewidth=1.8,
                          label=f"{CLASS_LABELS[c]} (M={med:.3f}, IQR={iqrv:.3f}, f={f:.3f})")
            ax.set_xlabel(xlab)
            if gi == 0:   # clip only the Charged column; Neutral keeps full range
                ax.set_xlim(*xlim)
            # Stretch the y-axis upward so the legend sits above the peak instead
            # of on top of it (histograms keep their shape, just occupy less height).
            ax.set_ylim(top=ax.get_ylim()[1] * 1.54)   # 10% taller than the prior 1.4
            ax.legend(fontsize=12.4, loc="upper right", framealpha=0.9)   # 20% over 12.0
            ax.tick_params(axis="both", which="major", labelsize=19)

            if gi == 0:
                ax.set_ylabel("Density")
            if ri == 0:
                ax.set_title(gname)
    fig.suptitle(f"{GLOWUP} per-class kinematic residuals", y=1.005)
    fig.tight_layout()
    return fig


def _plot_jet_matching(jm: dict):
    """Jet matching efficiency (vs truth pT) and fake rate (vs reco pT), per method."""
    centers = np.sqrt(JET_PT_BINS[:-1] * JET_PT_BINS[1:])

    def ratio_err(num, den):
        num = num.astype(float)
        den = den.astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(den > 0, num / den, np.nan)
            e = np.where(den > 0, np.sqrt(np.clip(r * (1 - r), 0, None) / np.where(den > 0, den, 1)), np.nan)
        return r, e

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
    # Efficiency = matched truth / all truth jets
    for label, col, num in ((GLOWUP, GLOWUP_COLOR, jm["glow_eff_num"]),
                            (PUPPI_LABEL, PUPPI_COLOR, jm["puppi_eff_num"])):
        r, e = ratio_err(num, jm["truth_total"])
        axL.errorbar(centers, r, yerr=e, marker="o", capsize=2, color=col, label=label)
    axL.axhline(1.0, color="gray", lw=1, ls="--")
    axL.set_ylim(0, 1.05)
    axL.set_xlabel(r"truth jet $p_T$ [GeV]")
    axL.set_ylabel("matched / truth jets")
    axL.set_title("Jet matching efficiency")
    # Fake rate = (all reco - matched) / all reco jets
    for label, col, tot, matched in ((GLOWUP, GLOWUP_COLOR, jm["glow_total"], jm["glow_match_pred"]),
                                      (PUPPI_LABEL, PUPPI_COLOR, jm["puppi_total"], jm["puppi_match_pred"])):
        fake = np.clip(tot.astype(float) - matched.astype(float), 0, None)
        r, e = ratio_err(fake, tot)
        axR.errorbar(centers, r, yerr=e, marker="o", capsize=2, color=col, label=label)
    axR.set_ylim(0, None)
    axR.set_xlabel(r"reco jet $p_T$ [GeV]")
    axR.set_ylabel("unmatched / reco jets")
    axR.set_title("Jet fake rate")
    for ax in (axL, axR):
        ax.set_xscale("log")
        ax.legend()
    fig.suptitle(rf"Jet matching ({GLOWUP} & {PUPPI_LABEL} vs {TARGET}, $\Delta R<0.4$)", y=1.02)
    fig.tight_layout()
    return fig


def render_all(merged: dict) -> dict:
    """Render the figure set.

    Entries commented out below are diagnostics that no figure in the paper
    references; the plotting functions are kept, so re-enabling one is a matter
    of uncommenting its line. The shard aggregates they consume are still
    computed in :func:`analyze_shard`, so nothing about the map/reduce cost
    changes -- only what gets rendered and written.
    """
    methods = [(GLOWUP, merged["glow_res"]), (PUPPI_LABEL, merged["puppi_res"])]
    glow_only = [(GLOWUP, merged["glow_res"])]   # the jet_residual_box_* figures show GLOW-UP alone
    figs = {
        # app:eff_fake / sec:res_jets box plots
        "jet_residual_box_dpt": _plot_jet_residual_box_single(
            glow_only, "dpt_over_truth", r"Jet $\Delta p_T / p_T^{\mathrm{Target}}$", (-1.0, 1.2)),
        "jet_residual_box_deta": _plot_jet_residual_box_single(
            glow_only, "deta", r"Jet $\Delta\eta$", (-0.15, 0.15)),
        "jet_residual_box_dphi": _plot_jet_residual_box_single(
            glow_only, "dphi", r"Jet $\Delta\phi$", (-0.15, 0.15)),
        # sec:res_jets -- the four panels of fig:jetres_top / fig:jetres_higgs
        # (copied into the paper with a dihiggs_ prefix for the OOD sample).
        "jet_relative_pt": _plot_jet_single(merged, "dpt_over_truth"),
        "jet_delta_eta": _plot_jet_single(merged, "deta"),
        "jet_delta_phi": _plot_jet_single(merged, "dphi"),
        "jet_energy": _plot_jet_single(merged, "energy"),
        # Jet pT residual in absolute units, the GeV counterpart of
        # jet_relative_pt. Its panel is already in JET_PANELS (it is one of the
        # six in the 3x2 composite) and its aggregate is already computed per
        # shard, so this only renders a figure that was previously unemitted --
        # no --force needed on an existing cache.
        "jet_delta_pt": _plot_jet_single(merged, "dpt"),
        # Same relative-pT residual as jet_relative_pt, linear y instead of log.
        # The log axis is what makes the tails legible, but it also flattens the
        # core; on a linear axis the peak shape and the GLOW-UP/PUPPI difference
        # around zero are the readable part. The x window is narrowed to match:
        # the bins still run to +4, but past ~1 there is nothing a linear axis
        # can resolve, and the full range would squeeze the peak into a corner.
        # Extra headroom because an uncompressed peak leaves the 4-row legend
        # less room than the log version does.
        # The legend also shrinks: a linear count axis needs 5-digit tick labels
        # where the log one needed "10^4", which narrows the axes enough that the
        # full-size legend would overrun them.
        # All methods, particle- and calorimeter-level, on one axis.
        **({"jet_relative_pt_methods_linear": _plot_jet_relative_pt_methods(merged)}
           if merged.get("calo_res") is not None else {}),
        **({"jet_relative_pt_calo_vs_hscalo_linear": _plot_calo_vs_hscalo(merged),
            "jet_delta_eta_calo_vs_hscalo": _plot_calo_vs_hscalo(merged, "deta"),
            "jet_delta_phi_calo_vs_hscalo": _plot_calo_vs_hscalo(merged, "dphi"),
            # Box plots of the calo comparison, against the truth-HS calo jet and
            # binned in its pT -- the like-for-like counterpart of jet_residual_box_*.
            **{f"jet_residual_box_calo_{name}": _plot_jet_residual_box_single(
                [(CALO_GLOW_LABEL, merged["calo_res_hs"]["glowcaloinc"]),
                 (CALO_PUPPI_LABEL, merged["calo_res_hs"]["puppicalo"])],
                key, ylab, ylim,
                colors={CALO_GLOW_LABEL: GLOWUP_COLOR, CALO_PUPPI_LABEL: PUPPI_COLOR},
                xlabel=r"truth-HS calo jet $p_T$ bin [GeV]")
               for name, key, ylab, ylim in (
                   ("dpt", "dpt_over_truth", r"Jet $\Delta p_T / p_T^{\mathrm{HS\,calo}}$", (-1.0, 1.2)),
                   ("deta", "deta", r"Jet $\Delta\eta$", (-0.15, 0.15)),
                   ("dphi", "dphi", r"Jet $\Delta\phi$", (-0.15, 0.15)))},
            # Energy counterpart of jet_residual_box_calo_dpt: Delta E / E of the
            # same matched jet pairs, binned in the truth-HS calo jet energy.
            **({"jet_residual_box_calo_de": _plot_jet_residual_box_single(
                [(CALO_GLOW_LABEL, merged["calo_res_hs"]["glowcaloinc"]),
                 (CALO_PUPPI_LABEL, merged["calo_res_hs"]["puppicalo"])],
                "de_over_truth", r"Jet $\Delta E / E^{\mathrm{HS\,calo}}$", (-1.0, 1.2),
                colors={CALO_GLOW_LABEL: GLOWUP_COLOR, CALO_PUPPI_LABEL: PUPPI_COLOR},
                xlabel=r"truth-HS calo jet $E$ bin [GeV]",
                bin_key="truth_e", edges=JET_RES_E_EDGES)}
               if "de_over_truth" in merged["calo_res_hs"]["glowcaloinc"] else {})}
           if merged.get("calo_res_hs") is not None else {}),
        "jet_relative_pt_linear": _plot_jet_single(
            merged, "dpt_over_truth", logy=False, liny_headroom=1.9,
            xlim=(-1.0, 1.0), legend_fontsize=13.5),
        # Fig. 2a (sec:res_pileup)
        "track_f1_vs_pt": _plot_track_f1_from_counts(merged["track"]),
        # Fig. 3 (sec:res_particles)
        "n_particles_by_pt_bin": _plot_n_particles_from_counts(merged["npart_truth"], merged["npart_pred"]),
        # Fig. 2b (sec:res_pileup)
        "calo_pred_vs_truth_e_dist": _plot_calo_e_dist_from_hist(merged["calo"]),
        **({"calo_pred_vs_truth_e_dist_incidence":
                _plot_calo_e_dist_incidence(merged["calo_diag"])}
           if "e_hist_pred_inc" in merged.get("calo_diag", {}) else {}),

        # --- not in the paper ---------------------------------------------
        # jet_resolution is the 3x2 composite of the four panels kept above;
        # the paper uses the standalone panels instead.
        # "jet_resolution": _plot_jet_resolution(merged),
        # "jet_resolution_binned": _make_binned_plots(methods, edges=JET_RES_PT_EDGES),
        # "jet_resolution_iqr_binned": _plot_jet_iqr_binned(methods),
        # "track_pr_curve": _plot_track_pr_curve(merged["track"]),
        # "feature_scatter": _plot_feature_scatter_from_hist(merged["feat_hist"]),
        # **({
        #     f"class_counts_{('nocut' if c <= 0 else 'pt' + format(c, 'g').replace('.', 'p'))}":
        #         _plot_class_counts(merged["class_count"], k)
        #     for k, c in enumerate(CLASS_COUNT_PT_CUTS)
        # } if merged.get("class_count") is not None else {}),
        # **({"class_pt_spectrum": _plot_class_pt_spectrum(merged["class_pt_spec"])}
        #    if merged.get("class_pt_spec") is not None else {}),
        # "calo_recall_vs_pt": _plot_calo_recall_from_counts(merged["calo"]),
    }
    # Not in the paper:
    # if merged.get("jet_match") is not None:
    #     figs["jet_matching_efficiency"] = _plot_jet_matching(merged["jet_match"])
    # Per-class figures only when the (matched) per-class aggregates are present
    # (skipped silently for older caches that predate them — re-run with --force).
    pc = merged.get("per_class")
    if pc is not None:
        if "res_truth_n" in pc:
            # sec:res_particles, fig:ptrel_class
            figs["class_residuals"] = _plot_class_residuals(pc)
        # app:eff_fake, fig:class_eff / fig:class_fake / fig:class_pt
        figs["class_efficiency_vs_pt"] = _plot_class_efficiency(pc)
        if "fake_pred_n" in pc:
            figs["class_fake_rate_vs_pt"] = _plot_class_fake_rate_vs_pt(pc)
        if "eff_correct" in pc:
            figs["class_match_accuracy_vs_pt"] = _plot_class_match_accuracy_vs_pt(pc)
    # Track-stream and per-event figures. Guarded like the per-class block so an
    # older cache that predates these aggregates skips them rather than failing.
    if merged.get("track_dist") is not None:
        td = merged["track_dist"]
        for var in ("pt", "z0", "eta", "phi"):
            if f"{var}_truth" in td:
                figs[f"track_hs_{var}_dist"] = _plot_hs_dist(td, var, "track")
        # d0 only exists in H5s written after node_d0 was added to the writer.
        if "d0_truth" in td:
            figs["track_hs_d0_dist"] = _plot_hs_dist(td, "d0", "track")
    if merged.get("calo_dist") is not None:
        for var in ("eta", "phi"):
            figs[f"calo_hs_{var}_dist"] = _plot_hs_dist(merged["calo_dist"], var, "calo")
    if merged.get("particle_dist") is not None:
        for var in ("eta", "phi"):
            figs[f"particle_{var}_dist"] = _plot_particle_dist(
                merged["particle_dist"], var, int(merged["n_events"]))
            if f"{var}_c1_truth" in merged["particle_dist"]:
                figs[f"particle_{var}_dist_vs_pred"] = _plot_particle_dist_vs_pred(
                    merged["particle_dist"], var, int(merged["n_events"]))
        if "nevt_c1_truth" in merged["particle_dist"]:
            figs["particles_per_event_vs_pred"] = _plot_particles_per_event(
                merged["particle_dist"], int(merged["n_events"]))
        if "pt_c1_truth" in merged["particle_dist"]:
            figs["particle_pt_dist_vs_pred"] = _plot_particle_dist_vs_pred(
                merged["particle_dist"], "pt", int(merged["n_events"]))
    if merged.get("class_totals") is not None:
        figs["class_totals"] = _plot_class_totals(merged["class_totals"])
    if merged.get("hs_ratio") is not None:
        figs["hs_energy_ratio_by_type"] = _plot_hs_energy_ratio(merged["hs_ratio"])
        if "class_pt" in pc:
            figs["class_pt_distribution"] = _plot_class_pt_distribution(pc)

        # --- not in the paper ---------------------------------------------
        # figs["class_confusion_matrix"] = _plot_class_confusion(pc)
        # figs["class_pt_response"] = _plot_class_pt_response(pc)
        # figs["class_pt_resolution"] = _plot_class_pt_resolution(pc)
        # figs["class_f1_vs_pt"] = _plot_class_f1_vs_pt(pc)
    return figs


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_fig(fig, out_dir: Path, name: str, dpi: int) -> None:
    if fig is None:
        print(f"  [skip] {name}: figure is None")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    # PDF only -- that is what the paper includes. Add "png" back to the tuple
    # if a raster copy is wanted for slides.
    for ext in ("pdf",):
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
    p.add_argument("--best-json-calo", type=str, default=DEFAULT_BEST_JSON_CALO,
                   help="PUPPI parameters for the calo-jet comparison (PUPPI calo jets only).")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Output directory (default: <first-h5-dir>/paper_plots).")
    p.add_argument("--state-dir", type=str, default=None,
                   help="Per-shard aggregate cache (default: <out-dir>/state).")
    p.add_argument("--workers", type=int, default=None,
                   help="Worker processes (default: min(cpu_count, n_shards)).")
    p.add_argument("--force", action="store_true", help="Recompute all shards (ignore cache).")
    p.add_argument("--no-per-class", dest="per_class", action="store_false", default=True,
                   help="Skip the per-class figures (confusion / response / resolution / "
                        "efficiency). They assume matched shards (predict_only=False).")
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
    # Unlike --best-json above, a missing calo tuning is an error, not a silent
    # fall-back to defaults that would quietly change the calo comparison.
    if not Path(args.best_json_calo).is_file():
        raise FileNotFoundError(f"calo PUPPI tuning not found: {args.best_json_calo}")
    puppi_params_calo = json.loads(Path(args.best_json_calo).read_text())["best_params"]
    print(f"Loaded calo PUPPI params ({Path(args.best_json_calo).name}): {puppi_params_calo}", flush=True)
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
                                   args.events_per_file, jet_cfg, subtract_pu, puppi_params,
                                   args.per_class, puppi_params_calo): f
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
