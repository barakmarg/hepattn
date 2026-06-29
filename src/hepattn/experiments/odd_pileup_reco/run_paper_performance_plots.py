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
from hepattn.experiments.odd_pileup_reco.reco_analysis import (
    CLASS_COLORS,
    CLASS_LABELS,
    cluster_jets,
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
#DEFAULT_PARQUET_DIR = "/storage/agrp/barakma/PileupODD/data/ttbar_pu200_all_vertices_paper"
DEFAULT_PARQUET_DIR = "/storage/agrp/barakma/PileupODD/data/dihiggs_pu200_all_vertices_paper"
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
FEAT_LABELS = [r"$p_T$ [GeV]", r"$\eta$", r"$\phi$ [rad]"]
FEAT_ROWS = ["All", "Charged", "Neutral"]
NPART_PT_BINS = (0.0, 2.0, 5.0, 10.0, 20.0, 50.0, 200.0)   # 0-1 and 1-2 merged into 0-2
CALO_E_BINS = np.logspace(np.log10(0.05), np.log10(2000.0), 60)
RESIDUAL_KEYS = ("dpt", "dpt_over_truth", "deta", "dphi", "truth_pt")
IND_THRESHOLD = 0.5
PARQUET_EVENTS_PER_FILE = 100   # event_ids per parquet shard file (NNNNN = eid // this)

# ── Per-particle-class performance (assumes MATCHED objects: predict_only=False) ─
N_CLASSES = 5                                            # real classes 0..4 (5 = null)
PERF_PT_BINS = np.logspace(np.log10(1.0), np.log10(500.0), 16)   # 15 truth-pT bins
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

    return {"conf": conf, "conf_pt": conf_pt, "eff_denom": eff_denom,
            "eff_reco": eff_reco, "eff_correct": eff_correct, "resp": resp,
            "fake_pred_n": fake_pred_n, "fake_wrong_n": fake_wrong_n}


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
                  jet_cfg: dict, subtract_pu: bool, puppi_params: dict,
                  per_class: bool = True) -> dict:
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

    # Per-class performance aggregates (assume matched objects; predict_only=False).
    per_class_agg = _per_class_counts(data) if per_class else None
    if per_class_agg is not None:
        per_class_agg.update(_class_residual_counts(data))

    return {
        "n_events": n_events,
        "per_class": per_class_agg,
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
    nb_calo = len(CALO_E_BINS) - 1
    m["calo"] = {k: np.sum([a["calo"].get(k, np.zeros(nb_calo, dtype=np.int64)) for a in aggs], axis=0)
                 for k in ("truth", "recovered", "pred", "hs_truth", "hs_recovered", "hs_pred")}
    if aggs[0].get("per_class") is not None:
        pcs = [a["per_class"] for a in aggs]
        m["per_class"] = {k: np.sum([pc[k] for pc in pcs], axis=0) for k in pcs[0]}
    if aggs[0].get("jet_match") is not None:
        jms = [a["jet_match"] for a in aggs]
        m["jet_match"] = {k: np.sum([jm[k] for jm in jms], axis=0) for k in jms[0]}
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
    "dpt_over_truth": (r"Jet relative $\Delta p_T / p_T^{\mathrm{Target}}$", np.linspace(-1.0, 4.0, 110), False, True),
    "energy": ("Jet energy [GeV]", None, False, True),
}


def _draw_jet_panel(ax, merged: dict, key: str, legend_fontsize: float = 11) -> None:
    """Draw one jet-resolution panel onto ``ax`` (shared by the 3x2 grid and the
    standalone single-panel figures so they stay identical)."""
    from scipy.stats import iqr

    g, pu = merged["glow_res"], merged["puppi_res"]
    nc, en = merged["nconst"], merged["energy"]
    xlabel, bins, density, logy = JET_PANELS[key]

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

    def lbl(name, d):
        d = d[np.isfinite(d)]
        if d.size == 0:
            return name
        s = rf"{name}  $\mu$={np.mean(d):.3f}, IQR={iqr(d):.3f}"
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

    # Only GLOW-UP is filled; PUPPI and Target are distinct outlines.
    ax.hist(glow, bins=b, histtype="stepfilled", alpha=0.45, linewidth=1.6,
            density=density, color=GLOWUP_COLOR, edgecolor=GLOWUP_COLOR, label=lbl(GLOWUP, glow))
    pud = np.asarray(series("puppi"), dtype=float)
    if pud.size:
        ax.hist(pud, bins=b, histtype="step", linestyle="-", linewidth=2.2,
                density=density, color=PUPPI_COLOR, label=lbl(PUPPI_LABEL, pud))
    if key in ("nconst", "energy"):
        td = np.asarray(series("truth"), dtype=float)
        if td.size:
            ax.hist(td, bins=b, histtype="step", linestyle="--", linewidth=1.8,
                    density=density, color="black", label=lbl(TARGET, td))

    # Top headroom so the upper-right legend never hides the central peak
    # (this was missing for the linear deta/dphi panels).
    if logy:
        ax.set_yscale("log")
        ylo, yhi = ax.get_ylim()
        ax.set_ylim(max(ylo, 1e-4), yhi * 1.6)
    else:
        ylo, yhi = ax.get_ylim()
        ax.set_ylim(ylo, yhi * 1.5)
    if key in ("deta", "dphi"):
        ax.set_xlim(-0.15, 0.15)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density" if density else "Count")
    ax.legend(fontsize=legend_fontsize, loc="upper right", framealpha=0.9,
              markerscale=1.6, handlelength=2.2)


def _plot_jet_resolution(merged: dict):
    """Reference 3x2 jet-resolution panels, mu + IQR + f labels; GLOW-UP/PUPPI/Target."""
    keys = ["deta", "dphi", "nconst", "dpt", "dpt_over_truth", "energy"]
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    for idx, key in enumerate(keys):
        _draw_jet_panel(axes[idx // 2, idx % 2], merged, key)
    fig.suptitle(f"Jet resolution: {GLOWUP} vs {PUPPI_LABEL} (vs {TARGET})")
    fig.tight_layout()
    return fig


def _plot_jet_single(merged: dict, key: str):
    """Standalone single-panel version of one jet-resolution distribution."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    _draw_jet_panel(ax, merged, key, legend_fontsize=14)
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
        ax.grid(alpha=0.3)
        ax.set_xlabel(r"truth jet $p_T$ bin [GeV]")
        ax.set_xticks(centers)
        ax.set_xticklabels(labels_x, rotation=30, ha="right")
    axes[0].set_ylabel("IQR of residual")
    axes[0].legend(fontsize=9, loc="best")
    fig.tight_layout()
    return fig


def _plot_jet_residual_boxes(methods):
    """Jet residuals as box-per-pT-bin, GLOW-UP & PUPPI. The per-bin jet count is
    annotated on top of each box (N=...)."""
    edges = JET_RES_PT_EDGES
    nbin = len(edges) - 1
    labels_x = [
        f"[{int(edges[i])},{('∞' if np.isinf(edges[i + 1]) else int(edges[i + 1]))})"
        for i in range(nbin)
    ]
    residual_keys = [
        ("dpt_over_truth", r"Jet $\Delta p_T / p_T^{\mathrm{Target}}$", (-1.0, 1.2)),
        ("deta", r"Jet $\Delta\eta$", (-0.15, 0.15)),
        ("dphi", r"Jet $\Delta\phi$", (-0.15, 0.15)),
    ]
    method_colors = {GLOWUP: GLOWUP_COLOR, PUPPI_LABEL: PUPPI_COLOR}
    offsets = {0: -0.2, 1: 0.2}
    width = 0.32
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for col, (key, ylab, ylim) in enumerate(residual_keys):
        ax = axes[col]
        trans = ax.get_xaxis_transform()
        for mi, (label, res) in enumerate(methods):
            vals = np.asarray(res.get(key, []), dtype=float)
            tpt = np.asarray(res.get("truth_pt", []), dtype=float)
            if vals.size == 0:
                continue
            idx = np.digitize(tpt, edges) - 1
            pb = [vals[(idx == b) & np.isfinite(vals)] for b in range(nbin)]
            counts = [v.size for v in pb]
            data = [v if v.size else np.array([np.nan]) for v in pb]
            off = offsets.get(mi, 0.0)
            bx = ax.boxplot(data, positions=np.arange(nbin) + off, widths=width,
                            patch_artist=True, showfliers=False, medianprops={"color": "black"})
            for b in bx["boxes"]:
                b.set_facecolor(method_colors.get(label, "gray"))
                b.set_alpha(0.65)
            for b in range(nbin):
                if counts[b] > 0:
                    ax.text(b + off, 0.985, f"N={counts[b]:,}", transform=trans, rotation=90,
                            va="top", ha="center", fontsize=11, color=method_colors.get(label, "gray"))
        if ylim:
            ax.set_ylim(*ylim)
        ax.axhline(0.0, color="gray", lw=1, ls=":")
        ax.set_ylabel(ylab)
        ax.set_xlabel(r"truth jet $p_T$ bin [GeV]")
        ax.set_xticks(np.arange(nbin))
        ax.set_xticklabels(labels_x, rotation=30, ha="right")
        ax.grid(True, axis="y", alpha=0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=method_colors[m], alpha=0.65, label=m)
               for m, _ in methods]
    axes[0].legend(handles=handles, fontsize=10, loc="lower left")
    fig.suptitle(rf"Jet residual box plots vs truth $p_T$ (N = jets per bin): {GLOWUP} vs {PUPPI_LABEL}", y=1.02)
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
    ax.set_xlabel(r"Track $p_T$ [GeV]")
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.1)
    ax.set_title("Track F1 vs pT")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_n_particles_from_counts(npart_truth: np.ndarray, npart_pred: np.ndarray):
    """Per-event particle multiplicity per pT bin as grouped box plots
    (Target vs GLOW-UP), one pair of boxes per pT bin."""
    n_bins = len(NPART_PT_BINS) - 1
    bin_labels = [
        f"{NPART_PT_BINS[i]:g}-{NPART_PT_BINS[i + 1]:g}" for i in range(n_bins)
    ]
    fig, ax = plt.subplots(figsize=(1.6 * n_bins + 3, 5.5))
    width = 0.32
    pos = np.arange(n_bins)
    box_t = ax.boxplot([npart_truth[:, i] for i in range(n_bins)],
                       positions=pos - width / 2 - 0.02, widths=width, patch_artist=True,
                       showfliers=False, medianprops={"color": "black"})
    box_p = ax.boxplot([npart_pred[:, i] for i in range(n_bins)],
                       positions=pos + width / 2 + 0.02, widths=width, patch_artist=True,
                       showfliers=False, medianprops={"color": "black"})
    for b in box_t["boxes"]:
        b.set_facecolor(TARGET_COLOR)
        b.set_alpha(0.7)
    for b in box_p["boxes"]:
        b.set_facecolor(GLOWUP_COLOR)
        b.set_alpha(0.7)
    ax.set_xticks(pos)
    ax.set_xticklabels(bin_labels)
    ax.set_xlabel(r"Particle $p_T$ bin [GeV]")
    ax.set_ylabel("Particles / event")
    ax.set_title(f"Particle multiplicity per event by $p_T$ bin ({GLOWUP} vs {TARGET})")
    ax.grid(True, axis="y", alpha=0.3)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=TARGET_COLOR, alpha=0.7, label=TARGET),
        plt.Rectangle((0, 0), 1, 1, facecolor=GLOWUP_COLOR, alpha=0.7, label=GLOWUP),
    ]
    ax.legend(handles=handles, fontsize=11)
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
                im = ax.pcolormesh(edges, edges, np.ma.masked_where(h.T == 0, h.T),
                                   norm=LogNorm(vmin=1), cmap="viridis")
                lo, hi = edges[0], edges[-1]
                ax.plot([lo, hi], [lo, hi], ls="--", color="red", lw=1.0)
                ax.text(0.03, 0.97, f"n={int(h.sum()):,}", transform=ax.transAxes,
                        va="top", ha="left", fontsize=8, color="white",
                        bbox={"facecolor": "black", "alpha": 0.35, "pad": 2})
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("counts (log scale)", fontsize=8)
            else:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel(f"{TARGET} {FEAT_LABELS[i]}")
            ax.set_ylabel(f"{GLOWUP} {FEAT_LABELS[i]}")
            ax.set_title(FEAT_ROWS[j])
    fig.suptitle(f"Feature density (hist2d, log color scale): {TARGET} vs {GLOWUP}", y=1.01)
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
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def _plot_calo_e_dist_from_hist(calo: dict):
    edges = CALO_E_BINS
    truth, pred = calo["truth"].astype(float), calo["pred"].astype(float)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.stairs(truth, edges, color=TARGET_COLOR, linewidth=2.0, label=f"{TARGET} (truth HS)  n={int(truth.sum()):,}")
    ax.stairs(pred, edges, color=GLOWUP_COLOR, linewidth=2.0, label=f"{GLOWUP} (pred HS)  n={int(pred.sum()):,}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Calorimeter cluster energy [GeV]")
    ax.set_ylabel("Clusters")
    ax.set_title("Calorimeter cluster energy in pileup removal: pred vs truth HS")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
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
    ax.grid(True, which="both", alpha=0.3)
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
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
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
    ax.grid(True, which="both", alpha=0.3)
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
            err = np.where(den[c] > 0, np.sqrt(np.clip(e * (1 - e), 0, None) / np.where(den[c] > 0, den[c], 1)), np.nan)
        ax.errorbar(centers, e, yerr=err, marker="o", capsize=2, color=CLASS_COLORS[c], label=CLASS_LABELS[c])
    ax.axhline(1.0, color="gray", lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.1)
    ax.set_xlabel(r"truth $p_T$ [GeV]")
    ax.set_ylabel("reconstruction efficiency")
    ax.set_title(f"{GLOWUP} reconstruction efficiency per class")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
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
            err = np.where(den[c] > 0, np.sqrt(np.clip(fr * (1 - fr), 0, None) / np.where(den[c] > 0, den[c], 1)), np.nan)
        ax.errorbar(centers, fr, yerr=err, marker="o", capsize=2, color=CLASS_COLORS[c], label=CLASS_LABELS[c])
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(r"Pred $p_T$ [GeV]")
    ax.set_ylabel("Fake rate")
    ax.set_title(f"{GLOWUP} per-class fake rate vs $p_T$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def _plot_class_residuals(pc: dict):
    """GLOW-UP kinematic residuals grouped Charged / Neutral, overlaying the
    classes as normalized (density) step histograms. Rows = Charged / Neutral,
    cols = (dpT/pT, deta, dphi)."""
    cols = [
        ("res_dptrel", DPTREL_BINS, r"$(p_T^{\mathrm{reco}} - p_T^{\mathrm{truth}}) / p_T^{\mathrm{truth}}$"),
        ("res_deta", DANG_BINS, r"$\eta^{\mathrm{reco}} - \eta^{\mathrm{truth}}$"),
        ("res_dphi", DANG_BINS, r"$\phi^{\mathrm{reco}} - \phi^{\mathrm{truth}}$"),
    ]
    groups = [("Charged", [0, 1, 2]), ("Neutral", [3, 4])]   # class indices
    truth_n = pc["res_truth_n"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), squeeze=False)
    for gi, (gname, classes) in enumerate(groups):
        for ci, (key, bins, xlab) in enumerate(cols):
            ax = axes[gi][ci]
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
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
            if ci == 0:
                ax.set_ylabel(f"{gname}\nDensity")
            if ci == 1:
                ax.set_title(gname)
    fig.suptitle(f"{GLOWUP} per-class kinematic residuals (matched objects)", y=1.005)
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
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.suptitle(rf"Jet matching ({GLOWUP} & {PUPPI_LABEL} vs {TARGET}, $\Delta R<0.4$)", y=1.02)
    fig.tight_layout()
    return fig


def render_all(merged: dict) -> dict:
    methods = [(GLOWUP, merged["glow_res"]), (PUPPI_LABEL, merged["puppi_res"])]
    figs = {
        "jet_resolution": _plot_jet_resolution(merged),
        "jet_resolution_binned": _make_binned_plots(methods, edges=JET_RES_PT_EDGES),
        "jet_resolution_iqr_binned": _plot_jet_iqr_binned(methods),
        "jet_residual_boxes": _plot_jet_residual_boxes(methods),
        # Same panels as jet_resolution, each as its own standalone figure.
        "jet_relative_pt": _plot_jet_single(merged, "dpt_over_truth"),
        "jet_delta_eta": _plot_jet_single(merged, "deta"),
        "jet_delta_phi": _plot_jet_single(merged, "dphi"),
        "jet_energy": _plot_jet_single(merged, "energy"),
        "track_f1_vs_pt": _plot_track_f1_from_counts(merged["track"]),
        "n_particles_by_pt_bin": _plot_n_particles_from_counts(merged["npart_truth"], merged["npart_pred"]),
        "feature_scatter": _plot_feature_scatter_from_hist(merged["feat_hist"]),
        "calo_recall_vs_pt": _plot_calo_recall_from_counts(merged["calo"]),
        "calo_pred_vs_truth_e_dist": _plot_calo_e_dist_from_hist(merged["calo"]),
    }
    if merged.get("jet_match") is not None:
        figs["jet_matching_efficiency"] = _plot_jet_matching(merged["jet_match"])
    # Per-class figures only when the (matched) per-class aggregates are present
    # (skipped silently for older caches that predate them — re-run with --force).
    pc = merged.get("per_class")
    if pc is not None:
        figs["class_confusion_matrix"] = _plot_class_confusion(pc)
        figs["class_pt_response"] = _plot_class_pt_response(pc)
        figs["class_pt_resolution"] = _plot_class_pt_resolution(pc)
        figs["class_efficiency_vs_pt"] = _plot_class_efficiency(pc)
        figs["class_f1_vs_pt"] = _plot_class_f1_vs_pt(pc)
        if "fake_pred_n" in pc:
            figs["class_fake_rate_vs_pt"] = _plot_class_fake_rate_vs_pt(pc)
        if "res_truth_n" in pc:
            figs["class_residuals"] = _plot_class_residuals(pc)
    return figs


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
                                   args.per_class): f
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
