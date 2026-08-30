"""Jet-resolution comparison plot using PUPPI **charged-subtract** as the
PUPPI overlay (replaces ``run_puppi_jet_resolution_perfect.py``).

The other panels (pflow ML, calo, calo-HS, truth) come from the H5 file
exactly as in the perfect-PFlow variant. PUPPI jets are computed from the
parquet ``ttbar_pu200_all_vertices_chunked/`` dataset, with the parquet
events **explicitly aligned to the H5 event_numbers** so per-event
matching against the H5 truth jets is consistent.

This is the strongest PUPPI baseline we have: cluster-space inputs with
truth-aware ``tracks_mask`` AND per-cluster subtraction of BOTH HS-charged
AND PU-charged calo deposits. See ``puppi_charged_subtract.py``.

Jet algorithm
-------------
The ``--jet-algorithm`` flag switches between ``antikt`` (default, LHC
standard) and ``kt``. Empirically, anti-kT yields slightly better numbers
across **all** methods on this dataset (cleaner tails, less PU-radiation
distortion of jet footprints). Critically, the **relative ordering of
methods is preserved** under both algorithms — the ML PFlow model still
outperforms PUPPI under kT, anti-kT, and at all jet-pT bins. The PUPPI
weights themselves are jet-algorithm-agnostic (computed per-particle
before clustering), so the optuna-tuned params in ``--best-json`` transfer
without retuning.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt

from hepattn.experiments.odd_pileup_reco.puppi_charged_subtract import (
    cluster_puppi_charged_subtract_jets,
    compute_puppi_weights_charged_subtract,
    load_charged_subtract_events,
)
from hepattn.experiments.odd_pileup_reco.reco_analysis import (
    cluster_calo_hs_jets,
    cluster_jets,
    get_jet_residuals,
    load_pflow_data,
    match_jets,
    plot_jet_resolution_with_calo,
)
#/storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco/logs/odd_pflow_reco_20260421-T104425/ckpts/epoch=099-val_loss=13.99981__test_dihiggs.h5

DEFAULT_H5 = (
    "/storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco/"
    "logs/odd_pflow_reco_20260421-T104425/ckpts/"
    #"epoch=099-val_loss=13.99981__test_dihiggs.h5"
    "epoch=099-val_loss=13.99981__test_latest_256_dim.h5"

)
DEFAULT_PARQUET_DIR = (
    "/storage/agrp/barakma/PileupODD/data/ttbar_pu200_all_vertices_chunked"
)
DEFAULT_BEST_JSON = (
    "/storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco/"
    "optuna_puppi_charged_subtract/puppi_charged_subtract_v1_best.json"
)
DEFAULT_OUT = (
    "/storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco/"
    "puppi_jet_res_2000_ttbar_veriftagain_charged_subtract_full.png"
)
DEFAULT_BINNED_OUT = (
    "/storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco/"
    "puppi_jet_res_2000_ttbar_veriftagain_charged_subtract_full_binned.png"
)

PT_BIN_EDGES = (10.0, 20.0, 50.0, 90.0, 150.0, 300.0, 500.0, float("inf"))


def _match_and_residuals(
    pred_jets: dict,
    truth_jets_root: dict,
    prefix: str,
    dr_cut: float = 0.4,
) -> dict[str, np.ndarray]:
    """Match `pred_jets[prefix]_jet_*` to truth jets in `truth_jets_root` and
    return per-matched-jet residuals (includes `truth_pt` for binning).
    Returns empty arrays if `pred_jets` is None or no matches.
    """
    empty = {k: np.array([]) for k in ("dpt_over_truth", "deta", "dphi", "truth_pt")}
    if pred_jets is None:
        return empty
    pkey = f"{prefix}_jet"
    n_pred = np.array([len(e) for e in pred_jets[f"{pkey}_pt"]])
    n_truth = np.array([len(e) for e in truth_jets_root["truth_jet_pt"]])
    mask = (n_pred > 0) & (n_truth > 0)
    if not mask.any():
        return empty
    tr_ix, pr_ix, _ = match_jets(
        pred_jets[f"{pkey}_pt"][mask], pred_jets[f"{pkey}_eta"][mask], pred_jets[f"{pkey}_phi"][mask],
        truth_jets_root["truth_jet_pt"][mask], truth_jets_root["truth_jet_eta"][mask], truth_jets_root["truth_jet_phi"][mask],
        dr_cut=dr_cut,
    )
    res = get_jet_residuals(
        tr_ix, pr_ix,
        truth_jets_root["truth_jet_pt"][mask], truth_jets_root["truth_jet_eta"][mask], truth_jets_root["truth_jet_phi"][mask],
        pred_jets[f"{pkey}_pt"][mask], pred_jets[f"{pkey}_eta"][mask], pred_jets[f"{pkey}_phi"][mask],
    )
    return res


def _binned_mean_iqr(values: np.ndarray, truth_pt: np.ndarray,
                     edges=PT_BIN_EDGES) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each pT bin, return (mean, iqr, n)."""
    n_bins = len(edges) - 1
    means = np.full(n_bins, np.nan)
    iqrs = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    if len(values) == 0:
        return means, iqrs, counts
    finite = np.isfinite(values) & np.isfinite(truth_pt)
    v = values[finite]
    t = truth_pt[finite]
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        sel = (t >= lo) & (t < hi)
        if sel.sum() < 2:
            counts[i] = int(sel.sum())
            if sel.sum() == 1:
                means[i] = float(v[sel][0])
            continue
        chunk = v[sel]
        means[i] = float(np.mean(chunk))
        q1, q3 = np.percentile(chunk, [25, 75])
        iqrs[i] = float(q3 - q1)
        counts[i] = int(sel.sum())
    return means, iqrs, counts


def _make_binned_plots(
    methods: list[tuple[str, dict[str, np.ndarray]]],
    edges=PT_BIN_EDGES,
) -> plt.Figure:
    """methods: list of (label, residuals_dict).
    Produces a 2x3 figure: rows=(mean, IQR), cols=(rel pT, eta, phi).
    """
    centers = np.arange(len(edges) - 1)
    labels_x = [
        f"[{int(edges[i])},{('∞' if np.isinf(edges[i+1]) else int(edges[i+1]))})"
        for i in range(len(edges) - 1)
    ]
    residual_keys = [
        ("dpt_over_truth", r"$\Delta p_T / p_T^{\mathrm{truth}}$"),
        ("deta",           r"$\Delta\eta$"),
        ("dphi",           r"$\Delta\phi$"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    for col, (key, latex) in enumerate(residual_keys):
        ax_mean = axes[0, col]
        ax_iqr = axes[1, col]
        for label, res in methods:
            if len(res.get(key, [])) == 0:
                continue
            means, iqrs, counts = _binned_mean_iqr(res[key], res["truth_pt"], edges=edges)
            ax_mean.plot(centers, means, marker="o", label=label)
            ax_iqr.plot(centers, iqrs, marker="o", label=label)
        ax_mean.axhline(0.0, color="gray", linewidth=0.8, linestyle=":")
        ax_mean.set_title(f"mean {latex} vs truth $p_T$")
        ax_iqr.set_title(f"IQR {latex} vs truth $p_T$")
        ax_mean.grid(alpha=0.3)
        ax_iqr.grid(alpha=0.3)
        ax_iqr.set_xlabel(r"truth jet $p_T$ bin [GeV]")
        if col == 0:
            ax_mean.set_ylabel("mean of residual")
            ax_iqr.set_ylabel("IQR of residual")

    for ax in axes[-1, :]:
        ax.set_xticks(centers)
        ax.set_xticklabels(labels_x, rotation=30, ha="right")

    axes[0, 0].legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def _h5_event_numbers(h5_path: str) -> list[int]:
    """Read the per-event ``event_number`` array from the H5 file."""
    with h5py.File(h5_path, "r") as f:
        ev = f["events"][:]
        # ``events`` is a structured array with a single field ``event_number``.
        return [int(x[0]) for x in ev]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5", type=str, default=DEFAULT_H5)
    p.add_argument("--parquet-dir", type=str, default=DEFAULT_PARQUET_DIR)
    p.add_argument("--best-json", type=str, default=DEFAULT_BEST_JSON,
                   help="JSON file with optuna-tuned PUPPI params.")
    p.add_argument("--out", type=str, default=DEFAULT_OUT)
    p.add_argument("--binned-out", type=str, default=DEFAULT_BINNED_OUT,
                   help="Path for the per-pT-bin (mean & IQR) plots.")
    p.add_argument("--event-start", type=int, default=None)
    p.add_argument("--event-stop", type=int, default=None)
    p.add_argument("--jet-R", type=float, default=0.7)
    p.add_argument("--min-constituents", type=int, default=3)
    p.add_argument("--min-pt", type=float, default=10.0)
    p.add_argument("--no-subtract-pu", action="store_true", default=False,
                   help="Sanity-check: subtract only HS-charged (= old puppi_perfect).")
    p.add_argument("--jet-algorithm", type=str, default="antikt", choices=["antikt", "kt"],
                   help="FastJet clustering algorithm. anti-kT (default) is the LHC "
                        "standard and gives slightly better numbers across all methods; "
                        "the relative ML > PUPPI ordering holds under either choice.")
    p.add_argument("--dpi", type=int, default=120)
    args = p.parse_args()

    # ---- 1. Load H5 (pflow ML, calo, truth) ----
    data = load_pflow_data(args.h5, event_start=args.event_start, event_stop=args.event_stop)
    n_events_h5 = data["pflow_class"].shape[0]
    print(f"Loaded {Path(args.h5).name}: n_events_h5 = {n_events_h5}", flush=True)

    jets = cluster_jets(
        data,
        jet_R=args.jet_R,
        min_constituents=args.min_constituents,
        min_pt=args.min_pt,
        jet_algorithm=args.jet_algorithm,
    )

    # ---- 2. Load parquet events ALIGNED to H5 event_numbers ----
    h5_eids = _h5_event_numbers(args.h5)
    if args.event_start is not None or args.event_stop is not None:
        s = args.event_start or 0
        e = args.event_stop if args.event_stop is not None else len(h5_eids)
        h5_eids = h5_eids[s:e]
    if len(h5_eids) != n_events_h5:
        print(f"  event_number count {len(h5_eids)} != loaded n_events {n_events_h5} — slicing to match",
              file=sys.stderr)
        h5_eids = h5_eids[:n_events_h5]
    print(f"Loading {len(h5_eids)} aligned parquet events ...", flush=True)
    events = load_charged_subtract_events(args.parquet_dir, event_ids=h5_eids)
    # Sanity-check alignment
    if list(events["event_id"]) != list(h5_eids):
        raise RuntimeError("parquet event order does not match H5 event_numbers")

    # ---- 3. Compute PUPPI weights and jets ----
    bp = {}
    if args.best_json and Path(args.best_json).is_file():
        bp = json.loads(Path(args.best_json).read_text()).get("best_params", {})
        print(f"  Loaded best params: {bp}", flush=True)
    subtract_pu = not args.no_subtract_pu
    print(f"Computing PUPPI weights (subtract_pu_charged={subtract_pu}) ...", flush=True)
    weights = compute_puppi_weights_charged_subtract(events, subtract_pu_charged=subtract_pu, **bp)
    puppi_jets = cluster_puppi_charged_subtract_jets(
        events,
        jet_R=args.jet_R,
        min_constituents=args.min_constituents,
        min_pt=args.min_pt,
        weights=weights,
        subtract_pu_charged=subtract_pu,
        jet_algorithm=args.jet_algorithm,
    )

    # ---- 4. Plot — use the existing shared layout from reco_analysis.py ----
    fig = plot_jet_resolution_with_calo(
        jets, data,
        jet_R=args.jet_R,
        min_constituents=args.min_constituents,
        min_pt=args.min_pt,
        puppi_jets=puppi_jets,
        puppi_label="PUPPI (charged subtract)",
        show_calo=False,
        jet_algorithm=args.jet_algorithm,
    )
    if fig is None:
        print("plot_jet_resolution_with_calo returned None (raw node fields missing).", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved {out}")

    # ---- 5. Binned (mean & IQR vs truth-pT) plots ----
    print("Building per-pT-bin (mean & IQR) plots ...", flush=True)
    calo_hs_jets = cluster_calo_hs_jets(
        data, jet_R=args.jet_R, min_constituents=args.min_constituents, min_pt=args.min_pt,
        jet_algorithm=args.jet_algorithm,
    )

    methods = [
        ("PFlow ML",              _match_and_residuals(jets,         jets, "pflow")),
        ("Calo-HS",               _match_and_residuals(calo_hs_jets, jets, "calo_hs")),
        ("PUPPI (charged sub)",   _match_and_residuals(puppi_jets,   jets, "puppi")),
    ]
    binned_fig = _make_binned_plots(methods)
    binned_out = Path(args.binned_out)
    binned_out.parent.mkdir(parents=True, exist_ok=True)
    binned_fig.savefig(binned_out, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved {binned_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
