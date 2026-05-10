"""Verify a chosen puppi-perfect config on held-out events.

Usage:
    python _verify.py --event-start 100 --event-stop 600
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/storage/agrp/barakma/hepattn/src")

from hepattn.experiments.odd_pileup_reco.puppi_perfect import (
    compute_puppi_weights_perfect,
    cluster_puppi_perfect_jets,
)
from hepattn.experiments.odd_pileup_reco.reco_analysis import (
    cluster_jets, get_jet_residuals, load_pflow_data, match_jets,
)
from scipy.stats import iqr as scipy_iqr


DEFAULT_H5 = (
    "/storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco/"
    "logs/odd_pflow_reco_20260421-T104425/ckpts/"
    "epoch=099-val_loss=13.99981__test_latest_256_dim.h5"
)


def evaluate(data, n_truth, truth_pt, truth_eta, truth_phi, label, **puppi_params):
    weights = compute_puppi_weights_perfect(data, **puppi_params)
    jets = cluster_puppi_perfect_jets(data, jet_R=0.7, min_constituents=3, min_pt=10.0, weights=weights)
    n_p = np.array([len(e) for e in jets["puppi_jet_pt"]])
    mask = (n_p > 0) & (n_truth > 0)
    tr_ix, pr_ix, _ = match_jets(
        jets["puppi_jet_pt"][mask], jets["puppi_jet_eta"][mask], jets["puppi_jet_phi"][mask],
        truth_pt[mask], truth_eta[mask], truth_phi[mask], dr_cut=0.4,
    )
    res = get_jet_residuals(
        tr_ix, pr_ix,
        truth_pt[mask], truth_eta[mask], truth_phi[mask],
        jets["puppi_jet_pt"][mask], jets["puppi_jet_eta"][mask], jets["puppi_jet_phi"][mask],
    )
    d = res["dpt_over_truth"]
    bias = float(np.nanmedian(d))
    spread = float(scipy_iqr(d))
    print(f"  {label:<35s} bias={bias:+.4f}  IQR={spread:.4f}  |b|+IQR={abs(bias) + spread:.4f}  N={len(d)}")
    return abs(bias) + spread


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", default=DEFAULT_H5)
    p.add_argument("--event-start", type=int, default=100)
    p.add_argument("--event-stop", type=int, default=2000)
    p.add_argument("--top-n", type=int, default=20, help="Re-evaluate top-N optuna trials.")
    args = p.parse_args()

    data = load_pflow_data(args.h5, event_start=args.event_start, event_stop=args.event_stop)
    print(f"Held-out: {data['pflow_class'].shape[0]} events from {args.event_start} to {args.event_stop}")

    jets = cluster_jets(data, jet_R=0.7, min_constituents=3, min_pt=10.0)
    truth_pt, truth_eta, truth_phi = jets["truth_jet_pt"], jets["truth_jet_eta"], jets["truth_jet_phi"]
    n_truth = np.array([len(e) for e in truth_pt])
    print(f"  Truth jets total = {sum(n_truth)}")

    print("\nDefaults (current puppi_perfect.py, mnp=7.0):")
    evaluate(data, n_truth, truth_pt, truth_eta, truth_phi, "default (mnp=7.0)", min_neutral_pt=7.0)

    print(f"\nTop-{args.top_n} optuna trials re-evaluated on held-out:")
    import optuna
    db_path = Path("/storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco/optuna_puppi_perfect/puppi_perfect_v1.db")
    study = optuna.load_study(study_name="puppi_perfect_v1", storage=f"sqlite:///{db_path}")
    sorted_trials = sorted(
        [t for t in study.trials if t.value is not None and t.value < 1e5],
        key=lambda t: t.value,
    )
    rows = []
    for t in sorted_trials[: args.top_n]:
        v_search = t.value
        v_holdout = evaluate(
            data, n_truth, truth_pt, truth_eta, truth_phi,
            f"trial#{t.number} (search={v_search:.4f})",
            **t.params,
        )
        rows.append((t.number, v_search, v_holdout, t.params))

    rows.sort(key=lambda r: r[2])
    print("\nMost robust (best held-out among top-N):")
    n0, vs, vh, params = rows[0]
    print(f"  trial#{n0}  search={vs:.4f}  held-out={vh:.4f}")
    for k, v in params.items():
        print(f"    {k:<22s} = {v}")


if __name__ == "__main__":
    main()
