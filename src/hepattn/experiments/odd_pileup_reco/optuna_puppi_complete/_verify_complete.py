"""Verify top-N puppi_complete optuna trials on held-out events."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "/storage/agrp/barakma/hepattn/src")
from hepattn.experiments.odd_pileup_reco.puppi_complete import (
    cluster_puppi_complete_jets, cluster_truth_hs_jets,
    compute_puppi_weights_complete, load_truth_particles,
)
from hepattn.experiments.odd_pileup_reco.reco_analysis import (
    get_jet_residuals, match_jets,
)
from scipy.stats import iqr as scipy_iqr


def evaluate(particles, n_truth, truth_pt, truth_eta, truth_phi, truth_nc, label, **puppi_params):
    weights = compute_puppi_weights_complete(particles, **puppi_params)
    jets = cluster_puppi_complete_jets(particles, jet_R=0.7, min_constituents=3, min_pt=10.0, weights=weights)
    n_p = np.array([len(x) for x in jets["puppi_jet_pt"]])
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
    nc_p_a = jets["puppi_jet_nconst"][mask]; nc_t_a = truth_nc[mask]
    nc_p, nc_t = [], []
    for i, (ti, pi) in enumerate(zip(tr_ix, pr_ix)):
        if len(ti):
            nc_p.append(nc_p_a[i][pi]); nc_t.append(nc_t_a[i][ti])
    if nc_p:
        ncp = np.concatenate(nc_p).astype(float); nct = np.concatenate(nc_t).astype(float)
        nc_rel = float(np.mean((ncp - nct) / np.clip(nct, 1, None)))
    else:
        nc_rel = float("nan")
    obj = abs(bias) + spread + 0.5 * abs(nc_rel)
    print(f"  {label:<40s} bias={bias:+.3f} IQR={spread:.3f} nc_rel={nc_rel:+.3f} obj={obj:.3f} N={len(d)}")
    return obj


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--event-start", type=int, default=100)
    p.add_argument("--event-stop", type=int, default=600)
    p.add_argument("--top-n", type=int, default=15)
    args = p.parse_args()

    particles = load_truth_particles(
        "/storage/agrp/barakma/PileupODD/data/ttbar_pu200_all_vertices_chunked",
        event_start=args.event_start, event_stop=args.event_stop,
    )
    print(f"Held-out: {len(particles['pt'])} events ({args.event_start}–{args.event_stop})")
    truth_jets = cluster_truth_hs_jets(particles, jet_R=0.7, min_constituents=3, min_pt=10.0)
    truth_pt   = truth_jets["truth_jet_pt"]
    truth_eta  = truth_jets["truth_jet_eta"]
    truth_phi  = truth_jets["truth_jet_phi"]
    truth_nc   = truth_jets["truth_jet_nconst"]
    n_truth = np.array([len(x) for x in truth_pt])
    print(f"  Truth jets = {n_truth.sum()}\n")

    import optuna
    db = Path("/storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco/optuna_puppi_complete/puppi_complete_v1.db")
    study = optuna.load_study(study_name="puppi_complete_v1", storage=f"sqlite:///{db}")
    top = sorted([t for t in study.trials if t.value is not None and t.value < 1e5], key=lambda t: t.value)[:args.top_n]

    rows = []
    for t in top:
        obj = evaluate(particles, n_truth, truth_pt, truth_eta, truth_phi, truth_nc,
                       f"trial#{t.number}(search={t.value:.3f})", **t.params)
        rows.append((t.number, t.value, obj, t.params))

    rows.sort(key=lambda r: r[2])
    print(f"\nMost robust: trial#{rows[0][0]}  search={rows[0][1]:.4f}  held-out-obj={rows[0][2]:.4f}")
    for k, v in rows[0][3].items():
        print(f"  {k:<22s} = {v}")


if __name__ == "__main__":
    main()
