"""Verify v2 (nconst-aware) optuna configs on held-out events."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/storage/agrp/barakma/hepattn/src")

from hepattn.experiments.odd_pileup_reco.puppi_perfect import (
    compute_puppi_weights_perfect, cluster_puppi_perfect_jets,
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


def evaluate(data, n_truth, truth_pt, truth_eta, truth_phi, truth_nc, label, **puppi_params):
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

    nc_p_arr = jets["puppi_jet_nconst"][mask]
    nc_t_arr = truth_nc[mask]
    nc_vals_p, nc_vals_t = [], []
    for i, (ti, pi) in enumerate(zip(tr_ix, pr_ix)):
        if len(ti):
            nc_vals_p.append(nc_p_arr[i][pi])
            nc_vals_t.append(nc_t_arr[i][ti])
    if nc_vals_p:
        nc_p = np.concatenate(nc_vals_p).astype(float)
        nc_t = np.concatenate(nc_vals_t).astype(float)
        nc_rel = float(np.mean((nc_p - nc_t) / np.clip(nc_t, 1, None)))
    else:
        nc_rel = float("nan")

    obj = abs(bias) + spread + 0.5 * abs(nc_rel)
    print(f"  {label:<40s}  bias={bias:+.3f}  IQR={spread:.3f}  nc_rel={nc_rel:+.3f}  obj={obj:.3f}  N={len(d)}")
    return obj, bias, spread, nc_rel


def main():
    data = load_pflow_data(DEFAULT_H5, event_start=100, event_stop=2000)
    print(f"Held-out: {data['pflow_class'].shape[0]} events (100-2000)")
    jets = cluster_jets(data, jet_R=0.7, min_constituents=3, min_pt=10.0)
    truth_pt   = jets["truth_jet_pt"]
    truth_eta  = jets["truth_jet_eta"]
    truth_phi  = jets["truth_jet_phi"]
    truth_nc   = jets["truth_jet_nconst"]
    n_truth = np.array([len(e) for e in truth_pt])
    print(f"  Truth jets = {sum(n_truth)}")

    print("\nV1 defaults (mnp=7.0, R0=0.2, apply_lv_adjust=True):")
    evaluate(data, n_truth, truth_pt, truth_eta, truth_phi, truth_nc,
             "v1-defaults (mnp=7.0)", min_neutral_pt=7.0,
             R0=0.2, rms_pt_min=0.1, min_neutral_pt_slope=0.0,
             min_weight=0.01, eta_max_extrap=2.0, apply_lv_adjust=True)

    print("\nV1 optuna-best (trial#133):")
    evaluate(data, n_truth, truth_pt, truth_eta, truth_phi, truth_nc,
             "v1-t133",
             R0=0.154, rms_pt_min=0.976, min_neutral_pt=1.04,
             min_neutral_pt_slope=0.465, min_weight=0.055,
             eta_max_extrap=2.0, apply_lv_adjust=False)

    print("\nV2 top-20 re-evaluated:")
    import optuna
    db = Path("/storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco/optuna_puppi_perfect/puppi_perfect_v2.db")
    study = optuna.load_study(study_name="puppi_perfect_v2", storage=f"sqlite:///{db}")
    top20 = sorted(
        [t for t in study.trials if t.value is not None and t.value < 1e5],
        key=lambda t: t.value,
    )[:20]

    rows = []
    for t in top20:
        obj, bias, spread, nc_rel = evaluate(
            data, n_truth, truth_pt, truth_eta, truth_phi, truth_nc,
            f"v2-t{t.number}(s={t.value:.3f})",
            **t.params,
        )
        rows.append((t.number, t.value, obj, bias, spread, nc_rel, t.params))

    rows.sort(key=lambda r: r[2])
    print(f"\nBest on held-out: trial#{rows[0][0]}  search={rows[0][1]:.4f}  held-out-obj={rows[0][2]:.4f}")
    for k, v in rows[0][6].items():
        print(f"  {k:<22s} = {v}")


if __name__ == "__main__":
    main()
