"""Optuna hyperparameter search for the perfect-PFlow PUPPI variant.

Optimizes against jet ``dpt_over_truth`` only (NOT jet # constituents):

    objective = |bias| + IQR    (both on (pt_pred - pt_truth) / pt_truth)

The truth jets and the truth/predicted match list are computed once;
each trial only re-runs ``compute_puppi_weights_perfect`` and the
PUPPI jet clustering + matching against truth jets.

Usage::

    python -m hepattn.experiments.odd_pileup_reco.optuna_puppi_perfect \
        --n-events 100 --n-trials 100
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


DEFAULT_H5 = (
    "/storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco/"
    "logs/odd_pflow_reco_20260421-T104425/ckpts/"
    "epoch=099-val_loss=13.99981__test_latest_256_dim.h5"
)


def _quiet_tqdm() -> None:
    """Disable tqdm progress bars from puppi.py / reco_analysis.py during trials."""
    import tqdm
    import functools

    class _NoBar:
        def __init__(self, iterable=None, *args, **kwargs):
            self.iterable = iterable
        def __iter__(self):
            return iter(self.iterable) if self.iterable is not None else iter(())
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            return False
        def update(self, *a, **k):
            pass
        def close(self):
            pass
        def set_description(self, *a, **k):
            pass

    tqdm.tqdm = _NoBar  # type: ignore[assignment]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5", type=str, default=DEFAULT_H5)
    p.add_argument("--n-events", type=int, default=100,
                   help="Number of events to load for the study (kept small so each trial is fast).")
    p.add_argument("--n-trials", type=int, default=100)
    p.add_argument("--jet-R", type=float, default=0.7)
    p.add_argument("--min-constituents", type=int, default=3)
    p.add_argument("--min-pt", type=float, default=10.0)
    p.add_argument("--dr-cut", type=float, default=0.4, help="Truth-match ΔR.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str,
                   default="/storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco/optuna_puppi_perfect")
    p.add_argument("--study-name", type=str, default="puppi_perfect_v2")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _quiet_tqdm()

    import optuna
    from scipy.stats import iqr as scipy_iqr

    from hepattn.experiments.odd_pileup_reco.puppi_perfect import (
        compute_puppi_weights_perfect,
        cluster_puppi_perfect_jets,
    )
    from hepattn.experiments.odd_pileup_reco.reco_analysis import (
        cluster_jets,
        get_jet_residuals,
        load_pflow_data,
        match_jets,
    )

    print(f"Loading {args.n_events} events from {args.h5} ...", flush=True)
    data = load_pflow_data(args.h5, event_start=0, event_stop=args.n_events)
    n_events = data["pflow_class"].shape[0]
    print(f"  loaded {n_events} events", flush=True)

    print("Clustering truth jets (one-time) ...", flush=True)
    jets = cluster_jets(
        data,
        jet_R=args.jet_R,
        min_constituents=args.min_constituents,
        min_pt=args.min_pt,
    )
    truth_pt    = jets["truth_jet_pt"]
    truth_eta   = jets["truth_jet_eta"]
    truth_phi   = jets["truth_jet_phi"]
    truth_nc    = jets["truth_jet_nconst"]
    n_truth = np.array([len(e) for e in truth_pt])
    print(f"  truth jets: total = {sum(n_truth)}, events with ≥1 = {(n_truth > 0).sum()}", flush=True)

    def objective(trial: optuna.Trial) -> float:
        # ---- search space ----
        R0                   = trial.suggest_float("R0", 0.10, 0.50)
        rms_pt_min           = trial.suggest_float("rms_pt_min", 0.05, 2.0, log=True)
        min_neutral_pt       = trial.suggest_float("min_neutral_pt", 0.5, 20.0, log=True)
        min_neutral_pt_slope = trial.suggest_float("min_neutral_pt_slope", 0.0, 2.0)
        min_weight           = trial.suggest_float("min_weight", 0.0, 0.5)
        eta_max_extrap       = trial.suggest_float("eta_max_extrap", 1.5, 3.0)
        apply_lv_adjust      = trial.suggest_categorical("apply_lv_adjust", [True, False])

        weights = compute_puppi_weights_perfect(
            data,
            R0=R0,
            rms_pt_min=rms_pt_min,
            min_neutral_pt=min_neutral_pt,
            min_neutral_pt_slope=min_neutral_pt_slope,
            min_weight=min_weight,
            eta_max_extrap=eta_max_extrap,
            apply_lv_adjust=apply_lv_adjust,
        )
        if weights is None:
            raise RuntimeError("compute_puppi_weights_perfect returned None — required fields missing")

        puppi_jets = cluster_puppi_perfect_jets(
            data,
            jet_R=args.jet_R,
            min_constituents=args.min_constituents,
            min_pt=args.min_pt,
            weights=weights,
        )
        if puppi_jets is None:
            return 1e6

        n_puppi = np.array([len(e) for e in puppi_jets["puppi_jet_pt"]])
        mask = (n_puppi > 0) & (n_truth > 0)
        if not mask.any():
            return 1e6

        tr_ix, pr_ix, _ = match_jets(
            puppi_jets["puppi_jet_pt"][mask],
            puppi_jets["puppi_jet_eta"][mask],
            puppi_jets["puppi_jet_phi"][mask],
            truth_pt[mask],
            truth_eta[mask],
            truth_phi[mask],
            dr_cut=args.dr_cut,
        )
        res = get_jet_residuals(
            tr_ix, pr_ix,
            truth_pt[mask], truth_eta[mask], truth_phi[mask],
            puppi_jets["puppi_jet_pt"][mask],
            puppi_jets["puppi_jet_eta"][mask],
            puppi_jets["puppi_jet_phi"][mask],
        )
        d = res["dpt_over_truth"]
        if len(d) < 5:
            return 1e6

        bias = float(np.nanmedian(d))
        spread = float(scipy_iqr(d))
        n_matched = int(len(d))

        # Constituent-count term: mean relative error on nconst for matched jets.
        # Compute per-event over matched pairs then pool.
        nc_puppi_vals, nc_truth_vals = [], []
        puppi_nc_arr = puppi_jets["puppi_jet_nconst"][mask]
        truth_nc_arr = truth_nc[mask]
        for i, (ti, pi) in enumerate(zip(tr_ix, pr_ix)):
            if len(ti) == 0:
                continue
            nc_puppi_vals.append(puppi_nc_arr[i][pi])
            nc_truth_vals.append(truth_nc_arr[i][ti])
        if nc_puppi_vals:
            nc_p = np.concatenate(nc_puppi_vals).astype(float)
            nc_t = np.concatenate(nc_truth_vals).astype(float)
            # Mean relative excess: positive = too many constituents.
            nc_rel_bias = float(np.mean((nc_p - nc_t) / np.clip(nc_t, 1, None)))
        else:
            nc_rel_bias = 0.0

        trial.set_user_attr("bias", bias)
        trial.set_user_attr("iqr", spread)
        trial.set_user_attr("nc_rel_bias", nc_rel_bias)
        trial.set_user_attr("n_matched_jets", n_matched)

        # Combined objective: kinematic (|bias|+IQR) + 0.5 * |nc relative bias|.
        # The 0.5 weight treats 100% nconst excess as equivalent to IQR=0.5 — a
        # meaningful but not dominating penalty so kinematics still matter.
        return abs(bias) + spread + 0.5 * abs(nc_rel_bias)

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=sampler,
        storage=f"sqlite:///{out_dir / (args.study_name + '.db')}",
        load_if_exists=True,
    )

    t0 = time.time()
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)
    elapsed = time.time() - t0
    print(f"\nDone: {len(study.trials)} trials in {elapsed:.1f}s "
          f"({elapsed / max(1, len(study.trials)):.2f}s/trial)", flush=True)

    print("\n=== Best trial ===")
    bt = study.best_trial
    print(f"  value (|bias|+IQR+0.5|nc_rel|) = {bt.value:.4f}")
    print(f"  bias              = {bt.user_attrs.get('bias'):+.4f}")
    print(f"  IQR               = {bt.user_attrs.get('iqr'):.4f}")
    print(f"  nc_rel_bias       = {bt.user_attrs.get('nc_rel_bias'):+.4f}")
    print(f"  n_matched_jets    = {bt.user_attrs.get('n_matched_jets')}")
    print(f"  params:")
    for k, v in bt.params.items():
        print(f"    {k:<22s} = {v}")

    summary = {
        "best_value": bt.value,
        "best_bias": bt.user_attrs.get("bias"),
        "best_iqr": bt.user_attrs.get("iqr"),
        "best_nc_rel_bias": bt.user_attrs.get("nc_rel_bias"),
        "best_n_matched_jets": bt.user_attrs.get("n_matched_jets"),
        "best_params": bt.params,
        "n_trials": len(study.trials),
        "n_events": n_events,
        "h5": args.h5,
    }
    (out_dir / f"{args.study_name}_best.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_dir / (args.study_name + '_best.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
