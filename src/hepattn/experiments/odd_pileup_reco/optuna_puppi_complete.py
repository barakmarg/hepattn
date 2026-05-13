"""Optuna hyperparameter search for puppi_complete (particle-space PUPPI).

Operates on the per-particle truth list, ~3000 particles/event including
PU — much denser than CMS PFlow space, so the CMS defaults and the
cluster-space tunes both over- or under-suppress. We re-tune for this regime.

Objective: |bias| + IQR + 0.5·|nc_rel| on Δpt/pt_truth + matched-jet nconst
ratio (same definition as ``optuna_puppi_perfect`` v2).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def _quiet_tqdm() -> None:
    import tqdm
    class _NoBar:
        def __init__(self, iterable=None, *a, **k): self.iterable = iterable
        def __iter__(self): return iter(self.iterable) if self.iterable is not None else iter(())
        def __enter__(self): return self
        def __exit__(self, *e): return False
        def update(self, *a, **k): pass
        def close(self): pass
        def set_description(self, *a, **k): pass
    tqdm.tqdm = _NoBar  # type: ignore[assignment]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquet-dir",
                   default="/storage/agrp/barakma/PileupODD/data/ttbar_pu200_all_vertices_chunked")
    p.add_argument("--n-events", type=int, default=100)
    p.add_argument("--n-trials", type=int, default=150)
    p.add_argument("--jet-R", type=float, default=0.7)
    p.add_argument("--min-constituents", type=int, default=3)
    p.add_argument("--min-pt", type=float, default=10.0)
    p.add_argument("--dr-cut", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir",
                   default="/storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco/optuna_puppi_complete")
    p.add_argument("--study-name", default="puppi_complete_v1")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _quiet_tqdm()

    import optuna
    from scipy.stats import iqr as scipy_iqr

    from hepattn.experiments.odd_pileup_reco.puppi_complete import (
        cluster_puppi_complete_jets, cluster_truth_hs_jets,
        compute_puppi_weights_complete, load_truth_particles,
    )
    from hepattn.experiments.odd_pileup_reco.reco_analysis import (
        get_jet_residuals, match_jets,
    )

    print(f"Loading {args.n_events} events ...", flush=True)
    particles = load_truth_particles(args.parquet_dir, event_start=0, event_stop=args.n_events)
    print(f"  {len(particles['pt'])} events, particles/event avg = "
          f"{np.mean([len(x) for x in particles['pt']]):.0f}", flush=True)

    print("Clustering truth-HS jets (one-time) ...", flush=True)
    truth_jets = cluster_truth_hs_jets(
        particles, jet_R=args.jet_R,
        min_constituents=args.min_constituents, min_pt=args.min_pt,
    )
    truth_pt   = truth_jets["truth_jet_pt"]
    truth_eta  = truth_jets["truth_jet_eta"]
    truth_phi  = truth_jets["truth_jet_phi"]
    truth_nc   = truth_jets["truth_jet_nconst"]
    n_truth = np.array([len(x) for x in truth_pt])
    print(f"  Truth jets total = {n_truth.sum()}", flush=True)

    def objective(trial: optuna.Trial) -> float:
        R0                   = trial.suggest_float("R0", 0.15, 0.50)
        dz_cut               = trial.suggest_float("dz_cut", 0.3, 2.0)
        rms_pt_min           = trial.suggest_float("rms_pt_min", 0.05, 2.0, log=True)
        min_neutral_pt       = trial.suggest_float("min_neutral_pt", 0.2, 5.0, log=True)
        min_neutral_pt_slope = trial.suggest_float("min_neutral_pt_slope", 0.0, 0.5)
        min_weight           = trial.suggest_float("min_weight", 0.0, 0.3)
        eta_max_extrap       = trial.suggest_float("eta_max_extrap", 1.5, 3.0)
        apply_lv_adjust      = trial.suggest_categorical("apply_lv_adjust", [True, False])

        weights = compute_puppi_weights_complete(
            particles,
            R0=R0, dz_cut=dz_cut, rms_pt_min=rms_pt_min,
            min_neutral_pt=min_neutral_pt, min_neutral_pt_slope=min_neutral_pt_slope,
            min_weight=min_weight, eta_max_extrap=eta_max_extrap,
            apply_lv_adjust=apply_lv_adjust,
        )
        puppi_jets = cluster_puppi_complete_jets(
            particles, jet_R=args.jet_R,
            min_constituents=args.min_constituents, min_pt=args.min_pt,
            weights=weights,
        )

        n_p = np.array([len(x) for x in puppi_jets["puppi_jet_pt"]])
        mask = (n_p > 0) & (n_truth > 0)
        if not mask.any():
            return 1e6
        tr_ix, pr_ix, _ = match_jets(
            puppi_jets["puppi_jet_pt"][mask],
            puppi_jets["puppi_jet_eta"][mask],
            puppi_jets["puppi_jet_phi"][mask],
            truth_pt[mask], truth_eta[mask], truth_phi[mask],
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

        nc_p_arr = puppi_jets["puppi_jet_nconst"][mask]
        nc_t_arr = truth_nc[mask]
        nc_p, nc_t = [], []
        for i, (ti, pi) in enumerate(zip(tr_ix, pr_ix)):
            if len(ti):
                nc_p.append(nc_p_arr[i][pi]); nc_t.append(nc_t_arr[i][ti])
        if nc_p:
            ncp = np.concatenate(nc_p).astype(float)
            nct = np.concatenate(nc_t).astype(float)
            nc_rel = float(np.mean((ncp - nct) / np.clip(nct, 1, None)))
        else:
            nc_rel = 0.0

        trial.set_user_attr("bias", bias)
        trial.set_user_attr("iqr", spread)
        trial.set_user_attr("nc_rel", nc_rel)
        trial.set_user_attr("n_matched", int(len(d)))
        return abs(bias) + spread + 0.5 * abs(nc_rel)

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=args.study_name, direction="minimize", sampler=sampler,
        storage=f"sqlite:///{out_dir / (args.study_name + '.db')}",
        load_if_exists=True,
    )
    t0 = time.time()
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)
    elapsed = time.time() - t0
    print(f"\nDone: {len(study.trials)} trials in {elapsed:.0f}s ({elapsed/max(1,len(study.trials)):.1f}s/trial)")

    bt = study.best_trial
    print(f"\n=== Best ===")
    print(f"  value (|b|+IQR+0.5|nc_rel|) = {bt.value:.4f}")
    print(f"  bias    = {bt.user_attrs.get('bias'):+.4f}")
    print(f"  IQR     = {bt.user_attrs.get('iqr'):.4f}")
    print(f"  nc_rel  = {bt.user_attrs.get('nc_rel'):+.4f}")
    print(f"  N       = {bt.user_attrs.get('n_matched')}")
    for k, v in bt.params.items():
        print(f"  {k:<22s} = {v}")
    (out_dir / f"{args.study_name}_best.json").write_text(json.dumps({
        "best_value": bt.value, "best_params": bt.params,
        "best_bias": bt.user_attrs.get("bias"),
        "best_iqr": bt.user_attrs.get("iqr"),
        "best_nc_rel": bt.user_attrs.get("nc_rel"),
        "best_n_matched": bt.user_attrs.get("n_matched"),
        "n_trials": len(study.trials), "n_events": args.n_events,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
