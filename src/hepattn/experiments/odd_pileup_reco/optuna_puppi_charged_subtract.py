"""Optuna hyperparameter search for puppi_charged_subtract.

Cluster-space PUPPI with full charged (HS+PU) calo subtraction. The neutral
residual is much smaller than in puppi_perfect (which subtracts only HS-
charged), so the min_neutral_pt thresholds want to be lower and the α-shape
parameters re-tuned.

Two-stage selection:

1. **Search**: TPE optimiser runs ``--n-trials`` trials on ``--n-events``
   events. Objective = ``|bias| + IQR + 0.5·|nc_rel|`` on Δpt/pt_truth.
2. **Validation**: top ``--top-k`` trials are re-evaluated on
   ``--n-validation-events`` *different* events (held-out, taken from
   ``event_start = n_events``). The trial with the best **validation**
   value is chosen as the final best. This rejects trials that got lucky
   on the small search set.

The saved ``*_best.json`` contains the validation-selected params plus the
full top-K validation table for transparency.
"""
from __future__ import annotations

import argparse
import json
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


def _cluster_truth(events, args):
    from hepattn.experiments.odd_pileup_reco.puppi_charged_subtract import (
        cluster_truth_hs_jets_from_events,
    )
    tj = cluster_truth_hs_jets_from_events(
        events, jet_R=args.jet_R,
        min_constituents=args.min_constituents, min_pt=args.min_pt,
        jet_algorithm=args.jet_algorithm,
    )
    return {
        "pt": tj["truth_jet_pt"],
        "eta": tj["truth_jet_eta"],
        "phi": tj["truth_jet_phi"],
        "nc": tj["truth_jet_nconst"],
        "n": np.array([len(x) for x in tj["truth_jet_pt"]]),
    }


def _evaluate_params(events, truth, params, *, subtract_pu, args) -> dict:
    """Compute (value, bias, iqr, nc_rel, n_matched) for one parameter set
    on the provided events + already-clustered truth-HS jets.
    """
    from scipy.stats import iqr as scipy_iqr

    from hepattn.experiments.odd_pileup_reco.puppi_charged_subtract import (
        cluster_puppi_charged_subtract_jets, compute_puppi_weights_charged_subtract,
    )
    from hepattn.experiments.odd_pileup_reco.reco_analysis import (
        get_jet_residuals, match_jets,
    )

    weights = compute_puppi_weights_charged_subtract(
        events, subtract_pu_charged=subtract_pu, **params,
    )
    puppi_jets = cluster_puppi_charged_subtract_jets(
        events, jet_R=args.jet_R,
        min_constituents=args.min_constituents, min_pt=args.min_pt,
        weights=weights, subtract_pu_charged=subtract_pu,
        jet_algorithm=args.jet_algorithm,
    )

    n_p = np.array([len(x) for x in puppi_jets["puppi_jet_pt"]])
    mask = (n_p > 0) & (truth["n"] > 0)
    if not mask.any():
        return {"value": 1e6, "bias": float("nan"), "iqr": float("nan"),
                "nc_rel": float("nan"), "n_matched": 0}
    tr_ix, pr_ix, _ = match_jets(
        puppi_jets["puppi_jet_pt"][mask],
        puppi_jets["puppi_jet_eta"][mask],
        puppi_jets["puppi_jet_phi"][mask],
        truth["pt"][mask], truth["eta"][mask], truth["phi"][mask],
        dr_cut=args.dr_cut,
    )
    res = get_jet_residuals(
        tr_ix, pr_ix,
        truth["pt"][mask], truth["eta"][mask], truth["phi"][mask],
        puppi_jets["puppi_jet_pt"][mask],
        puppi_jets["puppi_jet_eta"][mask],
        puppi_jets["puppi_jet_phi"][mask],
    )
    d = res["dpt_over_truth"]
    if len(d) < 5:
        return {"value": 1e6, "bias": float("nan"), "iqr": float("nan"),
                "nc_rel": float("nan"), "n_matched": int(len(d))}
    bias = float(np.nanmedian(d))
    spread = float(scipy_iqr(d))

    nc_p_arr = puppi_jets["puppi_jet_nconst"][mask]
    nc_t_arr = truth["nc"][mask]
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

    return {
        "value": abs(bias) + spread + 0.5 * abs(nc_rel),
        "bias": bias, "iqr": spread, "nc_rel": nc_rel,
        "n_matched": int(len(d)),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquet-dir",
                   default="/storage/agrp/barakma/PileupODD/data/ttbar_pu200_all_vertices_chunked")
    p.add_argument("--n-events", type=int, default=100,
                   help="Events used for the TPE search.")
    p.add_argument("--n-validation-events", type=int, default=1000,
                   help="Events used to re-evaluate the top-K trials "
                        "(disjoint from the search set).")
    p.add_argument("--top-k", type=int, default=10,
                   help="How many of the best search trials to re-evaluate "
                        "on the validation set before picking the final best.")
    p.add_argument("--n-trials", type=int, default=200)
    p.add_argument("--jet-R", type=float, default=0.7)
    p.add_argument("--min-constituents", type=int, default=3)
    p.add_argument("--min-pt", type=float, default=10.0)
    p.add_argument("--dr-cut", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir",
                   default="/storage/agrp/barakma/hepattn/src/hepattn/experiments/"
                           "odd_pileup_reco/optuna_puppi_charged_subtract")
    p.add_argument("--study-name", default="puppi_charged_subtract_v1")
    p.add_argument("--no-subtract-pu", action="store_true", default=False,
                   help="If set, subtract only HS-charged (= puppi_perfect mode) for sanity check.")
    p.add_argument("--jet-algorithm", type=str, default="antikt", choices=["antikt", "kt"],
                   help="FastJet clustering algorithm (default anti-kT, LHC standard).")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _quiet_tqdm()

    import optuna

    from hepattn.experiments.odd_pileup_reco.puppi_charged_subtract import (
        load_charged_subtract_events,
    )

    print(f"Loading {args.n_events} search events ...", flush=True)
    events = load_charged_subtract_events(
        args.parquet_dir, event_start=0, event_stop=args.n_events,
    )
    print(f"  {len(events['node_pt'])} events, "
          f"tracks/ev mean = {np.mean([t.sum() for t in events['node_is_track']]):.0f}, "
          f"clusters/ev mean = "
          f"{np.mean([len(t) - t.sum() for t in events['node_is_track']]):.0f}",
          flush=True)

    print(f"Clustering truth-HS jets (one-time, {args.jet_algorithm} R={args.jet_R}) ...",
          flush=True)
    truth = _cluster_truth(events, args)
    print(f"  Truth jets total = {truth['n'].sum()}", flush=True)

    subtract_pu = not args.no_subtract_pu

    def objective(trial: optuna.Trial) -> float:
        params = {
            "R0":                   trial.suggest_float("R0", 0.10, 0.40),
            "rms_pt_min":           trial.suggest_float("rms_pt_min", 0.02, 1.0, log=True),
            # Lower-bound MinNeutralPt cuts: full subtraction leaves only soft
            # neutral residual, so a few-GeV floor is plenty.
            "min_neutral_pt":       trial.suggest_float("min_neutral_pt", 0.05, 3.0, log=True),
            "min_neutral_pt_slope": trial.suggest_float("min_neutral_pt_slope", 0.0, 0.5),
            "min_weight":           trial.suggest_float("min_weight", 0.0, 0.5),
            # Widened from [1.5, 3.0] so the LV-adjust branch (which uses
            # this same cut for its LV-track sample) has more room to find
            # the right operating point jointly with apply_lv_adjust.
            "eta_max_extrap":       trial.suggest_float("eta_max_extrap", 1.0, 3.5),
            "apply_lv_adjust":      trial.suggest_categorical("apply_lv_adjust", [True, False]),
        }
        m = _evaluate_params(events, truth, params, subtract_pu=subtract_pu, args=args)
        trial.set_user_attr("bias", m["bias"])
        trial.set_user_attr("iqr", m["iqr"])
        trial.set_user_attr("nc_rel", m["nc_rel"])
        trial.set_user_attr("n_matched", m["n_matched"])
        return m["value"]

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=args.study_name, direction="minimize", sampler=sampler,
        storage=f"sqlite:///{out_dir / (args.study_name + '.db')}",
        load_if_exists=True,
    )
    t0 = time.time()
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)
    elapsed = time.time() - t0
    print(f"\nDone: {len(study.trials)} trials in {elapsed:.0f}s "
          f"({elapsed/max(1,len(study.trials)):.1f}s/trial)")

    # --- Search-best summary ---
    bt = study.best_trial
    print("\n=== Search-best (on {} events) ===".format(args.n_events))
    print(f"  value (|b|+IQR+0.5|nc_rel|) = {bt.value:.4f}")
    print(f"  bias    = {bt.user_attrs.get('bias'):+.4f}")
    print(f"  IQR     = {bt.user_attrs.get('iqr'):.4f}")
    print(f"  nc_rel  = {bt.user_attrs.get('nc_rel'):+.4f}")
    print(f"  N       = {bt.user_attrs.get('n_matched')}")

    # --- Validation pass: re-evaluate top-K on held-out events ---
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE
                 and t.value is not None and t.value < 1e5]
    completed.sort(key=lambda t: t.value)
    top = completed[:args.top_k]
    print(f"\nLoading {args.n_validation_events} validation events "
          f"(event_start={args.n_events}) ...", flush=True)
    val_events = load_charged_subtract_events(
        args.parquet_dir,
        event_start=args.n_events,
        event_stop=args.n_events + args.n_validation_events,
    )
    print(f"  {len(val_events['node_pt'])} validation events loaded.", flush=True)
    print(f"Clustering validation truth-HS jets ...", flush=True)
    val_truth = _cluster_truth(val_events, args)
    print(f"  Validation truth jets total = {val_truth['n'].sum()}", flush=True)

    print(f"\n=== Validating top-{len(top)} trials on {args.n_validation_events} events ===")
    val_table = []
    t1 = time.time()
    for rank, tr in enumerate(top, start=1):
        m = _evaluate_params(val_events, val_truth, tr.params,
                             subtract_pu=subtract_pu, args=args)
        row = {
            "rank": rank,
            "trial_number": tr.number,
            "params": tr.params,
            "search_value": tr.value,
            "search_bias": tr.user_attrs.get("bias"),
            "search_iqr": tr.user_attrs.get("iqr"),
            "search_nc_rel": tr.user_attrs.get("nc_rel"),
            "val_value": m["value"],
            "val_bias": m["bias"],
            "val_iqr": m["iqr"],
            "val_nc_rel": m["nc_rel"],
            "val_n_matched": m["n_matched"],
        }
        val_table.append(row)
        print(f"  [{rank:2d}] trial #{tr.number:<4d}  "
              f"search={tr.value:.4f}  "
              f"val={m['value']:.4f}  "
              f"(bias={m['bias']:+.4f}  IQR={m['iqr']:.4f}  "
              f"nc_rel={m['nc_rel']:+.4f}  N={m['n_matched']})",
              flush=True)
    print(f"  Validation pass took {time.time()-t1:.0f}s.", flush=True)

    if not val_table:
        print("WARNING: no completed trials to validate; falling back to search-best.")
        chosen = {
            "params": bt.params, "trial_number": bt.number,
            "search_value": bt.value, "val_value": None,
            "val_bias": None, "val_iqr": None, "val_nc_rel": None, "val_n_matched": None,
        }
    else:
        # Pick the trial with the smallest validation value.
        chosen = min(val_table, key=lambda r: r["val_value"])

    print("\n=== Final (validation-selected) best ===")
    print(f"  trial #{chosen.get('trial_number')}")
    print(f"  search value = {chosen.get('search_value'):.4f}")
    if chosen.get("val_value") is not None:
        print(f"  val value    = {chosen['val_value']:.4f}")
        print(f"  val bias     = {chosen['val_bias']:+.4f}")
        print(f"  val IQR      = {chosen['val_iqr']:.4f}")
        print(f"  val nc_rel   = {chosen['val_nc_rel']:+.4f}")
        print(f"  val N        = {chosen['val_n_matched']}")
    for k, v in chosen["params"].items():
        print(f"  {k:<22s} = {v}")

    (out_dir / f"{args.study_name}_best.json").write_text(json.dumps({
        "best_params": chosen["params"],
        "best_value": chosen.get("val_value"),
        "best_bias": chosen.get("val_bias"),
        "best_iqr": chosen.get("val_iqr"),
        "best_nc_rel": chosen.get("val_nc_rel"),
        "best_n_matched": chosen.get("val_n_matched"),
        "selected_from_trial": chosen.get("trial_number"),
        "selected_search_value": chosen.get("search_value"),
        "search_best_value": bt.value,
        "search_best_params": bt.params,
        "n_trials": len(study.trials),
        "n_events": args.n_events,
        "n_validation_events": args.n_validation_events,
        "top_k": args.top_k,
        "subtract_pu_charged": subtract_pu,
        "jet_algorithm": args.jet_algorithm,
        "jet_R": args.jet_R,
        "top_k_validation": val_table,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
