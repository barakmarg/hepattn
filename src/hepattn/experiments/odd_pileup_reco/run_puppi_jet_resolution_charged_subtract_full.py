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
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

from hepattn.experiments.odd_pileup_reco.puppi_charged_subtract import (
    cluster_puppi_charged_subtract_jets,
    compute_puppi_weights_charged_subtract,
    load_charged_subtract_events,
)
from hepattn.experiments.odd_pileup_reco.reco_analysis import (
    cluster_jets,
    load_pflow_data,
    plot_jet_resolution_with_calo,
)
#/storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco/logs/odd_pflow_reco_20260421-T104425/ckpts/epoch=099-val_loss=13.99981__test_dihiggs.h5

DEFAULT_H5 = (
    "/storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco/"
    "logs/odd_pflow_reco_20260421-T104425/ckpts/"
    "epoch=099-val_loss=13.99981__test_dihiggs.h5"
)
DEFAULT_PARQUET_DIR = (
    "/storage/agrp/barakma/PileupODD/data/dihiggs_pu200_all_vertices_chunked"
)
DEFAULT_BEST_JSON = (
    "/storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco/"
    "optuna_puppi_charged_subtract/puppi_charged_subtract_v1_best.json"
)
DEFAULT_OUT = (
    "/storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco/"
    "puppi_jet_res_2000_ddhiggs_charged_subtract_full.png"
)


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
    p.add_argument("--event-start", type=int, default=None)
    p.add_argument("--event-stop", type=int, default=None)
    p.add_argument("--jet-R", type=float, default=0.7)
    p.add_argument("--min-constituents", type=int, default=3)
    p.add_argument("--min-pt", type=float, default=10.0)
    p.add_argument("--no-subtract-pu", action="store_true", default=False,
                   help="Sanity-check: subtract only HS-charged (= old puppi_perfect).")
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
    )

    # ---- 4. Plot — use the existing shared layout from reco_analysis.py ----
    fig = plot_jet_resolution_with_calo(
        jets, data,
        jet_R=args.jet_R,
        puppi_jets=puppi_jets,
        puppi_label="PUPPI (charged subtract)",
    )
    if fig is None:
        print("plot_jet_resolution_with_calo returned None (raw node fields missing).", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
