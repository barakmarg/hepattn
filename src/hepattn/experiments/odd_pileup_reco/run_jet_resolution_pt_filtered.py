"""Jet-resolution plot with a low-pT constituent filter.

Same as ``run_puppi_jet_resolution.py`` but particles with pT < ``--pt-cut``
(GeV) are masked out of the pflow and truth indicators before jet clustering,
so they do not contribute as jet constituents.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from hepattn.experiments.odd_pileup_reco.reco_analysis import (
    cluster_jets,
    load_pflow_data,
    plot_jet_resolution_with_calo,
)


DEFAULT_H5 = (
    "/storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco/"
    "logs/odd_pflow_reco_20260421-T104425/ckpts/"
    "epoch=099-val_loss=13.99981__test_latest_256_dim.h5"
)
DEFAULT_OUT = (
    "/storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco/"
    "jet_resolution_pt_filtered.png"
)


def apply_pt_filter(data: dict, pt_cut: float) -> None:
    """Zero out indicators for particles below pt_cut (GeV), in-place."""
    data["pflow_indicator"] = (
        data["pflow_indicator"] & (data["pflow_ptetaphi"][..., 0] >= pt_cut)
    )
    data["truth_indicator"] = (
        data["truth_indicator"] & (data["truth_ptetaphi"][..., 0] >= pt_cut)
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5", type=str, default=DEFAULT_H5)
    p.add_argument("--out", type=str, default=DEFAULT_OUT)
    p.add_argument("--event-start", type=int, default=None)
    p.add_argument("--event-stop", type=int, default=None)
    p.add_argument("--pt-cut", type=float, default=0.25,
                   help="Drop particles with pT below this value (GeV) before clustering.")
    p.add_argument("--jet-R", type=float, default=0.7)
    p.add_argument("--min-constituents", type=int, default=3)
    p.add_argument("--min-pt", type=float, default=10.0)
    p.add_argument("--dpi", type=int, default=120)
    args = p.parse_args()

    data = load_pflow_data(args.h5, event_start=args.event_start, event_stop=args.event_stop)
    print(f"Loaded {Path(args.h5).name}: n_events = {data['pflow_class'].shape[0]}")

    apply_pt_filter(data, args.pt_cut)
    n_pflow = data["pflow_indicator"].sum()
    n_truth = data["truth_indicator"].sum()
    print(f"After pT >= {args.pt_cut:g} GeV filter: pflow={n_pflow:,}  truth={n_truth:,} particles")

    jets = cluster_jets(
        data,
        jet_R=args.jet_R,
        min_constituents=args.min_constituents,
        min_pt=args.min_pt,
    )

    fig = plot_jet_resolution_with_calo(jets, data, jet_R=args.jet_R)
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
