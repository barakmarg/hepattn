"""Generate the PUPPI jet-resolution comparison plot.

Loads a prediction-writer H5, clusters jets for PFlow / Truth / Proxy and
the calo / calo-HS / PUPPI baselines, and saves the 3x2 jet-resolution
figure produced by ``plot_jet_resolution_with_calo``.
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
    "puppi_jet_res_2000.png"
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5", type=str, default=DEFAULT_H5, help="prediction-writer H5 path")
    p.add_argument("--out", type=str, default=DEFAULT_OUT, help="output figure path")
    p.add_argument("--event-start", type=int, default=None)
    p.add_argument("--event-stop", type=int, default=None, help="exclusive; default = all events")
    p.add_argument("--jet-R", type=float, default=0.7)
    p.add_argument("--min-constituents", type=int, default=3)
    p.add_argument("--min-pt", type=float, default=10.0)
    p.add_argument("--dpi", type=int, default=120)
    args = p.parse_args()

    data = load_pflow_data(args.h5, event_start=args.event_start, event_stop=args.event_stop)
    print(f"Loaded {Path(args.h5).name}: n_events = {data['pflow_class'].shape[0]}")

    jets = cluster_jets(
        data,
        jet_R=args.jet_R,
        min_constituents=args.min_constituents,
        min_pt=args.min_pt,
    )

    fig = plot_jet_resolution_with_calo(jets, data, jet_R=args.jet_R)
    if fig is None:
        print("plot_jet_resolution_with_calo returned None (raw node fields missing)", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
