"""Render the jet-resolution comparison plot with the **perfect-PFlow PUPPI** variant.

Same plot as ``run_puppi_jet_resolution.py``, but the PUPPI overlay is computed
using ``compute_puppi_weights_perfect`` / ``cluster_puppi_perfect_jets``:

- LV tracks (truth ``tracks_mask=1``) added to jet inputs at full track pT.
- PU tracks (``tracks_mask=0``) excluded.
- Clusters PUPPI-weighted with their **neutral-remainder** pT
  (``node_e − calo_charged_e``), so adding LV tracks doesn't double-count
  HS-charged calo deposits.

This is the strongest possible PUPPI baseline: it uses truth charged-particle
cluster contributions (``calo_charged_e``) — i.e., perfect track→cluster
linking — so it's not a fair "no-truth-on-clusters" comparison. Useful as a
ceiling on what PUPPI could deliver if PFlow linking were perfect.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from hepattn.experiments.odd_pileup_reco.puppi_perfect import cluster_puppi_perfect_jets
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
    "puppi_jet_res_2000_perfect.png"
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5", type=str, default=DEFAULT_H5)
    p.add_argument("--out", type=str, default=DEFAULT_OUT)
    p.add_argument("--event-start", type=int, default=None)
    p.add_argument("--event-stop", type=int, default=None)
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

    puppi_jets_perfect = cluster_puppi_perfect_jets(
        data,
        jet_R=args.jet_R,
        min_constituents=args.min_constituents,
        min_pt=args.min_pt,
    )
    if puppi_jets_perfect is None:
        print("cluster_puppi_perfect_jets returned None (calo_charged_e missing).", file=sys.stderr)
        return 1

    fig = plot_jet_resolution_with_calo(
        jets, data,
        jet_R=args.jet_R,
        min_constituents=args.min_constituents,
        min_pt=args.min_pt,
        puppi_jets=puppi_jets_perfect,
        puppi_label="PUPPI (perfect PFlow)",
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
