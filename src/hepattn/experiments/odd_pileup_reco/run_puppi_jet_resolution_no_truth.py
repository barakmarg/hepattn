"""Render the jet-resolution comparison plot with the **truth-free** PUPPI variant.

Same plot as ``run_puppi_jet_resolution.py``, but the PUPPI overlay is computed
using ``compute_puppi_weights_no_truth`` (Δz CHS from ``node_z0`` instead of the
truth ``tracks_mask``).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from hepattn.experiments.odd_pileup_reco.puppi import (
    cluster_puppi_jets,
    compute_puppi_weights_no_truth,
)
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
    "puppi_jet_res_2000_no_truth.png"
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
    p.add_argument("--dz-cut", type=float, default=0.7,
                   help="Δz threshold for CHS-style LV/PU track classification.")
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

    weights_nt = compute_puppi_weights_no_truth(data, dz_cut=args.dz_cut)
    if weights_nt is None:
        print("compute_puppi_weights_no_truth returned None (node_z0 missing).", file=sys.stderr)
        return 1

    puppi_jets_nt = cluster_puppi_jets(
        data,
        jet_R=args.jet_R,
        min_constituents=args.min_constituents,
        min_pt=args.min_pt,
        weights=weights_nt,
    )
    if puppi_jets_nt is None:
        print("cluster_puppi_jets returned None.", file=sys.stderr)
        return 1

    fig = plot_jet_resolution_with_calo(
        jets, data,
        jet_R=args.jet_R,
        puppi_jets=puppi_jets_nt,
        puppi_label=f"PUPPI (no-truth, dz<{args.dz_cut})",
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
