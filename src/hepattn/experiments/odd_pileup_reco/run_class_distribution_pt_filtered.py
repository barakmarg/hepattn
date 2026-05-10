"""Render ``reco_analysis/class_distribution`` with a low-pT filter.

Same plot as ``plot_class_distribution`` from ``reco_analysis``, but truth and
pflow slots are restricted to ``pT >= --pt-cut`` (GeV) before the class
histogram is built. Useful for checking whether the apparent
"missing particles / over-predicted null" gap is dominated by very soft
(<300 MeV) particles that the network cannot reconstruct.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from hepattn.experiments.odd_pileup_reco.reco_analysis import (
    CLASS_LABELS,
    NUM_CLASSES,
    load_pflow_data,
)


DEFAULT_H5 = (
    "/storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco/"
    "logs/odd_pflow_reco_20260421-T104425/ckpts/"
    "epoch=099-val_loss=13.99981__test_latest_256_dim.h5"
)
DEFAULT_OUT = (
    "/storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco/"
    "class_distribution_pt_filtered.png"
)


def plot_class_distribution_pt_filtered(data: dict, pt_cut: float) -> plt.Figure:
    """Class distribution with a per-slot ``pT >= pt_cut`` (GeV) filter."""
    truth_cls = data["truth_class"].astype(np.int64, copy=False)
    pflow_cls = data["pflow_class"].astype(np.int64, copy=False)
    truth_pt = data["truth_ptetaphi"][..., 0]
    pflow_pt = data["pflow_ptetaphi"][..., 0]

    null_cls = NUM_CLASSES - 1
    n_real = null_cls  # plot non-null classes only

    truth_keep = (truth_cls >= 0) & (truth_cls < null_cls) & (truth_pt >= pt_cut)
    pflow_keep = (pflow_cls >= 0) & (pflow_cls < null_cls) & (pflow_pt >= pt_cut)

    truth_sel = truth_cls[truth_keep]
    pflow_sel = pflow_cls[pflow_keep]

    truth_counts = np.bincount(truth_sel, minlength=n_real)[:n_real]
    pflow_counts = np.bincount(pflow_sel, minlength=n_real)[:n_real]

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.arange(-0.5, n_real + 0.5)
    ax.hist(truth_sel, bins=bins,
            histtype="stepfilled", alpha=0.5, label="Truth")
    ax.hist(pflow_sel, bins=bins,
            histtype="step", label="PFlow")

    y_peak = np.maximum(truth_counts, pflow_counts).astype(float)
    y_offset = max(float(y_peak.max()) * 0.03, 1.0)
    for i in range(n_real):
        ax.text(
            i,
            float(y_peak[i]) + y_offset,
            f"T:{int(truth_counts[i]):,}\nP:{int(pflow_counts[i]):,}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xlabel("Class")
    ax.set_ylabel("Particles")
    ax.set_xticks(np.arange(n_real))
    ax.set_xticklabels(CLASS_LABELS[:n_real])
    ax.set_ylim(top=float(y_peak.max()) * 1.35)
    ax.legend()
    ax.set_title(
        f"Particle class distribution (pT >= {pt_cut:g} GeV, null excluded)\n"
        f"Truth total: {int(truth_counts.sum()):,}   "
        f"PFlow total: {int(pflow_counts.sum()):,}"
    )
    fig.tight_layout()
    return fig


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5", type=str, default=DEFAULT_H5)
    p.add_argument("--out", type=str, default=DEFAULT_OUT)
    p.add_argument("--event-start", type=int, default=None)
    p.add_argument("--event-stop", type=int, default=None)
    p.add_argument("--pt-cut", type=float, default=0.3,
                   help="Drop slots with pT below this threshold (GeV).")
    p.add_argument("--dpi", type=int, default=120)
    args = p.parse_args()

    data = load_pflow_data(args.h5, event_start=args.event_start, event_stop=args.event_stop)
    print(f"Loaded {Path(args.h5).name}: n_events = {data['pflow_class'].shape[0]}")

    fig = plot_class_distribution_pt_filtered(data, pt_cut=args.pt_cut)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
