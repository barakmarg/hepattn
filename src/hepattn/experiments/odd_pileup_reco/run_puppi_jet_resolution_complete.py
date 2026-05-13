"""Render jet-resolution plot for the **complete CMS-faithful PUPPI** variant.

Input: truth particles (HS + PU) from the chunked parquet dataset
(``ttbar_pu200_all_vertices_chunked``). PUPPI operates on the full particle
list with full reconstructed-like kinematics — its only blind spot is
per-particle HS/PU labels for neutrals.

The reference jets are clustered from truth HS particles only
(``vertex_primary == 1``). PUPPI jets are clustered from PUPPI-weighted
particles. We compute Δpt/pt_truth residuals on Hungarian-matched jet pairs
and report bias / IQR / N-constituent ratio.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hepattn.experiments.odd_pileup_reco.puppi_complete import (
    cluster_puppi_complete_jets,
    cluster_truth_hs_jets,
    compute_puppi_weights_complete,
    load_truth_particles,
)
from hepattn.experiments.odd_pileup_reco.reco_analysis import (
    get_jet_residuals,
    match_jets,
)


DEFAULT_DIR = "/storage/agrp/barakma/PileupODD/data/ttbar_pu200_all_vertices_chunked"
DEFAULT_OUT = (
    "/storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco/"
    "puppi_jet_res_complete.png"
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquet-dir", default=DEFAULT_DIR)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--n-events", type=int, default=100)
    p.add_argument("--event-start", type=int, default=0)
    p.add_argument("--jet-R", type=float, default=0.7)
    p.add_argument("--min-constituents", type=int, default=3)
    p.add_argument("--min-pt", type=float, default=10.0)
    p.add_argument("--dr-cut", type=float, default=0.4)
    # PUPPI tunables — defaults from particle-space optuna v1 (trial#133,
    # 150 trials × 100 events, verified on held-out 500 events:
    # bias=-0.032, IQR=0.416, nc_rel=-0.012).
    p.add_argument("--R0", type=float, default=0.354)
    p.add_argument("--dz-cut", type=float, default=1.997)
    p.add_argument("--rms-pt-min", type=float, default=0.066)
    p.add_argument("--min-neutral-pt", type=float, default=0.918)
    p.add_argument("--min-neutral-pt-slope", type=float, default=0.0085)
    p.add_argument("--min-weight", type=float, default=0.293)
    p.add_argument("--eta-max-extrap", type=float, default=1.637)
    p.add_argument("--no-lv-adjust", action="store_true", default=False)
    p.add_argument("--dpi", type=int, default=120)
    args = p.parse_args()

    print(f"Loading {args.n_events} events from {args.parquet_dir} ...", flush=True)
    particles = load_truth_particles(
        args.parquet_dir,
        event_start=args.event_start,
        event_stop=args.event_start + args.n_events,
    )
    n_events = len(particles["pt"])
    npart = np.array([len(x) for x in particles["pt"]])
    n_hs = np.array([int((vp == 1).sum()) for vp in particles["vertex_primary"]])
    print(f"  {n_events} events, particles/event: mean={npart.mean():.0f}, "
          f"HS/event: mean={n_hs.mean():.0f}", flush=True)

    print("Computing PUPPI weights ...", flush=True)
    weights = compute_puppi_weights_complete(
        particles,
        R0=args.R0,
        dz_cut=args.dz_cut,
        rms_pt_min=args.rms_pt_min,
        min_neutral_pt=args.min_neutral_pt,
        min_neutral_pt_slope=args.min_neutral_pt_slope,
        min_weight=args.min_weight,
        eta_max_extrap=args.eta_max_extrap,
        apply_lv_adjust=not args.no_lv_adjust,
    )

    print("Clustering jets ...", flush=True)
    puppi_jets = cluster_puppi_complete_jets(
        particles,
        jet_R=args.jet_R,
        min_constituents=args.min_constituents,
        min_pt=args.min_pt,
        weights=weights,
    )
    truth_jets = cluster_truth_hs_jets(
        particles,
        jet_R=args.jet_R,
        min_constituents=args.min_constituents,
        min_pt=args.min_pt,
    )

    n_p = np.array([len(x) for x in puppi_jets["puppi_jet_pt"]])
    n_t = np.array([len(x) for x in truth_jets["truth_jet_pt"]])
    mask = (n_p > 0) & (n_t > 0)
    print(f"  Truth jets total: {n_t.sum()}, PUPPI jets total: {n_p.sum()}", flush=True)

    tr_ix, pr_ix, _ = match_jets(
        puppi_jets["puppi_jet_pt"][mask],
        puppi_jets["puppi_jet_eta"][mask],
        puppi_jets["puppi_jet_phi"][mask],
        truth_jets["truth_jet_pt"][mask],
        truth_jets["truth_jet_eta"][mask],
        truth_jets["truth_jet_phi"][mask],
        dr_cut=args.dr_cut,
    )
    res = get_jet_residuals(
        tr_ix, pr_ix,
        truth_jets["truth_jet_pt"][mask],
        truth_jets["truth_jet_eta"][mask],
        truth_jets["truth_jet_phi"][mask],
        puppi_jets["puppi_jet_pt"][mask],
        puppi_jets["puppi_jet_eta"][mask],
        puppi_jets["puppi_jet_phi"][mask],
    )

    # nconst ratio on matched pairs
    nc_p_arr = puppi_jets["puppi_jet_nconst"][mask]
    nc_t_arr = truth_jets["truth_jet_nconst"][mask]
    nc_p_list, nc_t_list = [], []
    for i, (ti, pi) in enumerate(zip(tr_ix, pr_ix)):
        if len(ti):
            nc_p_list.append(nc_p_arr[i][pi])
            nc_t_list.append(nc_t_arr[i][ti])
    nc_p = np.concatenate(nc_p_list) if nc_p_list else np.array([])
    nc_t = np.concatenate(nc_t_list) if nc_t_list else np.array([])

    d = res["dpt_over_truth"]
    from scipy.stats import iqr
    bias = float(np.nanmedian(d))
    spread = float(iqr(d))
    nc_rel = float(np.mean((nc_p - nc_t) / np.clip(nc_t, 1, None))) if nc_p.size else float("nan")
    print(f"\nResolution: bias={bias:+.4f}  IQR={spread:.4f}  nc_rel={nc_rel:+.4f}  N_matched={len(d)}",
          flush=True)

    # ---- Plot ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    bins = np.linspace(-1.0, 4.0, 110)
    ax.hist(d, bins=bins, density=True, histtype="step", linewidth=1.8,
            label=rf"PUPPI  $\mu$={bias:+.3f}, IQR={spread:.3f}")
    ax.set_xlabel(r"$(p_T^{\rm puppi} - p_T^{\rm truth})/p_T^{\rm truth}$")
    ax.set_ylabel("density")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.set_title("Jet relative Δpt")

    ax = axes[0, 1]
    nc_bins = np.arange(0, max(nc_p.max() if nc_p.size else 50,
                               nc_t.max() if nc_t.size else 50) + 2)
    ax.hist(nc_t, bins=nc_bins, histtype="step", linewidth=1.8,
            label=f"truth-HS μ={nc_t.mean():.1f}")
    ax.hist(nc_p, bins=nc_bins, histtype="step", linewidth=1.8,
            label=f"PUPPI μ={nc_p.mean():.1f}")
    ax.set_xlabel("Jet # constituents")
    ax.legend(fontsize=9)
    ax.set_title(f"# constituents (matched, nc_rel={nc_rel:+.3f})")

    ax = axes[1, 0]
    ax.hist(res["deta"], bins=np.linspace(-0.2, 0.2, 50), histtype="step", linewidth=1.8)
    ax.set_xlabel(r"$\Delta\eta$"); ax.set_title("Jet Δη")

    ax = axes[1, 1]
    ax.hist(res["dphi"], bins=np.linspace(-0.2, 0.2, 50), histtype="step", linewidth=1.8)
    ax.set_xlabel(r"$\Delta\phi$"); ax.set_title("Jet Δφ")

    fig.suptitle(f"Complete PUPPI vs truth-HS jets (n_events={n_events})", fontsize=11)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
