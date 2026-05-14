"""Per-PU-level analysis plots for the PU-conditioned reco model.

Consumes the per-PU H5 files produced by ``run_forward_pass_per_pu_level``
(one file per PU level). For each H5 it loads events, runs the analysis,
and discards intermediates before moving on — memory stays bounded.

Produces:

1. Jet matching residuals — mean & IQR vs PU level
   a) relative pt diff (pred - truth) / truth
   b) eta diff
   c) phi diff
2. Reco particle efficiency / accuracy / purity vs PU level
   (charged, neutral, total)
3. Per-PU-level jet resolution figure
   (same 6-panel layout as ``plot_jet_resolution_with_calo`` but with
   one curve per PU level instead of PUPPI / calo baselines)
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import iqr

from hepattn.experiments.odd_pileup_reco_pu_cond.eval_data import load_eval_data_from_h5
from hepattn.experiments.odd_pileup_reco_pu_cond.reco_analysis import (
    NUM_CLASSES,
    cluster_jets,
    get_jet_residuals,
    match_jets,
    pflow_data_from_eval_dicts,
)


DEFAULT_H5_GLOB = (
    "/storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco_pu_cond/"
    "logs/odd_pflow_reco_pu_cond_20260510-T144919/ckpts/"
    "epoch=039-val_loss=12.39195__test_latest_256_dim_pu*.h5"
)
DEFAULT_OUT_DIR = (
    "/storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco_pu_cond/"
    "pu_level_plots"
)
DEFAULT_N_PER_PU = -1  # -1 = use all events in each file

CHARGED_CLASSES = (0, 1, 2)
NEUTRAL_CLASSES = (3, 4)
ALL_VALID_CLASSES = tuple(range(NUM_CLASSES - 1))

GROUP_DEFS = {
    "charged": CHARGED_CLASSES,
    "neutral": NEUTRAL_CLASSES,
    "total": ALL_VALID_CLASSES,
}
GROUP_COLORS = {"charged": "#d62728", "neutral": "#2ca02c", "total": "#4c72b0"}
GROUP_MARKERS = {"charged": "o", "neutral": "s", "total": "^"}


def log(msg: str) -> None:
    print(msg, flush=True)


def detect_pu_level(h5_path: Path) -> int:
    """Determine the PU level of a single-PU H5 file.

    Prefer the per-row ``pu_level`` field (one unique value). Fall back to
    parsing ``_pu<int>`` from the filename.
    """
    with h5py.File(h5_path, "r") as f:
        if "events" in f and "pu_level" in f["events"].dtype.names:
            arr = f["events"]["pu_level"][:]
        elif "pu_level" in f:
            arr = f["pu_level"][:]
        else:
            arr = None
    if arr is not None and len(arr) > 0:
        uniq = np.unique(arr)
        if len(uniq) == 1:
            return int(uniq[0])
        raise ValueError(
            f"{h5_path} has multiple PU levels ({uniq.tolist()}); expected one-PU-per-file"
        )

    # filename fallback: ..._pu<int>.h5
    import re
    m = re.search(r"_pu(\d+)", h5_path.stem)
    if m:
        return int(m.group(1))
    raise KeyError(f"could not determine PU level for {h5_path}")


def n_rows_in_h5(h5_path: Path) -> int:
    with h5py.File(h5_path, "r") as f:
        return int(f["events"].shape[0])


def _concat_nonempty(arrs) -> np.ndarray:
    chunks = [a for a in arrs if len(a) > 0]
    return np.concatenate(chunks) if chunks else np.array([])


def _concat_jet_energy(jet_dict: dict, prefix: str) -> np.ndarray:
    e_chunks = []
    for pt, eta, mass in zip(
        jet_dict[f"{prefix}_jet_pt"],
        jet_dict[f"{prefix}_jet_eta"],
        jet_dict[f"{prefix}_jet_mass"],
    ):
        if len(pt) == 0:
            continue
        e2 = (pt * np.cosh(eta)) ** 2 + np.maximum(mass, 0.0) ** 2
        e = np.sqrt(np.clip(e2, 0.0, None))
        e = e[np.isfinite(e) & (e > 0)]
        if len(e) > 0:
            e_chunks.append(e)
    return np.concatenate(e_chunks) if e_chunks else np.array([])


def compute_particle_metrics(pflow_data: dict) -> dict:
    """Return efficiency, accuracy, purity per class group on already-loaded data."""
    truth_cls = pflow_data["truth_class"]
    pred_cls = pflow_data["pflow_class"]
    null = NUM_CLASSES - 1

    truth_valid_any = truth_cls < null
    pred_valid_any = pred_cls < null

    out: dict[str, dict] = {}
    for gname, gset in GROUP_DEFS.items():
        gset_arr = np.array(gset, dtype=np.int64)
        truth_in_g = np.isin(truth_cls, gset_arr) & truth_valid_any
        pred_in_g = np.isin(pred_cls, gset_arr) & pred_valid_any

        n_truth = int(truth_in_g.sum())
        n_pred = int(pred_in_g.sum())
        n_eff = int((truth_in_g & pred_in_g).sum())

        acc_denom_mask = truth_in_g & pred_valid_any
        n_acc_denom = int(acc_denom_mask.sum())
        n_acc_correct = int((acc_denom_mask & (truth_cls == pred_cls)).sum())

        out[gname] = {
            "efficiency": (n_eff / n_truth) if n_truth else np.nan,
            "purity": (n_eff / n_pred) if n_pred else np.nan,
            "accuracy": (n_acc_correct / n_acc_denom) if n_acc_denom else np.nan,
            "n_truth": n_truth,
            "n_pred": n_pred,
        }
    return out


def process_pu_file(
    h5_path: Path,
    pu: int,
    n_events: int,
    jet_R: float,
    min_constituents: int,
    min_pt: float,
    dr_cut: float,
) -> dict:
    """Load one PU H5, run jet clustering & matching, return summary."""
    total = n_rows_in_h5(h5_path)
    n_take = total if n_events < 0 else min(n_events, total)

    log(f"\n=== PU {pu}: {h5_path.name}  loading {n_take}/{total} events ===")
    _, _, reco_data = load_eval_data_from_h5(h5_path, event_start=0, event_stop=n_take)
    pflow_data = pflow_data_from_eval_dicts(reco_data)
    del reco_data

    metrics = compute_particle_metrics(pflow_data)

    log(f"PU {pu}: clustering jets...")
    jets = cluster_jets(
        pflow_data, jet_R=jet_R, min_constituents=min_constituents, min_pt=min_pt,
    )
    del pflow_data

    n_p = np.array([len(e) for e in jets["pflow_jet_pt"]])
    n_t = np.array([len(e) for e in jets["truth_jet_pt"]])
    keep = (n_p > 0) & (n_t > 0)

    if keep.any():
        log(f"PU {pu}: matching jets on {int(keep.sum())} events...")
        ti, pi, _ = match_jets(
            jets["pflow_jet_pt"][keep], jets["pflow_jet_eta"][keep], jets["pflow_jet_phi"][keep],
            jets["truth_jet_pt"][keep], jets["truth_jet_eta"][keep], jets["truth_jet_phi"][keep],
            dr_cut=dr_cut,
        )
        residuals = get_jet_residuals(
            ti, pi,
            jets["truth_jet_pt"][keep], jets["truth_jet_eta"][keep], jets["truth_jet_phi"][keep],
            jets["pflow_jet_pt"][keep], jets["pflow_jet_eta"][keep], jets["pflow_jet_phi"][keep],
        )
    else:
        residuals = {k: np.array([]) for k in ("dpt", "dpt_over_truth", "deta", "dphi", "truth_pt")}

    nconst = _concat_nonempty(jets["pflow_jet_nconst"])
    energy = _concat_jet_energy(jets, "pflow")
    del jets

    log(
        f"PU {pu} done. matched-jets={len(residuals['dpt'])}  "
        f"eff(total)={metrics['total']['efficiency']:.3f}  "
        f"pur(total)={metrics['total']['purity']:.3f}"
    )
    return {"residuals": residuals, "nconst": nconst, "energy": energy, "metrics": metrics}


def plot_jet_residual_mean_iqr_vs_pu(per_pu: dict, out_dir: Path) -> None:
    fields = [
        ("dpt_over_truth", r"$\Delta p_T / p_T^{\mathrm{truth}}$", "jet_dpt_rel_vs_pu.png"),
        ("deta", r"$\Delta\eta$", "jet_deta_vs_pu.png"),
        ("dphi", r"$\Delta\phi$", "jet_dphi_vs_pu.png"),
    ]
    pu_sorted = sorted(per_pu.keys())
    for key, label, fname in fields:
        means, iqrs, counts = [], [], []
        for pu in pu_sorted:
            d = per_pu[pu]["residuals"][key]
            d = d[np.isfinite(d)]
            counts.append(len(d))
            means.append(float(np.mean(d)) if len(d) else np.nan)
            iqrs.append(float(iqr(d)) if len(d) else np.nan)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
        ax1.plot(pu_sorted, means, "o-", color="#4c72b0", linewidth=1.8, markersize=8)
        ax1.set_xlabel("PU level"); ax1.set_ylabel(rf"Mean {label}")
        ax1.set_xticks(pu_sorted); ax1.grid(alpha=0.3)
        ax1.axhline(0.0, color="grey", linewidth=0.8, linestyle=":")

        ax2.plot(pu_sorted, iqrs, "s-", color="#dd8452", linewidth=1.8, markersize=8)
        ax2.set_xlabel("PU level"); ax2.set_ylabel(rf"IQR {label}")
        ax2.set_xticks(pu_sorted); ax2.grid(alpha=0.3)

        for ax, ys in ((ax1, means), (ax2, iqrs)):
            for x, y, n in zip(pu_sorted, ys, counts):
                if np.isfinite(y):
                    ax.annotate(f"n={n}", (x, y), textcoords="offset points",
                                xytext=(0, 8), ha="center", fontsize=8, color="grey")

        fig.suptitle(f"Jet matched-pair {label} vs PU level")
        out = out_dir / fname
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        log(f"Saved {out}")


def plot_particle_metrics_vs_pu(per_pu: dict, out_dir: Path) -> None:
    pu_sorted = sorted(per_pu.keys())
    for mname in ("efficiency", "accuracy", "purity"):
        fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
        for gname in ("charged", "neutral", "total"):
            ys = [per_pu[pu]["metrics"][gname][mname] for pu in pu_sorted]
            ax.plot(
                pu_sorted, ys,
                marker=GROUP_MARKERS[gname], color=GROUP_COLORS[gname],
                linewidth=1.8, markersize=8, label=gname.capitalize(),
            )
        ax.set_xlabel("PU level"); ax.set_ylabel(mname.capitalize())
        ax.set_xticks(pu_sorted); ax.set_ylim(0.0, 1.02); ax.grid(alpha=0.3)
        ax.legend(); ax.set_title(f"Reco particle {mname} vs PU level")
        out = out_dir / f"particle_{mname}_vs_pu.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        log(f"Saved {out}")


def plot_jet_resolution_per_pu(per_pu: dict, out_dir: Path) -> None:
    pu_sorted = sorted(per_pu.keys())
    if not pu_sorted:
        return
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(len(pu_sorted) - 1, 1)) for i in range(len(pu_sorted))]

    def _bins_from_all(field_key, n_bins=80, clip_pcts=(0.5, 99.5)):
        chunks = [per_pu[pu]["residuals"][field_key] for pu in pu_sorted
                  if len(per_pu[pu]["residuals"][field_key])]
        if not chunks:
            return np.linspace(-1.0, 1.0, n_bins)
        all_d = np.concatenate(chunks)
        all_d = all_d[np.isfinite(all_d)]
        if len(all_d) < 2:
            return np.linspace(-1.0, 1.0, n_bins)
        lo, hi = np.percentile(all_d, clip_pcts)
        if hi <= lo:
            hi = lo + 1e-6
        return np.linspace(lo, hi, n_bins)

    bins_deta = np.linspace(-0.2, 0.2, 50)
    bins_dphi = np.linspace(-0.2, 0.2, 50)
    bins_dpt = _bins_from_all("dpt", n_bins=90)
    bins_dpt_rel = np.linspace(-1.0, 4.0, 110)

    nc_chunks = [per_pu[pu]["nconst"] for pu in pu_sorted if len(per_pu[pu]["nconst"])]
    if nc_chunks:
        all_nc = np.concatenate(nc_chunks)
        bins_nc = np.arange(all_nc.min() - 0.5, all_nc.max() + 1.5)
    else:
        bins_nc = np.linspace(0, 10, 11)

    e_chunks = [per_pu[pu]["energy"] for pu in pu_sorted if len(per_pu[pu]["energy"])]
    if e_chunks:
        all_e = np.concatenate(e_chunks); all_e = all_e[all_e > 0]
        if len(all_e) > 1:
            lo, hi = np.percentile(all_e, [0.5, 99.5])
            bins_e = np.linspace(max(lo, 1e-3), max(hi, lo * 1.05), 80)
        else:
            bins_e = np.linspace(0, 200, 80)
    else:
        bins_e = np.linspace(0, 200, 80)

    configs = [
        ("deta", r"Jet $\Delta\eta$", bins_deta, True),
        ("dphi", r"Jet $\Delta\phi$", bins_dphi, True),
        ("nconst", "Jet # Constituents", bins_nc, True),
        ("dpt", r"Jet absolute $\Delta p_T$ [GeV]", bins_dpt, True),
        ("dpt_over_truth",
         r"Jet relative $\Delta p_T = (p_T^{\mathrm{pred}} - p_T^{\mathrm{truth}}) / p_T^{\mathrm{truth}}$",
         bins_dpt_rel, False),
        ("energy", "Jet energy [GeV]", bins_e, False),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)
    for idx, (key, xlabel, bins, density) in enumerate(configs):
        ax = axes[idx // 2, idx % 2]
        for color, pu in zip(colors, pu_sorted):
            if key == "nconst":
                d = per_pu[pu]["nconst"]
            elif key == "energy":
                d = per_pu[pu]["energy"]
            else:
                d = per_pu[pu]["residuals"][key]
            d = d[np.isfinite(d)]
            if len(d) == 0:
                continue
            label = rf"PU {pu}  $\mu$={np.mean(d):.3f}, IQR={iqr(d):.3f}"
            ax.hist(d, bins=bins, histtype="step", linewidth=1.8,
                    density=density, color=color, label=label)
        if key == "energy":
            ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density" if density else "Counts")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle("Jet resolution per PU level (PFlow vs Truth)")
    out = out_dir / "jet_resolution_per_pu.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved {out}")


def resolve_h5_paths(args) -> list[Path]:
    if args.h5:
        return [Path(p) for p in args.h5]
    matches = sorted(glob.glob(args.h5_glob))
    if not matches:
        raise FileNotFoundError(f"no H5 files matched glob: {args.h5_glob}")
    return [Path(p) for p in matches]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5", type=str, nargs="+", default=None,
                   help="explicit list of per-PU H5 files (overrides --h5-glob)")
    p.add_argument("--h5-glob", type=str, default=DEFAULT_H5_GLOB,
                   help="glob matching per-PU H5 files")
    p.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    p.add_argument("--n-per-pu", type=int, default=DEFAULT_N_PER_PU,
                   help="cap events per PU file (-1 = use all in file; default: -1)")
    p.add_argument("--jet-R", type=float, default=0.7)
    p.add_argument("--min-constituents", type=int, default=3)
    p.add_argument("--min-pt", type=float, default=10.0)
    p.add_argument("--dr-cut", type=float, default=0.4)
    args = p.parse_args()

    h5_paths = resolve_h5_paths(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"Found {len(h5_paths)} per-PU H5 file(s):")
    pu_to_path: dict[int, Path] = {}
    for path in h5_paths:
        pu = detect_pu_level(path)
        if pu in pu_to_path:
            log(f"  WARN: duplicate PU {pu} — keeping {pu_to_path[pu].name}, "
                f"dropping {path.name}")
            continue
        pu_to_path[pu] = path
        log(f"  PU {pu}: {path.name}")

    if not pu_to_path:
        log("No usable H5 files; aborting.")
        return 1

    per_pu: dict[int, dict] = {}
    for pu in sorted(pu_to_path.keys(), reverse=True):
        per_pu[pu] = process_pu_file(
            pu_to_path[pu], pu,
            n_events=args.n_per_pu,
            jet_R=args.jet_R, min_constituents=args.min_constituents,
            min_pt=args.min_pt, dr_cut=args.dr_cut,
        )

    log("\n=== Plot 1: jet residual mean & IQR vs PU level ===")
    plot_jet_residual_mean_iqr_vs_pu(per_pu, out_dir)

    log("\n=== Plot 2: particle efficiency / accuracy / purity vs PU level ===")
    plot_particle_metrics_vs_pu(per_pu, out_dir)

    log("\n=== Plot 3: 6-panel jet resolution histograms per PU level ===")
    plot_jet_resolution_per_pu(per_pu, out_dir)

    log(f"\nAll plots saved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
