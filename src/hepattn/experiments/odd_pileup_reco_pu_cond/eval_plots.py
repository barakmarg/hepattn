"""Offline evaluation plots for the Two-Stream MaskFormer.

Data is loaded from prediction writer H5 files (canonical format). Two paths:

1. Load existing H5 directly (no model/GPU needed)::

    from hepattn.experiments.odd_pileup_reco_pu_cond.eval_plots import run_eval
    figs, track, cluster, reco = run_eval(
        h5_path="logs/.../epoch=042-val_loss=13.01940__test.h5"
    )

2. Run forward pass (produces H5 via PflowPredictionWriter, then loads it)::

    figs, track, cluster, reco = run_eval(
        ckpt_path="...", config_path="...", data_dir="..."
    )
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from hepattn.experiments.odd_pileup_reco_pu_cond.eval_data import (
    load_eval_data_from_h5,
    run_forward_pass,
)

# ── Configuration ──────────────────────────────────────────────────────────
CONFIG_PATH = "/storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco_pu_cond/configs/base.yaml"
RECO_TRUTH_PT_BINS = np.array([0.0, 0.2, 0.5, 0.9, 2.0, 4.0, 10.0, 20.0, np.inf], dtype=np.float64)
RECO_NUM_CLASSES = 6
CALO_PRED_THRESHOLD = 0.2
CALO_HS_FRAC_THRESHOLD = 0.05
CALO_HS_ENERGY_THRESHOLD = 0.15


def plot_incidence_matrix_event(
    reco_data: dict,
    event_index: int,
    incidence_key: str = "pred_incidence",
    num_nodes: int = 1400,
    mode: str = "pileup",
    truth_key: str = "truth_incidence",
    max_objects: int | None = None,
    pred_threshold: float = 0.0,
    truth_threshold: float = 0.0,
    tracks_only: bool = False,
    pred_track_proxy: bool = False,
    exclude_pileup_row: bool = False,
) -> tuple[plt.Figure, dict[str, float | int | tuple[int, ...]]]:
    """Plot reco-space incidence diagnostics for one event.

    Parameters
    ----------
    reco_data : dict
        In-memory output dict produced by load_eval_data_from_h5.
    event_index : int
        Event to visualize (supports negative indexing).
    incidence_key : str
        Prediction incidence key in reco_data.
    num_nodes : int
        Number of reco-node columns to use (1400 by default).
    mode : str
        One of: "pileup", "binary_compare", "raw_compare".
    truth_key : str
        Truth incidence key in reco_data (used by compare modes).
    max_objects : int | None
        Optional cap on number of object rows.
    pred_threshold : float
        Threshold for binary prediction incidence (binary_compare).
    truth_threshold : float
        Threshold for binary truth incidence (binary_compare).
    tracks_only : bool
        If True, restrict columns to reco nodes identified as tracks.
    pred_track_proxy : bool
        If True in binary_compare mode, keep only one predicted track per object
        (highest-score track above threshold), matching old charged-track proxy style.
    exclude_pileup_row : bool
        If True, drop row 0 before plotting/statistics. Useful when comparing
        to non-pileup-style object rows.
    """

    def _read_event_matrix(key: str) -> tuple[np.ndarray, int]:
        if key not in reco_data:
            raise KeyError(
                f"Key '{key}' not found in reco_data. "
                f"Available keys: {sorted(reco_data.keys())}. "
                "Reload reco_data with load_eval_data_from_h5(...) before calling this function."
            )

        inc_all = np.asarray(reco_data[key])
        if inc_all.ndim != 3:
            raise ValueError(
                f"Expected incidence array with shape (events, objects, nodes), got {inc_all.shape}"
            )

        n_events_local = inc_all.shape[0]
        evt = event_index
        if evt < 0:
            evt = n_events_local + evt
        if evt < 0 or evt >= n_events_local:
            raise IndexError(f"event_index={evt} out of range for {n_events_local} events")

        mat_local = inc_all[evt]
        if mat_local.ndim != 2:
            raise ValueError(f"Expected per-event incidence to be 2D, got {mat_local.shape}")
        if mat_local.shape[1] < num_nodes:
            raise ValueError(
                f"Requested num_nodes={num_nodes}, but incidence has only {mat_local.shape[1]} nodes"
            )

        mat_local = mat_local[:, :num_nodes]
        if max_objects is not None:
            n_obj = int(max_objects)
            if n_obj <= 0:
                raise ValueError(f"max_objects must be positive when provided, got {max_objects}")
            n_obj = min(n_obj, mat_local.shape[0])
            mat_local = mat_local[:n_obj]

        return mat_local, evt

    pred_mat, resolved_event_index = _read_event_matrix(incidence_key)
    obj_row_offset = 0
    if exclude_pileup_row:
        if pred_mat.shape[0] < 2:
            raise ValueError("Cannot exclude pileup row: incidence has fewer than 2 object rows")
        pred_mat = pred_mat[1:]
        obj_row_offset = 1

    reco_node_valid = None
    if "reco_node_valid" in reco_data:
        node_valid_all = np.asarray(reco_data["reco_node_valid"])
        if node_valid_all.ndim == 2 and resolved_event_index < node_valid_all.shape[0]:
            reco_node_valid = node_valid_all[resolved_event_index, :num_nodes].astype(bool)

    truth_valid_evt = None
    if "truth_valid" in reco_data:
        truth_valid_all = np.asarray(reco_data["truth_valid"])
        if truth_valid_all.ndim == 2 and resolved_event_index < truth_valid_all.shape[0]:
            truth_valid_evt = truth_valid_all[
                resolved_event_index,
                obj_row_offset:obj_row_offset + pred_mat.shape[0],
            ].astype(bool)

    n_tracks = None
    if "reco_n_tracks" in reco_data:
        n_tracks_all = np.asarray(reco_data["reco_n_tracks"]).reshape(-1)
        if resolved_event_index < len(n_tracks_all):
            n_tracks = int(max(0, min(int(n_tracks_all[resolved_event_index]), pred_mat.shape[1])))

    track_mask = None
    if "reco_is_track" in reco_data:
        reco_is_track_all = np.asarray(reco_data["reco_is_track"])
        if reco_is_track_all.ndim == 2 and resolved_event_index < reco_is_track_all.shape[0]:
            track_mask = reco_is_track_all[resolved_event_index, :pred_mat.shape[1]].astype(bool)

    if tracks_only:
        if track_mask is None:
            raise KeyError(
                "tracks_only=True requires reco_data['reco_is_track']. "
                "Reload reco_data from a v2 prediction-writer H5 with node_metadata."
            )
        pred_mat = pred_mat[:, track_mask]
        if reco_node_valid is not None:
            reco_node_valid = reco_node_valid[track_mask]
        n_tracks = pred_mat.shape[1]

    if mode == "pileup":
        if exclude_pileup_row:
            raise ValueError("exclude_pileup_row is not supported in mode='pileup'")
        pileup_row = pred_mat[0]
        mat_sum = np.sum(pred_mat)
        pileup_sum = np.sum(pileup_row)

        stats: dict[str, float | int | tuple[int, ...]] = {
            "event_index": int(resolved_event_index),
            "matrix_shape": tuple(int(x) for x in pred_mat.shape),
            "matrix_sum": float(mat_sum),
            "pileup_sum": float(pileup_sum),
            "pileup_fraction_of_total": float(pileup_sum / mat_sum),
            "pileup_min": float(np.min(pileup_row)),
            "pileup_max": float(np.max(pileup_row)),
            "pileup_mean": float(np.mean(pileup_row)),
            "pileup_std": float(np.std(pileup_row)),
            "pileup_nan_count": int(np.isnan(pileup_row).sum()),
            "pileup_posinf_count": int(np.isposinf(pileup_row).sum()),
            "pileup_neginf_count": int(np.isneginf(pileup_row).sum()),
            "matrix_nan_count": int(np.isnan(pred_mat).sum()),
            "matrix_posinf_count": int(np.isposinf(pred_mat).sum()),
            "matrix_neginf_count": int(np.isneginf(pred_mat).sum()),
        }

        if n_tracks is not None:
            stats["n_tracks"] = int(n_tracks)

        fig, (ax_mat, ax_pu) = plt.subplots(
            2,
            1,
            figsize=(15, 9),
            gridspec_kw={"height_ratios": [3.2, 1.2]},
            constrained_layout=True,
        )

        im = ax_mat.imshow(pred_mat, aspect="auto", interpolation="nearest", cmap="viridis")
        fig.colorbar(im, ax=ax_mat, pad=0.01, label="Incidence value")
        ax_mat.set_title(
            f"Reco incidence matrix | event={resolved_event_index} | shape={pred_mat.shape[0]}x{pred_mat.shape[1]}"
        )
        ax_mat.set_ylabel("Object index (row 0 = pileup token)")
        ax_mat.set_xlabel("Filtered node index")
        if n_tracks is not None and n_tracks > 0 and n_tracks < pred_mat.shape[1]:
            ax_mat.axvline(n_tracks, color="red", linestyle="--", linewidth=1.2)

        x = np.arange(pred_mat.shape[1], dtype=np.int64)
        ax_pu.plot(x, pileup_row, color="#d62728", linewidth=1.0)
        ax_pu.set_title("Pileup token row (row 0) across filtered nodes")
        ax_pu.set_xlabel("Filtered node index")
        ax_pu.set_ylabel("Incidence")
        ax_pu.grid(True, alpha=0.25)
        if n_tracks is not None and n_tracks > 0 and n_tracks < pred_mat.shape[1]:
            ax_pu.axvline(n_tracks, color="red", linestyle="--", linewidth=1.2)

        stats_lines = [
            f"matrix_sum={stats['matrix_sum']:.6g}",
            f"pileup_sum={stats['pileup_sum']:.6g}",
            f"pileup_fraction={stats['pileup_fraction_of_total']:.6g}",
            f"pileup_mean={stats['pileup_mean']:.6g}",
            f"pileup_std={stats['pileup_std']:.6g}",
            f"pileup_min={stats['pileup_min']:.6g}",
            f"pileup_max={stats['pileup_max']:.6g}",
            f"pileup_nan={stats['pileup_nan_count']} +inf={stats['pileup_posinf_count']} -inf={stats['pileup_neginf_count']}",
        ]
        if n_tracks is not None:
            stats_lines.append(f"n_tracks={n_tracks}")
        ax_pu.text(
            1.01,
            0.5,
            "\n".join(stats_lines),
            transform=ax_pu.transAxes,
            va="center",
            ha="left",
            fontsize=9,
            family="monospace",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
        )

        return fig, stats

    if mode not in {"binary_compare", "raw_compare"}:
        raise ValueError(f"Unsupported mode='{mode}'. Use one of: 'pileup', 'binary_compare', 'raw_compare'")

    truth_mat, _ = _read_event_matrix(truth_key)
    if exclude_pileup_row:
        truth_mat = truth_mat[obj_row_offset:obj_row_offset + pred_mat.shape[0]]
    if tracks_only and track_mask is not None:
        truth_mat = truth_mat[:, track_mask]

    if mode == "binary_compare":
        if pred_track_proxy:
            if track_mask is None and not tracks_only:
                raise KeyError(
                    "pred_track_proxy=True requires reco_data['reco_is_track'] unless tracks_only=True. "
                    "Reload reco_data from a v2 prediction-writer H5 with node_metadata."
                )

            pred_binary = np.zeros_like(pred_mat, dtype=bool)
            local_track_mask = np.ones(pred_mat.shape[1], dtype=bool) if tracks_only else track_mask
            pred_track_scores = np.where(local_track_mask[None, :], pred_mat, -np.inf)
            best_idx = np.argmax(pred_track_scores, axis=1)
            best_val = pred_track_scores[np.arange(pred_mat.shape[0]), best_idx]
            valid_rows = best_val > pred_threshold
            if np.any(valid_rows):
                row_idx = np.where(valid_rows)[0]
                pred_binary[row_idx, best_idx[row_idx]] = True
        else:
            pred_binary = pred_mat > pred_threshold

        truth_binary = truth_mat > truth_threshold

        matches = truth_binary & pred_binary
        pred_only = pred_binary & (~truth_binary)
        truth_only = truth_binary & (~pred_binary)

        n_matches = int(np.count_nonzero(matches))
        n_pred_only = int(np.count_nonzero(pred_only))
        n_truth_only = int(np.count_nonzero(truth_only))
        n_total_truth = int(np.count_nonzero(truth_binary))
        n_total_pred = int(np.count_nonzero(pred_binary))

        def _safe_pct(num: int, den: int) -> float:
            return float(100.0 * num / den) if den > 0 else 0.0

        match_eff = _safe_pct(n_matches, n_total_truth)
        fp_rate = _safe_pct(n_pred_only, n_total_pred)
        fn_rate = _safe_pct(n_truth_only, n_total_truth)

        display_pred = np.zeros_like(pred_binary, dtype=np.int32)
        display_truth = np.zeros_like(truth_binary, dtype=np.int32)
        display_pred[matches] = 1
        display_pred[pred_only] = 2
        display_truth[matches] = 1
        display_truth[truth_only] = 3

        cmap = ListedColormap(["white", "lime", "gold", "red"])
        fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)

        title_core = (
            "Incidence comparison\n"
            f"match={n_matches} ({match_eff:.1f}%) | "
            f"fp={n_pred_only} ({fp_rate:.1f}%) | "
            f"fn={n_truth_only} ({fn_rate:.1f}%)"
        )

        for ax, data, name in [
            (axes[0], display_pred, "Pred view"),
            (axes[1], display_truth, "Truth view"),
        ]:
            ax.imshow(data, interpolation="nearest", aspect="auto", cmap=cmap, vmin=0, vmax=3)
            if n_tracks is not None and n_tracks > 0 and n_tracks < data.shape[1]:
                ax.axvline(n_tracks, color="blue", linestyle="--", linewidth=1.2)
            ax.set_title(f"{name}\n{title_core}")
            ax.set_xlabel("Filtered node index")
            ax.set_ylabel("Object index")

        stats = {
            "event_index": int(resolved_event_index),
            "matrix_shape": tuple(int(x) for x in pred_mat.shape),
            "n_matches": n_matches,
            "n_pred_only": n_pred_only,
            "n_truth_only": n_truth_only,
            "n_total_pred": n_total_pred,
            "n_total_truth": n_total_truth,
            "match_eff_pct": float(match_eff),
            "fp_rate_pct": float(fp_rate),
            "fn_rate_pct": float(fn_rate),
            "pred_threshold": float(pred_threshold),
            "truth_threshold": float(truth_threshold),
            "tracks_only": int(bool(tracks_only)),
            "pred_track_proxy": int(bool(pred_track_proxy)),
            "exclude_pileup_row": int(bool(exclude_pileup_row)),
        }
        if n_tracks is not None:
            stats["n_tracks"] = int(n_tracks)

        return fig, stats

    pred_plot = pred_mat
    truth_plot = truth_mat

    if reco_node_valid is not None:
        pred_plot = pred_plot * reco_node_valid[None, :]
    if truth_valid_evt is not None:
        truth_plot = truth_plot * truth_valid_evt[:, None]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    axes[0].imshow(pred_plot, interpolation="nearest", aspect="auto")
    axes[0].set_title("PFlow incidence (reco-node space)")
    axes[0].set_xlabel("Filtered node index")
    axes[0].set_ylabel("Object index")

    axes[1].imshow(truth_plot, interpolation="nearest", aspect="auto")
    axes[1].set_title("Truth incidence (reco-node space)")
    axes[1].set_xlabel("Filtered node index")
    axes[1].set_ylabel("Object index")

    if n_tracks is not None and n_tracks > 0 and n_tracks < pred_plot.shape[1]:
        axes[0].axvline(n_tracks, color="red", linestyle="--", linewidth=1.2)
        axes[1].axvline(n_tracks, color="red", linestyle="--", linewidth=1.2)

    stats = {
        "event_index": int(resolved_event_index),
        "matrix_shape": tuple(int(x) for x in pred_plot.shape),
        "pred_sum": float(np.sum(pred_plot)),
        "truth_sum": float(np.sum(truth_plot)),
        "pred_nan_count": int(np.isnan(pred_plot).sum()),
        "truth_nan_count": int(np.isnan(truth_plot).sum()),
        "tracks_only": int(bool(tracks_only)),
        "exclude_pileup_row": int(bool(exclude_pileup_row)),
    }
    if n_tracks is not None:
        stats["n_tracks"] = int(n_tracks)
    if reco_node_valid is not None:
        stats["n_valid_nodes"] = int(np.count_nonzero(reco_node_valid))
    if truth_valid_evt is not None:
        stats["n_valid_truth_objects"] = int(np.count_nonzero(truth_valid_evt))

    return fig, stats


def compute_deltaR_window_stats(
    dataset,
    window_sizes: list[int] = (32, 64, 80, 128, 256),
    n_sample: int = 200,
) -> dict:
    """Compute mean/max delta R within Morton-sorted windows across sampled events.

    Vectorized per offset (numpy inner loop) — efficient for n_sample~200 events.

    Parameters
    ----------
    dataset : ODDDatasetPileup
        Dataset with full_data_array, track_cumsum, cluster_cumsum.
    window_sizes : iterable of int
    n_sample : int
        Events to sample (capped at dataset size).

    Returns
    -------
    dict : {window_size -> {"mean_dR": list[float], "max_dR": list[float]}}
    """
    import torch
    from hepattn.experiments.odd_pileup_maskformer.pflow_data import morton_encode

    window_sizes = sorted(set(window_sizes))
    max_half_w = max(window_sizes) // 2

    results = {w: {"mean_dR": [], "max_dR": [], "mean_deta": [], "mean_dphi": []} for w in window_sizes}

    n_sample = min(n_sample, dataset.num_events)
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(dataset.num_events, size=n_sample, replace=False)

    for evt_idx in sample_indices:
        t_start = int(dataset.track_cumsum[evt_idx])
        t_end = int(dataset.track_cumsum[evt_idx + 1])
        c_start = int(dataset.cluster_cumsum[evt_idx])
        c_end = int(dataset.cluster_cumsum[evt_idx + 1])

        eta = np.concatenate([
            dataset.full_data_array["track_eta"][t_start:t_end].numpy(),
            dataset.full_data_array["cluster_eta"][c_start:c_end].numpy(),
        ])
        phi = np.concatenate([
            dataset.full_data_array["track_phi"][t_start:t_end].numpy(),
            dataset.full_data_array["cluster_phi"][c_start:c_end].numpy(),
        ])
        n = len(eta)
        if n < 2:
            continue

        sort_idx = torch.argsort(
            morton_encode(torch.from_numpy(eta), torch.from_numpy(phi))
        ).numpy()
        eta = eta[sort_idx]
        phi = phi[sort_idx]

        half_w = min(max_half_w, n - 1)
        sum_dR = {w: 0.0 for w in window_sizes}
        sum_deta = {w: 0.0 for w in window_sizes}
        sum_dphi = {w: 0.0 for w in window_sizes}
        cnt_dR = {w: 0 for w in window_sizes}
        max_dR = {w: 0.0 for w in window_sizes}

        for d in range(1, half_w + 1):
            deta = eta[d:] - eta[:-d]
            dphi = phi[d:] - phi[:-d]
            dphi = np.arctan2(np.sin(dphi), np.cos(dphi))
            dR = np.sqrt(deta**2 + dphi**2)
            dR_sum = float(dR.sum())
            dR_max = float(dR.max())
            dR_cnt = len(dR)
            deta_sum = float(np.abs(deta).sum())
            dphi_sum = float(np.abs(dphi).sum())

            for w in window_sizes:
                if d <= w // 2:
                    sum_dR[w] += dR_sum
                    sum_deta[w] += deta_sum
                    sum_dphi[w] += dphi_sum
                    cnt_dR[w] += dR_cnt
                    if dR_max > max_dR[w]:
                        max_dR[w] = dR_max

        for w in window_sizes:
            if cnt_dR[w] > 0:
                results[w]["mean_dR"].append(sum_dR[w] / cnt_dR[w])
                results[w]["max_dR"].append(max_dR[w])
                results[w]["mean_deta"].append(sum_deta[w] / cnt_dR[w])
                results[w]["mean_dphi"].append(sum_dphi[w] / cnt_dR[w])

    return results


def reco_plots(
    reco_data: dict,
    truth_pt_bins: np.ndarray | list[float] = RECO_TRUTH_PT_BINS,
) -> dict[str, plt.Figure]:
    """Create reconstruction diagnostics binned in truth particle pt.

    Produces:
    - pt residual (pred-truth)/truth by truth-pt bin
    - eta residual (pred-truth) by truth-pt bin
    - phi residual (pred-truth, wrapped) by truth-pt bin
    - predicted vs truth class histogram by truth-pt bin
    - class count residual (N_pred - N_truth) by truth-pt bin
    """
    figs: dict[str, plt.Figure] = {}

    required = {
        "pred_class", "truth_class", "pred_valid", "truth_valid",
        "pred_pt", "truth_pt", "pred_eta", "truth_eta",
        "pred_sinphi", "pred_cosphi", "truth_sinphi", "truth_cosphi",
    }
    if not required.issubset(set(reco_data.keys())):
        return figs

    pt_edges = np.asarray(truth_pt_bins, dtype=np.float64)
    if pt_edges.ndim != 1 or len(pt_edges) < 2:
        raise ValueError("truth_pt_bins must be a 1D array with at least two edges")

    def _flat(name: str):
        return np.asarray(reco_data[name]).reshape(-1)

    pred_class = _flat("pred_class").astype(np.int64)
    truth_class = _flat("truth_class").astype(np.int64)
    pred_valid = _flat("pred_valid").astype(bool)
    truth_valid = _flat("truth_valid").astype(bool)

    pred_pt = _flat("pred_pt")
    truth_pt = _flat("truth_pt")
    pred_eta = _flat("pred_eta")
    truth_eta = _flat("truth_eta")
    pred_phi = np.arctan2(_flat("pred_sinphi"), _flat("pred_cosphi"))
    truth_phi = np.arctan2(_flat("truth_sinphi"), _flat("truth_cosphi"))

    finite = (
        np.isfinite(pred_pt) & np.isfinite(truth_pt)
        & np.isfinite(pred_eta) & np.isfinite(truth_eta)
        & np.isfinite(pred_phi) & np.isfinite(truth_phi)
    )
    base_mask = truth_valid & finite
    paired_mask = base_mask & pred_valid

    pt_res = (pred_pt - truth_pt) / np.clip(np.abs(truth_pt), 1e-8, None)
    eta_res = pred_eta - truth_eta
    phi_res = np.arctan2(np.sin(pred_phi - truth_phi), np.cos(pred_phi - truth_phi))

    def _pt_bin_label(i: int) -> str:
        lo = pt_edges[i]
        hi = pt_edges[i + 1]
        if np.isinf(hi):
            return f"[{lo:g}, inf)"
        return f"[{lo:g}, {hi:g})"

    def _in_pt_bin(i: int) -> np.ndarray:
        lo = pt_edges[i]
        hi = pt_edges[i + 1]
        mask = truth_pt >= lo
        if np.isfinite(hi):
            mask &= truth_pt < hi
        return mask

    def _plot_residual_grid(values: np.ndarray, title: str, xlabel: str, key: str):
        fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True)
        axes = axes.flatten()

        sample = values[paired_mask & np.isfinite(values)]
        if len(sample) > 10:
            lo, hi = np.percentile(sample, [0.5, 99.5])
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                lo, hi = float(sample.min()), float(sample.max())
            if lo == hi:
                lo -= 1.0
                hi += 1.0
            hist_range = (lo, hi)
        else:
            hist_range = None

        for i in range(len(pt_edges) - 1):
            ax = axes[i]
            mask = paired_mask & _in_pt_bin(i) & np.isfinite(values)
            if mask.any():
                vals = values[mask]
                q1, q2, q3 = np.percentile(vals, [25.0, 50.0, 75.0])
                iqr = q3 - q1
                mean = float(np.mean(vals))
                std = float(np.std(vals))

                ax.hist(vals, bins=80, range=hist_range, log=True, color="#1f77b4", alpha=0.85)
                ax.axvline(q1, color="#2ca02c", linestyle="--", linewidth=1.6, alpha=0.95)
                ax.axvline(q3, color="#2ca02c", linestyle="--", linewidth=1.6, alpha=0.95)
                ax.axvline(q2, color="#d62728", linestyle=":", linewidth=1.8, alpha=0.95)
                ax.axvline(mean, color="#9467bd", linestyle="-.", linewidth=1.8, alpha=0.95)
                ax.text(
                    0.03,
                    0.97,
                    f"mean={mean:.3g}\nstd={std:.3g}\nIQR={iqr:.3g}\nQ1={q1:.3g}\nQ3={q3:.3g}",
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=8,
                    bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
                )
            else:
                ax.text(0.5, 0.5, "no entries", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(_pt_bin_label(i), fontsize=10)
            ax.grid(True, alpha=0.25)

        fig.suptitle(title)
        fig.supxlabel(xlabel)
        fig.supylabel("Count (log scale)")
        fig.tight_layout()
        figs[key] = fig

    _plot_residual_grid(
        pt_res,
        "Reco pt residual by truth pt bin",
        "(pred_pt - truth_pt) / truth_pt",
        "reco/pt_residual_by_truth_pt",
    )
    _plot_residual_grid(
        eta_res,
        "Reco eta residual by truth pt bin",
        "pred_eta - truth_eta",
        "reco/eta_residual_by_truth_pt",
    )
    _plot_residual_grid(
        phi_res,
        "Reco phi residual by truth pt bin",
        "wrapped(pred_phi - truth_phi)",
        "reco/phi_residual_by_truth_pt",
    )

    # Predicted vs truth class histogram in each truth-pt bin.
    class_ids = np.arange(RECO_NUM_CLASSES)
    fig_cls, axes_cls = plt.subplots(2, 4, figsize=(20, 8), sharex=True, sharey=True)
    axes_cls = axes_cls.flatten()
    width = 0.42
    for i in range(len(pt_edges) - 1):
        ax = axes_cls[i]
        mask = base_mask & _in_pt_bin(i)
        if mask.any():
            truth_counts = np.bincount(np.clip(truth_class[mask], 0, RECO_NUM_CLASSES - 1), minlength=RECO_NUM_CLASSES)
            pred_counts = np.bincount(np.clip(pred_class[mask], 0, RECO_NUM_CLASSES - 1), minlength=RECO_NUM_CLASSES)
            ax.bar(class_ids - width / 2, truth_counts, width=width, label="truth", color="#4c72b0", alpha=0.85)
            ax.bar(class_ids + width / 2, pred_counts, width=width, label="pred", color="#dd8452", alpha=0.85)
            ax.set_yscale("log")
        else:
            ax.text(0.5, 0.5, "no entries", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(_pt_bin_label(i), fontsize=10)
        ax.set_xticks(class_ids)
        ax.grid(True, alpha=0.25)

    axes_cls[0].legend(loc="upper right")
    fig_cls.suptitle("Predicted vs truth particle class by truth pt bin")
    fig_cls.supxlabel("Particle class")
    fig_cls.supylabel("Count (log scale)")
    fig_cls.tight_layout()
    figs["reco/class_pred_vs_truth_by_truth_pt"] = fig_cls

    # Residual number of particles by class: N_pred - N_truth in each truth-pt bin.
    class_residual = np.zeros((len(pt_edges) - 1, RECO_NUM_CLASSES), dtype=np.int64)
    for i in range(len(pt_edges) - 1):
        mask = base_mask & _in_pt_bin(i)
        if mask.any():
            truth_counts = np.bincount(np.clip(truth_class[mask], 0, RECO_NUM_CLASSES - 1), minlength=RECO_NUM_CLASSES)
            pred_counts = np.bincount(np.clip(pred_class[mask], 0, RECO_NUM_CLASSES - 1), minlength=RECO_NUM_CLASSES)
            class_residual[i] = pred_counts - truth_counts

    vmax = int(np.max(np.abs(class_residual))) if class_residual.size else 1
    vmax = max(vmax, 1)
    fig_res, ax_res = plt.subplots(figsize=(12, 5))
    im = ax_res.imshow(class_residual, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    for i in range(class_residual.shape[0]):
        for j in range(class_residual.shape[1]):
            ax_res.text(j, i, f"{class_residual[i, j]:d}", ha="center", va="center", fontsize=9)
    ax_res.set_xticks(np.arange(RECO_NUM_CLASSES))
    ax_res.set_yticks(np.arange(len(pt_edges) - 1))
    ax_res.set_yticklabels([_pt_bin_label(i) for i in range(len(pt_edges) - 1)])
    ax_res.set_xlabel("Particle class")
    ax_res.set_ylabel("Truth pt bin")
    ax_res.set_title("Class count residual by truth pt bin (N_pred - N_truth)")
    cbar = fig_res.colorbar(im, ax=ax_res)
    cbar.set_label("N_pred - N_truth")
    fig_res.tight_layout()
    figs["reco/class_count_residual_by_truth_pt"] = fig_res

    # Truth-vs-pred kinematic histograms (all / charged / neutral)
    charged_mask = paired_mask & (truth_class < 3)
    neutral_mask = paired_mask & (truth_class >= 3) & (truth_class < (RECO_NUM_CLASSES - 1))

    def _finite_percentile_range(a: np.ndarray, b: np.ndarray, default: tuple[float, float]) -> tuple[float, float]:
        vals = np.concatenate([a, b])
        vals = vals[np.isfinite(vals)]
        if len(vals) < 2:
            return default
        lo, hi = np.percentile(vals, [0.5, 99.5])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = float(vals.min()), float(vals.max())
        if lo == hi:
            lo -= 1.0
            hi += 1.0
        return lo, hi

    def _plot_truth_vs_pred_kinematics(mask: np.ndarray, title: str, key: str) -> None:
        if not np.any(mask):
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        t_pt = truth_pt[mask]
        p_pt = pred_pt[mask]
        pt_vals = np.concatenate([t_pt, p_pt])
        pt_vals = pt_vals[np.isfinite(pt_vals) & (pt_vals > 0)]
        if len(pt_vals) >= 2:
            pt_lo, pt_hi = np.percentile(pt_vals, [0.5, 99.5])
            pt_lo = max(float(pt_lo), 1e-4)
            pt_hi = max(float(pt_hi), pt_lo * 1.1)
            pt_bins = np.logspace(np.log10(pt_lo), np.log10(pt_hi), 80)
        else:
            pt_bins = 80

        eta_lo, eta_hi = _finite_percentile_range(truth_eta[mask], pred_eta[mask], (-5.0, 5.0))
        eta_bins = np.linspace(eta_lo, eta_hi, 80)
        phi_bins = np.linspace(-np.pi, np.pi, 80)

        axes[0].hist(t_pt, bins=pt_bins, histtype="step", linewidth=2.0, label="truth", color="#4c72b0")
        axes[0].hist(p_pt, bins=pt_bins, histtype="step", linewidth=2.0, label="pred", color="#dd8452")
        if not np.isscalar(pt_bins):
            axes[0].set_xscale("log")
        axes[0].set_yscale("log")
        axes[0].set_title("pt")
        axes[0].set_xlabel("pt")
        axes[0].set_ylabel("Count (log scale)")
        axes[0].grid(True, alpha=0.25)

        axes[1].hist(truth_eta[mask], bins=eta_bins, histtype="step", linewidth=2.0, label="truth", color="#4c72b0")
        axes[1].hist(pred_eta[mask], bins=eta_bins, histtype="step", linewidth=2.0, label="pred", color="#dd8452")
        axes[1].set_yscale("log")
        axes[1].set_title("eta")
        axes[1].set_xlabel("eta")
        axes[1].grid(True, alpha=0.25)

        axes[2].hist(truth_phi[mask], bins=phi_bins, histtype="step", linewidth=2.0, label="truth", color="#4c72b0")
        axes[2].hist(pred_phi[mask], bins=phi_bins, histtype="step", linewidth=2.0, label="pred", color="#dd8452")
        axes[2].set_yscale("log")
        axes[2].set_title("phi")
        axes[2].set_xlabel("phi")
        axes[2].grid(True, alpha=0.25)

        axes[0].legend(loc="best")
        fig.suptitle(title)
        fig.tight_layout()
        figs[key] = fig

    _plot_truth_vs_pred_kinematics(
        paired_mask,
        "Reco kinematics: truth vs pred (all valid particles)",
        "reco/kinematics_truth_vs_pred_all",
    )
    _plot_truth_vs_pred_kinematics(
        charged_mask,
        "Reco kinematics: truth vs pred (charged particles)",
        "reco/kinematics_truth_vs_pred_charged",
    )
    _plot_truth_vs_pred_kinematics(
        neutral_mask,
        "Reco kinematics: truth vs pred (neutral particles)",
        "reco/kinematics_truth_vs_pred_neutral",
    )

    return figs


def make_plots(
    track_data: dict,
    cluster_data: dict,
    reco_data: dict | None = None,
    truth_pt_bins: np.ndarray | list[float] = RECO_TRUTH_PT_BINS,
) -> dict[str, plt.Figure]:
    """Reproduce all validation plots from ODDPFlowTwoStream."""
    import time
    from hepattn.experiments.odd_pileup_maskformer.plots import PhysicsPlotter

    figs = {}

    def _plot(name, fn, *args, **kwargs):
        t0 = time.perf_counter()
        figs[name] = fn(*args, **kwargs)
        print(f"  {name:<40s} {time.perf_counter() - t0:.2f}s")

    # ── Calo energy composition (truth only) ──
    if cluster_data.get("neutral_e") is not None and len(cluster_data.get("neutral_e", [])) > 0:
        _plot("calo/neutral_energy_frac", PhysicsPlotter.plot_calo_neutral_energy_frac,
              cluster_data["neutral_e"], cluster_data["total_e"])
        _plot("calo/charged_energy_frac", PhysicsPlotter.plot_calo_charged_energy_frac,
              cluster_data["charged_e"], cluster_data["total_e"])
        _plot("calo/cluster_energy_dist", PhysicsPlotter.plot_calo_cluster_energy_dist,
              cluster_data["total_e"])

    # ── Calo fraction plots ──
    if cluster_data.get("pred_frac") is not None and len(cluster_data.get("pred_frac", [])) > 0:
        _plot("calo/energy_corr", PhysicsPlotter.plot_energy_correlation,
              cluster_data["pred_frac"], cluster_data["total_e"], cluster_data["true_hs_e"])
        _plot("calo/frac_corr", PhysicsPlotter.plot_calo_frac_correlation,
              cluster_data["pred_frac"], cluster_data["true_frac"])
        _plot("calo/frac_dist", PhysicsPlotter.plot_calo_frac_distribution,
              cluster_data["pred_frac"], cluster_data["true_frac"])
        _plot("calo/energy_resid", PhysicsPlotter.plot_energy_residual,
              cluster_data["pred_frac"], cluster_data["total_e"], cluster_data["true_hs_e"])
        _plot("calo/hs_energy_dist", PhysicsPlotter.plot_hs_energy_distribution,
              cluster_data["pred_frac"], cluster_data["total_e"], cluster_data["true_hs_e"])
        if cluster_data.get("event_idx") is not None:
            _plot("calo/event_hs_energy_ratio", PhysicsPlotter.plot_calo_event_hs_energy_ratio,
                  cluster_data["pred_frac"], cluster_data["total_e"],
                  cluster_data["true_hs_e"], cluster_data["event_idx"])

    # ── Per-event mask energy ratio (binary mask, no fraction regression) ──
    if cluster_data.get("evt_pred_mask_e") is not None and len(cluster_data.get("evt_pred_mask_e", [])) > 0:
        _plot("calo/mask_energy_ratio", PhysicsPlotter.plot_calo_mask_energy_ratio,
              cluster_data["mask_pred"], cluster_data["mask_truth"],
              cluster_data["total_e"], cluster_data["true_hs_e"], cluster_data["event_idx"])

    # ── Per-event neutral/charged HS energy histograms ──
    if cluster_data.get("evt_pred_neutral_e") is not None and len(cluster_data.get("evt_pred_neutral_e", [])) > 0:
        _plot("calo/hs_energy_residual_by_type", PhysicsPlotter.plot_hs_energy_residual_by_type,
              cluster_data["evt_pred_neutral_e"], cluster_data["evt_truth_neutral_e"],
              cluster_data["evt_pred_charged_e"], cluster_data["evt_truth_charged_e"])
        _plot("calo/hs_energy_ratio_by_type", PhysicsPlotter.plot_hs_energy_ratio_by_type,
              cluster_data["evt_pred_neutral_e"], cluster_data["evt_truth_neutral_e"],
              cluster_data["evt_pred_charged_e"], cluster_data["evt_truth_charged_e"])

    # ── Calo mask F1 vs threshold ──
    if cluster_data.get("calo_mask_probs") is not None and len(cluster_data.get("calo_mask_probs", [])) > 0:
        _plot("calo/mask_f1_vs_threshold", PhysicsPlotter.plot_calo_mask_f1_vs_threshold,
              cluster_data["calo_mask_probs"], cluster_data["mask_truth"], cluster_data["total_e"])
        if cluster_data.get("true_frac") is not None:
            _plot("calo/mask_f1_vs_threshold_by_hs_frac", PhysicsPlotter.plot_calo_mask_f1_vs_threshold_by_hs_frac,
                  cluster_data["calo_mask_probs"], cluster_data["mask_truth"], cluster_data["true_frac"])

        # ── F1 & recall split by neutral vs charged cluster type ──
        if cluster_data.get("neutral_e") is not None:
            _plot("calo/mask_f1_recall_neutral_clusters", PhysicsPlotter.plot_calo_mask_f1_recall_neutral,
                  cluster_data["calo_mask_probs"], cluster_data["mask_truth"],
                  cluster_data["neutral_e"], cluster_data["charged_e"])
            _plot("calo/mask_f1_recall_charged_clusters", PhysicsPlotter.plot_calo_mask_f1_recall_charged,
                  cluster_data["calo_mask_probs"], cluster_data["mask_truth"],
                  cluster_data["neutral_e"], cluster_data["charged_e"])

    # ── HS fraction category counts ──
    if cluster_data.get("true_frac") is not None and len(cluster_data.get("true_frac", [])) > 0:
        _plot("calo/hs_frac_category_counts", PhysicsPlotter.plot_calo_hs_frac_category_counts,
              cluster_data["true_frac"])

    # ── Cluster swap ΔR ──
    if cluster_data.get("mask_pred") is not None and len(cluster_data.get("mask_pred", [])) > 0:
        is_fn = cluster_data["mask_truth"].astype(bool) & ~cluster_data["mask_pred"].astype(bool)
        is_fp = ~cluster_data["mask_truth"].astype(bool) & cluster_data["mask_pred"].astype(bool)
        if cluster_data.get("pred_frac") is not None and len(cluster_data.get("pred_frac", [])) > 0:
            t0 = time.perf_counter()
            for bin_key, bin_fig in PhysicsPlotter.plot_calo_mask_errors_by_energy(
                cluster_data["mask_pred"], cluster_data["mask_truth"],
                cluster_data["total_e"], cluster_data["true_hs_e"], cluster_data["pred_frac"],
            ).items():
                figs[f"calo/mask_errors_by_energy/{bin_key}"] = bin_fig
            print(f"  {'calo/mask_errors_by_energy':<40s} {time.perf_counter() - t0:.2f}s")
        else:
            print("  calo/mask_errors_by_energy               skipped (pred_frac unavailable)")
        _plot("calo/mistag_eta", PhysicsPlotter.plot_calo_mistag_vs_eta,
              cluster_data["mask_pred"], cluster_data["mask_truth"], cluster_data["eta"])
        _plot("calo/mask_metrics_vs_eta", PhysicsPlotter.plot_calo_mask_metrics_vs_eta,
              cluster_data["mask_pred"], cluster_data["mask_truth"], cluster_data["eta"])
        if cluster_data.get("phi") is not None:
            _plot("calo/mask_metrics_vs_phi", PhysicsPlotter.plot_calo_mask_metrics_vs_phi,
                  cluster_data["mask_pred"], cluster_data["mask_truth"], cluster_data["phi"])
        if cluster_data.get("total_e") is not None:
            _plot("calo/mask_metrics_vs_energy", PhysicsPlotter.plot_calo_mask_metrics_vs_cluster_energy,
                  cluster_data["mask_pred"], cluster_data["mask_truth"], cluster_data["total_e"])
        if cluster_data.get("true_frac") is not None:
            _plot("calo/mask_metrics_vs_hs_frac", PhysicsPlotter.plot_calo_mask_metrics_vs_hs_frac,
                  cluster_data["mask_pred"], cluster_data["mask_truth"], cluster_data["true_frac"])

    # ── Track plots ──
    if track_data.get("probs") is not None and len(track_data.get("probs", [])) > 0:
        _plot("track/score_dist", PhysicsPlotter.plot_track_score_distribution,
              track_data["probs"], track_data["truth"])
        _plot("track/eff_vs_pt", PhysicsPlotter.plot_track_efficiency_vs_pt,
              track_data["probs"], track_data["truth"], track_data["pt"])
        _plot("track/f1_vs_pt", PhysicsPlotter.plot_track_f1_vs_pt,
              track_data["probs"], track_data["truth"], track_data["pt"])
        _plot("track/rej_vs_pt", PhysicsPlotter.plot_pileup_rejection_vs_pt,
              track_data["probs"], track_data["truth"], track_data["pt"])
        _plot("track/mistag_eta", PhysicsPlotter.plot_mistag_rate_vs_eta,
              track_data["probs"], track_data["truth"], track_data["eta"])
        _plot("track/score_by_pt", PhysicsPlotter.plot_track_score_by_pt,
              track_data["probs"], track_data["truth"], track_data["pt"])
        _plot("track/f1_vs_threshold", PhysicsPlotter.plot_track_f1_vs_threshold,
              track_data["probs"], track_data["truth"], track_data["pt"])
        _plot("track/roc", PhysicsPlotter.plot_roc_curve,
              track_data["probs"], track_data["truth"])
        if track_data.get("z0") is not None and len(track_data.get("z0", [])) > 0:
            _plot("track/z0_dist", PhysicsPlotter.plot_track_z0_distribution,
                  track_data["probs"], track_data["truth"], track_data["z0"])
        _plot("track/pt_dist", PhysicsPlotter.plot_track_pt_distribution,
              track_data["probs"], track_data["truth"], track_data["pt"])

    if reco_data is not None:
        figs.update(reco_plots(reco_data, truth_pt_bins=truth_pt_bins))

    return figs


def _print_data_summary(track_data: dict, cluster_data: dict, reco_data: dict) -> None:
    track_count = len(track_data.get("probs", []))
    cluster_count = len(cluster_data.get("total_e", []))
    reco_count = int(np.asarray(
        reco_data.get("truth_pt", reco_data.get("truth_class", reco_data.get("pred_class", [])))
    ).size)
    print(f"  Tracks: {track_count} nodes")
    print(f"  Clusters: {cluster_count} nodes")
    print(f"  Reco objects: {reco_count} slots")


def load_eval(
    h5_path: str | Path | None = None,
    *,
    ckpt_path: str | Path | None = None,
    config_path: str | Path = CONFIG_PATH,
    data_dir: str | None = None,
    files: list[str | Path] | None = None,
    num_events: int = -1,
    batch_size: int = 48,
    num_workers: int = 1,
    accelerator: str = "auto",
) -> tuple[dict, dict, dict]:
    """Load evaluation data from H5 file, or run forward pass to generate one.

    Args:
        h5_path: Path to an existing prediction writer H5 file. If given,
            loads directly — no model or GPU needed.
        ckpt_path: Path to model checkpoint (used if h5_path is None).
        config_path: Path to YAML config (used if h5_path is None).
        data_dir: Directory with parquet files (used if h5_path is None).
        files: Explicit list of parquet files (used if h5_path is None).
        num_events: Number of events to load (-1 for all).
        batch_size: Batch size for forward pass.
        num_workers: DataLoader workers.
        accelerator: Lightning accelerator ("auto", "gpu", "cpu").

    Returns:
        (track_data, cluster_data, reco_data) dicts ready for ``make_plots()``.
    """
    if h5_path is not None:
        print(f"Loading evaluation data from {h5_path}")
        track_data, cluster_data, reco_data = load_eval_data_from_h5(h5_path)
    else:
        if ckpt_path is None:
            raise ValueError("Provide either h5_path (existing H5) or ckpt_path (for forward pass).")
        print("Running forward pass to generate predictions H5...")
        generated_h5 = run_forward_pass(
            ckpt_path=ckpt_path,
            config_path=config_path,
            data_dir=data_dir,
            files=files,
            num_events=num_events,
            batch_size=batch_size,
            num_workers=num_workers,
            accelerator=accelerator,
        )
        print(f"Loading evaluation data from {generated_h5}")
        track_data, cluster_data, reco_data = load_eval_data_from_h5(generated_h5)

    _print_data_summary(track_data, cluster_data, reco_data)
    return track_data, cluster_data, reco_data


def run_eval(
    h5_path: str | Path | None = None,
    *,
    ckpt_path: str | Path | None = None,
    config_path: str | Path = CONFIG_PATH,
    data_dir: str | None = None,
    files: list[str | Path] | None = None,
    num_events: int = -1,
    batch_size: int = 48,
    num_workers: int = 1,
    accelerator: str = "auto",
    truth_pt_bins: np.ndarray | list[float] = RECO_TRUTH_PT_BINS,
    reco_analysis: bool = False,
    reco_analysis_do_jets: bool = True,
    reco_analysis_event_indices: list[int] | None = None,
) -> tuple[dict[str, plt.Figure], dict, dict, dict]:
    """Full eval pipeline: load data + generate plots.

    Args:
        h5_path: Path to existing prediction writer H5 (no model/GPU needed if given).
        ckpt_path: Model checkpoint path (used when h5_path is None).
        config_path: YAML config path (used when h5_path is None).
        data_dir: Parquet data directory (used when h5_path is None).
        files: Explicit parquet file list (used when h5_path is None).
        num_events: Events to process (-1 for all).
        batch_size: Forward-pass batch size.
        num_workers: DataLoader worker count.
        accelerator: Lightning accelerator ("auto", "gpu", "cpu").
        truth_pt_bins: pt bin edges for efficiency/purity curves.
        reco_analysis: Also run particle-level reco analysis (``reco_analysis.py``).
        reco_analysis_do_jets: Enable FastJet clustering in reco analysis (requires fastjet).
        reco_analysis_event_indices: Event indices for event displays in reco analysis.

    Returns:
        (figs, track_data, cluster_data, reco_data)
    """
    track_data, cluster_data, reco_data = load_eval(
        h5_path=h5_path,
        ckpt_path=ckpt_path,
        config_path=config_path,
        data_dir=data_dir,
        files=files,
        num_events=num_events,
        batch_size=batch_size,
        num_workers=num_workers,
        accelerator=accelerator,
    )

    print("Making plots...")
    figs = make_plots(
        track_data,
        cluster_data,
        reco_data=reco_data,
        truth_pt_bins=truth_pt_bins,
    )
    print(f"  Generated {len(figs)} plots: {list(figs.keys())}")

    if reco_analysis and reco_data:
        from hepattn.experiments.odd_pileup_reco_pu_cond.reco_analysis import (
            pflow_data_from_eval_dicts,
            run_reco_analysis,
        )
        print("Running particle-level reco analysis...")
        pflow_data = pflow_data_from_eval_dicts(reco_data)
        reco_figs = run_reco_analysis(
            pflow_data,
            do_jets=reco_analysis_do_jets,
            event_display_indices=reco_analysis_event_indices,
        )
        figs.update(reco_figs)
        print(f"  Total figures after reco analysis: {len(figs)}")

    return figs, track_data, cluster_data, reco_data


def make_data_plots(dataset_or_loader) -> dict[str, plt.Figure]:
    """Plots that need only the raw dataset — no model inference required."""
    from torch.utils.data import DataLoader

    import time
    from hepattn.experiments.odd_pileup_maskformer.plots import PhysicsPlotter

    dataset = dataset_or_loader.dataset if isinstance(dataset_or_loader, DataLoader) else dataset_or_loader

    figs = {}

    def _plot(name, fn, *args, **kwargs):
        t0 = time.perf_counter()
        figs[name] = fn(*args, **kwargs)
        print(f"  {name:<40s} {time.perf_counter() - t0:.2f}s")

    stats = compute_deltaR_window_stats(dataset, window_sizes=[32, 64, 80, 128, 256], n_sample=200)
    _plot("data/deltaR_window_analysis", PhysicsPlotter.plot_deltaR_window_analysis, stats)
    _plot("data/eta_window_analysis",    PhysicsPlotter.plot_eta_window_analysis,    stats)
    _plot("data/phi_window_analysis",    PhysicsPlotter.plot_phi_window_analysis,    stats)

    # ── Truth particle class histogram (including null/pileup) ──
    # Shows two bars per class: raw PDG class vs. after trackless reclassification.
    # Trackless routing: charged hadron (0) → neutral hadron (3),
    #                    electron (1) → photon (4), muon (2) → neutral hadron (3).
    if hasattr(dataset, "full_data_array") and "particle_class" in dataset.full_data_array:
        t0 = time.perf_counter()
        try:
            num_objects = getattr(dataset, "num_objects", 400)

            classes_raw = dataset.full_data_array["particle_class"].numpy().copy()

            # Apply trackless reclassification using particle_has_track
            classes_routed = classes_raw.copy()
            if "particle_has_track" in dataset.full_data_array:
                has_track = dataset.full_data_array["particle_has_track"].numpy().astype(bool)
                trackless = ~has_track
                trackless_chhad_e = trackless & (classes_routed < 2)   # cls 0→3, cls 1→4
                trackless_muon    = trackless & (classes_routed == 2)   # cls 2→3
                classes_routed[trackless_chhad_e] += 3
                classes_routed[trackless_muon] = 3

            # Null slot counts (same for both: null is not affected by trackless routing)
            n_particles_per_event = np.diff(dataset.particle_cumsum).astype(np.int64)
            n_null = int(np.maximum(0, num_objects - 1 - n_particles_per_event).sum())
            n_null += dataset.num_events  # pileup tokens (1 per event)

            counts_raw    = np.bincount(np.clip(classes_raw,    0, 5), minlength=6)
            counts_routed = np.bincount(np.clip(classes_routed, 0, 5), minlength=6)
            counts_raw[5]    += n_null
            counts_routed[5] += n_null

            class_names = [
                "Charged Hadron\n(cls 0)",
                "Electron\n(cls 1)",
                "Muon\n(cls 2)",
                "Neutral Hadron\n(cls 3)",
                "Photon\n(cls 4)",
                "Null / Pileup\n(cls 5)",
            ]
            colors_raw    = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b2", "#937860"]
            colors_routed = ["#a8c4e0", "#f0c4a0", "#a8d4b4", "#e8a4a4", "#c4b8d8", "#c8b8a8"]

            x = np.arange(6)
            w = 0.42
            fig_cls, ax_cls = plt.subplots(figsize=(11, 5))

            bars_raw    = ax_cls.bar(x - w/2, counts_raw,    width=w, color=colors_raw,
                                     edgecolor="white", linewidth=0.5, alpha=0.95, label="Raw PDG class")
            bars_routed = ax_cls.bar(x + w/2, counts_routed, width=w, color=colors_routed,
                                     edgecolor="grey",  linewidth=0.5, alpha=0.95, label="After trackless routing")

            for bar, count in zip(bars_raw, counts_raw):
                if count > 0:
                    ax_cls.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.08,
                                f"{count:,}", ha="center", va="bottom", fontsize=7, color="#333333")
            for bar, count in zip(bars_routed, counts_routed):
                if count > 0:
                    ax_cls.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.08,
                                f"{count:,}", ha="center", va="bottom", fontsize=7, color="#555555")

            ax_cls.set_xticks(x)
            ax_cls.set_xticklabels(class_names, fontsize=10)
            ax_cls.set_ylabel("Count")
            ax_cls.set_title(
                f"Truth particle class distribution  |  {dataset.num_events} events"
                f"  |  {len(classes_raw):,} real particles  |  {num_objects} query slots"
            )
            ax_cls.set_yscale("log")
            ax_cls.grid(True, alpha=0.25, axis="y")
            ax_cls.legend(loc="upper right")
            fig_cls.tight_layout()
            figs["data/truth_particle_class_dist"] = fig_cls
            print(f"  {'data/truth_particle_class_dist':<40s} {time.perf_counter() - t0:.2f}s")
        except Exception as exc:
            print(f"  data/truth_particle_class_dist skipped: {exc}")

    # ── HS mask energy efficiency vs threshold ──
    if ("deps_hard_scatter_energy_deps_in_cluster" in dataset.full_data_array
            and "deps_cluster_idx" in dataset.full_data_array
            and "total_cluster_energy" in dataset.full_data_array
            and hasattr(dataset, "deps_cumsum")):
        n_clusters_total = int(dataset.cluster_cumsum[-1])
        true_hs_e = np.zeros(n_clusters_total, dtype=np.float32)
        d_energy = dataset.full_data_array["deps_hard_scatter_energy_deps_in_cluster"].numpy()
        d_cluster_idx_local = dataset.full_data_array["deps_cluster_idx"].numpy().astype(int)
        for evt_idx in range(dataset.num_events):
            d_start = int(dataset.deps_cumsum[evt_idx])
            d_end = int(dataset.deps_cumsum[evt_idx + 1])
            c_start = int(dataset.cluster_cumsum[evt_idx])
            if d_end > d_start:
                global_idx = d_cluster_idx_local[d_start:d_end] + c_start
                np.add.at(true_hs_e, global_idx, d_energy[d_start:d_end])
        total_cluster_e = dataset.full_data_array["total_cluster_energy"].numpy()
        true_frac = true_hs_e / (total_cluster_e + 1e-6)
        _plot("data/mask_hs_energy_efficiency", PhysicsPlotter.plot_mask_hs_energy_efficiency,
              true_hs_e, true_frac)

    return figs


def run_data_plots(
    config_path: str | Path = CONFIG_PATH,
    files: list[str | Path] | None = None,
    data_dir: str | None = None,
    num_events: int = -1,
    num_workers: int = 1,
) -> dict[str, plt.Figure]:
    """Generate all data-only plots without loading a model."""
    import yaml
    from hepattn.experiments.odd_pileup_reco_pu_cond.pflow_data import ODDDatasetPileup

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]

    if files is not None:
        files_list = [Path(f) for f in files]
        filepath = str(files_list[0].parent)
    elif data_dir is not None:
        files_list = None
        filepath = data_dir
    else:
        raise ValueError("Provide either files or data_dir.")

    dataset = ODDDatasetPileup(
        filepath=filepath,
        inputs=data_cfg["inputs"],
        targets=data_cfg["targets"],
        scale_dict_path=data_cfg["scale_dict_path"],
        num_events=num_events,
        max_nodes=data_cfg["max_nodes"],
        incidence_cutval=data_cfg.get("incidence_cutval", 0.01),
        hard_scatter_energy_threshold=data_cfg.get("hard_scatter_energy_threshold", 0.03),
        window_size=data_cfg.get("window_size", 512),
        files_list=files_list,
        is_inference=False,
    )
    return make_data_plots(dataset)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Usage: python eval_plots.py <h5_path> [output_dir]
        h5_file = sys.argv[1]
        out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(h5_file).parent.parent / "eval_plots"
    else:
        print("Usage: python eval_plots.py <h5_path> [output_dir]")
        sys.exit(1)

    figs, track_data, cluster_data, reco_data = run_eval(h5_path=h5_file)

    out_dir.mkdir(exist_ok=True)
    for name, fig in figs.items():
        fname = name.replace("/", "_") + ".png"
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved {len(figs)} plots to {out_dir}")
