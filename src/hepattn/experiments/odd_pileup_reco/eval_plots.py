"""
Offline evaluation script for the Two-Stream MaskFormer.

Loads a checkpoint and data files, runs inference, and produces
all validation plots from ODDPFlowTwoStream.on_validation_epoch_end().

Usage (standalone):
    python eval_plots.py

Usage (notebook):
    from hepattn.experiments.odd_pileup_reco.eval_plots import run_eval

    # Specific files
    figs, track, cluster = run_eval(files=[
        "/storage/agrp/barakma/PileupODD/data/ttbar_pu200/target_particles-0042.parquet",
        "/storage/agrp/barakma/PileupODD/data/ttbar_pu200/target_particles-0043.parquet",
    ])

    # Or all files in a directory
    figs, track, cluster = run_eval(data_dir="/storage/agrp/barakma/PileupODD/data/ttbar_pu200")
"""

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# ── Configuration ──────────────────────────────────────────────────────────
CKPT_PATH = "/storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco/logs/odd_pflow_reco_20260331-T135040/ckpts/epoch=008-val_loss=25.69795.ckpt"
CONFIG_PATH = "/storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco/configs/base.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 48
RECO_TRUTH_PT_BINS = np.array([0.0, 0.2, 0.5, 0.9, 2.0, 4.0, 10.0, 20.0, np.inf], dtype=np.float64)
RECO_NUM_CLASSES = 6
CALO_PRED_THRESHOLD = 0.2
CALO_HS_FRAC_THRESHOLD = 0.05
CALO_HS_ENERGY_THRESHOLD = 0.15


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
        t_end   = int(dataset.track_cumsum[evt_idx + 1])
        c_start = int(dataset.cluster_cumsum[evt_idx])
        c_end   = int(dataset.cluster_cumsum[evt_idx + 1])

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
        sum_dR   = {w: 0.0 for w in window_sizes}
        sum_deta = {w: 0.0 for w in window_sizes}
        sum_dphi = {w: 0.0 for w in window_sizes}
        cnt_dR   = {w: 0   for w in window_sizes}
        max_dR   = {w: 0.0 for w in window_sizes}

        for d in range(1, half_w + 1):
            deta = eta[d:] - eta[:-d]
            dphi = phi[d:] - phi[:-d]
            dphi = np.arctan2(np.sin(dphi), np.cos(dphi))
            dR = np.sqrt(deta**2 + dphi**2)
            dR_sum   = float(dR.sum())
            dR_max   = float(dR.max())
            dR_cnt   = len(dR)
            deta_sum = float(np.abs(deta).sum())
            dphi_sum = float(np.abs(dphi).sum())

            for w in window_sizes:
                if d <= w // 2:
                    sum_dR[w]   += dR_sum
                    sum_deta[w] += deta_sum
                    sum_dphi[w] += dphi_sum
                    cnt_dR[w]   += dR_cnt
                    if dR_max > max_dR[w]:
                        max_dR[w] = dR_max

        for w in window_sizes:
            if cnt_dR[w] > 0:
                results[w]["mean_dR"].append(sum_dR[w]   / cnt_dR[w])
                results[w]["max_dR"].append(max_dR[w])
                results[w]["mean_deta"].append(sum_deta[w] / cnt_dR[w])
                results[w]["mean_dphi"].append(sum_dphi[w] / cnt_dR[w])

    return results


def load_config(config_path: str = CONFIG_PATH) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_dataloader(
    cfg: dict,
    files: list[str | Path] | None = None,
    data_dir: str | None = None,
    num_events: int = -1,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 1,
) -> DataLoader:
    """Create a DataLoader directly from a list of parquet files or a directory.

    Args:
        cfg: Parsed YAML config dict.
        files: Explicit list of parquet file paths. Takes priority over data_dir.
        data_dir: Directory to glob for parquet files (used if files is None).
        num_events: Number of events to load (-1 for all).
        batch_size: Batch size.
        num_workers: DataLoader workers.
    """
    from hepattn.experiments.odd_pileup_reco.pflow_data import ODDDatasetPileup

    data_cfg = cfg["data"]

    if files is not None:
        files_list = [Path(f) for f in files]
        filepath = str(files_list[0].parent)
    elif data_dir is not None:
        files_list = None
        filepath = data_dir
    else:
        raise ValueError("Provide either `files` (list of parquet paths) or `data_dir`.")

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
    print(f"Dataset: {len(dataset)} events from {len(files_list) if files_list else 'dir'} file(s)")

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def load_model(ckpt_path: str = CKPT_PATH, device: str = DEVICE):
    from hepattn.experiments.odd_pileup_reco.lightning_module import ODDPFlowTwoStream

    model = ODDPFlowTwoStream.load_from_checkpoint(ckpt_path, map_location=device)
    model.eval()
    model.to(device)
    return model


def collect_predictions(model, dataloader, device: str = DEVICE, max_batches: int | None = None):
    """Run inference on dataloader and accumulate data for plots.

    Returns (track_data, cluster_data, reco_data) dicts with numpy arrays.
    """
    track_data = defaultdict(list)
    cluster_data = defaultdict(list)
    reco_data = defaultdict(list)
    event_counter = 0
    warned_no_cluster_nodes = False
    warned_no_reco_head = False
    warned_no_reco_truth = False
    var_transform = getattr(getattr(dataloader.dataset, "scaler", None), "transforms", {})

    def _inverse_if_needed(x: torch.Tensor, field: str) -> torch.Tensor:
        y = x.detach().cpu().float().unsqueeze(-1)
        if field in var_transform:
            y = var_transform[field].inverse_transform(y)
        return y.squeeze(-1)

    with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16):
        for i, batch in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break

            inputs, targets = batch
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            targets = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in targets.items()}

            outputs = model.model(inputs, targets=targets)
            preds = model.model.predict(outputs)

            labels = targets
            node_valid = labels["node_valid"].bool()
            is_track = labels["node_is_track"].bool().squeeze(-1)

            # ── Track data (Stream A) ──
            track_final = preds.get("track_final", {})
            track_node_mask = node_valid & is_track

            if track_node_mask.any() and "mask" in track_final:
                track_prob = track_final["mask"]["pflow_node_prob"].squeeze(-2)[track_node_mask]
                track_truth = labels["tracks_mask"][track_node_mask].int()

                track_data["probs"].append(track_prob.float().cpu().numpy())
                track_data["truth"].append(track_truth.cpu().numpy())
                track_data["pt"].append(labels["node_pt"][track_node_mask].float().cpu().numpy())
                track_data["eta"].append(labels["node_eta"][track_node_mask].float().cpu().numpy())
                track_data["z0"].append(labels["node_z0"][track_node_mask].float().cpu().numpy())

            # ── Cluster data (Stream B) ──
            calo_final = preds.get("calo_final", {})
            cluster_node_mask = node_valid & (~is_track)

            if (not cluster_node_mask.any()) and (not warned_no_cluster_nodes):
                print("Warning: no cluster nodes in batch; cluster diagnostics may stay empty.")
                warned_no_cluster_nodes = True

            if cluster_node_mask.any():
                node_e = labels["node_e"][cluster_node_mask]
                true_hs_energy = labels["calo_hard_scatter_energy"][cluster_node_mask]

                cluster_data["total_e"].append(node_e.float().cpu().numpy())
                cluster_data["true_hs_e"].append(true_hs_energy.float().cpu().numpy())
                cluster_data["eta"].append(labels["node_eta"][cluster_node_mask].float().cpu().numpy())
                cluster_data["phi"].append(labels["node_phi"][cluster_node_mask].float().cpu().numpy())

                # Fraction regression may be absent in some checkpoints.
                if "calo_fraction" in calo_final and "calo_hs_fraction" in calo_final["calo_fraction"]:
                    calo_frac_pred = calo_final["calo_fraction"]["calo_hs_fraction"].squeeze(-1)[cluster_node_mask]
                    calo_frac_true = labels["calo_hard_scatter_energy_frac"][cluster_node_mask]
                    cluster_data["pred_frac"].append(calo_frac_pred.float().cpu().numpy())
                    cluster_data["true_frac"].append(calo_frac_true.float().cpu().numpy())

                # Truth neutral/charged energy per cluster (for composition plot)
                if "calo_hs_neutral_energy" in labels:
                    cluster_data["neutral_e"].append(labels["calo_hs_neutral_energy"][cluster_node_mask].float().cpu().numpy())
                    cluster_data["charged_e"].append(labels["calo_hs_charged_energy"][cluster_node_mask].float().cpu().numpy())

                # Calo mask predictions for cluster swap plot
                if "calo_mask" in calo_final and "calo_node_prob" in calo_final["calo_mask"]:
                    calo_prob_flat = calo_final["calo_mask"]["calo_node_prob"][cluster_node_mask]
                    calo_hs_e_flat = labels["calo_hard_scatter_energy"][cluster_node_mask]
                    cluster_data["mask_pred"].append((calo_prob_flat > CALO_PRED_THRESHOLD).cpu().numpy())
                    cluster_data["calo_mask_probs"].append(calo_prob_flat.float().cpu().numpy())
                    calo_hs_frac_flat = labels["calo_hard_scatter_energy_frac"][cluster_node_mask]
                    cluster_data["mask_truth"].append(
                        ((calo_hs_frac_flat > CALO_HS_FRAC_THRESHOLD) & (calo_hs_e_flat > CALO_HS_ENERGY_THRESHOLD)).cpu().numpy()
                    )

                # Per-event indices
                counts = cluster_node_mask.sum(dim=-1).cpu().numpy()
                event_indices = np.repeat(
                    np.arange(event_counter, event_counter + len(counts)),
                    counts,
                )
                cluster_data["event_idx"].append(event_indices)
                event_counter += len(counts)

                # Per-event mask energy sums (binary mask, no fraction regression)
                if "calo_mask" in calo_final and "calo_node_prob" in calo_final["calo_mask"]:
                    pred_mask_b = (calo_final["calo_mask"]["calo_node_prob"] > CALO_PRED_THRESHOLD).float()
                    truth_mask_b = (
                        (labels["calo_hard_scatter_energy_frac"] > CALO_HS_FRAC_THRESHOLD)
                        & (labels["calo_hard_scatter_energy"] > CALO_HS_ENERGY_THRESHOLD)
                    ).float()
                    node_e_b = labels["node_e"]
                    cluster_valid = cluster_node_mask.float()
                    cluster_data["evt_pred_mask_e"].append(
                        (pred_mask_b * cluster_valid * node_e_b).sum(dim=-1).cpu().numpy())
                    cluster_data["evt_truth_mask_e"].append(
                        (truth_mask_b * cluster_valid * node_e_b).sum(dim=-1).cpu().numpy())

                # Per-event neutral/charged HS energy (mask-weighted sums)
                if "calo_mask" in calo_final and "calo_node_prob" in calo_final["calo_mask"] and "calo_hs_neutral_energy" in labels:
                    pred_mask_b = (calo_final["calo_mask"]["calo_node_prob"] > CALO_PRED_THRESHOLD).float()  # (B, N)
                    truth_mask_b = (
                        (labels["calo_hard_scatter_energy_frac"] > CALO_HS_FRAC_THRESHOLD)
                        & (labels["calo_hard_scatter_energy"] > CALO_HS_ENERGY_THRESHOLD)
                    ).float()  # (B, N)

                    cluster_valid = cluster_node_mask.float()  # (B, N)
                    neutral_e = labels["calo_hs_neutral_energy"]  # (B, N)
                    charged_e = labels["calo_hs_charged_energy"]  # (B, N)

                    cluster_data["evt_pred_neutral_e"].append(
                        (pred_mask_b * cluster_valid * neutral_e).sum(dim=-1).cpu().numpy())
                    cluster_data["evt_truth_neutral_e"].append(
                        (truth_mask_b * cluster_valid * neutral_e).sum(dim=-1).cpu().numpy())
                    cluster_data["evt_pred_charged_e"].append(
                        (pred_mask_b * cluster_valid * charged_e).sum(dim=-1).cpu().numpy())
                    cluster_data["evt_truth_charged_e"].append(
                        (truth_mask_b * cluster_valid * charged_e).sum(dim=-1).cpu().numpy())

            # ── Reconstruction data (Stream C) ──
            reco_final = preds.get("reco_final", {})
            if (not reco_final) and (not warned_no_reco_head):
                print("Warning: no reco_final predictions; this checkpoint/model may not include Stream C outputs.")
                warned_no_reco_head = True
            reco_cls = reco_final.get("classification", {})
            reco_reg = reco_final.get("regression", {})

            pred_class_t = reco_cls.get("reco_pflow_class")
            pred_valid_t = reco_cls.get("reco_pflow_valid")
            truth_class_t = labels.get("reco_particle_class")
            if truth_class_t is None:
                truth_class_t = labels.get("particle_class")

            truth_valid_t = labels.get("reco_particle_valid")
            if truth_valid_t is None:
                truth_valid_t = labels.get("particle_valid")

            found_reco_truth = False

            if pred_class_t is not None:
                reco_data["pred_class"].append(pred_class_t.detach().cpu().numpy().astype(np.int64))
                if pred_valid_t is None:
                    pred_valid_t = pred_class_t < (RECO_NUM_CLASSES - 1)
                reco_data["pred_valid"].append(pred_valid_t.detach().cpu().numpy().astype(bool))

            if truth_class_t is not None:
                reco_data["truth_class"].append(truth_class_t.detach().cpu().numpy().astype(np.int64))
                if truth_valid_t is None:
                    truth_valid_t = truth_class_t < (RECO_NUM_CLASSES - 1)
                reco_data["truth_valid"].append(truth_valid_t.detach().cpu().numpy().astype(bool))
                found_reco_truth = True

            for field in ["pt", "eta", "sinphi", "cosphi"]:
                pred_key = f"reco_pflow_{field}"
                truth_key = f"reco_particle_{field}"
                truth_alt_key = f"particle_{field}"

                if pred_key in reco_reg:
                    pred_field = _inverse_if_needed(reco_reg[pred_key], field)
                    reco_data[f"pred_{field}"].append(pred_field.numpy())

                if truth_key in labels:
                    truth_field = _inverse_if_needed(labels[truth_key], field)
                    reco_data[f"truth_{field}"].append(truth_field.numpy())
                    found_reco_truth = True
                elif truth_alt_key in labels:
                    truth_field = _inverse_if_needed(labels[truth_alt_key], field)
                    reco_data[f"truth_{field}"].append(truth_field.numpy())
                    found_reco_truth = True

            if (not found_reco_truth) and (not warned_no_reco_truth):
                print("Warning: no truth reco labels found in targets (expected reco_particle_* or particle_* keys).")
                warned_no_reco_truth = True

            if (i + 1) % 10 == 0:
                print(f"  Batch {i + 1} done")

    # Concatenate
    track_data = {k: np.concatenate(v) for k, v in track_data.items() if v}
    cluster_data = {k: np.concatenate(v) for k, v in cluster_data.items() if v}
    reco_data = {k: np.concatenate(v) for k, v in reco_data.items() if v}
    return track_data, cluster_data, reco_data


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


def load_eval(
    ckpt_path: str = CKPT_PATH,
    config_path: str = CONFIG_PATH,
    device: str = DEVICE,
    files: list[str | Path] | None = None,
    data_dir: str | None = None,
    num_events: int = -1,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 1,
    max_batches: int | None = None,
):
    """Full eval pipeline. Returns (track_data, cluster_data, model, loader).

    Args:
        files: List of parquet file paths to evaluate on.
        data_dir: Directory to glob for parquet files (used if files is None).
        num_events: Number of events to load (-1 for all).
        max_batches: Stop after this many batches (None for all).
    """
    print(f"Loading config from {config_path}")
    cfg = load_config(config_path)

    print("Building dataloader...")
    loader = build_dataloader(cfg, files=files, data_dir=data_dir, num_events=num_events, batch_size=batch_size, num_workers=num_workers)
    print(f"  {len(loader)} batches")

    print(f"Loading model from {ckpt_path}")
    model = load_model(ckpt_path, device=device)

    print("Running inference...")
    track_data, cluster_data, reco_data = collect_predictions(model, loader, device=device, max_batches=max_batches)
    cluster_arr = (
        cluster_data["pred_frac"]
        if "pred_frac" in cluster_data
        else cluster_data.get("total_e", np.array([]))
    )
    reco_arr = (
        reco_data["truth_pt"]
        if "truth_pt" in reco_data
        else reco_data["truth_class"]
        if "truth_class" in reco_data
        else reco_data.get("pred_class", np.array([]))
    )
    cluster_count = int(np.asarray(cluster_arr).size)
    reco_count = int(np.asarray(reco_arr).size)
    print(f"  Tracks: {len(track_data.get('probs', []))} nodes")
    print(f"  Clusters: {cluster_count} nodes")
    print(f"  Reco objects: {reco_count} slots")
    return track_data, cluster_data,reco_data,model,loader

def run_eval(
    ckpt_path: str = CKPT_PATH,
    config_path: str = CONFIG_PATH,
    device: str = DEVICE,
    files: list[str | Path] | None = None,
    data_dir: str | None = None,
    num_events: int = -1,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 1,
    max_batches: int | None = None,
):
    """Full eval pipeline. Returns (figs, track_data, cluster_data).

    Args:
        files: List of parquet file paths to evaluate on.
        data_dir: Directory to glob for parquet files (used if files is None).
        num_events: Number of events to load (-1 for all).
        max_batches: Stop after this many batches (None for all).
    """
    print(f"Loading config from {config_path}")
    cfg = load_config(config_path)

    print("Building dataloader...")
    loader = build_dataloader(cfg, files=files, data_dir=data_dir, num_events=num_events, batch_size=batch_size, num_workers=num_workers)
    print(f"  {len(loader)} batches")

    print(f"Loading model from {ckpt_path}")
    model = load_model(ckpt_path, device=device)

    print("Running inference...")
    track_data, cluster_data, reco_data = collect_predictions(model, loader, device=device, max_batches=max_batches)
    cluster_arr = (
        cluster_data["pred_frac"]
        if "pred_frac" in cluster_data
        else cluster_data.get("total_e", np.array([]))
    )
    reco_arr = (
        reco_data["truth_pt"]
        if "truth_pt" in reco_data
        else reco_data["truth_class"]
        if "truth_class" in reco_data
        else reco_data.get("pred_class", np.array([]))
    )
    cluster_count = int(np.asarray(cluster_arr).size)
    reco_count = int(np.asarray(reco_arr).size)
    print(f"  Tracks: {len(track_data.get('probs', []))} nodes")
    print(f"  Clusters: {cluster_count} nodes")
    print(f"  Reco objects: {reco_count} slots")

    print("Making plots...")
    figs = make_plots(
        track_data,
        cluster_data,
        reco_data=reco_data,
        truth_pt_bins=RECO_TRUTH_PT_BINS,
    )
    figs.update(make_data_plots(loader.dataset))
    print(f"  Generated {len(figs)} plots: {list(figs.keys())}")

    return figs, track_data, cluster_data


def make_data_plots(dataset_or_loader) -> dict[str, plt.Figure]:
    from torch.utils.data import DataLoader
    dataset = dataset_or_loader.dataset if isinstance(dataset_or_loader, DataLoader) else dataset_or_loader
    """Plots that need only the raw dataset — no model inference required."""
    import time
    from hepattn.experiments.odd_pileup_maskformer.plots import PhysicsPlotter

    figs = {}

    def _plot(name, fn, *args, **kwargs):
        t0 = time.perf_counter()
        figs[name] = fn(*args, **kwargs)
        print(f"  {name:<40s} {time.perf_counter() - t0:.2f}s")

    stats = compute_deltaR_window_stats(dataset, window_sizes=[32, 64, 80, 128, 256], n_sample=200)
    _plot("data/deltaR_window_analysis", PhysicsPlotter.plot_deltaR_window_analysis, stats)
    _plot("data/eta_window_analysis",    PhysicsPlotter.plot_eta_window_analysis,    stats)
    _plot("data/phi_window_analysis",    PhysicsPlotter.plot_phi_window_analysis,    stats)

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
    config_path: str = CONFIG_PATH,
    files: list[str | Path] | None = None,
    data_dir: str | None = None,
    num_events: int = -1,
    num_workers: int = 1,
) -> dict[str, plt.Figure]:
    """Generate all data-only plots without loading a model.

    Usage:
        figs = run_data_plots(data_dir="/storage/.../ttbar_pu200")
        figs = run_data_plots(files=["file1.parquet", "file2.parquet"])
    """
    cfg = load_config(config_path)
    loader = build_dataloader(cfg, files=files, data_dir=data_dir, num_events=num_events, num_workers=num_workers)
    return make_data_plots(loader.dataset)


if __name__ == "__main__":
    figs, track_data, cluster_data = run_eval()

    # Save all figures to disk
    out_dir = Path(CKPT_PATH).parent.parent / "eval_plots"
    out_dir.mkdir(exist_ok=True)
    for name, fig in figs.items():
        fname = name.replace("/", "_") + ".png"
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved {len(figs)} plots to {out_dir}")
