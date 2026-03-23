"""
Offline evaluation script for the Two-Stream MaskFormer.

Loads a checkpoint and data files, runs inference, and produces
all validation plots from ODDPFlowTwoStream.on_validation_epoch_end().

Usage (standalone):
    python eval_plots.py

Usage (notebook):
    from hepattn.experiments.odd_pileup_two_stream.eval_plots import run_eval

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
CKPT_PATH = "/storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_two_stream/logs/odd_pflow_two_stream_20260316-T193015/ckpts/epoch=015-val_loss=6.57477.ckpt"
CONFIG_PATH = "/storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_two_stream/configs/base.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 48


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
    from hepattn.experiments.odd_pileup_maskformer.pflow_data import ODDDatasetPileup

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
    from hepattn.experiments.odd_pileup_two_stream.lightning_module import ODDPFlowTwoStream

    model = ODDPFlowTwoStream.load_from_checkpoint(ckpt_path, map_location=device)
    model.eval()
    model.to(device)
    return model


def collect_predictions(model, dataloader, device: str = DEVICE, max_batches: int | None = None):
    """Run inference on dataloader and accumulate data for plots.

    Returns (track_data, cluster_data) dicts with numpy arrays.
    """
    track_data = defaultdict(list)
    cluster_data = defaultdict(list)
    event_counter = 0

    with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16):
        for i, batch in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break

            inputs, targets = batch
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            targets = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in targets.items()}

            outputs = model.model(inputs)
            preds = model.model.predict(outputs)

            labels = targets
            node_valid = labels["node_valid"].bool()
            is_track = labels["node_is_track"].bool()

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

            if cluster_node_mask.any() and "calo_fraction" in calo_final:
                calo_frac_pred = calo_final["calo_fraction"]["calo_hs_fraction"].squeeze(-1)[cluster_node_mask]
                calo_frac_true = labels["calo_hard_scatter_energy_frac"][cluster_node_mask]
                node_e = labels["node_e"][cluster_node_mask]
                true_hs_energy = labels["calo_hard_scatter_energy"][cluster_node_mask]

                cluster_data["pred_frac"].append(calo_frac_pred.float().cpu().numpy())
                cluster_data["true_frac"].append(calo_frac_true.float().cpu().numpy())
                cluster_data["total_e"].append(node_e.float().cpu().numpy())
                cluster_data["true_hs_e"].append(true_hs_energy.float().cpu().numpy())
                cluster_data["eta"].append(labels["node_eta"][cluster_node_mask].float().cpu().numpy())
                cluster_data["phi"].append(labels["node_phi"][cluster_node_mask].float().cpu().numpy())

                # Truth neutral/charged energy per cluster (for composition plot)
                if "calo_hs_neutral_energy" in labels:
                    cluster_data["neutral_e"].append(labels["calo_hs_neutral_energy"][cluster_node_mask].float().cpu().numpy())
                    cluster_data["charged_e"].append(labels["calo_hs_charged_energy"][cluster_node_mask].float().cpu().numpy())

                # Calo mask predictions for cluster swap plot
                if "calo_mask" in calo_final:
                    calo_prob_flat = calo_final["calo_mask"]["calo_node_prob"][cluster_node_mask]
                    calo_hs_e_flat = labels["calo_hard_scatter_energy"][cluster_node_mask]
                    cluster_data["mask_pred"].append((calo_prob_flat > 0.5).cpu().numpy())
                    cluster_data["calo_mask_probs"].append(calo_prob_flat.float().cpu().numpy())
                    cluster_data["mask_truth"].append((calo_hs_e_flat > 0.15).cpu().numpy())

                # Per-event indices
                counts = cluster_node_mask.sum(dim=-1).cpu().numpy()
                event_indices = np.repeat(
                    np.arange(event_counter, event_counter + len(counts)),
                    counts,
                )
                cluster_data["event_idx"].append(event_indices)
                event_counter += len(counts)

                # Per-event neutral/charged HS energy (mask-weighted sums)
                if "calo_mask" in calo_final and "calo_hs_neutral_energy" in labels:
                    pred_mask_b = (calo_final["calo_mask"]["calo_node_prob"] > 0.5).float()  # (B, N)
                    truth_mask_b = (
                        (labels["calo_hard_scatter_energy_frac"] > 0.05)
                        & (labels["calo_hard_scatter_energy"] > 0.15)
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

            if (i + 1) % 10 == 0:
                print(f"  Batch {i + 1} done")

    # Concatenate
    track_data = {k: np.concatenate(v) for k, v in track_data.items() if v}
    cluster_data = {k: np.concatenate(v) for k, v in cluster_data.items() if v}
    return track_data, cluster_data


def make_plots(track_data: dict, cluster_data: dict) -> dict[str, plt.Figure]:
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
        t0 = time.perf_counter()
        for bin_key, bin_fig in PhysicsPlotter.plot_calo_mask_errors_by_energy(
            cluster_data["mask_pred"], cluster_data["mask_truth"],
            cluster_data["total_e"], cluster_data["true_hs_e"], cluster_data["pred_frac"],
        ).items():
            figs[f"calo/mask_errors_by_energy/{bin_key}"] = bin_fig
        print(f"  {'calo/mask_errors_by_energy':<40s} {time.perf_counter() - t0:.2f}s")
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
    track_data, cluster_data = collect_predictions(model, loader, device=device, max_batches=max_batches)
    print(f"  Tracks: {len(track_data.get('probs', []))} nodes")
    print(f"  Clusters: {len(cluster_data.get('pred_frac', []))} nodes")
    return track_data, cluster_data,model,loader

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
    track_data, cluster_data = collect_predictions(model, loader, device=device, max_batches=max_batches)
    print(f"  Tracks: {len(track_data.get('probs', []))} nodes")
    print(f"  Clusters: {len(cluster_data.get('pred_frac', []))} nodes")

    print("Making plots...")
    figs = make_plots(track_data, cluster_data)
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
