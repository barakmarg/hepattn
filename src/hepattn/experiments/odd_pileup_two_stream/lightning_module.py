from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics as tm
from torch import nn

from hepattn.experiments.odd_pileup_maskformer.plots import PhysicsPlotter
from hepattn.models.wrapper import ModelWrapper


class ODDPFlowTwoStream(ModelWrapper):
    """Lightning module for the Two-Stream MaskFormer.

    Handles the two-stream output format where predictions are keyed as
    track_final/calo_final instead of just final.
    """

    def __init__(
        self,
        name: str,
        model: nn.Module,
        lrs_config: dict,
        optimizer: str = "AdamW",
        mtl: bool = False,
    ):
        super().__init__(name, model, lrs_config, optimizer, mtl)

        # Track classification metrics (from Stream A)
        self.track_f1 = tm.classification.BinaryF1Score()
        self.track_precision = tm.classification.BinaryPrecision()
        self.track_recall = tm.classification.BinaryRecall()

        # Calo mask metrics (from Stream B)
        self.calo_mask_f1 = tm.classification.BinaryF1Score()
        self.calo_mask_precision = tm.classification.BinaryPrecision()
        self.calo_mask_recall = tm.classification.BinaryRecall()

        # Accumulation buffers for end-of-epoch physics plots
        self._val_track_data: dict[str, list] = defaultdict(list)
        self._val_cluster_data: dict[str, list] = defaultdict(list)

    # ------------------------------------------------------------------
    # Override log_task_metrics to handle two-stream output format
    # ------------------------------------------------------------------

    def log_task_metrics(self, preds, targets, stage):
        # Task metrics are handled by log_custom_metrics for the two-stream model
        pass

    # ------------------------------------------------------------------
    # Validation accumulation hooks
    # ------------------------------------------------------------------

    def on_validation_epoch_start(self) -> None:
        self._val_track_data = defaultdict(list)
        self._val_cluster_data = defaultdict(list)

    def on_validation_epoch_end(self) -> None:
        if not self._val_track_data and not self._val_cluster_data:
            return

        track = {k: np.concatenate(v) for k, v in self._val_track_data.items()}
        cluster = {k: np.concatenate(v) for k, v in self._val_cluster_data.items()}

        figs = {}
        if cluster.get("pred_frac") is not None and len(cluster["pred_frac"]) > 0:
            figs["calo/energy_corr"] = PhysicsPlotter.plot_energy_correlation(
                cluster["pred_frac"], cluster["total_e"], cluster["true_hs_e"]
            )
            figs["calo/frac_corr"] = PhysicsPlotter.plot_calo_frac_correlation(
                cluster["pred_frac"], cluster["true_frac"]
            )
            figs["calo/frac_dist"] = PhysicsPlotter.plot_calo_frac_distribution(
                cluster["pred_frac"], cluster["true_frac"]
            )
            figs["calo/energy_resid"] = PhysicsPlotter.plot_energy_residual(
                cluster["pred_frac"], cluster["total_e"], cluster["true_hs_e"]
            )
            figs["calo/hs_energy_dist"] = PhysicsPlotter.plot_hs_energy_distribution(
                cluster["pred_frac"], cluster["total_e"], cluster["true_hs_e"]
            )

        if track.get("probs") is not None and len(track["probs"]) > 0:
            figs["track/score_dist"] = PhysicsPlotter.plot_track_score_distribution(
                track["probs"], track["truth"]
            )
            figs["track/eff_vs_pt"] = PhysicsPlotter.plot_track_efficiency_vs_pt(
                track["probs"], track["truth"], track["pt"]
            )
            figs["track/rej_vs_pt"] = PhysicsPlotter.plot_pileup_rejection_vs_pt(
                track["probs"], track["truth"], track["pt"]
            )
            figs["track/mistag_eta"] = PhysicsPlotter.plot_mistag_rate_vs_eta(
                track["probs"], track["truth"], track["eta"]
            )
            figs["track/score_by_pt"] = PhysicsPlotter.plot_track_score_by_pt(
                track["probs"], track["truth"], track["pt"]
            )
            figs["track/roc"] = PhysicsPlotter.plot_roc_curve(
                track["probs"], track["truth"]
            )
            if track.get("z0") is not None and len(track["z0"]) > 0:
                figs["track/z0_dist"] = PhysicsPlotter.plot_track_z0_distribution(
                    track["probs"], track["truth"], track["z0"]
                )
            figs["track/pt_dist"] = PhysicsPlotter.plot_track_pt_distribution(
                track["probs"], track["truth"], track["pt"]
            )

        # Log to CometML
        if self.logger is not None and hasattr(self.logger, "experiment"):
            exp = self.logger.experiment
            if hasattr(exp, "log_figure"):
                for name, fig in figs.items():
                    exp.log_figure(figure_name=name, figure=fig, step=self.current_epoch)
                    plt.close(fig)
                return
        # Fallback: just close figures without logging
        for fig in figs.values():
            plt.close(fig)

    # ------------------------------------------------------------------
    # Per-batch metrics + accumulation
    # ------------------------------------------------------------------

    def log_custom_metrics(self, preds, labels, stage):
        kwargs = {"sync_dist": True, "batch_size": 1}

        # Skip detailed metrics during training for speed
        if stage == "train":
            return

        node_valid = labels["node_valid"].bool()
        is_track = labels["node_is_track"].bool()

        # === Track Classification Metrics (Stream A) ===
        track_final = preds.get("track_final", {})
        track_node_mask = node_valid & is_track

        if track_node_mask.any() and "mask" in track_final:
            # Mask task output shape: (B, 1, N) → squeeze query dim → (B, N)
            track_prob = track_final["mask"]["pflow_node_prob"].squeeze(-2)[track_node_mask]
            track_truth = labels["tracks_mask"][track_node_mask].int()

            self.track_f1(track_prob, track_truth)
            self.log(f"{stage}/track_f1", self.track_f1, **kwargs)

            self.track_precision(track_prob, track_truth)
            self.log(f"{stage}/track_precision", self.track_precision, **kwargs)

            self.track_recall(track_prob, track_truth)
            self.log(f"{stage}/track_recall", self.track_recall, **kwargs)

            # Accumulate for epoch-end plots
            if stage == "val":
                self._val_track_data["probs"].append(track_prob.detach().float().cpu().numpy())
                self._val_track_data["truth"].append(track_truth.detach().cpu().numpy())
                self._val_track_data["pt"].append(labels["node_pt"][track_node_mask].detach().float().cpu().numpy())
                self._val_track_data["eta"].append(labels["node_eta"][track_node_mask].detach().float().cpu().numpy())
                self._val_track_data["z0"].append(labels["node_z0"][track_node_mask].detach().float().cpu().numpy())

        # === Vz Regression Metrics (Stream A) ===
        if "vz_regression" in track_final:
            pred_vz = track_final["vz_regression"]["pflow_vz"]  # (B, 1)
            true_vz = labels["particle_vz"]  # (B, 1)
            vz_mae = F.l1_loss(pred_vz, true_vz)
            self.log(f"{stage}/vz_mae", vz_mae, **kwargs)

        # === Calo Mask Metrics (Stream B) ===
        calo_final = preds.get("calo_final", {})
        cluster_node_mask = node_valid & (~is_track)

        if cluster_node_mask.any() and "calo_mask" in calo_final:
            calo_mask_prob = calo_final["calo_mask"]["calo_node_prob"][cluster_node_mask]
            calo_hs_energy = labels["calo_hard_scatter_energy"]
            # Use the same threshold from the task config (default 0.15)
            calo_mask_truth = (calo_hs_energy[cluster_node_mask] > 0.15).int()

            self.calo_mask_f1(calo_mask_prob, calo_mask_truth)
            self.log(f"{stage}/calo_mask_f1", self.calo_mask_f1, **kwargs)

            self.calo_mask_precision(calo_mask_prob, calo_mask_truth)
            self.log(f"{stage}/calo_mask_precision", self.calo_mask_precision, **kwargs)

            self.calo_mask_recall(calo_mask_prob, calo_mask_truth)
            self.log(f"{stage}/calo_mask_recall", self.calo_mask_recall, **kwargs)

        # === Cluster Fraction Metrics (Stream B) ===
        if cluster_node_mask.any() and "calo_fraction" in calo_final:
            calo_frac_pred = calo_final["calo_fraction"]["calo_hs_fraction"].squeeze(-1)[cluster_node_mask]
            calo_frac_true = labels["calo_hard_scatter_energy_frac"][cluster_node_mask]

            # Overall MAE
            mae = F.l1_loss(calo_frac_pred, calo_frac_true)
            self.log(f"{stage}/calo_frac_mae", mae, **kwargs)

            # Per-type MAE based on true HS energy fraction
            type_masks = {
                "pu_only": calo_frac_true < 0.02,
                "mix_pu": (calo_frac_true >= 0.03) & (calo_frac_true <= 0.30),
                "balanced": (calo_frac_true > 0.30) & (calo_frac_true < 0.70),
                "mix_hs": (calo_frac_true >= 0.70) & (calo_frac_true <= 0.97),
                "hs_only": calo_frac_true > 0.98,
            }
            for type_name, type_mask in type_masks.items():
                if type_mask.any():
                    mae = F.l1_loss(calo_frac_pred[type_mask], calo_frac_true[type_mask])
                    self.log(f"{stage}/calo_frac_mae_{type_name}", mae, **kwargs)

            # Hard scatter energy ratio
            node_e = labels["node_e"][cluster_node_mask]
            true_hs_energy = labels["calo_hard_scatter_energy"][cluster_node_mask]
            sum_true = true_hs_energy.sum()
            if sum_true > 0:
                pred_hs_energy = calo_frac_pred * node_e
                hs_energy_ratio = pred_hs_energy.sum() / sum_true
                self.log(f"{stage}/calo_hs_energy_ratio", hs_energy_ratio, **kwargs)

            # Accumulate for epoch-end plots
            if stage == "val":
                self._val_cluster_data["pred_frac"].append(calo_frac_pred.detach().float().cpu().numpy())
                self._val_cluster_data["true_frac"].append(calo_frac_true.detach().float().cpu().numpy())
                self._val_cluster_data["total_e"].append(node_e.detach().float().cpu().numpy())
                self._val_cluster_data["true_hs_e"].append(true_hs_energy.detach().float().cpu().numpy())
