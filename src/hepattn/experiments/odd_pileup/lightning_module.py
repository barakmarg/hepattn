import torch
import torch.nn.functional as F
import torchmetrics as tm
from torch import nn

from hepattn.models.wrapper import ModelWrapper


class ODDPFlow(ModelWrapper):
    def __init__(
        self,
        name: str,
        model: nn.Module,
        lrs_config: dict,
        optimizer: str = "AdamW",
        mtl: bool = False,
    ):
        super().__init__(name, model, lrs_config, optimizer, mtl)

        # Track classification metrics
        self.track_accuracy = tm.classification.BinaryAccuracy()
        self.track_precision = tm.classification.BinaryPrecision()
        self.track_recall = tm.classification.BinaryRecall()

    def log_custom_metrics(self, preds, labels, stage):
        kwargs = {"sync_dist": True, "batch_size": 1}

        # Skip detailed metrics during training for speed
        if stage == "train":
            return

        final_preds = preds["final"]
        node_valid = labels["node_valid"].bool()
        is_track = labels["is_track"].bool()

        # --- Track Classification Metrics (only on track nodes) ---
        track_node_mask = node_valid & is_track
        if track_node_mask.any():
            track_prob = final_preds["track_prob"].squeeze(-1)[track_node_mask]
            track_truth = labels["tracks_mask"][track_node_mask].int()

            self.track_accuracy(track_prob, track_truth)
            self.log(f"{stage}/track_accuracy", self.track_accuracy, **kwargs)

            self.track_precision(track_prob, track_truth)
            self.log(f"{stage}/track_precision", self.track_precision, **kwargs)

            self.track_recall(track_prob, track_truth)
            self.log(f"{stage}/track_recall", self.track_recall, **kwargs)

        # --- Cluster Metrics (only on cluster nodes) ---
        cluster_node_mask = node_valid & (~is_track)
        if cluster_node_mask.any():
            calo_frac_pred = final_preds["calo_hs_fraction"].squeeze(-1)[cluster_node_mask]
            calo_frac_true = labels["calo_hard_scatter_energy_frac"][cluster_node_mask]

            # Overall MAE
            mae = F.l1_loss(calo_frac_pred, calo_frac_true)
            self.log(f"{stage}/calo_frac_mae", mae, **kwargs)

            # Per-type MSE based on true HS energy fraction
            # a) pileup only:       < 2%  HS
            # b) mix pileup:       3-30%  HS
            # c) balanced:        30-70%  HS
            # d) mix hard scatter: 70-97% HS (3-30% pileup)
            # e) hs only:          > 98%  HS
            type_masks = {
                "pu_only":   calo_frac_true < 0.02,
                "mix_pu":   (calo_frac_true >= 0.03) & (calo_frac_true <= 0.30),
                "balanced": (calo_frac_true > 0.30)  & (calo_frac_true < 0.70),
                "mix_hs":   (calo_frac_true >= 0.70) & (calo_frac_true <= 0.97),
                "hs_only":   calo_frac_true > 0.98,
            }
            for type_name, type_mask in type_masks.items():
                if type_mask.any():
                    mse = F.mse_loss(calo_frac_pred[type_mask], calo_frac_true[type_mask])
                    self.log(f"{stage}/calo_frac_mse_{type_name}", mse, **kwargs)

            # Hard scatter energy ratio: sum(predicted HS energy) / sum(true HS energy)
            node_e = labels["node_e"][cluster_node_mask]
            true_hs_energy = labels["calo_hard_scatter_energy"][cluster_node_mask]
            sum_true = true_hs_energy.sum()
            if sum_true > 0:
                pred_hs_energy = calo_frac_pred * node_e
                hs_energy_ratio = pred_hs_energy.sum() / sum_true
                self.log(f"{stage}/calo_hs_energy_ratio", hs_energy_ratio, **kwargs)
