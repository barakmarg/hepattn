import torch
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

        # --- Cluster Energy Fraction MAE (only on cluster nodes) ---
        cluster_node_mask = node_valid & (~is_track)
        if cluster_node_mask.any():
            calo_frac_pred = final_preds["calo_hs_fraction"].squeeze(-1)[cluster_node_mask]
            calo_frac_true = labels["calo_hard_scatter_energy_frac"][cluster_node_mask]
            mae = nn.functional.l1_loss(calo_frac_pred, calo_frac_true)
            self.log(f"{stage}/calo_frac_mae", mae, **kwargs)
