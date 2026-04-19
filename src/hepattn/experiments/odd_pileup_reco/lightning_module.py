from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics as tm
from torch import nn

from hepattn.experiments.odd_pileup_maskformer.plots import PhysicsPlotter
from hepattn.experiments.odd_pileup_reco.reco_analysis import (
    cluster_jets,
    plot_calo_mask_energy_purity,
    plot_class_distribution,
    plot_jet_resolution_with_calo,
    pflow_data_from_eval_dicts,
)
from hepattn.models.wrapper import ModelWrapper


class MaskInference:
    """Mask evaluation utilities for reconstruction metrics."""

    @staticmethod
    def exact_match(pred, tgt):
        if len(tgt) == 0:
            return torch.tensor(torch.nan)
        return (pred == tgt).all(-1).float().mean()

    @staticmethod
    def eff(pred, tgt):
        return ((pred & tgt).sum(-1) / tgt.sum(-1)).mean()

    @staticmethod
    def pur(pred, tgt):
        return ((pred & tgt).sum(-1) / pred.sum(-1)).mean()


class ODDPFlowTwoStream(ModelWrapper):
    """Lightning module for the Two-Stream MaskFormer with Reconstruction.

    Handles the two/three-stream output format where predictions are keyed as
    track_final/calo_final/reco_final instead of just final.
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

        self.MI = MaskInference

        # Track classification metrics (from Stream A)
        self.track_f1 = tm.classification.BinaryF1Score()
        self.track_precision = tm.classification.BinaryPrecision()
        self.track_recall = tm.classification.BinaryRecall()

        # Calo mask metrics (from Stream B)
        self.calo_mask_f1 = tm.classification.BinaryF1Score()
        self.calo_mask_precision = tm.classification.BinaryPrecision()
        self.calo_mask_recall = tm.classification.BinaryRecall()

        # Reconstruction metrics (from Stream C)
        self.obj_accuracy_micro = tm.classification.MulticlassAccuracy(num_classes=6, average="micro")
        self.obj_accuracy_macro = tm.classification.MulticlassAccuracy(num_classes=6, average="macro")
        self.reco_eff = tm.classification.BinaryRecall()
        self.reco_pur = tm.classification.BinaryPrecision()

        # Accumulation buffers for end-of-epoch physics plots
        self._val_track_data: dict[str, list] = defaultdict(list)
        self._val_cluster_data: dict[str, list] = defaultdict(list)
        self._val_reco_data: dict[str, list] = defaultdict(list)
        self._val_jet_data: dict[str, list] = defaultdict(list)
        self._val_jet_event_count: int = 0

    # ------------------------------------------------------------------
    # Override training/validation steps to pass targets to forward
    # (needed for teacher forcing in Stream C)
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs, targets=targets)

        losses = self.model.loss(outputs, targets)
        total_loss = self.log_losses(losses, "train")

        for layer_name, layer_losses in losses.items():
            for task_name, task_losses in layer_losses.items():
                for loss_name, loss_value in task_losses.items():
                    if isinstance(loss_value, torch.Tensor):
                        if torch.isnan(loss_value).any():
                            print(f"NaN in loss: {layer_name}_{task_name}_{loss_name} at batch {batch_idx}")
                        if torch.isinf(loss_value).any():
                            print(f"Inf in loss: {layer_name}_{task_name}_{loss_name} at batch {batch_idx}")

        if batch_idx % self.trainer.log_every_n_steps == 0:
            preds = self.predict(outputs)
            self.log_metrics(preds, targets, "train")

        if self.mtl:
            self.mlt_opt(losses, outputs)
            return None

        return total_loss

    def validation_step(self, batch):
        inputs, targets = batch
        outputs = self.model(inputs, targets=targets)

        losses = self.model.loss(outputs, targets)
        self.log_losses(losses, "val")

        preds = self.predict(outputs)
        self.log_metrics(preds, targets, "val")
        return outputs, preds, losses

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs, targets=targets)

        losses = self.model.loss(outputs, targets)
        self.log_losses(losses, "test")

        preds = self.predict(outputs)
        self.log_metrics(preds, targets, "test")
        return outputs, preds, losses

    # ------------------------------------------------------------------
    # Override log_task_metrics to handle multi-stream output format
    # ------------------------------------------------------------------

    def log_task_metrics(self, preds, targets, stage):
        pass

    # ------------------------------------------------------------------
    # Validation accumulation hooks
    # ------------------------------------------------------------------

    def on_validation_epoch_start(self) -> None:
        self._val_track_data = defaultdict(list)
        self._val_cluster_data = defaultdict(list)
        self._val_reco_data = defaultdict(list)
        self._val_jet_data = defaultdict(list)
        self._val_jet_event_count = 0
        self._val_event_counter = 0

    def on_validation_epoch_end(self) -> None:
        if not self._val_track_data and not self._val_cluster_data and not self._val_reco_data and not self._val_jet_data:
            return

        track = {k: np.concatenate(v) for k, v in self._val_track_data.items()}
        cluster = {k: np.concatenate(v) for k, v in self._val_cluster_data.items()}
        reco = {k: np.concatenate(v) for k, v in self._val_reco_data.items()}

        figs = {}

        if reco.get("pflow_class") is not None and len(reco["pflow_class"]) > 0:
            figs["reco/class_distribution"] = plot_class_distribution(reco)
        if cluster.get("evt_pred_neutral_e") is not None and len(cluster.get("evt_pred_neutral_e", [])) > 0:
            figs["calo/hs_energy_residual_by_type"] = PhysicsPlotter.plot_hs_energy_residual_by_type(
                cluster["evt_pred_neutral_e"], cluster["evt_truth_neutral_e"],
                cluster["evt_pred_charged_e"], cluster["evt_truth_charged_e"],
            )
            figs["calo/hs_energy_ratio_by_type"] = PhysicsPlotter.plot_hs_energy_ratio_by_type(
                cluster["evt_pred_neutral_e"], cluster["evt_truth_neutral_e"],
                cluster["evt_pred_charged_e"], cluster["evt_truth_charged_e"],
            )

        if track.get("probs") is not None and len(track["probs"]) > 0:
            figs["track/score_dist"] = PhysicsPlotter.plot_track_score_distribution(
                track["probs"], track["truth"]
            )
            figs["track/eff_vs_pt"] = PhysicsPlotter.plot_track_efficiency_vs_pt(
                track["probs"], track["truth"], track["pt"]
            )
            figs["track/f1_vs_pt"] = PhysicsPlotter.plot_track_f1_vs_pt(
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
            if track.get("z0") is not None and len(track["z0"]) > 0:
                figs["track/z0_dist"] = PhysicsPlotter.plot_track_z0_distribution(
                    track["probs"], track["truth"], track["z0"]
                )
            figs["track/pt_dist"] = PhysicsPlotter.plot_track_pt_distribution(
                track["probs"], track["truth"], track["pt"]
            )

        # Jet resolution with calo (first 1000 validation events)
        if self._val_jet_data.get("node_valid"):
            jet = {k: np.concatenate(v) for k, v in self._val_jet_data.items()}

            # Inverse-transform particle kinematics from scaled → physical space
            scaler = None
            try:
                scaler = self.trainer.datamodule.scaler
            except AttributeError:
                pass
            if scaler is not None:
                for _field in ("pt", "eta", "sinphi", "cosphi"):
                    _tr = scaler.transforms[_field]
                    for _prefix in ("pred", "truth"):
                        _key = f"{_prefix}_{_field}"
                        if _key in jet:
                            _arr = jet[_key].astype(np.float32)
                            _nan = np.isnan(_arr)
                            _t = torch.from_numpy(np.where(_nan, 0.0, _arr))
                            _out = _tr.inverse_transform(_t).numpy()
                            _out[_nan] = np.nan
                            jet[_key] = _out

            # Build data dict for reco_analysis clustering functions
            _jet_reco = {
                "truth_class": jet.get("truth_class"),
                "pred_class":  jet.get("pred_class"),
                "truth_pt":    jet.get("truth_pt"),
                "truth_eta":   jet.get("truth_eta"),
                "truth_sinphi": jet.get("truth_sinphi"),
                "truth_cosphi": jet.get("truth_cosphi"),
                "pred_pt":     jet.get("pred_pt"),
                "pred_eta":    jet.get("pred_eta"),
                "pred_sinphi": jet.get("pred_sinphi"),
                "pred_cosphi": jet.get("pred_cosphi"),
                "node_valid":    jet.get("node_valid"),
                "node_is_track": jet.get("node_is_track"),
                "node_eta":      jet.get("node_eta"),
                "node_phi":      jet.get("node_phi"),
                "node_e":        jet.get("node_e"),
                "calo_hs_energy": jet.get("calo_hs_energy"),
            }
            # Calo mask energy purity plot
            _purity_data = {
                "node_e":        jet.get("node_e"),
                "calo_hs_energy": jet.get("calo_hs_energy"),
                "calo_hs_frac":  jet.get("calo_hs_frac"),
                "node_is_track": jet.get("node_is_track"),
                "node_valid":    jet.get("node_valid"),
                "calo_prob":     jet.get("calo_prob"),
                "reco_node_indices": jet.get("reco_node_indices"),
                "reco_is_track":     jet.get("reco_is_track"),
            }
            _purity_fig = plot_calo_mask_energy_purity(_purity_data)
            if _purity_fig is not None:
                figs["reco_analysis/calo_mask_hs_energy_purity"] = _purity_fig

            try:
                _data = pflow_data_from_eval_dicts(_jet_reco)
                _jets = cluster_jets(_data)
                _fig = plot_jet_resolution_with_calo(_jets, _data, compare_jets=None)
                if _fig is not None:
                    figs["reco/jet_resolution_with_calo"] = _fig
            except Exception as _e:
                print(f"Jet resolution plot failed: {_e}")

        # Log to CometML
        if self.logger is not None and hasattr(self.logger, "experiment"):
            exp = self.logger.experiment
            if hasattr(exp, "log_figure"):
                for name, fig in figs.items():
                    exp.log_figure(figure_name=name, figure=fig, step=self.current_epoch)
                    plt.close(fig)
                return
        for fig in figs.values():
            plt.close(fig)

    # ------------------------------------------------------------------
    # Per-batch metrics + accumulation
    # ------------------------------------------------------------------

    def log_custom_metrics(self, preds, labels, stage):
        kwargs = {"sync_dist": True, "batch_size": 1}

        if stage == "train":
            return

        node_valid = labels["node_valid"].bool()
        is_track = labels["node_is_track"].bool()

        # === Track Classification Metrics (Stream A) ===
        track_final = preds.get("track_final", {})
        track_node_mask = node_valid & is_track

        if track_node_mask.any() and "mask" in track_final:
            track_prob = track_final["mask"]["pflow_node_prob"].squeeze(-2)[track_node_mask]
            track_truth = labels["tracks_mask"][track_node_mask].int()

            self.track_f1(track_prob, track_truth)
            self.log(f"{stage}/track_f1", self.track_f1, **kwargs)

            self.track_precision(track_prob, track_truth)
            self.log(f"{stage}/track_precision", self.track_precision, **kwargs)

            self.track_recall(track_prob, track_truth)
            self.log(f"{stage}/track_recall", self.track_recall, **kwargs)

            if stage == "val":
                self._val_track_data["probs"].append(track_prob.detach().float().cpu().numpy())
                self._val_track_data["truth"].append(track_truth.detach().cpu().numpy())
                self._val_track_data["pt"].append(labels["node_pt"][track_node_mask].detach().float().cpu().numpy())
                self._val_track_data["eta"].append(labels["node_eta"][track_node_mask].detach().float().cpu().numpy())
                self._val_track_data["z0"].append(labels["node_z0"][track_node_mask].detach().float().cpu().numpy())

        # === Vz Regression Metrics (Stream A) ===
        if "vz_regression" in track_final:
            pred_vz = track_final["vz_regression"]["pflow_vz"]
            true_vz = labels["particle_vz"]
            vz_mae = F.l1_loss(pred_vz, true_vz)
            self.log(f"{stage}/vz_mae", vz_mae, **kwargs)

        # === Calo Mask Metrics (Stream B) ===
        calo_final = preds.get("calo_final", {})
        cluster_node_mask = node_valid & (~is_track)

        if cluster_node_mask.any() and "calo_mask" in calo_final:
            calo_mask_prob = calo_final["calo_mask"]["calo_node_prob"][cluster_node_mask]
            calo_hs_energy = labels["calo_hard_scatter_energy"]
            calo_hs_frac = labels["calo_hard_scatter_energy_frac"]
            calo_mask_task = self.model.calo_tasks[0]
            calo_mask_truth = (
                (calo_hs_frac[cluster_node_mask] > calo_mask_task.hs_frac_threshold)
                & (calo_hs_energy[cluster_node_mask] > calo_mask_task.hs_energy_threshold)
            ).int()

            self.calo_mask_f1(calo_mask_prob, calo_mask_truth)
            self.log(f"{stage}/calo_mask_f1", self.calo_mask_f1, **kwargs)

            self.calo_mask_precision(calo_mask_prob, calo_mask_truth)
            self.log(f"{stage}/calo_mask_precision", self.calo_mask_precision, **kwargs)

            self.calo_mask_recall(calo_mask_prob, calo_mask_truth)
            self.log(f"{stage}/calo_mask_recall", self.calo_mask_recall, **kwargs)

        # === Calo Mask Validation Accumulation (Stream B) ===
        if cluster_node_mask.any() and stage == "val":
            if "calo_mask" in calo_final:
                calo_prob_flat = calo_final["calo_mask"]["calo_node_prob"][cluster_node_mask]
                calo_hs_e_flat = labels["calo_hard_scatter_energy"][cluster_node_mask]
                calo_hs_frac_flat = labels["calo_hard_scatter_energy_frac"][cluster_node_mask]
                calo_mask_task = self.model.calo_tasks[0]
                self._val_cluster_data["mask_pred"].append((calo_prob_flat > calo_mask_task.pred_threshold).detach().cpu().numpy())
                self._val_cluster_data["calo_mask_probs"].append(calo_prob_flat.detach().float().cpu().numpy())
                self._val_cluster_data["mask_truth"].append(
                    ((calo_hs_frac_flat > calo_mask_task.hs_frac_threshold) & (calo_hs_e_flat > calo_mask_task.hs_energy_threshold)).detach().cpu().numpy()
                )

            counts = cluster_node_mask.sum(dim=-1).cpu().numpy()
            event_indices = np.repeat(
                np.arange(self._val_event_counter, self._val_event_counter + len(counts)),
                counts,
            )
            self._val_cluster_data["event_idx"].append(event_indices)
            self._val_event_counter += len(counts)

            if "calo_mask" in calo_final:
                calo_mask_task = self.model.calo_tasks[0]
                pred_mask_b = (calo_final["calo_mask"]["calo_node_prob"] > calo_mask_task.pred_threshold).float()
                truth_mask_b = (
                    (labels["calo_hard_scatter_energy_frac"] > calo_mask_task.hs_frac_threshold)
                    & (labels["calo_hard_scatter_energy"] > calo_mask_task.hs_energy_threshold)
                ).float()

                cluster_valid = cluster_node_mask.float()
                neutral_e = labels["calo_hs_neutral_energy"]
                charged_e = labels["calo_hs_charged_energy"]

                self._val_cluster_data["evt_pred_neutral_e"].append(
                    (pred_mask_b * cluster_valid * neutral_e).sum(dim=-1).detach().cpu().numpy())
                self._val_cluster_data["evt_truth_neutral_e"].append(
                    (truth_mask_b * cluster_valid * neutral_e).sum(dim=-1).detach().cpu().numpy())
                self._val_cluster_data["evt_pred_charged_e"].append(
                    (pred_mask_b * cluster_valid * charged_e).sum(dim=-1).detach().cpu().numpy())
                self._val_cluster_data["evt_truth_charged_e"].append(
                    (truth_mask_b * cluster_valid * charged_e).sum(dim=-1).detach().cpu().numpy())

        # === Reconstruction Metrics (Stream C) ===
        reco_final = preds.get("reco_final", {})

        if "classification" in reco_final:
            reco_target_obj = self.model.reco_target_object
            class_key = f"{reco_target_obj}_class"
            if class_key in labels:
                particle_class_preds = reco_final["classification"]["reco_pflow_class"]
                particle_class_labels = labels[class_key]

                self.obj_accuracy_micro(particle_class_preds.view(-1), particle_class_labels.view(-1))
                self.log(f"{stage}/reco_obj_class_accuracy_micro", self.obj_accuracy_micro, **kwargs)
                self.obj_accuracy_macro(particle_class_preds.view(-1), particle_class_labels.view(-1))
                self.log(f"{stage}/reco_obj_class_accuracy_macro", self.obj_accuracy_macro, **kwargs)

                if stage == "val":
                    self._val_reco_data["pflow_class"].append(particle_class_preds.detach().cpu().numpy())
                    self._val_reco_data["truth_class"].append(particle_class_labels.detach().cpu().numpy())

                    # Accumulate jet resolution data (first 1000 events only)
                    _MAX_JET_EVENTS = 1000
                    if self._val_jet_event_count < _MAX_JET_EVENTS:
                        _take = min(particle_class_preds.shape[0], _MAX_JET_EVENTS - self._val_jet_event_count)
                        self._val_jet_data["pred_class"].append(particle_class_preds[:_take].detach().cpu().numpy())
                        self._val_jet_data["truth_class"].append(particle_class_labels[:_take].cpu().numpy())
                        # Particle kinematics (scaled — inverse-transformed at epoch end)
                        if "regression" in reco_final:
                            reco_obj = self.model.reco_target_object
                            for _field in ("pt", "eta", "sinphi", "cosphi"):
                                _pk = f"reco_pflow_{_field}"
                                _tk = f"{reco_obj}_{_field}"
                                if _pk in reco_final["regression"]:
                                    self._val_jet_data[f"pred_{_field}"].append(
                                        reco_final["regression"][_pk][:_take].detach().float().cpu().numpy()
                                    )
                                if _tk in labels:
                                    self._val_jet_data[f"truth_{_field}"].append(
                                        labels[_tk][:_take].float().cpu().numpy()
                                    )
                        # Node-level data (physical space — no transform needed).
                        # node_valid/node_is_track are bool, rest may be bf16 under AMP.
                        for _nk in ("node_valid", "node_is_track"):
                            if _nk in labels:
                                self._val_jet_data[_nk].append(labels[_nk][:_take].cpu().numpy())
                        for _nk in ("node_eta", "node_phi", "node_e"):
                            if _nk in labels:
                                self._val_jet_data[_nk].append(labels[_nk][:_take].float().cpu().numpy())
                        if "calo_hard_scatter_energy" in labels:
                            self._val_jet_data["calo_hs_energy"].append(
                                labels["calo_hard_scatter_energy"][:_take].float().cpu().numpy()
                            )
                        if "calo_hard_scatter_energy_frac" in labels:
                            self._val_jet_data["calo_hs_frac"].append(
                                labels["calo_hard_scatter_energy_frac"][:_take].float().cpu().numpy()
                            )
                        # calo_prob: full node-space sigmoid probs from Stream B
                        _calo_final = preds.get("calo_final", {})
                        if "calo_mask" in _calo_final and "calo_node_prob" in _calo_final["calo_mask"]:
                            self._val_jet_data["calo_prob"].append(
                                _calo_final["calo_mask"]["calo_node_prob"][:_take].detach().float().cpu().numpy()
                            )
                        # reco_node_indices / reco_is_track for pred-path purity
                        if hasattr(self.model, "_reco_node_indices"):
                            _rni = self.model._reco_node_indices[:_take]  # (take, 1400)
                            self._val_jet_data["reco_node_indices"].append(_rni.cpu().numpy())
                            _is_track_full = labels["node_is_track"][:_take]  # (take, 5500)
                            _safe = _rni.clamp(0, _is_track_full.shape[1] - 1)
                            _rit = _is_track_full.gather(1, _safe)  # (take, 1400)
                            self._val_jet_data["reco_is_track"].append(_rit.cpu().numpy())
                        self._val_jet_event_count += _take

                truth_valid = particle_class_labels < 5
                pred_valid = particle_class_preds < 5
                self.reco_eff(pred_valid, truth_valid)
                self.reco_pur(pred_valid, truth_valid)
                self.log(f"{stage}/reco_eff", self.reco_eff, **kwargs)
                self.log(f"{stage}/reco_pur", self.reco_pur, **kwargs)

        if "mask" in reco_final:
            reco_target_obj = self.model.reco_target_object
            valid_key = f"{reco_target_obj}_valid"
            mask_key = f"{reco_target_obj}_node_valid"
            if valid_key in labels and mask_key in labels and hasattr(self.model, "_reco_node_indices"):
                truth_valid = labels[valid_key]
                pred_masks = list(reco_final["mask"].values())[0].squeeze()  # (B, num_objects, max_reco_nodes)

                # Reindex truth masks from full node space (5500) to reco node space (1400)
                reco_idx = self.model._reco_node_indices  # (B, max_reco_nodes)
                full_truth = labels[mask_key]  # (B, num_objects, max_nodes)
                truth_masks = full_truth.gather(2, reco_idx.unsqueeze(1).expand(-1, full_truth.shape[1], -1)).squeeze()

                if truth_valid.any():
                    tv = truth_valid.squeeze()
                    if pred_masks.dim() > 1 and tv.dim() > 0:
                        pm = pred_masks[tv]
                        tm_v = truth_masks[tv]
                        if len(pm) > 0 and pm.sum(-1).gt(0).any():
                            recall_idx = tm_v.sum(-1) > 0
                            if recall_idx.any():
                                self.log(f"{stage}/reco_mask_recall", self.MI.eff(pm[recall_idx], tm_v[recall_idx]), **kwargs)
                            pur_idx = pm.sum(-1) > 0
                            if pur_idx.any():
                                self.log(f"{stage}/reco_mask_purity", self.MI.pur(pm[pur_idx], tm_v[pur_idx]), **kwargs)
