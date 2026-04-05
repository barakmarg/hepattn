import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hepattn.models.dense import Dense
from hepattn.models.loss import loss_fns
from hepattn.models.task import Task


class CaloHitMaskTask(Task):
    """Multi-query calo hit mask with max-pooled master mask and deep supervision.

    Each of Q hybrid queries produces its own mask over all nodes via dot-product.
    The master mask is the max over all query masks, representing the overall
    calo HS classification. BCE + Dice loss is computed on the pooled master mask
    vs calo_is_hs target at every decoder layer.

    The per-query masks are used for masked cross-attention so each query
    only attends to the specific calo nodes it claimed.
    """

    def __init__(
        self,
        name: str,
        input_hit: str,
        input_object: str,
        losses: dict[str, float],
        dim: int,
        null_weight: float = 0.1,
        logit_scale: float = 4.0,
        pred_threshold: float = 0.5,
        attn_threshold: float = 0.1,
        mask_attn: bool = True,
        hs_energy_threshold: float = 0.15,
        hs_frac_threshold: float = 0.2,
        has_intermediate_loss: bool = True,
    ):
        super().__init__(has_intermediate_loss=has_intermediate_loss, permute_loss=False)
        self.name = name
        self.input_hit = input_hit
        self.input_object = input_object
        self.losses = losses
        self.null_weight = null_weight
        self.logit_scale = logit_scale
        self.pred_threshold = pred_threshold
        self.attn_threshold = attn_threshold
        self.mask_attn = mask_attn
        self.hs_energy_threshold = hs_energy_threshold
        self.hs_frac_threshold = hs_frac_threshold
        self.outputs = ["calo_node_logit", "per_query_logit"]

        self.object_net = Dense(dim, dim)
        self.hit_net = Dense(dim, dim)

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        x_object = self.object_net(x[f"{self.input_object}_embed"])  # (B, Q, D)
        x_hit = self.hit_net(x[f"{self.input_hit}_embed"])  # (B, N, D)

        # Per-query-per-node logits
        per_query_logit = self.logit_scale * torch.einsum("bqd,bnd->bqn", x_object, x_hit)
        # (B, Q, N)

        # Zero out invalid nodes
        node_valid = x[f"{self.input_hit}_valid"]
        if node_valid is not None:
            invalid_mask = ~node_valid.unsqueeze(-2).expand_as(per_query_logit)
            per_query_logit = per_query_logit.masked_fill(invalid_mask, -100.0)

        # Pool to master mask: max over query dimension
        master_logit = per_query_logit.max(dim=1)[0]  # (B, N)
        # Reshape to (B, 1, N) for compatibility with mask loss functions
        master_logit = master_logit.unsqueeze(1)

        return {
            "calo_node_logit": master_logit,  # (B, 1, N) for loss
            "per_query_logit": per_query_logit,  # (B, Q, N) for attention masking
        }

    def attn_mask(self, outputs: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        if not self.mask_attn:
            return {}

        # Per-query attention masks: each query attends to its claimed nodes
        attn_mask = outputs["per_query_logit"].detach().sigmoid() >= self.attn_threshold
        # (B, Q, N) — the calo-only restriction is handled by CaloMaskFormerDecoder's inverted prior

        return {self.input_hit: attn_mask}

    def predict(self, outputs: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        prob = outputs["calo_node_logit"].detach().sigmoid()  # (B, 1, N)
        return {
            "calo_node_prob": prob.squeeze(1),  # (B, N)
            "calo_node_valid": (prob >= self.pred_threshold).squeeze(1),  # (B, N)
        }

    def loss(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        is_track = targets["node_is_track"].bool().squeeze(-1)  # (B, N)
        valid = targets["node_valid"].bool()  # (B, N)

        # Target: calo cluster is hard scatter if HS fraction and absolute energy exceed thresholds
        calo_hs_energy = targets["calo_hard_scatter_energy"]  # (B, N)
        calo_hs_frac = targets["calo_hard_scatter_energy_frac"]  # (B, N)
        calo_is_hs = (calo_hs_frac > self.hs_frac_threshold) & (calo_hs_energy > self.hs_energy_threshold) & ~is_track & valid
        # Shape to (B, 1, N) to match master_logit
        target = calo_is_hs.unsqueeze(1).float()

        output = outputs["calo_node_logit"]  # (B, 1, N)

        # Only calo nodes contribute to loss (not tracks, not padding)
        input_pad_mask = valid & ~is_track  # (B, N)

        # Object valid mask: always True (single pooled object)
        object_valid = torch.ones(output.shape[0], 1, dtype=torch.bool, device=output.device)

        sample_weight = target + self.null_weight * (1 - target)

        losses = {}
        for loss_fn, loss_weight in self.losses.items():
            losses[loss_fn] = loss_weight * loss_fns[loss_fn](
                output, target, object_valid_mask=object_valid, input_pad_mask=input_pad_mask, sample_weight=sample_weight
            )
        return losses

    def cost(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        return {}


class PileupCaloFractionTaskV2(Task):
    """Calorimeter energy fraction regression conditioned on the global calo query.

    Same architecture as PileupCaloFractionTask but with two key changes:
    1. Uses query_embed[:, 0:1, :] (global calo query at slot 0) for conditioning
    2. L1 loss is ONLY computed on true HS clusters (calo_hard_scatter_energy > threshold)
    """

    def __init__(
        self,
        name: str,
        input_hit: str,
        input_object: str,
        dim: int,
        loss_weight: float = 1.0,
        signal_weight: float = 25.3,
        hs_energy_threshold: float = 0.15,
        hs_frac_threshold: float = 0.2,
        has_intermediate_loss: bool = False,
    ):
        super().__init__(has_intermediate_loss=has_intermediate_loss, permute_loss=False)
        self.name = name
        self.input_hit = input_hit
        self.input_object = input_object
        self.loss_weight = loss_weight
        self.signal_weight = signal_weight
        self.hs_energy_threshold = hs_energy_threshold
        self.hs_frac_threshold = hs_frac_threshold
        self.outputs = ["calo_frac"]

        self.query_net = Dense(dim, dim)
        self.hit_net = Dense(dim, dim)
        self.output_net = Dense(
            2 * dim,
            1,
            hidden_layers=[256, 128, 64, 32],
            activation=nn.SiLU(),
            final_activation=nn.Sigmoid(),
        )

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        # Use only the global calo query (slot 0) for conditioning
        q = self.query_net(x[f"{self.input_object}_embed"][:, 0:1, :])  # (B, 1, D)
        h = self.hit_net(x[f"{self.input_hit}_embed"])  # (B, N, D)
        q_expanded = q.expand_as(h)  # (B, N, D)
        combined = torch.cat([h, q_expanded], dim=-1)  # (B, N, 2D)
        frac = self.output_net(combined).squeeze(-1)  # (B, N)
        return {"calo_frac": frac}

    def predict(self, outputs: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        return {"calo_hs_fraction": outputs["calo_frac"].detach()}

    def loss(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        is_track = targets["node_is_track"].bool().squeeze(-1)
        valid = targets["node_valid"].bool()
        calo_hs_energy = targets["calo_hard_scatter_energy"]

        calo_hs_frac = targets["calo_hard_scatter_energy_frac"]

        # CRITICAL: Only compute loss on true HS clusters (fraction AND absolute energy thresholds)
        mask = valid & ~is_track & (calo_hs_frac > self.hs_frac_threshold) & (calo_hs_energy > self.hs_energy_threshold)

        if mask.any():
            pred_frac = outputs["calo_frac"][mask]
            E_total = targets["node_e"].squeeze(-1)[mask]
            E_HS_true = calo_hs_energy[mask]

            E_HS_pred = pred_frac * E_total
            loss = torch.mean(torch.abs(E_HS_pred - E_HS_true))
        else:
            loss = torch.tensor(0.0, device=outputs["calo_frac"].device, requires_grad=True)

        return {"l1": self.loss_weight * loss}

    def cost(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        return {}

    def attn_mask(self, outputs: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        return {}


class CaloNodeMaskTask(Task):
    """Node-classification calo mask. Classifies each node as HS/pileup
    directly from enriched node embeddings via a Dense MLP head.

    Drop-in replacement for CaloHitMaskTask: same loss/predict/output keys,
    but forward() uses a single Dense head instead of query-hit dot-product.
    """

    def __init__(
        self,
        name: str,
        input_hit: str,
        losses: dict[str, float],
        dim: int,
        net: nn.Module,
        null_weight: float = 0.05,
        pred_threshold: float = 0.2,
        hs_energy_threshold: float = 0.15,
        hs_frac_threshold: float = 0.05,
        has_intermediate_loss: bool = False,
    ):
        super().__init__(has_intermediate_loss=has_intermediate_loss, permute_loss=False)
        self.name = name
        self.input_hit = input_hit
        self.losses = losses
        self.null_weight = null_weight
        self.pred_threshold = pred_threshold
        self.hs_energy_threshold = hs_energy_threshold
        self.hs_frac_threshold = hs_frac_threshold
        self.outputs = ["calo_node_logit"]

        self.net = net

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        logit = self.net(x[f"{self.input_hit}_embed"]).squeeze(-1)  # (B, N)

        node_valid = x[f"{self.input_hit}_valid"]
        if node_valid is not None:
            logit = logit.masked_fill(~node_valid, -100.0)

        # (B, 1, N) for compatibility with mask loss functions
        return {"calo_node_logit": logit.unsqueeze(1)}

    def predict(self, outputs: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        prob = outputs["calo_node_logit"].detach().sigmoid()  # (B, 1, N)
        return {
            "calo_node_prob": prob.squeeze(1),  # (B, N)
            "calo_node_valid": (prob >= self.pred_threshold).squeeze(1),  # (B, N)
        }

    def loss(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        is_track = targets["node_is_track"].bool().squeeze(-1)
        valid = targets["node_valid"].bool()

        calo_hs_energy = targets["calo_hard_scatter_energy"]
        calo_hs_frac = targets["calo_hard_scatter_energy_frac"]
        calo_is_hs = (calo_hs_frac > self.hs_frac_threshold) & (calo_hs_energy > self.hs_energy_threshold) & ~is_track & valid

        target = calo_is_hs.unsqueeze(1).float()  # (B, 1, N)
        output = outputs["calo_node_logit"]  # (B, 1, N)

        input_pad_mask = valid & ~is_track
        object_valid = torch.ones(output.shape[0], 1, dtype=torch.bool, device=output.device)
        sample_weight = target + self.null_weight * (1 - target)

        losses = {}
        for loss_fn, loss_weight in self.losses.items():
            losses[loss_fn] = loss_weight * loss_fns[loss_fn](
                output, target, object_valid_mask=object_valid, input_pad_mask=input_pad_mask, sample_weight=sample_weight
            )
        return losses

    def cost(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        return {}

    def attn_mask(self, outputs: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        return {}


class CaloNodeFractionTask(Task):
    """Node-classification calo fraction. Regresses HS energy fraction
    directly from enriched node embeddings via a Dense MLP head.

    Drop-in replacement for PileupCaloFractionTaskV2: same loss/predict/output keys,
    but forward() uses a single Dense head instead of query-conditioned MLP.
    """

    def __init__(
        self,
        name: str,
        input_hit: str,
        dim: int,
        net: nn.Module,
        loss_weight: float = 0.01,
        hs_energy_threshold: float = 0.15,
        hs_frac_threshold: float = 0.05,
        has_intermediate_loss: bool = False,
    ):
        super().__init__(has_intermediate_loss=has_intermediate_loss, permute_loss=False)
        self.name = name
        self.input_hit = input_hit
        self.loss_weight = loss_weight
        self.hs_energy_threshold = hs_energy_threshold
        self.hs_frac_threshold = hs_frac_threshold
        self.outputs = ["calo_frac"]

        self.net = net

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        frac = self.net(x[f"{self.input_hit}_embed"]).squeeze(-1)  # (B, N)
        return {"calo_frac": frac}

    def predict(self, outputs: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        return {"calo_hs_fraction": outputs["calo_frac"].detach()}

    def loss(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        is_track = targets["node_is_track"].bool().squeeze(-1)
        valid = targets["node_valid"].bool()
        calo_hs_energy = targets["calo_hard_scatter_energy"]
        calo_hs_frac = targets["calo_hard_scatter_energy_frac"]

        mask = valid & ~is_track & (calo_hs_frac > self.hs_frac_threshold) & (calo_hs_energy > self.hs_energy_threshold)

        if mask.any():
            pred_frac = outputs["calo_frac"][mask]
            E_total = targets["node_e"].squeeze(-1)[mask]
            E_HS_true = calo_hs_energy[mask]
            E_HS_pred = pred_frac * E_total
            loss = torch.mean(torch.abs(E_HS_pred - E_HS_true))
        else:
            loss = torch.tensor(0.0, device=outputs["calo_frac"].device, requires_grad=True)

        return {"l1": self.loss_weight * loss}

    def cost(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        return {}

    def attn_mask(self, outputs: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        return {}
