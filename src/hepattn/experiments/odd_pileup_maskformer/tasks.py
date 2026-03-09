import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hepattn.models.dense import Dense
from hepattn.models.task import Task


class PileupTrackClassTask(Task):
    """Binary track classification conditioned on the query (Proxy PV).

    Concatenates query and node embeddings, then applies an MLP to produce
    per-node logits. Loss is computed only on track nodes.
    """

    def __init__(
        self,
        name: str,
        input_hit: str,
        input_object: str,
        dim: int,
        loss_weight: float = 1.0,
        pos_weight: float = 16.24,
        threshold: float = 0.5,
        has_intermediate_loss: bool = False,
    ):
        super().__init__(has_intermediate_loss=has_intermediate_loss, permute_loss=False)
        self.name = name
        self.input_hit = input_hit
        self.input_object = input_object
        self.loss_weight = loss_weight
        self.threshold = threshold
        self.outputs = ["track_logit"]

        self.register_buffer("pos_weight_val", torch.tensor(pos_weight))

        self.query_net = Dense(dim, dim)
        self.hit_net = Dense(dim, dim)
        self.output_net = Dense(2 * dim, 1, hidden_layers=[256, 128, 64, 32], activation=nn.SiLU())

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        q = self.query_net(x[f"{self.input_object}_embed"])  # (B, 1, D)
        h = self.hit_net(x[f"{self.input_hit}_embed"])  # (B, N, D)
        q_expanded = q.expand_as(h)  # (B, N, D)
        combined = torch.cat([h, q_expanded], dim=-1)  # (B, N, 2D)
        logit = self.output_net(combined).squeeze(-1)  # (B, N)
        return {"track_logit": logit}

    def predict(self, outputs: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        prob = outputs["track_logit"].detach().sigmoid()
        return {
            "track_prob": prob,
            "track_is_hs": prob >= self.threshold,
        }

    def loss(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        is_track = targets["node_is_track"].bool().squeeze(-1)
        valid = targets["node_valid"].bool()
        mask = valid & is_track

        if mask.any():
            pred = outputs["track_logit"][mask]
            target = targets["tracks_mask"].float()[mask]
            loss = F.binary_cross_entropy_with_logits(pred, target, pos_weight=self.pos_weight_val)
        else:
            loss = torch.tensor(0.0, device=outputs["track_logit"].device, requires_grad=True)

        return {"bce": self.loss_weight * loss}

    def cost(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        return {}

    def attn_mask(self, outputs: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        return {}


class PileupCaloFractionTask(Task):
    """Calorimeter energy fraction regression conditioned on the query (Proxy PV).

    Concatenates query and node embeddings, then applies an MLP with sigmoid
    to produce per-node energy fractions in [0, 1]. Loss is computed only on
    cluster nodes using energy-weighted L1.
    """

    def __init__(
        self,
        name: str,
        input_hit: str,
        input_object: str,
        dim: int,
        loss_weight: float = 1.0,
        signal_weight: float = 25.3,
        signal_threshold: float = 0.05,
        has_intermediate_loss: bool = False,
    ):
        super().__init__(has_intermediate_loss=has_intermediate_loss, permute_loss=False)
        self.name = name
        self.input_hit = input_hit
        self.input_object = input_object
        self.loss_weight = loss_weight
        self.signal_weight = signal_weight
        self.signal_threshold = signal_threshold
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
        q = self.query_net(x[f"{self.input_object}_embed"])  # (B, 1, D)
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
        mask = valid & ~is_track

        if mask.any():
            pred_frac = outputs["calo_frac"][mask]
            E_total = targets["node_e"].squeeze(-1)[mask]
            E_HS_true = targets["calo_hard_scatter_energy"][mask]
            is_signal = (E_HS_true > self.signal_threshold).float()
            weights = 1 + self.signal_weight * is_signal
            E_HS_pred = pred_frac * E_total
            loss = torch.mean(torch.abs(E_HS_pred - E_HS_true) * weights)
        else:
            loss = torch.tensor(0.0, device=outputs["calo_frac"].device, requires_grad=True)

        return {"l1": self.loss_weight * loss}

    def cost(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        return {}

    def attn_mask(self, outputs: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        return {}
