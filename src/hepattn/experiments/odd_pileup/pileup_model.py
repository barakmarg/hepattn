import torch
from torch import Tensor, nn
import torch.nn.functional as F
from hepattn.models.transformer import Encoder

class TracksMlp(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.mlp = net
    
    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)

class CaloMlp(nn.Module):
    def __init__(self, net: nn.Module, final_activation: nn.Module = None):
        super().__init__()
        self.mlp = net
        self.act = final_activation if final_activation else nn.Identity()
    
    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.mlp(x))

class PileupRemovalModel(nn.Module):
    def __init__(
        self,
        input_nets: nn.ModuleList,
        encoder: Encoder,
        tracks_mlp: nn.Module,
        calo_mlp: nn.Module,
        dim: int,
        raw_variables: list[str] | None = None,
        input_sort_field: str | None = None,
        track_loss_weight: float = 1.0,
        calo_loss_weight: float = 1.0,
        track_threshold: float = 0.5,
        track_pos_weight: float = 17.0,
        calo_signal_threshold: float = 0.05,
        calo_signal_weight: float = 15.5,
    ):
        super().__init__()
        self.input_nets = input_nets
        self.encoder = encoder
        self.tracks_mlp = tracks_mlp
        self.calo_mlp = calo_mlp
        self.dim = dim
        self.raw_variables = raw_variables or []
        self.input_sort_field = input_sort_field
        self.track_loss_weight = track_loss_weight
        self.calo_loss_weight = calo_loss_weight
        self.track_threshold = track_threshold
        self.track_pos_weight = track_pos_weight
        self.calo_signal_threshold = calo_signal_threshold
        self.calo_signal_weight = calo_signal_weight

        # Define tasks list for the ModelWrapper to iterate over if needed
        # (Even though we calculate loss internally, the wrapper might check this)
        self.tasks = nn.ModuleList([]) 

    def forward(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        # 1. Prepare Input names
        input_names = [input_net.input_name for input_net in self.input_nets]
        x = {}

        # 2. Pass Raw Variables (Important for Loss!)
        for raw_var in self.raw_variables:
            if raw_var in inputs:
                x[raw_var] = inputs[raw_var]

        # 3. Embed Inputs (InputNets)
        for input_net in self.input_nets:
            input_name = input_net.input_name
            x[input_name + "_embed"] = input_net(inputs)
            x[input_name + "_valid"] = inputs[input_name + "_valid"]
            
            # Helper mask to know which node came from where
            device = inputs[input_name + "_valid"].device
            x[f"key_is_{input_name}"] = torch.cat(
                [torch.full((inputs[i + "_valid"].shape[-1],), i == input_name, device=device, dtype=torch.bool) for i in input_names], dim=-1
            )

        # 4. Merge inputs (Tracks + Clusters)
        x["key_embed"] = torch.concatenate([x[input_name + "_embed"] for input_name in input_names], dim=-2)
        x["key_valid"] = torch.concatenate([x[input_name + "_valid"] for input_name in input_names], dim=-1)

        # Expose sort coordinate for windowed attention (e.g. deltaR_idx for Z-order sort)
        if self.input_sort_field is not None:
            x[f"key_{self.input_sort_field}"] = inputs.get(f"node_{self.input_sort_field}")

        # 5. Encoder
        if self.encoder is not None:
            # Note: The Encoder expects 'kv_mask' for masking
            x["key_embed"] = self.encoder(
                x["key_embed"], 
                x_sort_value=x.get(f"key_{self.input_sort_field}"), 
                kv_mask=x.get("key_valid")
            )
        # 6. Apply Heads
        latent = x["key_embed"]
        
        # Track Head (Logits)
        track_logits = self.tracks_mlp(latent) 
        
        # Calo Head (Fraction 0-1)
        calo_frac = self.calo_mlp(latent)
        # Return dictionary used by Loss and Predict
        return {
            "track_logits": track_logits,
            "calo_frac": calo_frac,
            "key_valid": x["key_valid"],
            # Raw variables passed through for loss calculation
            "node_e": x.get("node_e"),
            "is_track": x.get("node_is_track"),
        }

    def predict(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Inference output generation"""
        track_probs = torch.sigmoid(outputs["track_logits"])
        return {
            "final": {
                "track_is_hard_scatter": track_probs > self.track_threshold,
                "track_prob": track_probs,
                "calo_hs_fraction": outputs["calo_frac"],
            }
        }

    def loss(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        """Physics-Weighted Loss"""
        
        # 1. Create Masks
        valid = outputs["key_valid"].bool()
        # "node_is_track" comes from inputs, ensure it's bool
        is_track = outputs["is_track"].bool().squeeze(-1) 
        
        track_mask = valid & is_track
        cluster_mask = valid & (~is_track)

        # --- Track Loss (weighted BCE) ---
        if track_mask.any():
            track_pred = outputs["track_logits"].squeeze(-1)[track_mask]
            track_target = targets["tracks_mask"].float()[track_mask]
            pos_weight = torch.tensor(self.track_pos_weight, device=valid.device)
            loss_tracks = F.binary_cross_entropy_with_logits(track_pred, track_target, pos_weight=pos_weight)
        else:
            loss_tracks = torch.tensor(0.0, device=valid.device, requires_grad=True)

        # --- Cluster Loss (l1 on Energy) ---
        if cluster_mask.any():
            pred_alpha = outputs["calo_frac"].squeeze(-1)[cluster_mask]
            E_total = outputs["node_e"].squeeze(-1)[cluster_mask]
            E_HS_true = targets["calo_hard_scatter_energy"][cluster_mask]
            is_signal = (E_HS_true / (E_total + 1e-8) > self.calo_signal_threshold).float()
            weights = 1 + self.calo_signal_weight * is_signal
            E_HS_pred = pred_alpha * E_total
            loss_calo = torch.mean(torch.abs(E_HS_pred - E_HS_true) * weights)

        else:
            loss_calo = torch.tensor(0.0, device=valid.device, requires_grad=True)

        # ModelWrapper.log_losses expects nested dict: {layer: {task: {loss_name: value}}}
        return {
            "final": {
                "tracks": {"bce": self.track_loss_weight * loss_tracks},
                "calo": {"l1": self.calo_loss_weight * loss_calo},
            }
        }