
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from hepattn.models.dense import Dense
from hepattn.models.transformer import Encoder


class PileupRemovalModel(nn.Module):
    def __init__(
        self,
        input_nets: nn.ModuleList,
        encoder: Encoder,
        tracks_mlp: Dense,
        calo_mlp: Dense,
        vertex_mlp: Dense,
        dim: int,
        raw_variables: list[str] | None = None,
        input_sort_field: str | None = None,
        track_loss_weight: float = 1.0,
        calo_loss_weight: float = 1.0,
        vertex_loss_weight: float = 1.0,
        track_threshold: float = 0.5,
        track_pos_weight: float = 17.0,
        calo_signal_threshold: float = 0.05,
        calo_signal_weight: float = 15.5,
        enable_vertex_token: bool = False,
        use_global_context: bool = True,
    ):
        super().__init__()
        self.input_nets = input_nets
        self.encoder = encoder
        self.tracks_mlp = tracks_mlp
        self.calo_mlp = calo_mlp
        self.vertex_mlp = vertex_mlp
        self.dim = dim
        self.raw_variables = raw_variables or []
        self.input_sort_field = input_sort_field
        self.track_loss_weight = track_loss_weight
        self.calo_loss_weight = calo_loss_weight
        self.vertex_loss_weight = vertex_loss_weight
        self.track_threshold = track_threshold
        self.track_pos_weight = track_pos_weight
        self.calo_signal_threshold = calo_signal_threshold
        self.calo_signal_weight = calo_signal_weight
        self.enable_vertex_token = enable_vertex_token
        self.use_global_context = use_global_context

        # Learnable vertex token: a single global token prepended to each event's sequence.
        # Initialized to zero; the encoder learns to populate it with vertex-relevant info.
        if self.enable_vertex_token:
            self.vertex_token = nn.Parameter(torch.zeros(1, 1, dim))

        # Define tasks list for the ModelWrapper to iterate over if needed
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

        # 5a. Optionally append learnable vertex token to the node sequence.
        # The truth vz is stored separately to supervise the vertex regression head.
        n_vertex_tokens = 0
        if self.enable_vertex_token:
            B = x["key_embed"].shape[0]
            vtx_embed = self.vertex_token.expand(B, -1, -1)  # (B, 1, dim)
            vtx_valid = torch.ones(B, 1, dtype=torch.bool, device=vtx_embed.device)

            # Append to node sequence
            x["key_embed"] = torch.cat([x["key_embed"], vtx_embed], dim=1)   # (B, N+1, dim)
            x["key_valid"] = torch.cat([x["key_valid"], vtx_valid], dim=1)   # (B, N+1)

            # Sort value: place vertex token last (after all sorted regular tokens)
            if self.input_sort_field is not None:
                sort_vals = x[f"key_{self.input_sort_field}"]              # (B, N)
                vtx_sort = torch.full((sort_vals.shape[0], 1), float("inf"), device=sort_vals.device)
                x[f"key_{self.input_sort_field}"] = torch.cat([sort_vals, vtx_sort], dim=1)

            # Store truth vz to supervise the vertex regression head (not used as input)
            if "vertex_token_features" in inputs:
                x["true_vertex_z"] = inputs["vertex_token_features"]          # (B, 1, 1)

            n_vertex_tokens = 1

        # 5b. Encoder
        if self.encoder is not None:
            x["key_embed"] = self.encoder(
                x["key_embed"],
                x_sort_value=x.get(f"key_{self.input_sort_field}"),
                kv_mask=x.get("key_valid"),
                n_global_tokens=n_vertex_tokens,
            )

        # 5c. Extract and slice off vertex token before MLP heads.
        # The post-encoder vertex token captures global event context via attention.
        if n_vertex_tokens > 0:
            x["vtx_context"] = x["key_embed"][:, -1, :]          # (B, dim)
            x["key_embed"] = x["key_embed"][:, :-n_vertex_tokens, :]
            x["key_valid"] = x["key_valid"][:, :-n_vertex_tokens]

        latent = x["key_embed"]   # [B, N, D]
        mask = x["key_valid"].bool().unsqueeze(-1)   # [B, N, 1]

        # 6. Determine context for MLP heads:
        #    - Vertex token context (preferred when enable_vertex_token=True)
        #    - Global mean context as fallback
        if "vtx_context" in x:
            context = x["vtx_context"]                            # (B, dim)
        elif self.use_global_context:
            context = (latent * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)  # [B, D]
        else:
            context = None

        # 7. Apply Heads (conditioned on context)
        track_logits = self.tracks_mlp(latent, context=context)
        calo_frac = self.calo_mlp(latent, context=context)

        # 8. Vertex regression: predict truth vz from the post-encoder vertex token
        vertex_pred = None
        if "vtx_context" in x:
            vertex_pred = self.vertex_mlp(x["vtx_context"])       # (B, 1)

        return {
            "track_logits": track_logits,
            "calo_frac": calo_frac,
            "vertex_pred": vertex_pred,
            "true_vertex_z": x.get("true_vertex_z"),
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
                "vertex_z_pred": outputs["vertex_pred"],
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
            is_signal = (E_HS_true  > self.calo_signal_threshold).float()
            weights = 1 + self.calo_signal_weight * is_signal
            E_HS_pred = pred_alpha * E_total
            loss_calo = torch.mean(torch.abs(E_HS_pred - E_HS_true) * weights)
        else:
            loss_calo = torch.tensor(0.0, device=valid.device, requires_grad=True)

        # --- Vertex Regression Loss (MAE on normalized vz, std=84.891 mm) ---
        if outputs.get("vertex_pred") is not None and outputs.get("true_vertex_z") is not None:
            v_pred = outputs["vertex_pred"]                        # (B, 1)
            v_true = outputs["true_vertex_z"].squeeze(-1)          # (B, 1)
            loss_vertex = F.l1_loss(v_pred / 84.891, v_true / 84.891)
        else:
            loss_vertex = torch.tensor(0.0, device=valid.device, requires_grad=True)

        # ModelWrapper.log_losses expects nested dict: {layer: {task: {loss_name: value}}}
        return {
            "final": {
                "tracks": {"bce": self.track_loss_weight * loss_tracks},
                "calo": {"l1": self.calo_loss_weight * loss_calo},
                "vertex": {"mae": self.vertex_loss_weight * loss_vertex},
            }
        }
