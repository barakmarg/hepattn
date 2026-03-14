import torch
from torch import Tensor, nn

from hepattn.models.decoder import MaskFormerDecoder
from hepattn.models.task import IncidenceRegressionTask, ObjectClassificationTask


class TwoStreamMaskFormer(nn.Module):
    """Two-Stream Causally-Conditioned Hybrid Query MaskFormer.

    Stream A (Track MaskFormer): Finds PV and classifies HS tracks. Calo gradients
    are strictly blocked from entering this stream.

    Bridge: Extracts physical HS tracks (teacher-forced during training, predicted
    during inference), detaches them, and constructs hybrid queries for calo.

    Stream B (Calo MaskFormer): Hybrid queries (HS tracks + 1 global neutral query)
    execute a second MaskFormer loop over calo nodes using masked cross-attention.
    """

    def __init__(
        self,
        input_nets: nn.ModuleList,
        encoder: nn.Module | None,
        track_decoder: MaskFormerDecoder,
        track_tasks: nn.ModuleList,
        calo_decoder: MaskFormerDecoder,
        calo_tasks: nn.ModuleList,
        dim: int,
        max_hs_tracks: int = 200,
        teacher_forcing: bool = True,
        target_object: str = "particle",
        pooling: nn.Module | None = None,
        input_sort_field: str | None = None,
        raw_variables: list[str] | None = None,
    ):
        super().__init__()

        self.input_nets = input_nets
        self.encoder = encoder

        # Stream A
        self.track_decoder = track_decoder
        self.track_tasks = track_tasks
        self.track_decoder.tasks = track_tasks

        # Stream B
        self.calo_decoder = calo_decoder
        self.calo_tasks = calo_tasks
        self.calo_decoder.tasks = calo_tasks

        self.dim = dim
        self.max_hs_tracks = max_hs_tracks
        self.teacher_forcing = teacher_forcing
        self.target_object = target_object
        self.pooling = pooling
        self.input_sort_field = input_sort_field
        self.raw_variables = raw_variables or []

        # Learnable query parameters
        self.track_query_initial = nn.Parameter(torch.randn(1, dim))
        self.calo_global_query = nn.Parameter(torch.randn(1, dim))

        # Expose all tasks for ModelWrapper compatibility (parameter discovery, etc.)
        self.tasks = nn.ModuleList([*track_tasks, *calo_tasks])

    def forward(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        input_names = [input_net.input_name for input_net in self.input_nets]

        assert "key" not in input_names, "'key' input name is reserved."
        assert "query" not in input_names, "'query' input name is reserved."

        x = {}

        for raw_var in self.raw_variables:
            if raw_var in inputs:
                x[raw_var] = inputs[raw_var]

        # Store positional encodings if decoder needs them
        if self.track_decoder.preserve_posenc or self.calo_decoder.preserve_posenc:
            assert all(input_net.posenc is not None for input_net in self.input_nets)
            x["key_posenc"] = torch.concatenate([input_net.posenc(inputs) for input_net in self.input_nets], dim=-2)

        # Embed input objects
        for input_net in self.input_nets:
            input_name = input_net.input_name
            x[input_name + "_embed"] = input_net(inputs)
            x[input_name + "_valid"] = inputs[input_name + "_valid"]

            device = inputs[input_name + "_valid"].device
            x[f"key_is_{input_name}"] = torch.cat(
                [torch.full((inputs[i + "_valid"].shape[-1],), i == input_name, device=device, dtype=torch.bool) for i in input_names],
                dim=-1,
            )

        # Merge inputs
        x["key_embed"] = torch.concatenate([x[input_name + "_embed"] for input_name in input_names], dim=-2)
        x["key_valid"] = torch.concatenate([x[input_name + "_valid"] for input_name in input_names], dim=-1)

        batch_size = x["key_valid"].shape[0]

        if batch_size == 1 and x["key_valid"].all():
            x["key_valid"] = None

        # Sort for windowed attention
        if self.input_sort_field is not None:
            x[f"key_{self.input_sort_field}"] = torch.concatenate(
                [inputs[input_name + "_" + self.input_sort_field] for input_name in input_names], dim=-1
            )

        # Shared encoder
        if self.encoder is not None:
            x["key_embed"] = self.encoder(x["key_embed"], x_sort_value=x.get(f"key_{self.input_sort_field}"), kv_mask=x.get("key_valid"))

        # Unmerge back to per-input-type embeddings
        for input_name in input_names:
            x[input_name + "_embed"] = x["key_embed"][..., x[f"key_is_{input_name}"], :]

        # =====================================================================
        # STREAM A: Track MaskFormer (Isolated)
        # =====================================================================
        x_track = dict(x)  # shallow copy
        x_track["query_embed"] = self.track_query_initial.expand(batch_size, -1, -1)
        x_track["query_valid"] = torch.full((batch_size, 1), True, device=x["key_embed"].device)

        x_track, track_outputs = self.track_decoder(x_track, input_names)

        # Final track task outputs
        track_outputs["final"] = {}
        for task in self.track_tasks:
            track_outputs["final"][task.name] = task(x_track)
            if isinstance(task, IncidenceRegressionTask):
                x_track["incidence"] = track_outputs["final"][task.name][task.outputs[0]].detach()
            if isinstance(task, ObjectClassificationTask):
                x_track["class_probs"] = track_outputs["final"][task.name][task.outputs[0]].detach()

        # =====================================================================
        # BRIDGE: Construct Hybrid Calo Queries
        # =====================================================================
        is_track = x["node_is_track"].bool().squeeze(-1)  # (B, N)

        if self.training and self.teacher_forcing:
            # Teacher forcing: use ground truth HS track mask
            hs_track_mask = x["tracks_mask"].bool() & is_track  # (B, N)
        else:
            # Inference: use predicted HS tracks from Stream A
            mask_logits = track_outputs["final"]["mask"]["pflow_node_logit"]  # (B, 1, N)
            hs_track_mask = (mask_logits.squeeze(1).sigmoid() >= 0.5) & is_track  # (B, N)

        # CRITICAL: detach node embeddings to block calo gradients from entering encoder
        node_embed_detached = x["node_embed"].detach()  # (B, N, D)

        hybrid_queries, query_valid = self._build_hybrid_queries(
            node_embed_detached, hs_track_mask, batch_size
        )

        # =====================================================================
        # STREAM B: Calo MaskFormer
        # =====================================================================
        x_calo = dict(x)  # shallow copy of shared state
        x_calo["query_embed"] = hybrid_queries
        x_calo["query_valid"] = query_valid

        x_calo, calo_outputs = self.calo_decoder(x_calo, input_names)

        # Final calo task outputs
        calo_outputs["final"] = {}
        for task in self.calo_tasks:
            calo_outputs["final"][task.name] = task(x_calo)
            if isinstance(task, IncidenceRegressionTask):
                x_calo["incidence"] = calo_outputs["final"][task.name][task.outputs[0]].detach()
            if isinstance(task, ObjectClassificationTask):
                x_calo["class_probs"] = calo_outputs["final"][task.name][task.outputs[0]].detach()

        # =====================================================================
        # COMBINE OUTPUTS from both streams
        # =====================================================================
        outputs = {}
        for layer_name, layer_out in track_outputs.items():
            outputs[f"track_{layer_name}"] = layer_out
        for layer_name, layer_out in calo_outputs.items():
            outputs[f"calo_{layer_name}"] = layer_out

        return outputs

    def _build_hybrid_queries(
        self, node_embed: Tensor, hs_mask: Tensor, batch_size: int
    ) -> tuple[Tensor, Tensor]:
        """Build padded hybrid query tensor from detached HS track embeddings.

        Parameters
        ----------
        node_embed : Tensor
            Detached node embeddings (B, N, D).
        hs_mask : Tensor
            Boolean mask of HS tracks (B, N).
        batch_size : int
            Batch size.

        Returns
        -------
        queries : Tensor
            Hybrid queries (B, max_hs_tracks + 1, D).
            Slot 0 = global calo query (learnable), slots 1.. = HS track embeddings.
        valid : Tensor
            Boolean validity mask (B, max_hs_tracks + 1).
        """
        D = node_embed.shape[-1]
        device = node_embed.device
        Q = self.max_hs_tracks + 1

        queries = torch.zeros(batch_size, Q, D, device=device, dtype=node_embed.dtype)
        valid = torch.zeros(batch_size, Q, device=device, dtype=torch.bool)

        # Slot 0: global calo query (learnable, for isolated neutral energy)
        queries[:, 0, :] = self.calo_global_query
        valid[:, 0] = True

        # Slots 1..Q-1: HS track embeddings (padded)
        for b in range(batch_size):
            track_indices = hs_mask[b].nonzero(as_tuple=True)[0]
            n = min(len(track_indices), self.max_hs_tracks)
            if n > 0:
                queries[b, 1 : 1 + n, :] = node_embed[b, track_indices[:n], :]
                valid[b, 1 : 1 + n] = True

        return queries, valid

    def loss(self, outputs: dict, targets: dict) -> dict:
        losses = {}

        for layer_name, layer_out in outputs.items():
            if layer_name.startswith("track_"):
                real_layer = layer_name[len("track_"):]
                losses[layer_name] = {}
                for task in self.track_tasks:
                    if real_layer != "final" and not task.has_intermediate_loss:
                        continue
                    if task.name in layer_out:
                        losses[layer_name][task.name] = task.loss(layer_out[task.name], targets)
            elif layer_name.startswith("calo_"):
                real_layer = layer_name[len("calo_"):]
                losses[layer_name] = {}
                for task in self.calo_tasks:
                    if real_layer != "final" and not task.has_intermediate_loss:
                        continue
                    if task.name in layer_out:
                        losses[layer_name][task.name] = task.loss(layer_out[task.name], targets)

        return losses

    def predict(self, outputs: dict) -> dict:
        preds = {}

        for layer_name, layer_out in outputs.items():
            preds[layer_name] = {}
            if layer_name.startswith("track_"):
                real_layer = layer_name[len("track_"):]
                for task in self.track_tasks:
                    if real_layer != "final" and not task.has_intermediate_loss:
                        continue
                    if task.name in layer_out:
                        preds[layer_name][task.name] = task.predict(layer_out[task.name])
            elif layer_name.startswith("calo_"):
                real_layer = layer_name[len("calo_"):]
                for task in self.calo_tasks:
                    if real_layer != "final" and not task.has_intermediate_loss:
                        continue
                    if task.name in layer_out:
                        preds[layer_name][task.name] = task.predict(layer_out[task.name])

        # Gated inference: multiply calo fraction by calo mask prediction
        if "calo_final" in preds:
            calo_final = preds["calo_final"]
            if "calo_mask" in calo_final and "calo_fraction" in calo_final:
                calo_hs_prob = calo_final["calo_mask"]["calo_node_prob"]  # (B, N)
                calo_gate = (calo_hs_prob > 0.5).float()
                calo_final["calo_fraction"]["calo_hs_fraction"] = (
                    calo_final["calo_fraction"]["calo_hs_fraction"] * calo_gate
                )

        return preds
