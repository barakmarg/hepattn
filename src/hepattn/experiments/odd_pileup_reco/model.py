import torch
from torch import Tensor, nn

from hepattn.models.decoder import MaskFormerDecoder
from hepattn.models.task import IncidenceRegressionTask, ObjectClassificationTask


class TwoStreamMaskFormer(nn.Module):
    """Two-Stream Causally-Conditioned Hybrid Query MaskFormer with Reconstruction.

    Stream A (Track MaskFormer): Finds PV and classifies HS tracks.
    Bridge: Extracts HS tracks, constructs hybrid queries for calo.
    Stream B (Calo MaskFormer): Hybrid queries classify calo HS/pileup.
    Stream C (Reconstruction): Filters to HS nodes, runs standard MaskFormer
    decoder with Hungarian matching for particle reconstruction.
    """

    def __init__(
        self,
        input_nets: nn.ModuleList,
        encoder: nn.Module | None,
        track_decoder: MaskFormerDecoder,
        track_tasks: nn.ModuleList,
        calo_decoder: nn.Module,
        calo_tasks: nn.ModuleList,
        dim: int,
        max_hs_tracks: int = 200,
        num_latent_queries: int = 1,
        teacher_forcing: bool = True,
        target_object: str = "particle",
        pooling: nn.Module | None = None,
        input_sort_field: str | None = None,
        raw_variables: list[str] | None = None,
        # --- Stream C (reconstruction) ---
        reco_decoder: MaskFormerDecoder | None = None,
        reco_tasks: nn.ModuleList | None = None,
        reco_matcher: nn.Module | None = None,
        max_reco_calo_nodes: int = 1200,
        max_reco_nodes: int = 1400,
        reco_target_object: str = "reco_particle",
        num_reco_queries: int = 400,
        calo_hs_frac_threshold: float = 0.05,
        calo_hs_energy_threshold: float = 0.15,
        calo_pred_threshold: float = 0.2,
        reco_calo_noise_mean: int = 300,
        reco_calo_noise_std: float = 50.0,
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

        self.dim = dim
        self.max_hs_tracks = max_hs_tracks
        self.num_latent_queries = num_latent_queries
        self.teacher_forcing = teacher_forcing
        self.target_object = target_object
        self.pooling = pooling
        self.input_sort_field = input_sort_field
        self.raw_variables = raw_variables or []

        # Learnable query parameters
        self.track_query_initial = nn.Parameter(torch.randn(1, dim))
        self.calo_latent_queries = nn.Parameter(torch.randn(num_latent_queries, dim))

        # Stream C (reconstruction)
        self.reco_decoder = reco_decoder
        self.reco_tasks = reco_tasks or nn.ModuleList()
        self.reco_matcher = reco_matcher
        self.max_reco_calo_nodes = max_reco_calo_nodes
        self.max_reco_nodes = max_reco_nodes
        self.reco_target_object = reco_target_object
        self.num_reco_queries = num_reco_queries
        self.calo_hs_frac_threshold = calo_hs_frac_threshold
        self.calo_hs_energy_threshold = calo_hs_energy_threshold
        self.calo_pred_threshold = calo_pred_threshold
        self.reco_calo_noise_mean = reco_calo_noise_mean
        self.reco_calo_noise_std = reco_calo_noise_std

        if self.reco_decoder is not None:
            self.reco_decoder.tasks = self.reco_tasks
            self.reco_query_initial = nn.Parameter(torch.randn(num_reco_queries, dim))

        # Expose all tasks for ModelWrapper compatibility
        self.tasks = nn.ModuleList([*track_tasks, *calo_tasks, *self.reco_tasks])

    def forward(self, inputs: dict[str, Tensor], targets: dict[str, Tensor] | None = None) -> dict[str, Tensor]:
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

        # Save initial encoder output for skip connection (gradients flow back)
        initial_encoder_embed = x["key_embed"]

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
            hs_track_mask = x["tracks_mask"].bool() & is_track  # (B, N)
        else:
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
        # COMBINE OUTPUTS from Streams A and B
        # =====================================================================
        outputs = {}
        for layer_name, layer_out in track_outputs.items():
            outputs[f"track_{layer_name}"] = layer_out
        for layer_name, layer_out in calo_outputs.items():
            outputs[f"calo_{layer_name}"] = layer_out

        # =====================================================================
        # STREAM C: Reconstruction (optional)
        # =====================================================================
        if self.reco_decoder is not None:
            reco_outputs = self._forward_reco(
                x, initial_encoder_embed, track_outputs, calo_outputs,
                is_track, targets, input_names, batch_size,
            )
            for layer_name, layer_out in reco_outputs.items():
                outputs[f"reco_{layer_name}"] = layer_out

        return outputs

    def _forward_reco(
        self,
        x: dict[str, Tensor],
        initial_encoder_embed: Tensor,
        track_outputs: dict,
        calo_outputs: dict,
        is_track: Tensor,
        targets: dict[str, Tensor] | None,
        input_names: list[str],
        batch_size: int,
    ) -> dict[str, dict]:
        """Stream C: filter to HS nodes and run reconstruction decoder."""
        device = x["key_embed"].device
        node_valid = x.get("key_valid")
        if node_valid is None:
            node_valid = torch.ones(batch_size, x["key_embed"].shape[1], dtype=torch.bool, device=device)

        # --- Determine HS masks ---
        if self.training and self.teacher_forcing and targets is not None:
            # Teacher forcing: truth masks
            reco_track_mask = targets["tracks_mask"].bool() & is_track

            calo_hs_frac = targets["calo_hard_scatter_energy_frac"]
            calo_hs_energy = targets["calo_hard_scatter_energy"]
            truth_calo_mask = (
                (calo_hs_frac > self.calo_hs_frac_threshold)
                & (calo_hs_energy > self.calo_hs_energy_threshold)
                & ~is_track & node_valid
            )

            # Sample additional predicted-mask nodes to teach the reco to handle residual PU
            pred_calo_logits = calo_outputs["final"]["calo_mask"]["calo_node_logit"]  # (B, 1, N)
            pred_calo_mask = (pred_calo_logits.squeeze(1).sigmoid() >= self.calo_pred_threshold) & ~is_track & node_valid
            # Extra predicted nodes not already in truth
            extra_pred = pred_calo_mask & ~truth_calo_mask

            # Sample ~N(reco_calo_noise_mean, reco_calo_noise_std) extra nodes per batch element
            n_extra = int(max(0, torch.normal(
                mean=torch.tensor(float(self.reco_calo_noise_mean)),
                std=torch.tensor(self.reco_calo_noise_std),
            ).item()))

            sampled_extra = torch.zeros_like(extra_pred)
            for b in range(batch_size):
                extra_indices = extra_pred[b].nonzero(as_tuple=True)[0]
                if len(extra_indices) > 0 and n_extra > 0:
                    k = min(n_extra, len(extra_indices))
                    perm = torch.randperm(len(extra_indices), device=device)[:k]
                    sampled_extra[b, extra_indices[perm]] = True

            reco_calo_mask = truth_calo_mask | sampled_extra
        else:
            # Inference: use predicted masks
            track_logits = track_outputs["final"]["mask"]["pflow_node_logit"]
            reco_track_mask = (track_logits.squeeze(1).sigmoid() >= 0.5) & is_track

            calo_logits = calo_outputs["final"]["calo_mask"]["calo_node_logit"]
            reco_calo_mask = (calo_logits.squeeze(1).sigmoid() >= self.calo_pred_threshold) & ~is_track & node_valid

        # --- Top-k calo selection by energy ---
        node_e = x.get("node_e", torch.zeros(batch_size, x["key_embed"].shape[1], device=device))
        if node_e.dim() > 2:
            node_e = node_e.squeeze(-1)
        calo_energy_masked = torch.where(reco_calo_mask, node_e, torch.zeros_like(node_e))

        # Clamp k to actual available calo nodes
        max_k = min(self.max_reco_calo_nodes, calo_energy_masked.shape[-1])
        _, topk_indices = calo_energy_masked.topk(max_k, dim=-1, sorted=False)
        reco_calo_topk = torch.zeros_like(reco_calo_mask)
        reco_calo_topk.scatter_(1, topk_indices, True)
        reco_calo_topk = reco_calo_topk & reco_calo_mask

        # --- Combined HS node mask ---
        reco_node_mask = reco_track_mask | reco_calo_topk  # (B, N)

        # --- Skip connection with initial encoder output ---
        reco_embed = x["key_embed"] + initial_encoder_embed

        # --- Gather filtered nodes into padded tensor ---
        N_full = reco_embed.shape[1]
        D = reco_embed.shape[2]
        max_rn = self.max_reco_nodes

        filtered_embed = torch.zeros(batch_size, max_rn, D, device=device, dtype=reco_embed.dtype)
        filtered_valid = torch.zeros(batch_size, max_rn, device=device, dtype=torch.bool)
        filtered_is_track = torch.zeros(batch_size, max_rn, device=device, dtype=torch.float32)
        # Store original indices for target reindexing
        reco_node_indices = torch.zeros(batch_size, max_rn, device=device, dtype=torch.long)

        # Gather raw variables for reco tasks
        raw_keys = ["node_e", "node_pt", "node_eta", "node_sinphi", "node_cosphi", "node_is_track"]
        filtered_raw = {k: torch.zeros(batch_size, max_rn, device=device) for k in raw_keys}

        for b in range(batch_size):
            indices = reco_node_mask[b].nonzero(as_tuple=True)[0]
            n = min(len(indices), max_rn)
            if n > 0:
                filtered_embed[b, :n] = reco_embed[b, indices[:n]]
                filtered_valid[b, :n] = True
                filtered_is_track[b, :n] = is_track[b, indices[:n]].float()
                reco_node_indices[b, :n] = indices[:n]
                for k in raw_keys:
                    if k in x:
                        src = x[k][b]
                        if src.dim() > 1:
                            src = src.squeeze(-1)
                        filtered_raw[k][b, :n] = src[indices[:n]]

        # Store for target reindexing in loss()
        self._reco_node_indices = reco_node_indices
        self._reco_filtered_valid = filtered_valid

        # --- Build x_reco dict ---
        x_reco = {
            "key_embed": filtered_embed,
            "key_valid": filtered_valid,
            "key_is_node": torch.ones(max_rn, device=device, dtype=torch.bool),
            "query_embed": self.reco_query_initial.expand(batch_size, -1, -1),
            "query_valid": torch.full((batch_size, self.num_reco_queries), True, device=device),
            "node_is_track": filtered_is_track,
        }
        for k in raw_keys:
            x_reco[k] = filtered_raw[k]

        # --- Run reco decoder ---
        x_reco, reco_outputs = self.reco_decoder(x_reco, ["node"])

        # --- Final reco task outputs ---
        reco_outputs["final"] = {}
        for task in self.reco_tasks:
            reco_outputs["final"][task.name] = task(x_reco)
            if isinstance(task, IncidenceRegressionTask):
                x_reco["incidence"] = reco_outputs["final"][task.name][task.outputs[0]].detach()
            if isinstance(task, ObjectClassificationTask):
                x_reco["class_probs"] = reco_outputs["final"][task.name][task.outputs[0]].detach()

        return reco_outputs

    def _build_hybrid_queries(
        self, node_embed: Tensor, hs_mask: Tensor, batch_size: int
    ) -> tuple[Tensor, Tensor]:
        """Build padded hybrid query tensor from detached HS track embeddings."""
        D = node_embed.shape[-1]
        device = node_embed.device
        NL = self.num_latent_queries
        Q = self.max_hs_tracks + NL

        queries = torch.zeros(batch_size, Q, D, device=device, dtype=node_embed.dtype)
        valid = torch.zeros(batch_size, Q, device=device, dtype=torch.bool)

        queries[:, :NL, :] = self.calo_latent_queries
        valid[:, :NL] = True

        for b in range(batch_size):
            track_indices = hs_mask[b].nonzero(as_tuple=True)[0]
            n = min(len(track_indices), self.max_hs_tracks)
            if n > 0:
                queries[b, NL : NL + n, :] = node_embed[b, track_indices[:n], :]
                valid[b, NL : NL + n] = True

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

        # --- Reconstruction losses (Stream C) with Hungarian matching ---
        if self.reco_decoder is not None and any(k.startswith("reco_") for k in outputs):
            reco_targets = self._build_reco_targets(targets)
            reco_layer_outputs = {k[len("reco_"):]: v for k, v in outputs.items() if k.startswith("reco_")}

            if self.reco_matcher is not None:
                batch_idxs = torch.arange(reco_targets[f"{self.reco_target_object}_valid"].shape[0]).unsqueeze(1)

                costs = {}
                for layer_name, layer_outputs in reco_layer_outputs.items():
                    layer_costs = None
                    for task in self.reco_tasks:
                        if layer_name != "final" and not task.has_intermediate_loss:
                            continue
                        if task.name not in layer_outputs:
                            continue
                        task_costs = task.cost(layer_outputs[task.name], reco_targets)
                        for cost in task_costs.values():
                            layer_costs = cost if layer_costs is None else layer_costs + cost
                    if layer_costs is not None:
                        layer_costs = layer_costs.detach()
                    costs[layer_name] = layer_costs

                for layer_name, cost in costs.items():
                    if cost is None:
                        continue
                    pred_idxs = self.reco_matcher(cost, reco_targets[f"{self.reco_target_object}_valid"])
                    for task in self.reco_tasks:
                        if not task.permute_loss:
                            continue
                        if layer_name != "final" and not task.has_intermediate_loss:
                            continue
                        for output_name in task.outputs:
                            if output_name in reco_layer_outputs[layer_name].get(task.name, {}):
                                reco_layer_outputs[layer_name][task.name][output_name] = \
                                    reco_layer_outputs[layer_name][task.name][output_name][batch_idxs, pred_idxs]

            for layer_name, layer_outputs in reco_layer_outputs.items():
                losses[f"reco_{layer_name}"] = {}
                for task in self.reco_tasks:
                    if layer_name != "final" and not task.has_intermediate_loss:
                        continue
                    if task.name in layer_outputs:
                        losses[f"reco_{layer_name}"][task.name] = task.loss(layer_outputs[task.name], reco_targets)

        return losses

    def _build_reco_targets(self, targets: dict) -> dict:
        """Reindex reconstruction targets to filtered node space."""
        reco_targets = {}
        indices = self._reco_node_indices  # (B, max_reco_nodes)
        max_rn = indices.shape[1]

        # Copy scalar particle-level targets as-is
        for key, val in targets.items():
            if key.startswith("reco_particle_"):
                suffix = key[len("reco_particle_"):]
                # Rewrite key to match task's target_object prefix
                reco_targets[f"{self.reco_target_object}_{suffix}"] = val

        # Reindex node-level targets: (B, num_objects, max_nodes) -> (B, num_objects, max_reco_nodes)
        for key in ["node_valid", "node_incidence"]:
            src_key = f"{self.reco_target_object}_{key}"
            if src_key in reco_targets and reco_targets[src_key].dim() == 3:
                # (B, num_objects, max_nodes) -> gather along node dim
                num_objects = reco_targets[src_key].shape[1]
                idx_expanded = indices.unsqueeze(1).expand(-1, num_objects, -1)
                reco_targets[src_key] = torch.gather(reco_targets[src_key], 2, idx_expanded)

        # Also remap node_valid as a 2D mask for tasks that need it
        if "node_valid" in targets:
            reco_targets["node_valid"] = self._reco_filtered_valid

        # node_is_track in filtered space
        if "node_is_track" in targets:
            is_track_full = targets["node_is_track"]
            if is_track_full.dim() > 2:
                is_track_full = is_track_full.squeeze(-1)
            reco_targets["node_is_track"] = torch.gather(is_track_full, 1, indices)

        return reco_targets

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
            elif layer_name.startswith("reco_"):
                real_layer = layer_name[len("reco_"):]
                for task in self.reco_tasks:
                    if real_layer != "final" and not task.has_intermediate_loss:
                        continue
                    if task.name in layer_out:
                        preds[layer_name][task.name] = task.predict(layer_out[task.name])

        return preds
