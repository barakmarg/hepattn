from functools import partial

import torch
from torch import Tensor, nn

from hepattn.experiments.odd_pileup_reco.pileup_maskformer_decoder import PileupMaskFormerDecoder
from hepattn.models.attention import Attention, repad_from_flash_varlen, unpad_for_flash_varlen
from hepattn.models.decoder import BiCrossAttentionLayer
from hepattn.models.dense import Dense
from hepattn.models.task import IncidenceRegressionTask, ObjectClassificationTask
from hepattn.models.transformer import Residual


class CaloMaskFormerDecoder(PileupMaskFormerDecoder):
    """MaskFormerDecoder for the Calo stream with inverted physics prior.

    Inherits from PileupMaskFormerDecoder but:
    1. Inverts the physics prior: queries attend to calo nodes only (~node_is_track)
    2. Passes query_valid as q_mask so padded hybrid queries are masked in attention
    """

    def forward(self, x: dict[str, Tensor], input_names: list[str]) -> tuple[dict[str, Tensor], dict[str, dict]]:
        batch_size = x["query_embed"].shape[0]
        num_queries = x["query_embed"].shape[-2]
        num_constituents = x["key_embed"].shape[-2]
        self.log_step += 1

        outputs = {}

        for layer_index, decoder_layer in enumerate(self.decoder_layers):
            outputs[f"layer_{layer_index}"] = {}

            attn_masks = {}
            query_mask = x.get("query_valid")  # Pass query_valid to mask padded hybrid queries

            for task in self.tasks:
                if not task.has_intermediate_loss:
                    continue

                task_outputs = task(x)

                if isinstance(task, IncidenceRegressionTask):
                    x["incidence"] = task_outputs[task.outputs[0]].detach()
                if isinstance(task, ObjectClassificationTask):
                    x["class_probs"] = task_outputs[task.outputs[0]].detach()

                outputs[f"layer_{layer_index}"][task.name] = task_outputs

                task_attn_masks = task.attn_mask(task_outputs)
                for input_name, attn_mask in task_attn_masks.items():
                    if input_name in attn_masks:
                        attn_masks[input_name] |= attn_mask
                    else:
                        attn_masks[input_name] = attn_mask

            # Construct the full attention mask
            attn_mask = None
            if attn_masks and self.mask_attention:
                attn_mask = torch.full((batch_size, num_queries, num_constituents), True, device=x["key_embed"].device)
                for input_name, task_attn_mask in attn_masks.items():
                    attn_mask[..., x[f"key_is_{input_name}"]] = task_attn_mask

                # INVERTED physics prior: query can only attend to calo, never tracks
                if self.prior_mask_key in x:
                    prior = ~x[self.prior_mask_key].bool().unsqueeze(-2)  # (B, 1, N) — inverted!
                    attn_mask = attn_mask & prior

            # Log attention mask if requested
            if self.log_attn_mask and (attn_mask is not None) and (self.log_step % 1000 == 0):
                if not hasattr(self, "attn_masks_to_log"):
                    self.attn_masks_to_log = {}
                if layer_index == 0 or layer_index == len(self.decoder_layers) - 1:
                    self.attn_masks_to_log[layer_index] = {
                        "mask": attn_mask[0].detach().cpu().clone(),
                        "step": self.log_step,
                        "layer": layer_index,
                    }

            x = self.add_query_posenc(x)

            x["query_embed"], x["key_embed"] = decoder_layer(
                x["query_embed"], x["key_embed"], attn_mask=attn_mask, q_mask=query_mask, kv_mask=x.get("key_valid")
            )

            x = self.re_add_original_embeddings(x)

            for input_name in input_names:
                x[input_name + "_embed"] = x["key_embed"][..., x[f"key_is_{input_name}"], :]

        return x, outputs


class CaloFlashCrossAttentionDecoder(nn.Module):
    """Bidirectional cross-attention decoder with query self-attention, using flash-varlen.

    Each decoder layer: query self-attention → bidirectional cross-attention (queries ↔ calo).
    Hybrid queries = learnable latent queries + HS track embeddings.
    """

    def __init__(
        self,
        dim: int,
        num_decoder_layers: int,
        attn_kwargs: dict | None = None,
        sa_attn_kwargs: dict | None = None,
        dense_kwargs: dict | None = None,
        norm: str = "LayerNorm",
        hybrid_norm: bool = False,
    ) -> None:
        super().__init__()

        attn_kwargs = attn_kwargs or {}
        dense_kwargs = dense_kwargs or {}

        # Self-attention kwargs: derive from cross-attention kwargs, strip window_size
        _sa_attn_kwargs = {**attn_kwargs}
        _sa_attn_kwargs.pop("window_size", None)
        if sa_attn_kwargs:
            _sa_attn_kwargs.update(sa_attn_kwargs)

        # Build self-attention + cross-attention layers
        self.sa_layers = nn.ModuleList()
        self.layers = nn.ModuleList()

        for i in range(num_decoder_layers):
            qkv_norm = hybrid_norm
            hn = hybrid_norm if i > 0 else False
            attn_norm = norm if not hn else None
            dense_post_norm = not hn

            res = partial(Residual, dim=dim, norm=norm)

            # Query self-attention block (Attention + Dense)
            self.sa_layers.append(nn.ModuleList([
                res(Attention(dim, qkv_norm=qkv_norm, **_sa_attn_kwargs), norm=attn_norm),
                res(Dense(dim, **dense_kwargs), norm=norm, post_norm=dense_post_norm),
            ]))

            # Bidirectional cross-attention layer
            self.layers.append(BiCrossAttentionLayer(
                dim=dim,
                norm=norm,
                depth=i,
                dense_kwargs=dense_kwargs,
                attn_kwargs=attn_kwargs,
                hybrid_norm=hybrid_norm,
            ))

        # Not used internally, but set for interface compatibility with model.py
        self.preserve_posenc = False

    def forward(self, x: dict[str, Tensor], input_names: list[str]) -> tuple[dict[str, Tensor], dict[str, dict]]:
        query_embed = x["query_embed"]  # (B, Q, D) — latent queries + HS tracks
        query_valid = x["query_valid"]  # (B, Q)
        key_embed = x["key_embed"]  # (B, N, D) — all nodes (tracks + calo)
        key_valid = x["key_valid"]  # (B, N)
        node_is_track = x["node_is_track"].bool().squeeze(-1)  # (B, N)

        batch_size = query_embed.shape[0]
        num_queries = query_embed.shape[-2]
        num_keys = key_embed.shape[-2]

        # Isolate calo nodes only
        calo_mask = key_valid & ~node_is_track  # (B, N) — True for valid calo nodes

        # Unpad both sequences for flash-varlen
        q_flat, q_indices, q_varlen = unpad_for_flash_varlen(query_embed, query_valid)
        kv_flat, kv_indices, kv_varlen = unpad_for_flash_varlen(key_embed, calo_mask)

        # Self-attention varlen kwargs (same cu_seqlens for Q and K)
        sa_kwargs = {"varlen_kwargs": {
            "cu_seqlens": q_varlen["cu_seqlens"],
            "max_seqlen": q_varlen["max_seqlen"],
        }}

        # Cross-attention varlen kwargs
        ab_kwargs = {"varlen_kwargs": {
            "cu_seqlens_q": q_varlen["cu_seqlens"],
            "cu_seqlens_k": kv_varlen["cu_seqlens"],
            "max_seqlen_q": q_varlen["max_seqlen"],
            "max_seqlen_k": kv_varlen["max_seqlen"],
        }}
        ba_kwargs = {"varlen_kwargs": {
            "cu_seqlens_q": kv_varlen["cu_seqlens"],
            "cu_seqlens_k": q_varlen["cu_seqlens"],
            "max_seqlen_q": kv_varlen["max_seqlen"],
            "max_seqlen_k": q_varlen["max_seqlen"],
        }}

        # Run through layers: self-attention on queries, then bidirectional cross-attention
        for sa_layer, ca_layer in zip(self.sa_layers, self.layers):
            sa_attn, sa_dense = sa_layer
            q_flat = sa_dense(sa_attn(q_flat, **sa_kwargs))
            q_flat, kv_flat = ca_layer(q_flat, kv_flat, ab_kwargs=ab_kwargs, ba_kwargs=ba_kwargs)

        # Repad both sequences
        x["query_embed"] = repad_from_flash_varlen(q_flat, batch_size, num_queries, q_indices)
        calo_repadded = repad_from_flash_varlen(kv_flat, batch_size, num_keys, kv_indices)

        # Scatter updated calo embeddings back into key_embed
        x["key_embed"] = torch.where(calo_mask.unsqueeze(-1), calo_repadded, key_embed)

        # Unmerge per-input-type embeddings
        for input_name in input_names:
            x[input_name + "_embed"] = x["key_embed"][..., x[f"key_is_{input_name}"], :]

        return x, {}
