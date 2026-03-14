import torch
from torch import Tensor

from hepattn.experiments.odd_pileup_maskformer.decoder import PileupMaskFormerDecoder
from hepattn.models.task import IncidenceRegressionTask, ObjectClassificationTask


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
