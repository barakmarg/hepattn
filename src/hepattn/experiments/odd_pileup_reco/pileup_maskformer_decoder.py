import torch
from torch import Tensor

from hepattn.models.decoder import MaskFormerDecoder
from hepattn.models.task import IncidenceRegressionTask, ObjectClassificationTask


class PileupMaskFormerDecoder(MaskFormerDecoder):
    """MaskFormerDecoder with a hard physics prior on the attention mask.

    ANDs the learned attention mask with a binary prior (e.g. node_is_track)
    so the query can never attend to nodes where the prior is False.
    For pileup removal: the query only attends to tracks, never calo clusters.
    """

    def __init__(self, prior_mask_key: str = "node_is_track", **kwargs):
        super().__init__(**kwargs)
        self.prior_mask_key = prior_mask_key

    def forward(self, x: dict[str, Tensor], input_names: list[str]) -> tuple[dict[str, Tensor], dict[str, dict]]:
        batch_size = x["query_embed"].shape[0]
        num_constituents = x["key_embed"].shape[-2]
        self.log_step += 1

        outputs = {}

        for layer_index, decoder_layer in enumerate(self.decoder_layers):
            outputs[f"layer_{layer_index}"] = {}

            attn_masks = {}
            query_mask = None

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

                if self.use_query_masks:
                    task_query_mask = task.query_mask(task_outputs)
                    if task_query_mask is not None:
                        query_mask = task_query_mask if query_mask is None else query_mask | task_query_mask

            # Construct the full attention mask
            attn_mask = None
            if attn_masks and self.mask_attention:
                attn_mask = torch.full((batch_size, self.num_queries, num_constituents), True, device=x["key_embed"].device)
                for input_name, task_attn_mask in attn_masks.items():
                    attn_mask[..., x[f"key_is_{input_name}"]] = task_attn_mask

                # Physics prior: query can only attend to tracks, never calo clusters
                if self.prior_mask_key in x:
                    prior = x[self.prior_mask_key].bool().unsqueeze(-2)  # (B, 1, N)
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
