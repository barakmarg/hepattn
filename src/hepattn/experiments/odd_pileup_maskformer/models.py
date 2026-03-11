"""Shifted-window variants of Encoder and MaskFormer for phi-periodic Morton sorting.

These subclasses implement Swin Transformer-style shifted windows: the encoder
alternates between a base Morton ordering and one where phi is rotated by π, so that
tokens near the phi=±π boundary (which are genuine ΔR neighbours) share an attention
window in every other layer.
"""

import torch
from torch import Tensor

from hepattn.models.attention import repad_from_flash_varlen, unpad_for_flash_varlen
from hepattn.models.maskformer import MaskFormer
from hepattn.models.task import IncidenceRegressionTask, ObjectClassificationTask
from hepattn.models.transformer import Encoder


class ShiftedWindowEncoder(Encoder):
    """Encoder with Swin-style shifted-window Morton sorting.

    Alternates between a base and a phi-shifted Morton ordering every layer,
    ensuring particles near the phi=±π boundary share an attention window in
    at least every other layer.

    Accepts an extra `x_sort_value_shifted` argument in `forward`. When it is
    None the behaviour is identical to the parent `Encoder`.
    """

    def forward(
        self,
        x: Tensor,
        x_sort_value: Tensor | None = None,
        x_sort_value_shifted: Tensor | None = None,
        n_global_tokens: int = 0,
        **kwargs,
    ) -> Tensor:
        # Fall back to parent when no shifted index is supplied
        if x_sort_value_shifted is None:
            return super().forward(
                x, x_sort_value=x_sort_value, n_global_tokens=n_global_tokens, **kwargs
            )

        batch_size = x.shape[0]
        seq_len = x.shape[-2]
        kv_mask = kwargs.get("kv_mask")

        # For flash-varlen the window_size is passed directly to the kernel via
        # attn_kwargs; no mask_mod is needed.  score_mod is inherited from parent.
        initial_values = {} if self.value_residual else None

        # Alternate sort values: even layers → base, odd layers → shifted
        sort_values = [
            x_sort_value_shifted if i % 2 else x_sort_value
            for i in range(self.num_layers)
        ]

        prev_sort_idx = None
        prev_unpad_indices = None

        for i, layer in enumerate(self.layers):
            sv = sort_values[i]
            sort_idx = torch.argsort(sv, dim=-1)  # (B, N)

            # Sort x into the current layer's token order
            x = torch.gather(x, -2, sort_idx.unsqueeze(-1).expand_as(x))

            # Re-sort initial_values["v"] from the previous layer's order into the
            # current layer's order.  initial_values["v"] lives in the unpadded domain
            # with shape (total_valid, H, Dh); we repad, unsort, sort, re-unpad.
            if initial_values is not None and "v" in initial_values:
                v0 = initial_values["v"]  # (total_valid, H, Dh)
                head_shape = v0.shape[1:]  # (H, Dh)

                # repad: (total_valid, H*Dh) → (B, N, H*Dh)
                v0_padded = repad_from_flash_varlen(
                    v0.flatten(-2), batch_size, seq_len, prev_unpad_indices
                )
                v0_padded = v0_padded.unflatten(-1, head_shape)  # (B, N, H, Dh)

                # unsort back to original order using previous sort_idx
                unsort_prev = torch.argsort(prev_sort_idx, dim=-1)  # (B, N)
                expand = unsort_prev.unsqueeze(-1).unsqueeze(-1).expand_as(v0_padded)
                v0_padded = torch.gather(v0_padded, 1, expand)

                # sort into current layer's order
                expand = sort_idx.unsqueeze(-1).unsqueeze(-1).expand_as(v0_padded)
                v0_padded = torch.gather(v0_padded, 1, expand)

                # unpad again
                v0_flat, _, _ = unpad_for_flash_varlen(v0_padded.flatten(-2), kv_mask)
                initial_values["v"] = v0_flat.unflatten(-1, head_shape)

            # Unpad x for flash-varlen
            x_unpadded, unpad_indices, varlen_kwargs = unpad_for_flash_varlen(x, kv_mask)
            prev_unpad_indices = unpad_indices
            prev_sort_idx = sort_idx

            layer_kwargs = dict(kwargs)
            layer_kwargs["varlen_kwargs"] = varlen_kwargs
            layer_kwargs.pop("kv_mask", None)  # consumed by unpadding

            x_unpadded = layer(
                x_unpadded,
                attn_mask=None,
                score_mod=self.score_mod,
                initial_values=initial_values,
                **layer_kwargs,
            )

            # Repad
            x = repad_from_flash_varlen(x_unpadded, batch_size, seq_len, unpad_indices)

            # Unsort back to original token order
            unsort_idx = torch.argsort(sort_idx, dim=-1)
            x = torch.gather(x, -2, unsort_idx.unsqueeze(-1).expand_as(x))

        return x


class ShiftedWindowMaskFormer(MaskFormer):
    """MaskFormer that passes a shifted Morton index to ShiftedWindowEncoder.

    Merges `node_deltaR_idx_shifted` alongside `node_deltaR_idx` and forwards
    `x_sort_value_shifted` to the encoder.  All other behaviour is identical to
    the parent `MaskFormer`.
    """

    def forward(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        input_names = [net.input_name for net in self.input_nets]

        assert "key" not in input_names, "'key' input name is reserved."
        assert "query" not in input_names, "'query' input name is reserved."

        x = {}

        for raw_var in self.raw_variables:
            if raw_var in inputs:
                x[raw_var] = inputs[raw_var]

        if self.decoder.preserve_posenc:
            assert all(net.posenc is not None for net in self.input_nets)
            x["key_posenc"] = torch.concatenate(
                [net.posenc(inputs) for net in self.input_nets], dim=-2
            )

        for input_net in self.input_nets:
            name = input_net.input_name
            x[name + "_embed"] = input_net(inputs)
            x[name + "_valid"] = inputs[name + "_valid"]
            device = inputs[name + "_valid"].device
            x[f"key_is_{name}"] = torch.cat(
                [
                    torch.full(
                        (inputs[i + "_valid"].shape[-1],),
                        i == name,
                        device=device,
                        dtype=torch.bool,
                    )
                    for i in input_names
                ],
                dim=-1,
            )

        x["key_embed"] = torch.concatenate(
            [x[n + "_embed"] for n in input_names], dim=-2
        )
        x["key_valid"] = torch.concatenate(
            [x[n + "_valid"] for n in input_names], dim=-1
        )

        batch_size = x["key_valid"].shape[0]
        if batch_size == 1 and x["key_valid"].all():
            x["key_valid"] = None

        if self.input_sort_field is not None:
            x[f"key_{self.input_sort_field}"] = torch.concatenate(
                [inputs[n + "_" + self.input_sort_field] for n in input_names], dim=-1
            )
            shifted_field = self.input_sort_field + "_shifted"
            if all(n + "_" + shifted_field in inputs for n in input_names):
                x[f"key_{shifted_field}"] = torch.concatenate(
                    [inputs[n + "_" + shifted_field] for n in input_names], dim=-1
                )

        if self.encoder is not None:
            x["key_embed"] = self.encoder(
                x["key_embed"],
                x_sort_value=x.get(f"key_{self.input_sort_field}"),
                x_sort_value_shifted=x.get(f"key_{self.input_sort_field}_shifted"),
                kv_mask=x.get("key_valid"),
            )

        for name in input_names:
            x[name + "_embed"] = x["key_embed"][..., x[f"key_is_{name}"], :]

        x["query_embed"] = self.query_initial.expand(batch_size, -1, -1)
        x["query_valid"] = torch.full(
            (batch_size, self.num_queries), True, device=x["query_embed"].device
        )

        x, outputs = self.decoder(x, input_names)

        if self.pooling is not None:
            x_pooled = self.pooling(
                x[f"{self.pooling.input_name}_embed"],
                x[f"{self.pooling.input_name}_valid"],
            )
            x[f"{self.pooling.output_name}_embed"] = x_pooled

        outputs["final"] = {}
        for task in self.tasks:
            outputs["final"][task.name] = task(x)
            if isinstance(task, IncidenceRegressionTask):
                x["incidence"] = outputs["final"][task.name][task.outputs[0]].detach()
            if isinstance(task, ObjectClassificationTask):
                x["class_probs"] = outputs["final"][task.name][task.outputs[0]].detach()

        return outputs
