from models.transformer import Encoder
import torch
from torch import Tensor, nn


class TracksMlp(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.mlp = net
    
    def forward(self, inputs: dict[str, Tensor]) -> Tensor:
        x = inputs["key_embed"]
        return self.mlp(x)

class CaloMlp(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.mlp = net
    
    def forward(self, inputs: dict[str, Tensor]) -> Tensor:
        x = inputs["key_embed"]
        return self.mlp(x)

class PileupRemovalModel(nn.Module):
    def __init__(
        self,
        input_nets: nn.ModuleList,
        encoder: Encoder,
        tracks_mlp: nn.Module,
        calo_mlp: nn.Module,
    ):
        super().__init__()
        self.input_nets = input_nets
        self.encoder = encoder
        self.tracks_mlp = tracks_mlp
        self.calo_mlp = calo_mlp
    
    def forward(self, inputs: dict[str, Tensor]):
        # Atomic input names
        input_names = [input_net.input_name for input_net in self.input_nets]

        assert "key" not in input_names, "'key' input name is reserved."
        assert "query" not in input_names, "'query' input name is reserved."

        x = {}

        for raw_var in self.raw_variables:
            # If the raw variable is present in the inputs, add it directly to the output
            if raw_var in inputs:
                x[raw_var] = inputs[raw_var]

        # Store input positional encodings if we need to preserve them for the decoder
        if self.decoder.preserve_posenc:
            assert all(input_net.posenc is not None for input_net in self.input_nets)
            x["key_posenc"] = torch.concatenate([input_net.posenc(inputs) for input_net in self.input_nets], dim=-2)

        # Embed the input objects
        for input_net in self.input_nets:
            input_name = input_net.input_name
            x[input_name + "_embed"] = input_net(inputs)
            x[input_name + "_valid"] = inputs[input_name + "_valid"]

            # These slices can be used to pick out specific
            # objects after we have merged them all together
            # TODO: Clean this up
            device = inputs[input_name + "_valid"].device
            x[f"key_is_{input_name}"] = torch.cat(
                [torch.full((inputs[i + "_valid"].shape[-1],), i == input_name, device=device, dtype=torch.bool) for i in input_names], dim=-1
            )

        # Merge the input objects and he padding mask into a single set
        x["key_embed"] = torch.concatenate([x[input_name + "_embed"] for input_name in input_names], dim=-2)
        x["key_valid"] = torch.concatenate([x[input_name + "_valid"] for input_name in input_names], dim=-1)

        # calculate the batch size and combined number of input constituents
        batch_size = x["key_valid"].shape[0]

        # if all key_valid are true, then we can just set it to None
        if batch_size == 1 and x["key_valid"].all():
            x["key_valid"] = None

        # Also merge the field being used for sorting in window attention if requested
        if self.input_sort_field is not None:
            x[f"key_{self.input_sort_field}"] = torch.concatenate(
                [inputs[input_name + "_" + self.input_sort_field] for input_name in input_names], dim=-1
            )        
        
        # Pass merged input hits through the encoder
        if self.encoder is not None:
            # Note that a padded feature is a feature that is not valid!
            x["key_embed"] = self.encoder(x["key_embed"], x_sort_value=x.get(f"key_{self.input_sort_field}"), kv_mask=x.get("key_valid"))

        t = self.tracks_mlp(x)
        c = self.calo_mlp(x)
        return t, c