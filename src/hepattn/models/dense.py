import torch
from torch import Tensor, nn

from hepattn.models.activation import SwiGLU


class Dense(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int | None = None,
        hidden_layers: list[int] | None = None,
        hidden_dim_scale: int = 2,
        activation: nn.Module | None = None,
        final_activation: nn.Module | None = None,
        dropout: float = 0.0,
        bias: bool = True,
        norm_input: bool = False,
        context_size: int = 0,
    ) -> None:
        """A fully connected feed forward neural network, which can take
        in additional contextual information.

        Parameters
        ----------
        input_size : int
            Input size
        output_size : int
            Output size. If not specified this will be the same as the input size.
        hidden_layers : list, optional
            Number of nodes per layer, if not specified, the network will have
            a single hidden layer with size `input_size * hidden_dim_scale`.
        hidden_dim_scale : int, optional
            Scale factor for the hidden layer size.
        activation : nn.Module
            Activation function for hidden layers.
        final_activation : nn.Module, optional
            Activation function for the output layer.
        dropout : float, optional
            Apply dropout with the supplied probability.
        bias : bool, optional
            Whether to use bias in the linear layers.
        norm_input : bool, optional
            Whether to apply layer normalization to the input (before context concat).
        context_size : int, optional
            Size of the optional global context vector. When > 0, a context tensor
            can be passed to forward() and will be broadcast over the node dimension
            and concatenated to x before the first linear layer.
        """
        super().__init__()

        if output_size is None:
            output_size = input_size
        if hidden_layers is None:
            hidden_layers = [input_size * hidden_dim_scale]
        if activation is None:
            activation = SwiGLU()

        self.input_size = input_size
        self.output_size = output_size
        self.context_size = context_size
        gate = isinstance(activation, SwiGLU)

        # LayerNorm is stored separately so it can be applied before context concat
        self.input_norm = nn.LayerNorm(input_size) if norm_input else None

        # First layer takes input_size + context_size
        effective_input = input_size + context_size
        node_list = [effective_input, *hidden_layers]

        layers = []
        for i in range(len(node_list) - 1):
            in_dim = node_list[i]
            proj_dim = node_list[i + 1]
            proj_dim = proj_dim * 2 if gate else proj_dim

            # inner projection and activation
            layers.extend((nn.Linear(in_dim, proj_dim, bias=bias), activation))

            # maybe dropout
            if dropout:
                layers.append(nn.Dropout(dropout))

        # final projection and activation
        layers.append(nn.Linear(node_list[-1], output_size, bias=bias))
        if final_activation:
            layers.append(final_activation)

        # build the net
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor, context: Tensor | None = None) -> Tensor:
        if self.input_norm is not None:
            x = self.input_norm(x)
        if self.context_size > 0 and context is not None:
            # context: [B, D] → broadcast to [B, N, D] and concat along feature dim
            context_expanded = context.unsqueeze(1).expand(-1, x.shape[1], -1)
            x = torch.cat([x, context_expanded], dim=-1)
        return self.net(x)
