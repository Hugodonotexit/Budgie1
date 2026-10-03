"""The feed-forward block."""

from torch import nn

from .activations import CenteredDSiLU
from .cache import FFN_CONV
from .configuration_budgie import BudgieConfig
from .convolution import CausalConv


class FeedForward(nn.Module):
    """A causal conv, then d -> w1 -> ... -> wn -> d, n + 1 linear layers where the usual FFN has two.
    SiLU after every hidden layer except the middle one (n // 2), which gets the centered dSiLU."""

    def __init__(self, config: BudgieConfig, layer_idx: int):
        super().__init__()
        d = config.hidden_size
        widths = [d, *config.ffn_shapes[layer_idx], d]
        self.conv = CausalConv(d, config.ffn_conv_kernel, layer_idx, FFN_CONV)
        self.layers = nn.ModuleList([nn.Linear(widths[i], widths[i + 1], bias=False) for i in range(len(widths) - 1)])
        hidden = len(widths) - 2
        self.activations = nn.ModuleList([CenteredDSiLU() if i == hidden // 2 else nn.SiLU() for i in range(hidden)])

    def forward(self, x, cache, token_mask):
        h = self.conv(x, cache, token_mask)
        for linear, activation in zip(self.layers[:-1], self.activations):
            h = activation(linear(h))
        return self.layers[-1](h)
