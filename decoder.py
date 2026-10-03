"""One transformer layer."""

from transformers.modeling_layers import GradientCheckpointingLayer

from .attention import Attention
from .configuration_budgie import BudgieConfig
from .feedforward import FeedForward
from .norms import RMSNorm


class DecoderLayer(GradientCheckpointingLayer):
    """x += attention(norm(x)), then x += ffn(norm(x)). Returns (x, kv): the attention's K/V when
    this layer owns it, which the model hands to any layer that reads it (config.kv_share)."""

    def __init__(self, config: BudgieConfig, layer_idx: int):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Attention(config, layer_idx)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn = FeedForward(config, layer_idx)

    def forward(self, x, cos_sin, cache, shared_kv, token_mask, attention_mask, past):
        a, kv = self.self_attn(self.attn_norm(x), cos_sin, cache, shared_kv, token_mask, attention_mask, past)
        x = x + a
        return x + self.ffn(self.ffn_norm(x), cache, token_mask), kv
