"""The decoding cache: windowed K/V plus causal-conv states, per layer."""

from transformers.cache_utils import (
    Cache,
    LinearAttentionAndFullAttentionLayer,
    LinearAttentionAndSlidingWindowAttentionLayer,
    LinearAttentionLayer,
)

from .patterns import AttentionPattern

# Slots of the three causal convs in a layer's cache: before Q, before the FFN, before K/V.
Q_CONV, FFN_CONV, KV_CONV = 0, 1, 2
N_CONV_STATES = 3
BRANCH_CONV = 3  # the retention branch's conv; a slot is only allocated when config.linear_branch != "none"


class BudgieCache(Cache):
    """Per layer: K/V for exactly as many past tokens as a later query can still see
    (`AttentionPattern.keep_past`; everything for global layers; none for a KV-sharing reader),
    plus the three causal-conv states. Also the last few token ids, which the n-gram embeddings need.

    With the retention branch on there is a fourth conv slot per layer, and the branch of each block
    keeps its fp32 state in slot 0 of `recurrent_states` of that block's first G layer. `last_doc_pos`
    is each sequence's position within its current document after the last call (int64 [B])."""

    def __init__(self, config):
        n_states = N_CONV_STATES + (1 if config.linear_branch != "none" else 0)
        layers = []
        for i in range(config.num_hidden_layers):
            pattern = AttentionPattern.from_spec(config.layer_spec(i))
            if config.kv_owner(i) is not None:
                layers.append(LinearAttentionLayer(number_of_states=n_states))
            elif pattern.is_global:
                layers.append(LinearAttentionAndFullAttentionLayer(number_of_states=n_states))
            else:  # the layer keeps sliding_window - 1 past tokens
                layers.append(LinearAttentionAndSlidingWindowAttentionLayer(
                    sliding_window=pattern.keep_past + 1, number_of_states=n_states))
        super().__init__(layers=layers)
        self.last_ids = None
        self.last_doc_pos = None

    def reorder_cache(self, beam_idx):
        super().reorder_cache(beam_idx)
        if self.last_ids is not None:
            self.last_ids = self.last_ids.index_select(0, beam_idx.to(self.last_ids.device))
        if self.last_doc_pos is not None:
            self.last_doc_pos = self.last_doc_pos.index_select(0, beam_idx.to(self.last_doc_pos.device))
