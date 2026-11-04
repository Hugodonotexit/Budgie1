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
BRANCH_CONV = 3  # the retention branch's conv; a slot is only allocated when config.has_branch


class BudgieCache(Cache):
    """Per layer: K/V for exactly as many past tokens as a later query can still see
    (`AttentionPattern.keep_past`; everything for global layers; none for a KV-sharing reader),
    plus the three causal-conv states. Also the last few token ids, which the n-gram embeddings need.

    With the retention branch on there is a fourth conv slot per layer, and the branch of each block
    keeps its fp32 state in slot 0 of `recurrent_states` of the layer after that block's SUM. `last_doc_pos`
    is each sequence's position within its current document after the last call (int64 [B])."""

    def __init__(self, config):
        n_states = N_CONV_STATES + (1 if config.has_branch else 0)
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
        # Compressed + selected layers (sparse.py): layer -> (query group, S of the group before it, partial S of it)
        self.sparse_state = {}
        self.fast_engine = None  # the decode.FastDecoder currently holding this cache's state, if any
        self.no_fast = False     # set when this cache must stay on the eager path (padded batch, beam search...)

    def get_seq_length(self, layer_idx=0):
        if self.fast_engine is not None:
            return self.fast_engine.pos_host
        return super().get_seq_length(layer_idx)

    def _leave_fast(self):
        """Take the state back from the decode engine before anything edits the cache directly."""
        if self.fast_engine is not None:
            self.fast_engine.release()
        self.no_fast = True

    def _map_sparse(self, fn):
        self.sparse_state = {i: (g, fn(prev), fn(part)) for i, (g, prev, part) in self.sparse_state.items()}

    def crop(self, *args, **kwargs):
        self._leave_fast()
        self.sparse_state = {}   # a selection state cannot be rolled back; a sparse layer refuses to continue
        return super().crop(*args, **kwargs)

    def batch_repeat_interleave(self, repeats):
        self._leave_fast()
        self._map_sparse(lambda t: t.repeat_interleave(repeats, dim=0))
        return super().batch_repeat_interleave(repeats)

    def batch_select_indices(self, indices):
        self._leave_fast()
        self._map_sparse(lambda t: t[indices])
        return super().batch_select_indices(indices)

    def reorder_cache(self, beam_idx):
        self._leave_fast()
        super().reorder_cache(beam_idx)
        if self.last_ids is not None:
            self.last_ids = self.last_ids.index_select(0, beam_idx.to(self.last_ids.device))
        if self.last_doc_pos is not None:
            self.last_doc_pos = self.last_doc_pos.index_select(0, beam_idx.to(self.last_doc_pos.device))
        self._map_sparse(lambda t: t.index_select(0, beam_idx.to(t.device)))
