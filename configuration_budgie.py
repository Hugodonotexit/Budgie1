"""Configuration for Budgie, a decoder-only transformer with windowed, dilated and global attention."""

from transformers import PretrainedConfig

# One entry per layer kind. window: tokens each query sees (itself included); None = the whole
# prefix. dilation: the window is taken at stride `dilation` (token i sees i, i-d, i-2d, ...), so
# its reach is window * dilation. conv: kernel of the depthwise causal convs in front of Q and K/V.
# rope: rotary positions (False = none). kv_heads: key/value heads shared by the query heads.
# 2026-11-04: 7 layers per block at hidden 1536 / 12 heads, one global layer per block (was 12 layers at 1024 / 16 heads
# with N, D1-2, D128-256 and a second global kind G; configs saved before carry their own attention_types).
DEFAULT_ATTENTION_TYPES = {
    "S": {"window": 2048, "dilation": 1, "conv": 2, "rope": True, "kv_heads": 6},   # local, reach 2k
    "D1-4": {"window": 1024, "dilation": 4, "conv": 4, "rope": True, "kv_heads": 4},   # dilated local, reach 4k
    "D4-16": {"window": 4096, "dilation": 4, "conv": 4, "rope": True, "kv_heads": 4},   # dilated mid, reach 16k
    "D16-64": {"window": 16384, "dilation": 4, "conv": 4, "rope": True, "kv_heads": 2},   # dilated mid, reach 64k
    "D64-256": {"window": 65536, "dilation": 4, "conv": 4, "rope": True, "kv_heads": 2},   # dilated long, reach 256k
    "G_shared": {"window": None, "dilation": 1, "conv": 2, "rope": False, "kv_heads": 4},  # global, no positions
}

# "SUM" is not a layer: it marks where the retention branch's output is added back to the residual stream. The branch
# reads the stream at the start of the block and, with a SUM in the pattern, exists in every block. No SUM = no branch.
BRANCH_SUM = "SUM"
DEFAULT_PATTERN = ("S", "D1-4", "D4-16", "S", "D16-64", BRANCH_SUM, "D64-256", "G_shared")
# One multiplier per n-gram hash head. The heads must differ in their multiplier, not just a seed:
# a seed only shifts every hash by a constant, so n-grams that collide under one head would collide
# under all of them.
NGRAM_HASH_BASES = (1000003, 1000033, 1000037, 1000039, 1000081, 1000099, 1000117, 1000121)
SHARED_KINDS = ("D1-4", "D4-16", "D16-64", "D64-256", "G_shared")
LIN_RULES = ("retention", "gated_delta")
LINEAR_BRANCH_KEYS = ("lin_rule", "lin_heads", "lin_head_dim", "lin_conv_kernel", "lin_chunk", "lin_half_life_range", "lin_gate_init", "lin_head_gate",
                      "lin_base_heads", "lin_head_stages")


class BudgieConfig(PretrainedConfig):
    """Decoder-only transformer built from repeated blocks of attention layers.

    Depth: `block_pattern` (one entry per layer, keys of `attention_types`, plus at most one "SUM") repeated
    `num_blocks` times. "SUM" is not a layer, only the place where the retention branch is added back (below).
    num_hidden_layers (the entries other than SUM, times num_blocks) is derived and cannot be set.

    Every layer is  x += attn(norm(x))  then  x += ffn(norm(x)).
    Attention: Q reads a depthwise causal conv of norm(x), K/V read a second one; QK-norm; a learned
    sink logit per head (an extra softmax slot that attends to nothing); windowed / dilated / global
    per the layer's kind; grouped-query.

    FFN: a depthwise causal conv, then a stack of linear layers, deeper than the usual two. It is
    d -> w1 -> w2 -> ... -> wn -> d, with `intermediate_size` = [w1 .. wn] the hidden widths, so it
    has n + 1 matrices. After hidden layer k the activation is SiLU, except the middle one
    (k = n // 2), which is a dSiLU shifted so it is 0 at 0. [2048, 2048, 2048] at d = 1024 is
    W1 1024->2048, W2 2048->2048, W3 2048->2048, W4 2048->1024. An int w means
    [w] * ffn_hidden_layers. A list of such lists (one per block, or one per layer) gives
    different layers different shapes.

    `kv_share` maps a reader layer to the earlier owner layer of the same kind whose K/V it reuses
    (the reader keeps its own Q side and FFN, and has no K/V projection, K/V conv or K/V cache).
    The default shares between consecutive blocks: blocks 2, 4, ... read from blocks 1, 3, ... for the
    dilated and G_shared layers (every default kind but S). Pass {} to turn sharing off.

    Input: tokens embedded with a frequency-sorted adaptive embedding (clusters split at
    `adaptive_cutoffs` in frequency rank, each `adaptive_div` times narrower than the last, projected
    back up to hidden_size) plus hashed n-gram embeddings of the previous n tokens for each order
    in `ngram_orders`. Each order has a table of `ngram_buckets` rows of `ngram_dim` dims (default
    hidden_size // 8: many narrow rows collide far less than few wide ones for the same parameters).
    Every n-gram is hashed by `ngram_heads` independent hash functions, each into its own slice
    (ngram_buckets / ngram_heads rows) of that table; the heads' rows are concatenated and projected
    up to hidden_size by a learned Linear, so two n-grams that collide under one hash are still
    told apart by the others.
    Output: an adaptive softmax tied to the adaptive tables, with logits soft-capped
    at `logit_softcap`. `adaptive_cutoffs=[]` makes it an ordinary full softmax.
    The frequency order is a buffer of the model (`embed.rank_of`), identity until set.

    `global_logit_scale` (None = off) multiplies the queries of the global (window None) layers by
    max(1, ln(p + 1) / ln(L)) at absolute position p, {"type": "log", "trained_length": L}: a softmax over more
    keys than training saw is sharpened back in proportion to log(keys) (Scalable-Softmax). It is exactly 1
    for every position below L, so it changes nothing inside the trained length.

    Optional retention branch (see retention.py), on when `block_pattern` has a "SUM". It is a
    small linear-attention branch (`lin_heads` heads of `lin_head_dim`, decay half-lives spread
    log-uniformly over `lin_half_life_range`, `lin_chunk` tokens per chunk) that reads each block's
    input, in parallel with the local layers, and adds its gated output to the residual stream at the
    SUM, i.e. just before the layer that follows it (so SUM cannot be last). It resets at every document
    start (a `bos_token_id` token).
    `lin_rule` is its state update. "retention": fixed decay per head, every token added in with the same
    weight. "gated_delta" (Gated DeltaNet, Yang et al. 2025): before writing a token's value at its key, the
    state's current value at that key is erased (the delta rule), with a per-token write strength beta and a
    per-token decay that starts at the head's half-life and learns to shorten or lengthen it; less forgetful
    at long range because a new fact replaces the old one at its key instead of piling on top of it.
    A config.json written before `lin_rule` existed has a retention branch and loads as "retention".
    `lin_head_gate` adds a per-token, per-head sigmoid gate on the branch's head outputs.
    `lin_base_heads` is set when a trained branch was widened (grow_branch.py): its first `lin_base_heads`
    heads keep the half-lives they were trained with, and the other heads take the positions in between
    (see retention.branch_half_lives). None (not written to config.json) = every head is on one grid.
    A branch widened twice records every size in `lin_head_stages` (e.g. [4, 16, 32]); each stage's new heads
    go halfway between the heads that already exist, and `lin_base_heads` is then the first entry.
    Turned off (no SUM), the model has no extra modules or parameters and the `lin_*` fields are not
    written to config.json. A config.json from before SUM existed has `linear_branch`: "before_G" loads
    with a SUM put before the block's first G layer, "none" or no key loads without one.
    """

    model_type = "budgie"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=42000,
        hidden_size=1536,
        block_pattern=DEFAULT_PATTERN,
        num_blocks=8,
        intermediate_size=(2560, 2560, 2560),
        ffn_hidden_layers=3,
        num_attention_heads=12,
        head_dim=128,
        attention_types=None,
        kv_share=None,
        qk_norm=True,
        attention_sinks=True,
        ffn_conv_kernel=4,
        ngram_orders=(2, 3),
        ngram_buckets=262144,
        ngram_dim=None,
        ngram_heads=2,
        adaptive_cutoffs=(16384,),
        adaptive_div=2,
        logit_softcap=50.0,
        lin_rule="gated_delta",
        lin_heads=64,
        lin_head_dim=128,
        lin_conv_kernel=4,
        lin_chunk=512,
        lin_half_life_range=(1024, 1048576),
        lin_gate_init=0.05,
        lin_head_gate=True,
        lin_base_heads=None,
        lin_head_stages=None,
        rms_norm_eps=1e-5,
        rope_theta=500000.0,
        rope_extension=None,
        global_logit_scale=None,
        max_position_embeddings=1048576,
        initializer_range=0.025,
        fp32_residual=True,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=True,
        **kwargs,
    ):
        kwargs.pop("num_hidden_layers", None)  # derived
        self.block_pattern = list(block_pattern)
        self.num_blocks = int(num_blocks)
        self.attention_types = {k: dict(v) for k, v in (attention_types or DEFAULT_ATTENTION_TYPES).items()}
        unknown = sorted(set(self.layer_pattern) - set(self.attention_types))
        if unknown:
            raise ValueError(f"block_pattern uses {unknown}, which are not in attention_types {sorted(self.attention_types)}")
        if self.block_pattern.count(BRANCH_SUM) > 1 or self.block_pattern[-1:] == [BRANCH_SUM]:
            raise ValueError(f"block_pattern may have one {BRANCH_SUM!r}, followed by a layer in the same block, got {self.block_pattern}")
        for kind, spec in self.attention_types.items():
            if spec["window"] is not None and spec["window"] < 2:
                raise ValueError(f"attention type {kind}: window must be >= 2 or None, got {spec['window']}")
            if spec["dilation"] < 1 or spec["conv"] < 2:
                raise ValueError(f"attention type {kind}: need dilation >= 1 and conv >= 2, got {spec}")
            if num_attention_heads % spec["kv_heads"]:
                raise ValueError(f"attention type {kind}: {num_attention_heads} heads not divisible by kv_heads {spec['kv_heads']}")
        self.num_hidden_layers = len(self.layer_pattern) * self.num_blocks

        if isinstance(intermediate_size, (list, tuple)):
            intermediate_size = [list(map(int, w)) if isinstance(w, (list, tuple)) else int(w) for w in intermediate_size]
            per_layer_shapes = intermediate_size and isinstance(intermediate_size[0], list)
            if per_layer_shapes and len(intermediate_size) not in (self.num_blocks, self.num_hidden_layers):
                raise ValueError(
                    f"intermediate_size has {len(intermediate_size)} FFN shapes; expected {self.num_blocks} (per block) "
                    f"or {self.num_hidden_layers} (per layer)")
        widths = [w for shape in (intermediate_size if isinstance(intermediate_size, list) else [intermediate_size])
                  for w in (shape if isinstance(shape, list) else [shape])]
        if not widths or min(widths) < 2 or (isinstance(intermediate_size, list) and not intermediate_size):
            raise ValueError(f"intermediate_size must be non-empty with every width >= 2, got {intermediate_size}")
        self.intermediate_size = intermediate_size
        self.ffn_hidden_layers = int(ffn_hidden_layers)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.qk_norm = qk_norm
        self.attention_sinks = attention_sinks
        self.ffn_conv_kernel = ffn_conv_kernel
        self.ngram_orders = [int(n) for n in ngram_orders]
        self.ngram_buckets = int(ngram_buckets)
        self.ngram_dim = int(ngram_dim) if ngram_dim is not None else hidden_size // 8
        self.ngram_heads = int(ngram_heads)
        if self.ngram_dim < 1:
            raise ValueError(f"ngram_dim must be >= 1, got {self.ngram_dim}")
        if not 1 <= self.ngram_heads <= len(NGRAM_HASH_BASES):
            raise ValueError(f"ngram_heads must be between 1 and {len(NGRAM_HASH_BASES)}, got {self.ngram_heads}")
        if self.ngram_buckets % self.ngram_heads:
            raise ValueError(f"ngram_buckets {self.ngram_buckets} must divide evenly into ngram_heads {self.ngram_heads} slices")
        self.adaptive_cutoffs = [int(c) for c in adaptive_cutoffs]
        self.adaptive_div = adaptive_div
        self.logit_softcap = logit_softcap
        cuts = self.adaptive_cutoffs
        if cuts != sorted(set(cuts)) or (cuts and (cuts[0] < 1 or cuts[-1] >= vocab_size)):
            raise ValueError(f"adaptive_cutoffs must be increasing and inside (0, vocab_size), got {cuts}")
        for i in range(len(cuts) + 1):
            width = hidden_size // adaptive_div**i
            if width < 1:
                raise ValueError(f"adaptive cluster {i} would have width {width}; it must be >= 1")
        self.lin_heads, self.lin_head_dim = int(lin_heads), int(lin_head_dim)
        self.lin_conv_kernel, self.lin_chunk = int(lin_conv_kernel), int(lin_chunk)
        self.lin_half_life_range = [float(lin_half_life_range[0]), float(lin_half_life_range[1])]
        self.lin_gate_init, self.lin_head_gate = float(lin_gate_init), bool(lin_head_gate)
        self.lin_base_heads = None if lin_base_heads is None else int(lin_base_heads)
        stages = None if not lin_head_stages else [int(n) for n in lin_head_stages]
        given_stages = stages                                  # as written, for the check below
        if stages is not None and len(stages) == 2 and self.lin_base_heads in (None, stages[0]):
            self.lin_base_heads, stages = stages[0], None          # one widening: the older, shorter spelling
        self.lin_head_stages = stages
        if lin_rule not in LIN_RULES:
            raise ValueError(f"lin_rule must be one of {LIN_RULES}, got {lin_rule!r}")
        self.lin_rule = lin_rule
        if stages is not None:
            self.lin_base_heads = stages[0]
        if self.has_branch:
            if self.lin_head_dim < 1:
                raise ValueError(f"lin_head_dim must be >= 1, got {self.lin_head_dim}")
            lo, hi = self.lin_half_life_range
            if self.lin_heads < 1 or self.lin_conv_kernel < 2 or self.lin_chunk < 1 or not 0 < lo <= hi:
                raise ValueError("need lin_heads >= 1, lin_conv_kernel >= 2, lin_chunk >= 1 and 0 < lin_half_life_range[0] <= [1]")
            base = self.lin_base_heads
            if base is not None and not (1 <= base <= self.lin_heads and self.lin_heads % base == 0):
                raise ValueError(f"lin_base_heads must divide lin_heads ({self.lin_heads}), got {base}")
            if given_stages is not None and not (given_stages[-1] == self.lin_heads and given_stages[0] >= 1
                                                 and all(a < b and b % a == 0 for a, b in zip(given_stages, given_stages[1:]))):
                raise ValueError(f"lin_head_stages must increase, each a multiple of the one before, and end at lin_heads "
                                 f"({self.lin_heads}), got {given_stages}")
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        if rope_extension is not None:
            rope_extension = dict(rope_extension)
            if rope_extension.get("type") not in ("linear", "yarn") or float(rope_extension.get("factor", 0)) < 1:
                raise ValueError(f"rope_extension needs type 'linear' or 'yarn' and factor >= 1, got {rope_extension}")
            if rope_extension["type"] == "yarn" and not rope_extension.get("original_max_position_embeddings"):
                raise ValueError("rope_extension type 'yarn' needs original_max_position_embeddings")
        self.rope_extension = rope_extension
        if global_logit_scale is not None:
            global_logit_scale = dict(global_logit_scale)
            if global_logit_scale.get("type") != "log" or int(global_logit_scale.get("trained_length", 0)) < 2:
                raise ValueError(f"global_logit_scale needs type 'log' and trained_length >= 2, got {global_logit_scale}")
            global_logit_scale["trained_length"] = int(global_logit_scale["trained_length"])
        self.global_logit_scale = global_logit_scale
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.fp32_residual = fp32_residual
        self.use_cache = use_cache

        if kv_share is None:
            kv_share = self._default_kv_share()
        self.kv_share = {int(r): int(o) for r, o in kv_share.items()}  # JSON turns the keys into strings
        kinds = self.layer_kinds
        for reader, owner in self.kv_share.items():
            if not (0 <= owner < reader < self.num_hidden_layers) or kinds[owner] != kinds[reader] or owner in self.kv_share:
                raise ValueError(f"kv_share {reader} <- {owner}: the owner must be an earlier layer of the same kind that is not itself a reader")

        # Written out explicitly: save_pretrained only adds the entry for the class being
        # saved, so saving a BudgieForCausalLM would otherwise leave AutoModel unresolvable.
        kwargs.setdefault("auto_map", {
            "AutoConfig": "configuration_budgie.BudgieConfig",
            "AutoModel": "modeling_budgie.BudgieModel",
            "AutoModelForCausalLM": "modeling_budgie.BudgieForCausalLM",
        })
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.num_hidden_layers = len(self.layer_pattern) * self.num_blocks

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        config_dict = dict(config_dict)
        old = config_dict.pop("linear_branch", "none")   # written before SUM: the branch sat before the block's first G
        pattern = list(config_dict.get("block_pattern", DEFAULT_PATTERN))
        if old == "before_G" and BRANCH_SUM not in pattern and "G" in pattern:
            pattern.insert(pattern.index("G"), BRANCH_SUM)
            config_dict["block_pattern"] = pattern
        if BRANCH_SUM in config_dict.get("block_pattern", ()):
            config_dict.setdefault("lin_rule", "retention")   # written before lin_rule existed: the fixed-decay branch
        return super().from_dict(config_dict, **kwargs)

    def to_dict(self):
        """With the retention branch off, the branch fields are left out, so an old config.json and a
        newly saved one are identical."""
        d = super().to_dict()
        for key in ("rope_extension", "global_logit_scale"):
            if d.get(key) is None:
                d.pop(key, None)                    # unset: config.json stays as before the field existed
        if not self.has_branch:
            for key in LINEAR_BRANCH_KEYS:
                d.pop(key, None)
        else:
            for key in ("lin_base_heads", "lin_head_stages"):
                if d.get(key) is None:
                    d.pop(key, None)        # a branch that was never widened writes the same config.json as before
        return d

    def lin_stages(self):
        """The branch's head counts over its life, e.g. [4, 16, 32]: the first is the grid it was built on,
        each later one a widening (see retention.branch_half_lives). [lin_heads] if it was never widened."""
        if self.lin_head_stages:
            return list(self.lin_head_stages)
        return [self.lin_base_heads, self.lin_heads] if self.lin_base_heads not in (None, self.lin_heads) else [self.lin_heads]

    @property
    def layer_pattern(self):
        """block_pattern without the SUM: one entry per layer of a block."""
        return [k for k in self.block_pattern if k != BRANCH_SUM]

    @property
    def has_branch(self):
        return BRANCH_SUM in self.block_pattern

    @property
    def branch_pos(self):
        """Index in a block of the layer the branch's output is added in front of; None without a branch."""
        return self.block_pattern.index(BRANCH_SUM) if self.has_branch else None

    @property
    def layer_kinds(self):
        return self.layer_pattern * self.num_blocks

    def layer_spec(self, i):
        return self.attention_types[self.layer_kinds[i]]

    def kv_owner(self, i):
        """The layer whose K/V layer i reads, or None if layer i computes its own."""
        return self.kv_share.get(i)

    def _default_kv_share(self):
        """Odd-numbered blocks (2nd, 4th, ...) read the D4 / D8 / D16 / G layers of the block before them."""
        size, share = len(self.layer_pattern), {}
        for block in range(1, self.num_blocks, 2):
            for pos, kind in enumerate(self.layer_pattern):
                if kind in SHARED_KINDS:
                    share[block * size + pos] = (block - 1) * size + pos
        return share

    @property
    def ffn_shapes(self):
        """The hidden widths of every layer's FFN: a list (one per layer) of lists."""
        w = self.intermediate_size
        if not isinstance(w, list):
            return [[w] * self.ffn_hidden_layers] * self.num_hidden_layers
        if not isinstance(w[0], list):
            return [list(w)] * self.num_hidden_layers
        if len(w) == self.num_hidden_layers:
            return [list(s) for s in w]
        return [list(s) for s in w for _ in self.layer_pattern]  # one per block

    @property
    def adaptive_cuts(self):
        """Boundaries of the frequency-rank clusters: [0, *cutoffs, vocab_size]."""
        return [0, *self.adaptive_cutoffs, self.vocab_size]
