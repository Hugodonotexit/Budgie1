"""Budgie: the Hugging Face model classes. The layers they are built from each live in their own module:

    norms.py         RMSNorm, QKNorm
    activations.py   centered dSiLU
    rotary.py        RotaryEmbedding
    convolution.py   CausalConv (causal-conv1d kernel or PyTorch)
    sdpa.py          attention with a learned sink
    patterns.py      AttentionPattern: local / dilated / global
    attention.py     Attention
    feedforward.py   FeedForward
    decoder.py       DecoderLayer
    cache.py         BudgieCache
    embeddings.py    AdaptiveInput, HashedNgramEmbedding, BudgieEmbedding
    head.py          AdaptiveSoftmaxHead
    vision.py        BudgieVision (image input, when config.vision_config is set)

Loads through the Auto classes -- registered below for `import budgie`, and exported with the
checkpoint through `auto_map` for `trust_remote_code=True`:

    AutoConfig.from_pretrained(path)
    AutoModel.from_pretrained(path)              # BudgieModel, hidden states only
    AutoModelForCausalLM.from_pretrained(path)   # BudgieForCausalLM
"""

import torch
from torch import nn

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from .attention import Attention
from .cache import BudgieCache
from .configuration_budgie import DEFAULT_VISION_CONFIG, IMAGE_ID_KEYS, BudgieConfig, check_vision_config
from .convolution import CausalConv
from .decode import try_fast_decode
from .decoder import DecoderLayer
from .embeddings import BudgieEmbedding
from .head import AdaptiveSoftmaxHead
from .norms import QKNorm, RMSNorm
from .retention import RetentionBranch, document_positions, to_device
from .rotary import RotaryEmbedding
from .vision import BudgieVision, DepthwiseConv2d, VisionAttention


class BudgiePreTrainedModel(PreTrainedModel):
    config_class = BudgieConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_sdpa = True

    def _init_weights(self, module):
        super()._init_weights(module)  # Linear / Embedding / RMSNorm
        # Only what was not just loaded from a checkpoint (transformers flags those).
        with torch.no_grad():
            if isinstance(module, QKNorm) and not getattr(module.weight, "_is_hf_initialized", False):
                module.weight.fill_(1.0)  # transformers only recognises norm classes named ...RMSNorm
            elif isinstance(module, CausalConv) and not getattr(module.weight, "_is_hf_initialized", False):
                module.reset_parameters()
            elif isinstance(module, Attention):
                if module.sinks is not None and not getattr(module.sinks, "_is_hf_initialized", False):
                    module.sinks.zero_()
                if module.sparse is not None and not getattr(module.pool_score, "_is_hf_initialized", False):
                    module.pool_score.zero_()               # mean pooling to start
            elif isinstance(module, DepthwiseConv2d) and not getattr(module.weight, "_is_hf_initialized", False):
                module.reset_parameters()                   # 2-D vision convs start as the identity
            elif isinstance(module, VisionAttention):
                if module.pool_score is not None and not getattr(module.pool_score, "_is_hf_initialized", False):
                    module.pool_score.zero_()               # compressed vision blocks start as the plain mean
            elif isinstance(module, RetentionBranch):
                if not getattr(module.g, "_is_hf_initialized", False):
                    module.g.fill_(self.config.lin_gate_init)
                if not getattr(module.gain, "_is_hf_initialized", False):
                    module.gain.fill_(1.0)
                if module.rule == "gated_delta":
                    if not getattr(module.a_proj.weight, "_is_hf_initialized", False):
                        module.a_proj.weight.zero_()        # decay starts exactly at the head's half-life
                    if not getattr(module.dt_bias, "_is_hf_initialized", False):
                        module.dt_bias.zero_()
                    if not getattr(module.b_proj.bias, "_is_hf_initialized", False):
                        module.b_proj.bias.zero_()          # write strength starts near 0.5
            elif isinstance(module, AdaptiveSoftmaxHead) and module.cluster_vectors is not None \
                    and not getattr(module.cluster_vectors, "_is_hf_initialized", False):
                module.cluster_vectors.normal_(0.0, self.config.initializer_range)


class BudgieModel(BudgiePreTrainedModel):
    def __init__(self, config: BudgieConfig):
        super().__init__(config)
        self.embed = BudgieEmbedding(config)
        self.rotary = RotaryEmbedding(config.head_dim, config.rope_theta, getattr(config, "rope_extension", None))
        self.layers = nn.ModuleList([DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        # Image input (config.vision_config); no module at all for a text-only model.
        self.vision = BudgieVision(config) if config.has_vision else None
        # The retention branch of every block (a SUM in config.block_pattern); no module at all without one.
        self.branches = None
        if config.has_branch:
            self._sum_pos = config.branch_pos  # the layer of a block the branch's output is added in front of
            size = len(config.layer_pattern)
            self.branches = nn.ModuleList([RetentionBranch(config, b, b * size + self._sum_pos) for b in range(config.num_blocks)])
        # The layers whose K/V some later layer reads (config.kv_share): only these are kept while the stack runs.
        self._kv_last_reader = {}
        for j in range(config.num_hidden_layers):
            if config.kv_owner(j) is not None:
                self._kv_last_reader[config.kv_owner(j)] = j
        # Compressed + selected global layers (config.sparse_global) decode on the eager path only (decode.py).
        self.has_sparse_global = any(layer.self_attn.sparse is not None for layer in self.layers)
        self.record_branch_stats = False  # set by a caller that wants per-block branch statistics
        self.branch_stats = []
        # Set by a trainer (not saved): a torch.device to run every block's retention branch on, alongside
        # the block's local layers on the model's own GPU. Forwards without a cache only.
        self.branch_device = None
        # Set by set_long_context_mode (not saved): the HostPool and the segment length of the retention branch.
        self.offload_pool = None
        self.branch_segment = None
        self.gradient_checkpointing = False
        self.post_init()

    def set_long_context_mode(self, pool, branch_segment=None, compile_halves=False, chunk=None):
        """Training on sequences too long for whole-layer checkpointing (see offload.py): every layer is
        checkpointed in two halves with its input in pinned host memory, and the retention branches run
        `branch_segment` tokens at a time with the same treatment, on the model's own GPU (a `branch_device`
        copy of the whole sequence would not fit the other GPU either). `pool` None switches it all off.
        `chunk`: layer halves of sequences longer than this run a chunk at a time, forward and backward
        (chunked.py), which bounds the working set by the chunk instead of the sequence.
        The trainer turns Hugging Face's gradient checkpointing off while this is on."""
        self.offload_pool = pool
        self.branch_segment = branch_segment if pool is not None else None
        for layer in self.layers:
            layer.set_offload(pool, compile_halves, chunk)
        if pool is not None:
            pool.clear()

    def set_global_logit_scale(self, spec):
        """Set config.global_logit_scale on a built model ({"type": "log", "trained_length": L} or None); every
        global layer picks it up. Saved with the config."""
        self.config.global_logit_scale = None if spec is None else BudgieConfig(global_logit_scale=spec).global_logit_scale
        for layer in self.layers:
            layer.self_attn.set_global_logit_scale(self.config.global_logit_scale)

    def set_branch_head_mask(self, zero_first=0, mask=None):
        """Silence retention-branch heads in every block (an experiment, not saved): the first `zero_first`
        heads, or any per-head list `mask` of multipliers; (0, None) clears it. See RetentionBranch.set_head_mask."""
        if self.branches is None:
            if zero_first or mask is not None:
                raise ValueError("this model has no retention branch")
            return
        for branch in self.branches:
            if mask is None and zero_first:
                branch.set_head_mask([0.0] * zero_first + [1.0] * (branch.heads - zero_first))
            else:
                branch.set_head_mask(mask)

    def put_images(self, x, input_ids, pixel_values, pixel_attention_mask, spatial_shapes):
        """x with the <|image|> positions of input_ids replaced, in order (batch row by row, each image row-major), by
        the projected features of the images (BudgieProcessor's pixel_values / pixel_attention_mask / spatial_shapes).
        Ids with no <|image|> in them (a decode step after the prompt) leave x as it is, so generate() may pass the
        images on every step."""
        if self.vision is None:
            raise ValueError("pixel_values given, but this model has no vision_config (see BudgieForCausalLM.add_vision)")
        if input_ids is None or pixel_attention_mask is None or spatial_shapes is None:
            raise ValueError("pixel_values needs input_ids (to find the <|image|> positions), pixel_attention_mask and spatial_shapes")
        is_image = input_ids == self.config.vision_config["image_token_id"]
        n_slots = int(is_image.sum())
        if n_slots == 0:
            return x
        s = self.config.vision_config["pixel_shuffle"]
        need = int((spatial_shapes[:, 0] // s * (spatial_shapes[:, 1] // s)).sum())
        if n_slots != need:
            raise ValueError(f"input_ids have {n_slots} <|image|> tokens, but the {len(spatial_shapes)} images need {need} "
                             f"(processing_budgie.expand_images writes them)")
        feats = self.vision(pixel_values, pixel_attention_mask, spatial_shapes).to(x.dtype)
        return x.masked_scatter(is_image[..., None], feats)

    def get_input_embeddings(self):
        return self.embed.adaptive.tok[0]

    def set_input_embeddings(self, value):
        self.embed.adaptive.tok[0] = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values: Cache | None = None,
        inputs_embeds=None,
        use_cache=None,
        pixel_values=None,
        pixel_attention_mask=None,
        spatial_shapes=None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("specify exactly one of input_ids or inputs_embeds")
        cfg = self.config
        use_cache = cfg.use_cache if use_cache is None else use_cache
        if use_cache and not isinstance(past_key_values, BudgieCache):
            if past_key_values is not None and past_key_values.get_seq_length() > 0:
                raise ValueError("past_key_values must be a BudgieCache (the model's own cache), or empty")
            past_key_values = BudgieCache(cfg)  # replaces the generic cache generate() may have made
        cache = past_key_values if use_cache else None

        if cache is not None:
            hidden = try_fast_decode(self, cache, input_ids, attention_mask)  # graph-replayed single-token step
            if hidden is not None:
                return BaseModelOutputWithPast(last_hidden_state=hidden, past_key_values=cache)
            if cache.fast_engine is not None:  # an ordinary call after fast decoding: take the state back
                cache.fast_engine.release()

        x = self.embed(input_ids, cache) if inputs_embeds is None else inputs_embeds
        if pixel_values is not None:
            x = self.put_images(x, input_ids, pixel_values, pixel_attention_mask, spatial_shapes)
        T = x.shape[1]
        past = cache.get_seq_length() if cache is not None else 0
        if position_ids is None:
            position_ids = (torch.arange(T, device=x.device) + past)[None]
        if attention_mask is not None and (attention_mask.dim() != 2 or bool(attention_mask.all())):
            attention_mask = None  # nothing is padded
        token_mask = attention_mask[:, -T:] if attention_mask is not None else None
        cos_sin = self.rotary(position_ids)

        doc_pos = None
        if self.branches is not None:
            # Position within the current document, counted from the last bos (or from the start of the
            # sequence). Computed once here; with `inputs_embeds` there are no ids, so the whole
            # sequence is one document.
            is_bos = (input_ids == cfg.bos_token_id) if input_ids is not None else torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)
            doc_pos = document_positions(is_bos, cache.last_doc_pos if cache is not None else None, token_mask)
            self.branch_stats = []
            block_size = len(cfg.layer_pattern)
            segmented = cache is None and self.branch_segment is not None
            remote = self.branch_device if cache is None and not segmented and self.branch_device not in (None, x.device) else None
            if remote is not None:
                doc_pos_remote = to_device(doc_pos, remote)
                mask_remote = None if token_mask is None else to_device(token_mask, remote)

        shared = {}  # K/V of the owner layers, for the layers that read them
        for i, layer in enumerate(self.layers):
            if self.branches is not None:
                block, pos = divmod(i, block_size)
                branch = self.branches[block]
                if pos == 0:
                    block_input = x
                    branch.record_stats = self.record_branch_stats
                    if remote is not None:  # queued now, so the other GPU works through it during the local layers
                        pending = branch.forward_on(remote, block_input, doc_pos_remote, mask_remote)
                if pos == self._sum_pos:  # the branch reads the stream from the start of the block and is summed in at the SUM
                    if segmented and token_mask is None:
                        pool = self.offload_pool if torch.is_grad_enabled() else None
                        out = branch.forward_segmented(block_input, doc_pos, self.branch_segment, pool).to(x.dtype)
                    elif remote is None:
                        out = branch(block_input, cache, doc_pos, token_mask).to(x.dtype)
                    else:
                        out, pending = to_device(pending.to(x.dtype), x.device), None
                    if self.record_branch_stats:
                        local = (x - block_input).float()
                        stats = {k: v.to(x.device) for k, v in branch.stats.items()}
                        self.branch_stats.append({**stats, "local_rms": local.pow(2).mean(dim=(1, 2)).sqrt(), "mean_abs_g": branch.g.detach().float().abs().mean()})
                    x = x + out
            owner = cfg.kv_owner(i)
            x, kv = layer(x, cos_sin, cache, shared[owner] if owner is not None else None, token_mask, attention_mask, past)
            if owner is None and i in self._kv_last_reader:   # not every owner is read; its K/V must not outlive the layer then
                shared[i] = kv
            elif owner is not None and self._kv_last_reader[owner] == i:
                del shared[owner]                      # nothing reads it after this layer
        if self.branches is not None and cache is not None:
            cache.last_doc_pos = doc_pos[:, -1].clone()
        return BaseModelOutputWithPast(last_hidden_state=self.norm(x), past_key_values=cache)


class BudgieForCausalLM(BudgiePreTrainedModel, GenerationMixin):
    """BudgieModel plus the adaptive-softmax head, tied to the input tables. With labels the loss is
    returned and `logits` is None; without, `logits` are log-probabilities over the whole vocabulary
    in token-id order."""

    _tied_weights_keys = {}

    def __init__(self, config: BudgieConfig):
        super().__init__(config)
        self.model = BudgieModel(config)
        self.head = AdaptiveSoftmaxHead(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return None  # the head reuses the input tables

    def add_vision(self, vision_config=None):
        """Give a text model image input: set config.vision_config (default DEFAULT_VISION_CONFIG) and build the vision
        tower and projector on the model's device and dtype, initialised as the rest of Budgie is (Linear weights
        N(0, initializer_range), zero biases, unit norms). The tower is trained from scratch. Every text weight is
        left as it is. Returns the new BudgieVision."""
        cfg = self.config
        if self.model.vision is not None:
            raise ValueError("this model already has a vision tower")
        cfg.vision_config = check_vision_config(vision_config or DEFAULT_VISION_CONFIG, cfg.vocab_size,
                                                (cfg.pad_token_id, cfg.bos_token_id, cfg.eos_token_id))
        ref = next(self.parameters())
        vision = BudgieVision(cfg)
        vision.apply(self._init_weights)
        self.model.vision = vision.to(ref.device, ref.dtype)
        return self.model.vision

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values: Cache | None = None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        logits_to_keep: int | torch.Tensor = 0,
        pixel_values=None,
        pixel_attention_mask=None,
        spatial_shapes=None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if labels is not None and self.config.has_vision:
            v = self.config.vision_config               # the image span is input, never something to predict
            ids = torch.tensor([v[k] for k in IMAGE_ID_KEYS], device=labels.device)
            labels = labels.masked_fill(torch.isin(labels, ids), -100)
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
            **kwargs,
        )
        hidden, table = out.last_hidden_state, self.model.embed.adaptive
        if labels is not None:
            return CausalLMOutputWithPast(loss=self.head.loss(hidden, labels, table), logits=None, past_key_values=out.past_key_values)
        keep = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        return CausalLMOutputWithPast(logits=self.head.log_probs(hidden[:, keep, :], table), past_key_values=out.past_key_values)


AutoConfig.register(BudgieConfig.model_type, BudgieConfig, exist_ok=True)
AutoModel.register(BudgieConfig, BudgieModel, exist_ok=True)
AutoModelForCausalLM.register(BudgieConfig, BudgieForCausalLM, exist_ok=True)

# Written into config.json as `auto_map` and the source files copied next to the weights on
# save_pretrained, so a checkpoint directory loads with trust_remote_code=True on a machine
# that has never seen this package.
BudgieConfig.register_for_auto_class()
BudgieModel.register_for_auto_class("AutoModel")
BudgieForCausalLM.register_for_auto_class("AutoModelForCausalLM")
