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
from .configuration_budgie import BudgieConfig
from .convolution import CausalConv
from .decoder import DecoderLayer
from .embeddings import BudgieEmbedding
from .head import AdaptiveSoftmaxHead
from .norms import QKNorm, RMSNorm
from .retention import RetentionBranch, document_positions, to_device
from .rotary import RotaryEmbedding


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
            elif isinstance(module, Attention) and module.sinks is not None and not getattr(module.sinks, "_is_hf_initialized", False):
                module.sinks.zero_()
            elif isinstance(module, RetentionBranch):
                if not getattr(module.g, "_is_hf_initialized", False):
                    module.g.fill_(self.config.lin_gate_init)
                if not getattr(module.gain, "_is_hf_initialized", False):
                    module.gain.fill_(1.0)
            elif isinstance(module, AdaptiveSoftmaxHead) and module.cluster_vectors is not None \
                    and not getattr(module.cluster_vectors, "_is_hf_initialized", False):
                module.cluster_vectors.normal_(0.0, self.config.initializer_range)


class BudgieModel(BudgiePreTrainedModel):
    def __init__(self, config: BudgieConfig):
        super().__init__(config)
        self.embed = BudgieEmbedding(config)
        self.rotary = RotaryEmbedding(config.head_dim, config.rope_theta)
        self.layers = nn.ModuleList([DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        # The retention branch of every block (config.linear_branch); no module at all when it is off.
        self.branches = None
        if config.linear_branch != "none":
            if config.linear_branch != "before_G":
                raise NotImplementedError(f"linear_branch={config.linear_branch!r} is not implemented yet (only 'before_G')")
            self._g_pos = config.block_pattern.index("G")  # the first G layer of a block
            size = len(config.block_pattern)
            self.branches = nn.ModuleList([RetentionBranch(config, b, b * size + self._g_pos) for b in range(config.num_blocks)])
        self.record_branch_stats = False  # set by a caller that wants per-block branch statistics
        self.branch_stats = []
        # Set by a trainer (not saved): a torch.device to run every block's retention branch on, alongside
        # the block's local layers on the model's own GPU. Forwards without a cache only.
        self.branch_device = None
        self.gradient_checkpointing = False
        self.post_init()

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

        x = self.embed(input_ids, cache) if inputs_embeds is None else inputs_embeds
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
            block_size = len(cfg.block_pattern)
            remote = self.branch_device if cache is None and self.branch_device not in (None, x.device) else None
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
                if pos == self._g_pos:  # the branch reads the stream from the start of the block and writes just before G
                    if remote is None:
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
            if owner is None:
                shared[i] = kv
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
        **kwargs,
    ) -> CausalLMOutputWithPast:
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
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
