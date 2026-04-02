# -*- coding: utf-8 -*-

import math
from typing import Optional, Union

from transformers.configuration_utils import PretrainedConfig


class HymbaConfig(PretrainedConfig):
    """
    Configuration for LaCT-SWIGLU model.
    It implements the LaCT-SWIGLU layer mixed with in-layer sliding window attention

    Args:
        hidden_size (int, optional): The hidden size of the model. Defaults to 2048.
        num_hidden_layers (int, optional): The number of hidden layers in the model. Defaults to 24.
        num_attn_heads (int, optional): The number of attention heads in the model. Defaults to 32.
        num_lact_heads (int, optional): The number of feed-forward heads in the model. Defaults to 4.
    """

    model_type = "hymba"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 32000,
        tie_word_embeddings: bool = False,
        hidden_size: int = 2048,
        intermediate_size: Optional[int] = None,
        num_hidden_layers: int = 24,
        num_attn_heads: Optional[int] = 32,
        # num_attention_heads: Optional[int] = None,
        num_key_value_heads: Optional[int] = None,
        hidden_act: str = "swish",
        initializer_range: float = 0.006,
        rms_norm_eps: Optional[float] = None,
        use_cache: bool = True,
        calc_logits_for_entire_prompt: bool = False,
        output_router_logits: bool = False,
        router_aux_loss_coef: float = 0.001,
        pad_token_id: int = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        # sliding_window: Optional[int] = None,
        max_position_embeddings: int = 2048,
        orig_max_position_embeddings: Optional[int] = None,
        attention_dropout: float = 0.0,
        num_experts_per_tok: int = 2,
        num_experts: int = 16,
        use_mamba_kernels: bool = True,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_dt_rank: Union[int, str] = "auto",
        mamba_conv_bias: bool = True,
        mamba_proj_bias: bool = False,
        mamba_inner_layernorms: bool = True,
        kv_reuse_every_i_layer: int = -1,
        kv_reuse_group: Optional[list] = None,
        kv_weight_reuse: bool = False,
        global_attn_idx: Optional[list] = None,
        num_mamba: int = 1,
        attn_implementation_new: str = "sdpa",
        rope_type: Optional[str] = None,

        qkv_bias: bool = False,
        attn_qk_norm: bool = False,
        qkv_silu: bool = True,  # if True, apply silu to q, k, v.
        no_v_silu: bool = False,  # if True, don't apply silu to v, will overwrite qkv_silu.
        window_size: int = 2048,
        use_sliding_window: bool = True,
        layer_types: Optional[list] = None,
        rope_theta: Optional[float] = 10000.0,
        hidden_ratio: Optional[int] = 4,
        elementwise_affine: Optional[bool] = True,
        norm_eps: float = 1e-6,
        fuse_norm: bool = True,
        last_layer_fuse_norm: bool = True,
        fuse_swiglu: bool = True,
        fuse_cross_entropy: bool = True,
        **kwargs,
    ):
        if num_attention_heads is None:
            num_attention_heads = num_attn_heads
        if num_attn_heads is None:
            num_attn_heads = num_attention_heads
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.tie_word_embeddings = tie_word_embeddings
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attn_heads = num_attn_heads
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = norm_eps if rms_norm_eps is None else rms_norm_eps
        self.use_cache = use_cache
        self.calc_logits_for_entire_prompt = calc_logits_for_entire_prompt
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.sliding_window = window_size if use_sliding_window else None
        self.max_position_embeddings = max_position_embeddings
        self.orig_max_position_embeddings = orig_max_position_embeddings
        self.attention_dropout = attention_dropout
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.use_mamba_kernels = use_mamba_kernels
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.mamba_dt_rank = (
            math.ceil(self.hidden_size / 16)
            if mamba_dt_rank == "auto"
            else mamba_dt_rank
        )
        self.mamba_conv_bias = mamba_conv_bias
        self.mamba_proj_bias = mamba_proj_bias
        self.mamba_inner_layernorms = mamba_inner_layernorms
        self.kv_reuse_every_i_layer = kv_reuse_every_i_layer
        self.kv_reuse_group = kv_reuse_group
        self.kv_weight_reuse = kv_weight_reuse
        self.global_attn_idx = global_attn_idx
        self.num_mamba = num_mamba
        self.attn_implementation_new = attn_implementation_new
        self.attn_implementation = kwargs.pop("attn_implementation", attn_implementation_new)
        self.rope_type = rope_type
        self.qkv_bias = qkv_bias
        self.attn_qk_norm = attn_qk_norm
        self.qkv_silu = qkv_silu
        self.no_v_silu = no_v_silu
        self.window_size = window_size
        # When `use_sliding_window` is False, LaCT will use full attention
        # (i.e. no locality constraint) by passing `window_size=None` to the
        # attention layer. This mirrors the `use_sliding_window` switch used in
        # `Qwen3GDNConfig`, while keeping the existing `window_size` interface.
        self.use_sliding_window = use_sliding_window
        # Optional per-layer attention pattern, mirroring Qwen3GDNConfig.layer_types.
        # Expected entries (if provided) include:
        # - "full_attention": use global attention (window_size=None)
        # - "linear_attention": mapped to LaCT with sliding window enabled
        # - "sliding_attention": LaCT with sliding window enabled
        # If None, all layers fall back to the global `use_sliding_window` flag.
        self.layer_types = layer_types
        self.rope_theta = rope_theta

        self.hidden_ratio = hidden_ratio

        self.elementwise_affine = elementwise_affine
        self.norm_eps = norm_eps

        self.fuse_norm = fuse_norm
        self.last_layer_fuse_norm = last_layer_fuse_norm  # seems that you need to set this to False to use activation checkpointing for every layer.
        self.fuse_swiglu = fuse_swiglu
        self.fuse_cross_entropy = fuse_cross_entropy

        self.attn_hidden_size = kwargs.pop("attn_hidden_size", -1)
        self.kq_head_dim = kwargs.pop("kq_head_dim", -1)
        self.v_head_dim = kwargs.pop("v_head_dim", -1)
        self.kq_norm = kwargs.pop("kq_norm", None)
        self.rope = kwargs.pop("rope", False)
        self.num_memory_tokens = kwargs.pop("num_memory_tokens", 0)
        self.memory_tokens_interspersed_every = kwargs.pop(
            "memory_tokens_interspersed_every",
            0,
        )
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
