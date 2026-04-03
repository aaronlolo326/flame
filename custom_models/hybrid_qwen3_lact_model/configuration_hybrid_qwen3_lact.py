# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional

from transformers.configuration_utils import PretrainedConfig


class HybridQwen3LaCTConfig(PretrainedConfig):
    model_type = "hybrid_qwen3_lact"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 4096,
        intermediate_size: int = 22016,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 32,
        head_dim: Optional[int] = None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = 1000000.0,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        use_sliding_window: bool = True,
        sliding_window: Optional[int] = 4096,
        hybrid_layer_types: Optional[list[str]] = None,
        num_lact_heads: int = 4,
        inter_multi: int = 1,
        qkv_bias: bool = False,
        attn_qk_norm: bool = False,
        lact_chunk_size: int = 1024,
        use_muon: bool = False,
        lr_dim: int = 1,
        qkv_silu: bool = True,
        no_v_silu: bool = False,
        lr_parameterization: str = "mamba",
        learnable_ttt_scale: bool = True,
        ttt_inner_steps: int = 1,
        use_momentum: bool = True,
        ttt_loss_type: str = "dot_product",
        ttt_prenorm: bool = True,
        memory_update_phases: Optional[list[int]] = None,
        ttt_nope: bool = False,
        w0_w2_low_rank: int = 32,
        fw_init_gain: float = 0.5,
        use_fused_kernel: bool = True,
        fp32_states: bool = False,
        fuse_cross_entropy: bool = True,
        source_model_name_or_path: Optional[str] = None,
        source_model_type: Optional[str] = None,
        w0_init_strategy: str = "random_small",
        w0_init_scale: float = 0.1,
        pad_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads if head_dim is None else head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if use_sliding_window else None
        self.hybrid_layer_types = hybrid_layer_types or ["lact"] * num_hidden_layers
        if len(self.hybrid_layer_types) != num_hidden_layers:
            raise ValueError(
                f"hybrid_layer_types must have length {num_hidden_layers}, "
                f"got {len(self.hybrid_layer_types)}."
            )
        invalid = sorted(set(self.hybrid_layer_types) - {"fa", "lact"})
        if invalid:
            raise ValueError(
                "hybrid_layer_types must contain only 'fa' or 'lact'. "
                f"Got invalid entries: {invalid}."
            )

        self.num_lact_heads = num_lact_heads
        self.inter_multi = inter_multi
        self.qkv_bias = qkv_bias
        self.attn_qk_norm = attn_qk_norm
        self.lact_chunk_size = lact_chunk_size
        self.use_muon = use_muon
        self.lr_dim = lr_dim
        self.qkv_silu = qkv_silu
        self.no_v_silu = no_v_silu
        self.lr_parameterization = lr_parameterization
        self.learnable_ttt_scale = learnable_ttt_scale
        self.ttt_inner_steps = ttt_inner_steps
        self.use_momentum = use_momentum
        self.ttt_loss_type = ttt_loss_type
        self.ttt_prenorm = ttt_prenorm
        self.memory_update_phases = memory_update_phases
        self.ttt_nope = ttt_nope
        self.w0_w2_low_rank = w0_w2_low_rank
        self.fw_init_gain = fw_init_gain
        self.use_fused_kernel = use_fused_kernel
        self.fp32_states = fp32_states
        self.fuse_cross_entropy = fuse_cross_entropy
        self.source_model_name_or_path = source_model_name_or_path
        self.source_model_type = source_model_type
        self.w0_init_strategy = w0_init_strategy
        self.w0_init_scale = w0_init_scale
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
