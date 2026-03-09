# # -*- coding: utf-8 -*-

# from typing import Optional

# from transformers.configuration_utils import PretrainedConfig


# class TttE2EConfig(PretrainedConfig):
#     model_type = "ttt_e2e"
#     keys_to_ignore_at_inference = ["past_key_values"]

#     def __init__(
#         self,
#         vocab_size: int = 32000,
#         hidden_size: int = 768,
#         intermediate_size: int = 2048,
#         num_hidden_layers: int = 12,
#         num_attention_heads: int = 12,
#         sliding_window_size: int = 1024,
#         seq_modeling_block: str = "self_attention",
#         qk_norm: bool = True,
#         pre_norm: bool = True,
#         post_norm: bool = True,
#         rms_norm_eps: float = 1e-6,
#         initializer_range: float = 0.02,
#         resid_pdrop: float = 0.0,
#         embd_pdrop: float = 0.0,
#         attn_pdrop: float = 0.0,
#         rope_theta: float = 10000.0,
#         hidden_act: str = "silu",
#         train_mode: str = "pretrain",
#         mini_batch_size: int = 1024,
#         suffix_len: int = 0,
#         inner_lr: float = 1e-3,
#         force_eager_attention: bool = False,
#         tie_word_embeddings: bool = False,
#         use_cache: bool = True,
#         max_position_embeddings: Optional[int] = 2048,
#         pad_token_id: int = None,
#         bos_token_id: int = 1,
#         eos_token_id: int = 2,
#         **kwargs,
#     ):
#         self.vocab_size = vocab_size
#         self.hidden_size = hidden_size
#         self.intermediate_size = intermediate_size
#         self.num_hidden_layers = num_hidden_layers
#         self.num_attention_heads = num_attention_heads
#         self.sliding_window_size = sliding_window_size
#         self.seq_modeling_block = seq_modeling_block
#         self.qk_norm = qk_norm
#         self.pre_norm = pre_norm
#         self.post_norm = post_norm
#         self.rms_norm_eps = rms_norm_eps
#         self.initializer_range = initializer_range
#         self.resid_pdrop = resid_pdrop
#         self.embd_pdrop = embd_pdrop
#         self.attn_pdrop = attn_pdrop
#         self.rope_theta = rope_theta
#         self.hidden_act = hidden_act
#         self.train_mode = train_mode
#         self.mini_batch_size = mini_batch_size
#         self.suffix_len = suffix_len
#         self.inner_lr = inner_lr
#         self.force_eager_attention = force_eager_attention
#         self.use_cache = use_cache
#         self.max_position_embeddings = max_position_embeddings

#         super().__init__(
#             pad_token_id=pad_token_id,
#             bos_token_id=bos_token_id,
#             eos_token_id=eos_token_id,
#             tie_word_embeddings=tie_word_embeddings,
#             **kwargs,
#         )
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional, Union, Tuple

from transformers.configuration_utils import PretrainedConfig


IntOrIntTuple = Union[int, Tuple[int, int]]


class E2ETTTConfig(PretrainedConfig):
    """HF config for an E2E-TTT-style model (Torch draft)."""

    model_type = "e2e_ttt"

    def __init__(
        self,
        # tokenizer / io
        vocab_size: int = 32000,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: Optional[int] = None,
        tie_word_embeddings: bool = False,
        # transformer sizes
        hidden_size: int = 2048,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 32,
        intermediate_size: Optional[int] = None,
        hidden_act: str = "swish",
        rms_norm_eps: float = 1e-6,
        norm_eps: float = 1e-6,
        initializer_range: float = 0.02,
        # Architecture toggles (not framework-mandated): alignable with JAX variants.
        pre_norm: bool = True,
        post_norm: bool = True,
        qk_norm: bool = True,
        resid_pdrop: float = 0.0,
        embd_pdrop: float = 0.0,
        attn_pdrop: float = 0.0,
        # positions
        max_position_embeddings: int = 2048,
        rope_theta: float = 10000.0,
        # sliding-window attention
        window_size: int = 2048,
        # E2E-TTT knobs
        use_e2e_ttt: bool = True,
        suffix_frac: float = 0.25,
        mini_batch_size: int = 16,
        ilr_warmup_steps: int = 0,
        optimizer_inner: Optional[dict] = None,
        inner_steps_per_chunk: int = 1,
        ttt_mlp_only: bool = True,
        two_mlp_per_block: bool = True,
        detach_fast_weights: bool = False,
        # optional: update subset of MLP parameters
        inner_param_filter: str = "prime_mlp",  # prime_mlp | all_mlp
        # attention backend
        attn_backend: str = "flash",  # flash | sdpa
        fuse_cross_entropy: bool = True,
        fuse_norm: bool = True,
        last_layer_fuse_norm: bool = True,
        fuse_swiglu: bool = True,
        hidden_ratio: Optional[int] = 4,
        # debug / instrumentation
        debug_ttt_logs: bool = False,
        debug_ttt_log_every: int = 50,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        # sizes
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size or (4 * hidden_size)
        hidden_act = str(hidden_act).lower()
        if hidden_act == "silu":
            hidden_act = "swish"
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.norm_eps = norm_eps
        self.initializer_range = initializer_range
        self.pre_norm = bool(pre_norm)
        self.post_norm = bool(post_norm)
        self.qk_norm = bool(qk_norm)
        self.resid_pdrop = float(resid_pdrop)
        self.embd_pdrop = float(embd_pdrop)
        self.attn_pdrop = float(attn_pdrop)
        self.fuse_swiglu = fuse_swiglu
        self.hidden_ratio = hidden_ratio
        # positions
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        
        self.fuse_norm = fuse_norm
        self.last_layer_fuse_norm = last_layer_fuse_norm  # seems that you need to set this to False to use activation checkpointing for every layer.
        self.fuse_swiglu = fuse_swiglu
        self.fuse_cross_entropy = fuse_cross_entropy
        # SWA
        self.window_size = window_size

        # E2E-TTT
        self.use_e2e_ttt = use_e2e_ttt
        self.suffix_frac = float(suffix_frac)
        self.suffix_len = max(1, int(round(self.num_hidden_layers * self.suffix_frac)))
        self.mini_batch_size = int(mini_batch_size)
        
        
        resolved_inner_lr = 0.001
        resolved_optimizer_inner = dict(optimizer_inner) if optimizer_inner is not None else {}
        if "optimizer_type" not in resolved_optimizer_inner:
            resolved_optimizer_inner["optimizer_type"] = "sgd"
        if "lr" in resolved_optimizer_inner:
            resolved_inner_lr = float(resolved_optimizer_inner["lr"])
        

        ilr_init = None
        if "ilr_init" in resolved_optimizer_inner:
            ilr_init = float(resolved_optimizer_inner["ilr_init"])
        if "clip_gradient" not in resolved_optimizer_inner:
            resolved_optimizer_inner["clip_gradient"] = 0.0
        resolved_optimizer_inner["clip_gradient"] = float(resolved_optimizer_inner["clip_gradient"])
        resolved_optimizer_inner["optimizer_type"] = str(resolved_optimizer_inner["optimizer_type"])

        self.inner_lr = resolved_inner_lr
        self.optimizer_inner = resolved_optimizer_inner
        self.ilr_warmup_steps = int(ilr_warmup_steps)
        if self.ilr_warmup_steps < 0:
            raise ValueError("ilr_warmup_steps must be >= 0")
        self.ilr_init = float(resolved_inner_lr if ilr_init is None else ilr_init)
        self.inner_steps_per_chunk = int(inner_steps_per_chunk)
        self.ttt_mlp_only = bool(ttt_mlp_only)
        self.two_mlp_per_block = bool(two_mlp_per_block)
        self.detach_fast_weights = bool(detach_fast_weights)
        self.inner_param_filter = str(inner_param_filter)

        self.attn_backend = str(attn_backend)
        self.debug_ttt_logs = bool(debug_ttt_logs)
        self.debug_ttt_log_every = max(1, int(debug_ttt_log_every))

        # basic sanity
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be > 0")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.mini_batch_size <= 0:
            raise ValueError("mini_batch_size must be > 0")


class E2ETTTV2Config(E2ETTTConfig):
    model_type = "ttt_e2e_v2"
