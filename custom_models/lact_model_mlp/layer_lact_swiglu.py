from __future__ import annotations

import os
import math
import warnings
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from transformers.utils import logging

from fla.models.utils import Cache
from fla.modules import RMSNorm, RotaryEmbedding

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
except ImportError:
    warnings.warn(
        "Flash Attention is not installed. Please install it via `pip install flash-attn --no-build-isolation`",
        category=ImportWarning,
    )
    flash_attn_func = None
from .ttt_operation import (
    block_causal_lact_swiglu,
    prenorm_block_causal_lact_swiglu,
    l2_norm,
)

from .ttt_operation_fused_kernel import (
    postnorm_block_causal_lact_swiglu_fused_kernel_triton,
    prenorm_block_causal_lact_swiglu_fused_kernel_triton,
)


logger = logging.get_logger(__name__)


def _get_int_env(var_name: str, default: int) -> int:
    value = os.getenv(var_name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid integer for %s=%r; fallback to %d", var_name, value, default)
        return default


def inv_softplus(x):
    if isinstance(x, torch.Tensor):
        y = x + torch.log(-torch.expm1(-x))
    else:
        y = x + math.log(-math.expm1(-x))
    return y


class LaCTMLPSWIGLULayer(nn.Module):
    """
    魔改版架构：
    1. SWA (Sliding Window Attention) 负责局部的 Token 混合。
    2. Dynamic TTT-MLP (基于 LACT 大块更新) 接在 SWA 之后，作为 Channel Mixer。
    3. 内部使用纯 PyTorch bmm 手动求导，完全切断二阶导数计算图，消除 OOM 瓶颈。
    """
    def __init__(
        self,
        hidden_size: int,
        num_attn_heads: int,
        num_lact_heads: int,
        inter_multi: float,
        window_size: int,
        lact_chunk_size: int,
        qkv_bias: bool = False,
        attn_qk_norm: bool = True,
        qkv_silu: bool = True,
        no_v_silu: bool = False,
        lr_dim: int = 1,
        use_muon: bool = False,
        lr_parameterization: str = "mamba",
        learnable_ttt_scale: bool = False,
        ttt_prenorm: bool = False,
        ttt_nope: bool = False,
        rope_theta: float = 500000.0,
        layer_idx: int = None,
        max_position_embeddings: int = 2048,
        w0_w2_low_rank: int = -1,
        use_momentum: bool = False,
        ttt_loss_type: str = "dot_product",
        fw_init_gain: float = 0.5,  # init the fast weights
        use_fused_kernel: bool = False,
        fp32_states: bool = False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.layer_idx = layer_idx
        self.max_position_embeddings = max_position_embeddings
        self.lact_chunk_size = lact_chunk_size

        self.ttt_prenorm = ttt_prenorm
        self.use_muon = use_muon
        # ==========================================
        # 1. SWA (Token Mixer) 组件
        # ==========================================
        self.num_attn_heads = num_attn_heads
        self.window_size = window_size
        self.attn_head_dim = hidden_size // num_attn_heads

        self.attn_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.attn_o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.attn_qk_norm = attn_qk_norm
        if self.attn_qk_norm:
            self.q_norm = RMSNorm(self.hidden_size)
            self.k_norm = RMSNorm(self.hidden_size)
            
        self.rope_theta = rope_theta
        self.rotary = RotaryEmbedding(dim=self.attn_head_dim, base=self.rope_theta)

        # ==========================================
        # 2. Dynamic TTT-MLP (Channel Mixer) 组件
        # ==========================================
        self.num_fw_heads = int(num_lact_heads)
        self.use_ttt = self.num_fw_heads > 0
        self.fw_head_dim = hidden_size // self.num_fw_heads if self.use_ttt else 0
        
        self.d_in = self.fw_head_dim
        self.d_out = self.fw_head_dim
        self.d_h = int(self.d_in * inter_multi) if self.use_ttt else 0
        self.ttt_norm = (
            RMSNorm(self.fw_head_dim, elementwise_affine=True) if self.use_ttt else None
        )
        self.mlp_norm = RMSNorm(hidden_size)

        # 静态投影层：将 SWA 的输出投影为当前 Chunk 的局部 Q, K, V
        if self.use_ttt:
            self.mlp_qkv = nn.Linear(
                hidden_size, 3 * self.num_fw_heads * self.fw_head_dim, bias=False
            )
            self.mlp_o_proj = nn.Linear(
                self.num_fw_heads * self.fw_head_dim, hidden_size, bias=False
            )
        else:
            self.mlp_qkv = None
            self.mlp_o_proj = None

        # 快速权重的 Meta-Weights 初始化
        if self.use_ttt:
            self.w0 = nn.Parameter(
                torch.randn(self.num_fw_heads, self.d_h, self.d_in) / math.sqrt(self.d_in)
            )
            self.w2 = nn.Parameter(
                torch.randn(self.num_fw_heads, self.d_h, self.d_in) / math.sqrt(self.d_in)
            )
            self.w1 = nn.Parameter(
                torch.randn(self.num_fw_heads, self.d_out, self.d_h) / math.sqrt(self.d_h)
            )
        else:
            self.w0 = None
            self.w2 = None
            self.w1 = None

        # 学习率投影
        self.lr_dim = int(lr_dim * 3 * self.num_fw_heads) if self.use_ttt else 0
        self.lr_proj = nn.Linear(self.hidden_size, self.lr_dim) if self.use_ttt else None
        self.lr_parameterization = lr_parameterization
        if lr_parameterization.lower() == "mamba":
            base_lr = 0.001
            self.base_lr_inv = inv_softplus(base_lr)

        self.learnable_ttt_scale = learnable_ttt_scale and self.use_ttt
        if self.learnable_ttt_scale:
            self.ttt_scale_proj = nn.Linear(hidden_size, self.num_fw_heads)
        else:
            self.ttt_scale_proj = None

        self.use_momentum = use_momentum and self.use_ttt
        if self.use_momentum:
            self.momentum_proj = nn.Sequential(
                nn.Linear(hidden_size, self.num_fw_heads),
                nn.Sigmoid(),
            )
        else:
            self.momentum_proj = None
        self.fp32_states = fp32_states
        self.use_fused_kernel = use_fused_kernel
        self.log_chunk_diversity = os.getenv("LACT_LOG_CHUNK_DIVERSITY", "0") == "1"
        self.log_chunk_diversity_every = max(_get_int_env("LACT_LOG_CHUNK_DIVERSITY_EVERY", 1), 1)
        self.log_chunk_diversity_max_chunks = _get_int_env("LACT_LOG_CHUNK_DIVERSITY_MAX_CHUNKS", 0)
        self._diversity_log_forward_calls = 0
        if self.log_chunk_diversity:
            logger.info(
                "Enable LACT chunk diversity log for lact_model_mlp layer=%s every=%d max_chunks=%d",
                self.layer_idx,
                self.log_chunk_diversity_every,
                self.log_chunk_diversity_max_chunks,
            )

    @torch.no_grad()
    def _log_chunk_diversity(self, tensor: torch.Tensor, tensor_name: str, call_idx: int) -> None:
        if tensor.ndim != 3:
            return

        chunk_size = max(int(self.lact_chunk_size), 1)
        _, seq_len, feat_dim = tensor.shape
        if seq_len == 0:
            return

        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        chunks_to_log = (
            num_chunks
            if self.log_chunk_diversity_max_chunks <= 0
            else min(num_chunks, self.log_chunk_diversity_max_chunks)
        )

        for chunk_idx in range(chunks_to_log):
            start = chunk_idx * chunk_size
            end = min(seq_len, start + chunk_size)
            chunk = tensor[:, start:end, :].detach().float().reshape(-1, feat_dim)
            num_tokens = chunk.shape[0]

            if num_tokens < 2:
                logger.info(
                    "[chunk_diversity][model=lact_model_mlp][layer=%s][call=%d][tensor=%s][chunk=%d/%d] "
                    "tokens=%d (skip: <2 tokens)",
                    self.layer_idx,
                    call_idx,
                    tensor_name,
                    chunk_idx + 1,
                    num_chunks,
                    num_tokens,
                )
                continue

            chunk_normed = chunk / chunk.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            sum_vec = chunk_normed.sum(dim=0)
            offdiag_sum = (sum_vec * sum_vec).sum() - float(num_tokens)
            mean_pairwise_cos = (offdiag_sum / float(num_tokens * (num_tokens - 1))).item()

            token_var = chunk.var(dim=0, unbiased=False)
            token_var_mean = token_var.mean().item()
            token_var_max = token_var.max().item()

            chunk_centered = chunk - chunk.mean(dim=0, keepdim=True)
            cov_denom = float(max(num_tokens - 1, 1))
            cov = (chunk_centered.transpose(0, 1) @ chunk_centered) / cov_denom
            eigvals = torch.linalg.eigvalsh(cov).clamp_min(0.0)
            eigvals = torch.flip(eigvals, dims=(0,))
            eigvals_sum = float(eigvals.sum())

            if eigvals_sum <= 0.0:
                effective_rank = 0.0
                top_singular_values = [0.0, 0.0, 0.0]
            else:
                probs = eigvals / eigvals_sum
                entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum()
                effective_rank = torch.exp(entropy).item()
                topk = min(3, eigvals.shape[0])
                singular_vals = torch.sqrt(eigvals[:topk] * cov_denom)
                top_singular_values = [float(x) for x in singular_vals.tolist()]
                if topk < 3:
                    top_singular_values.extend([0.0] * (3 - topk))

            logger.info(
                "[chunk_diversity][model=lact_model_mlp][layer=%s][call=%d][tensor=%s][chunk=%d/%d] "
                "tokens=%d mean_pairwise_cos=%.6f effective_rank=%.3f top_singular_values=%s "
                "token_var_mean=%.6e token_var_max=%.6e",
                self.layer_idx,
                call_idx,
                tensor_name,
                chunk_idx + 1,
                num_chunks,
                num_tokens,
                mean_pairwise_cos,
                effective_rank,
                [round(v, 6) for v in top_singular_values],
                token_var_mean,
                token_var_max,
            )

        if chunks_to_log < num_chunks:
            logger.info(
                "[chunk_diversity][model=lact_model_mlp][layer=%s][call=%d][tensor=%s] "
                "skip remaining chunks: logged=%d total=%d",
                self.layer_idx,
                call_idx,
                tensor_name,
                chunks_to_log,
                num_chunks,
            )

    @torch.no_grad()
    def _log_fast_kv_chunk_diversity(self, fast_k: torch.Tensor, fast_v: torch.Tensor) -> None:
        if not self.log_chunk_diversity:
            return

        self._diversity_log_forward_calls += 1
        if self._diversity_log_forward_calls % self.log_chunk_diversity_every != 0:
            return

        call_idx = self._diversity_log_forward_calls
        self._log_chunk_diversity(fast_k, "fast_k", call_idx)
        self._log_chunk_diversity(fast_v, "fast_v", call_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [b, s, d]
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, seq_len, _ = hidden_states.size()

        # ==========================================
        # Part A: SWA (局部注意力)
        # ==========================================
        q, k, v = self.attn_qkv(hidden_states).chunk(3, dim=-1)

        if self.attn_qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        q = rearrange(q, "... (h d) -> ... h d", d=self.attn_head_dim)
        k = rearrange(k, "... (h d) -> ... h d", d=self.attn_head_dim)
        v = rearrange(v, "... (h d) -> ... h d", d=self.attn_head_dim)

        cu_seqlens = kwargs.get("cu_seqlens", None)
        seqlen_offset, max_seqlen = 0, seq_len

        # RoPE for Attention
        q, k = self.rotary(
            q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens,
        )

        if flash_attn_func is None:
            raise ImportError(
                "Please install Flash Attention via `pip install flash-attn --no-build-isolation` first"
            )

        # Flash Attention
        if attention_mask is not None:
            q, k, v, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                q, k, v, attention_mask, seq_len
            )
            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_q, max_seqlen_k = max_seq_lens
            attn_out = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                causal=(self.window_size is None),
                window_size=((-1, -1) if self.window_size is None else (self.window_size - 1, 0)),
            )
            attn_out = pad_input(attn_out, indices_q, batch_size, seq_len)
        elif cu_seqlens is not None:
            attn_out = flash_attn_varlen_func(
                q.squeeze(0), k.squeeze(0), v.squeeze(0),
                cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
                causal=(self.window_size is None),
                window_size=((-1, -1) if self.window_size is None else (self.window_size - 1, 0)),
            ).unsqueeze(0)
        else:
            attn_out = flash_attn_func(
                q, k, v, causal=(self.window_size is None),
                window_size=((-1, -1) if self.window_size is None else (self.window_size - 1, 0)),
            )

        attn_out = attn_out.reshape(batch_size, seq_len, -1)
        attn_out = self.attn_o_proj(attn_out)

        # SWA Residual
        x = hidden_states + attn_out
        o = x

        # ==========================================
        # Part B: Dynamic TTT-MLP 
        # ==========================================
        if self.use_ttt:
            mlp_in = self.mlp_norm(x)

            # 1. 投影出当前局部上下文的 Q, K, V 和 LR
            fast_qkv = self.mlp_qkv(mlp_in)
            fast_q, fast_k, fast_v = fast_qkv.chunk(3, dim=-1)

            fast_q = rearrange(fast_q, "b s (n_h d) -> (b n_h) s d", n_h=self.num_fw_heads)
            fast_k = rearrange(fast_k, "b s (n_h d) -> (b n_h) s d", n_h=self.num_fw_heads)
            fast_v = rearrange(fast_v, "b s (n_h d) -> (b n_h) s d", n_h=self.num_fw_heads)

            fast_q = l2_norm(F.silu(fast_q))
            fast_k = l2_norm(F.silu(fast_k))
            fast_v = F.silu(fast_v)
            self._log_fast_kv_chunk_diversity(fast_k, fast_v)

            lr = self.lr_proj(mlp_in)
            if self.lr_parameterization == "mamba":
                lr = torch.nn.functional.softplus(lr.float() + self.base_lr_inv)

            fw_lr = rearrange(lr, "b s (n_h lr_dim) -> (b n_h) s lr_dim", n_h=self.num_fw_heads)
            fw_lr1, fw_lr2, fw_lr3 = fw_lr.chunk(3, dim=-1)

            # 2. 拷贝 Meta-Weights 作为当前序列的起点
            fw_w0 = self.w0.repeat(batch_size, 1, 1).clone()
            fw_w1 = self.w1.repeat(batch_size, 1, 1).clone()
            fw_w2 = self.w2.repeat(batch_size, 1, 1).clone()

            if self.use_momentum:
                momentum = self.momentum_proj(hidden_states).float()  # [b, s, nh]
                momentum = rearrange(
                    momentum, "b s (n_h d) -> (b n_h) s d", n_h=self.num_fw_heads
                )
            else:
                momentum = None

            if self.fp32_states:
                # here we cast the fast weights to fp32, but all matmuls are still in bf16
                # only fast weight updates are in fp32.  This is similar to bf16 training of slow weights.
                fw_w0 = fw_w0.to(torch.float32)
                fw_w1 = fw_w1.to(torch.float32)
                fw_w2 = fw_w2.to(torch.float32)

            # [b * nh, s, d_ttt_head]
            if self.ttt_prenorm:
                # pre-norm version of ttt.   state = state + f(norm(state))
                if self.use_fused_kernel:
                    fw_x = prenorm_block_causal_lact_swiglu_fused_kernel_triton(
                        fw_w0,
                        fw_w1,
                        fw_w2,
                        fast_q,
                        fast_k,
                        fast_v,
                        fw_lr1,
                        fw_lr2,
                        fw_lr3,
                        chunk_size=self.lact_chunk_size,
                        use_muon=self.use_muon,
                        momentum=momentum,
                    )
                else:
                    fw_x = prenorm_block_causal_lact_swiglu(
                        fw_w0,
                        fw_w1,
                        fw_w2,
                        fast_q,
                        fast_k,
                        fast_v,
                        fw_lr1,
                        fw_lr2,
                        fw_lr3,
                        chunk_size=self.lact_chunk_size,
                        use_muon=self.use_muon,
                        momentum=momentum,
                    )
            else:
                # post-norm version of ttt.   state = norm(state + f(state))
                if self.use_fused_kernel:
                    fw_x = postnorm_block_causal_lact_swiglu_fused_kernel_triton(
                        fw_w0,
                        fw_w1,
                        fw_w2,
                        fast_q,
                        fast_k,
                        fast_v,
                        fw_lr1,
                        fw_lr2,
                        fw_lr3,
                        chunk_size=self.lact_chunk_size,
                        use_muon=self.use_muon,
                        momentum=momentum,
                    )
                else:
                    fw_x = block_causal_lact_swiglu(
                        fw_w0,
                        fw_w1,
                        fw_w2,
                        fast_q,
                        fast_k,
                        fast_v,
                        fw_lr1,
                        fw_lr2,
                        fw_lr3,
                        chunk_size=self.lact_chunk_size,
                        use_muon=self.use_muon,
                        momentum=momentum,
                    )

            # per-head output norm for ttt layer.
            ttt_x_normed = self.ttt_norm(fw_x)
            if self.learnable_ttt_scale:
                ttt_scale = F.silu(self.ttt_scale_proj(hidden_states), inplace=False)
                ttt_scale = rearrange(
                    ttt_scale, "b s (n_h d) -> (b n_h) s d", n_h=self.num_fw_heads
                )
                ttt_x_normed = ttt_x_normed * ttt_scale

            ttt_x_normed = rearrange(
                ttt_x_normed, "(b n_h) s d -> b s (n_h d)", n_h=self.num_fw_heads
            )
            o = o + ttt_x_normed
        ### TTT ends ###

        if self.use_ttt:
            o = self.mlp_o_proj(o)

        attentions = None
        return o, attentions, past_key_values

    def _upad_input(self, q, k, v, attention_mask, q_len):
        batch_size, seq_len, num_heads, head_dim = k.shape
        cache_mask = attention_mask[:, -seq_len:]
        seqlens = cache_mask.sum(-1, dtype=torch.int32)
        indices_k = torch.nonzero(cache_mask.flatten(), as_tuple=False).flatten()
        max_seqlen_k = seqlens.max().item()
        cu_seqlens_k = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))

        k = index_first_axis(k.reshape(batch_size * seq_len, num_heads, head_dim), indices_k)
        v = index_first_axis(v.reshape(batch_size * seq_len, num_heads, head_dim), indices_k)
        if q_len == seq_len:
            q = index_first_axis(
                q.reshape(batch_size * seq_len, self.num_attn_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_q = max_seqlen_k
            indices_q = indices_k
        elif q_len == 1:
            max_seqlen_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=q.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            q = q.squeeze(1)
        else:
            attention_mask = attention_mask[:, -q_len:]
            q, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(q, attention_mask)

        return (
            q,
            k,
            v,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_q, max_seqlen_k),
        )
