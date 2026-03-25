# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss

from ..lact_model.layer_lact_swiglu import LowRankFastWeight, inv_softplus
from ..lact_model.ttt_operation import (
    block_causal_lact_swiglu,
    l2_norm,
    prenorm_block_causal_lact_swiglu,
)
from ..lact_model.ttt_operation_fused_kernel import (
    postnorm_block_causal_lact_swiglu_fused_kernel_triton,
    prenorm_block_causal_lact_swiglu_fused_kernel_triton,
)
from .configuration_hybrid_qwen3_lact import HybridQwen3LaCTConfig

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Qwen3MLP(nn.Module):
    def __init__(self, config: HybridQwen3LaCTConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = getattr(F, config.hidden_act) if hasattr(F, config.hidden_act) else F.silu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 1000000.0) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.LongTensor, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        # Match Qwen3-style float32 RoPE computation for numerical stability.
        # Recompute from (dim, base) to avoid relying on non-persistent buffer materialization.
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)
                / self.dim
            )
        )
        pos = position_ids.to(device=device, dtype=torch.long)
        if not pos.is_contiguous():
            pos = pos.contiguous()
        inv_freq_expanded = inv_freq[None, :, None].expand(pos.shape[0], -1, 1)
        pos_expanded = pos[:, None, :].to(torch.float32)
        freqs = (inv_freq_expanded @ pos_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        # if not (torch.isfinite(cos).all() and torch.isfinite(sin).all()):
        #     # Defensive guard: keep forward finite even if upstream kernels return non-finite values.
        #     cos = torch.nan_to_num(cos, nan=1.0, posinf=1.0, neginf=1.0)
        #     sin = torch.nan_to_num(sin, nan=0.0, posinf=0.0, neginf=0.0)
        return cos.to(dtype), sin.to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def build_attention_mask(
    seq_len: int,
    device: torch.device,
    sliding_window: Optional[int],
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    pos = torch.arange(seq_len, device=device)
    allowed = pos[:, None] >= pos[None, :]
    if sliding_window is not None:
        allowed &= (pos[:, None] - pos[None, :]) < sliding_window
    if attention_mask is not None:
        allowed = allowed.unsqueeze(0) & attention_mask[:, None, None, :].bool()
        # Avoid all-masked query rows, which would lead to NaNs in softmax.
        has_valid_key = allowed.any(dim=-1, keepdim=True)
        eye = torch.eye(seq_len, device=device, dtype=torch.bool).unsqueeze(0).unsqueeze(1)
        allowed = torch.where(has_valid_key, allowed, eye)
    return allowed


class HybridQwen3LaCTBranch(nn.Module):
    def __init__(self, config: HybridQwen3LaCTConfig, layer_idx: int) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads for GQA. "
                f"Got {self.num_attention_heads} and {self.num_key_value_heads}."
            )
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.token_mixer_dim = self.num_attention_heads * self.head_dim
        self.num_fw_heads = config.num_lact_heads
        if self.num_fw_heads <= 0:
            raise ValueError(
                "num_lact_heads must be > 0 for layers configured as 'lact'. "
                f"Got num_lact_heads={self.num_fw_heads} at layer {layer_idx}."
            )
        if self.token_mixer_dim % self.num_fw_heads != 0:
            raise ValueError(
                "token_mixer_dim must be divisible by num_lact_heads. "
                f"Got token_mixer_dim={self.token_mixer_dim}, num_lact_heads={self.num_fw_heads}."
            )
        self.fw_head_dim = self.token_mixer_dim // self.num_fw_heads
        self.inter_multi = config.inter_multi
        self.attn_qk_norm = config.attn_qk_norm
        if self.attn_qk_norm:
            self.q_norm = RMSNorm(self.fw_head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.fw_head_dim, eps=config.rms_norm_eps)
        self.qkv_silu = config.qkv_silu
        self.no_v_silu = config.no_v_silu
        self.ttt_prenorm = config.ttt_prenorm
        self.ttt_nope = config.ttt_nope
        self.use_muon = config.use_muon
        self.use_momentum = config.use_momentum
        self.use_fused_kernel = config.use_fused_kernel
        self.fp32_states = config.fp32_states
        self.w0_w2_low_rank = config.w0_w2_low_rank
        self.lact_chunk_size = config.lact_chunk_size
        self.memory_update_phase = None
        if config.memory_update_phases is not None:
            self.memory_update_phase = config.memory_update_phases[layer_idx]

        d_in = self.fw_head_dim
        d_out = self.fw_head_dim
        d_h = int(d_in * self.inter_multi)
        self.d_h = d_h
        self.lr_dim = int(config.lr_dim * 3 * self.num_fw_heads)
        self.lr_parameterization = config.lr_parameterization
        self.base_lr_inv = inv_softplus(0.001)

        if self.w0_w2_low_rank > 0:
            self.w0 = LowRankFastWeight(self.num_fw_heads, d_h, d_in, self.w0_w2_low_rank, init_gain=config.fw_init_gain, add_identity=True)
            self.w2 = LowRankFastWeight(self.num_fw_heads, d_h, d_in, self.w0_w2_low_rank, init_gain=config.fw_init_gain, add_identity=True)
        else:
            self.w0 = nn.Parameter(torch.randn(self.num_fw_heads, d_h, d_in) / math.sqrt(d_in))
            self.w2 = nn.Parameter(torch.randn(self.num_fw_heads, d_h, d_in) / math.sqrt(d_in))
        self.w1 = nn.Parameter(torch.randn(self.num_fw_heads, d_out, d_h) / math.sqrt(d_h))

        self.lr_proj = nn.Linear(self.hidden_size, self.lr_dim)
        self.qk_scale = nn.Parameter(torch.ones(self.token_mixer_dim, 2))
        self.qk_offset = nn.Parameter(torch.zeros(self.token_mixer_dim, 2))
        self.learnable_ttt_scale = config.learnable_ttt_scale
        if self.learnable_ttt_scale:
            self.ttt_scale_proj = nn.Linear(self.hidden_size, self.num_fw_heads)
        self.ttt_norm = RMSNorm(self.fw_head_dim, eps=config.rms_norm_eps)
        if self.use_momentum:
            self.momentum_proj = nn.Sequential(nn.Linear(self.hidden_size, self.num_fw_heads), nn.Sigmoid())
        self.rotary = RotaryEmbedding(self.fw_head_dim, base=config.rope_theta)
        self.register_buffer("_metric_ttt_scale_sum", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("_metric_ttt_scale_abs_sum", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("_metric_ttt_scale_max", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("_metric_ttt_count", torch.zeros((), dtype=torch.float32), persistent=False)

    def _rescale_qk(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        qk_scale = self.qk_scale.view(1, 1, -1, 2)
        qk_offset = self.qk_offset.view(1, 1, -1, 2)
        q = q * qk_scale[:, :, :, 0] + qk_offset[:, :, :, 0]
        k = k * qk_scale[:, :, :, 1] + qk_offset[:, :, :, 1]
        return q, k

    def prepare_fast_qkv(
        self,
        *,
        fast_q: torch.Tensor,
        fast_k: torch.Tensor,
        fast_v: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, num_query_heads, query_head_dim = fast_q.shape
        if num_query_heads != self.num_attention_heads or query_head_dim != self.head_dim:
            raise ValueError(
                "Expected fast_q with shape [B, S, num_attention_heads, head_dim]. "
                f"Got {tuple(fast_q.shape)}."
            )
        if fast_k.shape[:2] != (batch_size, seq_len) or fast_v.shape[:2] != (batch_size, seq_len):
            raise ValueError(
                "fast_k and fast_v must match fast_q on batch/seq dimensions. "
                f"Got fast_k={tuple(fast_k.shape)}, fast_v={tuple(fast_v.shape)}, fast_q={tuple(fast_q.shape)}."
            )
        if fast_k.shape[2] != self.num_key_value_heads or fast_k.shape[3] != self.head_dim:
            raise ValueError(
                "Expected fast_k with shape [B, S, num_key_value_heads, head_dim]. "
                f"Got {tuple(fast_k.shape)}."
            )
        if fast_v.shape[2] != self.num_key_value_heads or fast_v.shape[3] != self.head_dim:
            raise ValueError(
                "Expected fast_v with shape [B, S, num_key_value_heads, head_dim]. "
                f"Got {tuple(fast_v.shape)}."
            )

        if self.num_key_value_groups != 1:
            fast_k = repeat_kv(fast_k.transpose(1, 2), self.num_key_value_groups).transpose(1, 2)
            fast_v = repeat_kv(fast_v.transpose(1, 2), self.num_key_value_groups).transpose(1, 2)

        fast_q = fast_q.reshape(batch_size, seq_len, -1)
        fast_k = fast_k.reshape(batch_size, seq_len, -1)
        fast_v = fast_v.reshape(batch_size, seq_len, -1)
        if (
            fast_q.shape[-1] != self.token_mixer_dim
            or fast_k.shape[-1] != self.token_mixer_dim
            or fast_v.shape[-1] != self.token_mixer_dim
        ):
            raise ValueError(
                "Expected flattened fast_q/fast_k/fast_v to have token_mixer_dim as last dim. "
                f"Got fast_q={fast_q.shape[-1]}, fast_k={fast_k.shape[-1]}, fast_v={fast_v.shape[-1]}, token_mixer_dim={self.token_mixer_dim}."
            )

        fast_q, fast_k = self._rescale_qk(fast_q, fast_k)

        fast_q = fast_q.view(batch_size, seq_len, self.num_fw_heads, self.fw_head_dim)
        fast_k = fast_k.view(batch_size, seq_len, self.num_fw_heads, self.fw_head_dim)
        fast_v = fast_v.view(batch_size, seq_len, self.num_fw_heads, self.fw_head_dim)

        if self.attn_qk_norm:
            fast_q = self.q_norm(fast_q)
            fast_k = self.k_norm(fast_k)

        if self.qkv_silu:
            fast_q = F.silu(fast_q)
            fast_k = F.silu(fast_k)
            if not self.no_v_silu:
                fast_v = F.silu(fast_v)

        fast_q = fast_q.permute(0, 2, 1, 3).reshape(batch_size * self.num_fw_heads, seq_len, self.fw_head_dim)
        fast_k = fast_k.permute(0, 2, 1, 3).reshape(batch_size * self.num_fw_heads, seq_len, self.fw_head_dim)
        fast_v = fast_v.permute(0, 2, 1, 3).reshape(batch_size * self.num_fw_heads, seq_len, self.fw_head_dim)

        fast_q = l2_norm(fast_q)
        fast_k = l2_norm(fast_k)

        if not self.ttt_nope:
            fast_q_rope = fast_q.view(batch_size, self.num_fw_heads, seq_len, self.fw_head_dim).transpose(1, 2)
            fast_k_rope = fast_k.view(batch_size, self.num_fw_heads, seq_len, self.fw_head_dim).transpose(1, 2)
            cos, sin = self.rotary(position_ids, fast_q.dtype, fast_q.device)
            fast_q_rope, fast_k_rope = apply_rotary_pos_emb(fast_q_rope, fast_k_rope, cos, sin)
            fast_q = fast_q_rope.transpose(1, 2).reshape(batch_size * self.num_fw_heads, seq_len, self.fw_head_dim)
            fast_k = fast_k_rope.transpose(1, 2).reshape(batch_size * self.num_fw_heads, seq_len, self.fw_head_dim)

        return fast_q, fast_k, fast_v

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        fast_q: torch.Tensor,
        fast_k: torch.Tensor,
        fast_v: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size, seq_len, _ = hidden_states.shape
        fast_q, fast_k, fast_v = self.prepare_fast_qkv(
            fast_q=fast_q,
            fast_k=fast_k,
            fast_v=fast_v,
            position_ids=position_ids,
        )

        if self.w0_w2_low_rank > 0:
            fw_w0 = self.w0().repeat(batch_size, 1, 1)
            fw_w2 = self.w2().repeat(batch_size, 1, 1)
        else:
            fw_w0 = self.w0.repeat(batch_size, 1, 1)
            fw_w2 = self.w2.repeat(batch_size, 1, 1)
        fw_w1 = self.w1.repeat(batch_size, 1, 1)

        lr = F.softplus(self.lr_proj(hidden_states).float() + self.base_lr_inv)
        fw_lr = lr.view(batch_size, seq_len, self.num_fw_heads, -1).permute(0, 2, 1, 3).reshape(batch_size * self.num_fw_heads, seq_len, -1)
        fw_lr1, fw_lr2, fw_lr3 = fw_lr.chunk(3, dim=-1)

        if self.use_momentum:
            momentum = self.momentum_proj(hidden_states).float()
            momentum = momentum.view(batch_size, seq_len, self.num_fw_heads, 1).permute(0, 2, 1, 3).reshape(batch_size * self.num_fw_heads, seq_len, 1)
        else:
            momentum = None

        if self.fp32_states:
            fw_w0 = fw_w0.float()
            fw_w1 = fw_w1.float()
            fw_w2 = fw_w2.float()

        use_fused = (
            self.use_fused_kernel
            and hidden_states.is_cuda
            and hidden_states.dtype == torch.bfloat16
            and self.memory_update_phase in (None, self.lact_chunk_size - 1)
        )
        if self.ttt_prenorm:
            if use_fused:
                fw_x = prenorm_block_causal_lact_swiglu_fused_kernel_triton(
                    fw_w0, fw_w1, fw_w2, fast_q, fast_k, fast_v, fw_lr1, fw_lr2, fw_lr3,
                    chunk_size=self.lact_chunk_size, use_muon=self.use_muon, momentum=momentum
                )
            else:
                fw_x = prenorm_block_causal_lact_swiglu(
                    fw_w0, fw_w1, fw_w2, fast_q, fast_k, fast_v, fw_lr1, fw_lr2, fw_lr3,
                    chunk_size=self.lact_chunk_size, update_phase=self.memory_update_phase,
                    use_muon=self.use_muon, momentum=momentum
                )
        else:
            if use_fused:
                fw_x = postnorm_block_causal_lact_swiglu_fused_kernel_triton(
                    fw_w0, fw_w1, fw_w2, fast_q, fast_k, fast_v, fw_lr1, fw_lr2, fw_lr3,
                    chunk_size=self.lact_chunk_size, use_muon=self.use_muon, momentum=momentum
                )
            else:
                fw_x = block_causal_lact_swiglu(
                    fw_w0, fw_w1, fw_w2, fast_q, fast_k, fast_v, fw_lr1, fw_lr2, fw_lr3,
                    chunk_size=self.lact_chunk_size, update_phase=self.memory_update_phase,
                    use_muon=self.use_muon, momentum=momentum
                )

        ttt_x = self.ttt_norm(fw_x)
        scale_stats = {
            "scale_mean": torch.zeros((), dtype=torch.float32, device=hidden_states.device),
            "scale_abs_mean": torch.zeros((), dtype=torch.float32, device=hidden_states.device),
            "scale_max": torch.zeros((), dtype=torch.float32, device=hidden_states.device),
        }
        if self.learnable_ttt_scale:
            ttt_scale = F.silu(self.ttt_scale_proj(hidden_states), inplace=False)
            scale_stats = {
                "scale_mean": ttt_scale.mean().float(),
                "scale_abs_mean": ttt_scale.abs().mean().float(),
                "scale_max": ttt_scale.max().float(),
            }
            ttt_scale = ttt_scale.view(batch_size, seq_len, self.num_fw_heads, 1).permute(0, 2, 1, 3).reshape(batch_size * self.num_fw_heads, seq_len, 1)
            ttt_x = ttt_x * ttt_scale

        ttt_x = ttt_x.view(batch_size, self.num_fw_heads, seq_len, self.fw_head_dim).transpose(1, 2).reshape(batch_size, seq_len, self.token_mixer_dim)
        return ttt_x, {
            "scale_mean": scale_stats["scale_mean"],
            "scale_abs_mean": scale_stats["scale_abs_mean"],
            "scale_max": scale_stats["scale_max"],
        }

    def update_runtime_metrics(self, stats: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            self._metric_ttt_scale_sum.add_(stats["scale_mean"].detach().to(torch.float32))
            self._metric_ttt_scale_abs_sum.add_(stats["scale_abs_mean"].detach().to(torch.float32))
            self._metric_ttt_scale_max.copy_(torch.maximum(self._metric_ttt_scale_max, stats["scale_max"].detach().to(torch.float32)))
            self._metric_ttt_count.add_(torch.ones((), dtype=torch.float32, device=self._metric_ttt_count.device))

    def consume_runtime_metrics(self) -> dict[str, float]:
        count = float(self._metric_ttt_count.item())
        if count == 0:
            return {}
        metrics = {
            "ttt_scale_mean": float((self._metric_ttt_scale_sum / self._metric_ttt_count).item()),
            "ttt_scale_abs_mean": float((self._metric_ttt_scale_abs_sum / self._metric_ttt_count).item()),
            "ttt_scale_max": float(self._metric_ttt_scale_max.item()),
        }
        self._metric_ttt_scale_sum.zero_()
        self._metric_ttt_scale_abs_sum.zero_()
        self._metric_ttt_scale_max.zero_()
        self._metric_ttt_count.zero_()
        return metrics


class HybridQwen3Attention(nn.Module):
    def __init__(self, config: HybridQwen3LaCTConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.hybrid_layer_types[layer_idx]
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.sliding_window = config.sliding_window if self.layer_type == "lact" and config.use_sliding_window else None

        self.q_proj = nn.Linear(config.hidden_size, self.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary = RotaryEmbedding(self.head_dim, base=config.rope_theta)
        self.lact_branch = HybridQwen3LaCTBranch(config, layer_idx) if self.layer_type == "lact" else None

    def _attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if (
            flash_attn_func is not None
            and attention_mask is None
            and query_states.is_cuda
            and key_states.is_cuda
            and value_states.is_cuda
        ):
            return flash_attn_func(
                query_states,
                key_states,
                value_states,
                causal=True,
                window_size=(-1, -1) if self.sliding_window is None else (self.sliding_window - 1, 0),
            )

        q = query_states.transpose(1, 2)
        k = key_states.transpose(1, 2)
        v = value_states.transpose(1, 2)
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)
        scores = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        mask = build_attention_mask(q.shape[-2], q.device, self.sliding_window, attention_mask)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        probs = F.dropout(probs, p=0.0 if not self.training else self.attention_dropout, training=self.training)
        return torch.matmul(probs, v).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        q_raw = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        k_raw = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        v_raw = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        q_attn = self.q_norm(q_raw)
        k_attn = self.k_norm(k_raw)
        cos, sin = self.rotary(position_ids, q_attn.dtype, q_attn.device)
        q_attn, k_attn = apply_rotary_pos_emb(q_attn, k_attn, cos, sin)

        attn_output = self._attention(q_attn, k_attn, v_raw, attention_mask)
        attn_output = attn_output.reshape(batch_size, seq_len, -1).contiguous()
        if self.lact_branch is not None:
            branch_output, branch_stats = self.lact_branch(
                hidden_states,
                position_ids,
                fast_q=q_raw,
                fast_k=k_raw,
                fast_v=v_raw,
            )
            self.lact_branch.update_runtime_metrics(branch_stats)
            attn_output = attn_output + branch_output
        
        attn_output = self.o_proj(attn_output)
        return attn_output


class HybridQwen3LaCTDecoderLayer(nn.Module):
    def __init__(self, config: HybridQwen3LaCTConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_type = config.hybrid_layer_types[layer_idx]
        self.self_attn = HybridQwen3Attention(config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_ids=position_ids, attention_mask=attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class HybridQwen3LaCTPreTrainedModel(PreTrainedModel):
    config_class = HybridQwen3LaCTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["HybridQwen3LaCTDecoderLayer"]

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

        if isinstance(module, HybridQwen3LaCTBranch):
            nn.init.ones_(module.qk_scale)
            nn.init.zeros_(module.qk_offset)
            if module.w0_w2_low_rank > 0:
                module.w0._init_weights()
                module.w2._init_weights()
            else:
                nn.init.normal_(module.w0, mean=0.0, std=1.0 / math.sqrt(module.fw_head_dim))
                nn.init.normal_(module.w2, mean=0.0, std=1.0 / math.sqrt(module.fw_head_dim))
            nn.init.normal_(module.w1, mean=0.0, std=1.0 / math.sqrt(module.d_h))


class HybridQwen3LaCTModel(HybridQwen3LaCTPreTrainedModel):
    def __init__(self, config: HybridQwen3LaCTConfig, **kwargs) -> None:
        super().__init__(config)
        del kwargs
        self.padding_idx = config.pad_token_id
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [HybridQwen3LaCTDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        del kwargs
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_len, _ = inputs_embeds.shape
        position_ids = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0).expand(batch_size, -1)

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            hidden_states = layer(hidden_states, attention_mask=attention_mask, position_ids=position_ids)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if not return_dict:
            return (hidden_states, None, all_hidden_states, None)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, hidden_states=all_hidden_states, past_key_values=None, attentions=None)


class HybridQwen3LaCTForCausalLM(HybridQwen3LaCTPreTrainedModel, GenerationMixin):

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"} # ["lm_head.weight"]
    def __init__(self, config: HybridQwen3LaCTConfig, **kwargs):
        super().__init__(config)
        del kwargs
        self.model = HybridQwen3LaCTModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def consume_ttt_runtime_metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        scale_means = []
        scale_abs_means = []
        scale_maxes = []
        for idx, layer in enumerate(self.model.layers):
            branch = getattr(layer.self_attn, "lact_branch", None)
            if branch is None:
                continue
            layer_metrics = branch.consume_runtime_metrics()
            if not layer_metrics:
                continue
            metrics[f"ttt/layer_{idx:02d}/scale_mean"] = layer_metrics["ttt_scale_mean"]
            metrics[f"ttt/layer_{idx:02d}/scale_abs_mean"] = layer_metrics["ttt_scale_abs_mean"]
            metrics[f"ttt/layer_{idx:02d}/scale_max"] = layer_metrics["ttt_scale_max"]
            scale_means.append(layer_metrics["ttt_scale_mean"])
            scale_abs_means.append(layer_metrics["ttt_scale_abs_mean"])
            scale_maxes.append(layer_metrics["ttt_scale_max"])
        if scale_means:
            metrics["ttt/global/scale_mean"] = sum(scale_means) / len(scale_means)
            metrics["ttt/global/scale_abs_mean"] = sum(scale_abs_means) / len(scale_abs_means)
            metrics["ttt/global/scale_max"] = max(scale_maxes)
        return metrics

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: int = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states if output_hidden_states is not None else False,
            return_dict=True if return_dict is None else return_dict,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        fuse_linear_and_cross_entropy = self.config.fuse_cross_entropy and self.training
        logits = None if fuse_linear_and_cross_entropy else self.lm_head(hidden_states[:, -logits_to_keep:])

        loss = None
        if labels is not None:
            if self.criterion is None:
                if fuse_linear_and_cross_entropy:
                    criterion = FusedLinearCrossEntropyLoss()
                elif self.config.fuse_cross_entropy:
                    criterion = FusedCrossEntropyLoss(inplace_backward=True)
                else:
                    criterion = nn.CrossEntropyLoss()
            else:
                criterion = self.criterion
            labels = labels.to(hidden_states.device)
            labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], criterion.ignore_index)), 1)
            if fuse_linear_and_cross_entropy:
                loss = criterion(hidden_states, labels, self.lm_head.weight, self.lm_head.bias)
            else:
                loss = criterion(logits.view(labels.numel(), -1), labels.view(-1))

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=outputs.hidden_states,
            attentions=None,
        )
