# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn import RMSNorm
from transformers.utils import logging
from .configuration_ttt_e2e import E2ETTTConfig

logger = logging.get_logger(__name__)

# class RMSNorm(nn.Module):
#     def __init__(self, dim: int, eps: float = 1e-6):
#         super().__init__()
#         self.eps = eps
#         self.weight = nn.Parameter(torch.ones(dim))

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     var = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    #     x = x * torch.rsqrt(var + self.eps)
    #     return x.to(self.weight.dtype) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if position_ids is None:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        else:
            pos = position_ids.to(device=device, dtype=self.inv_freq.dtype)
            if pos.dim() == 1:
                freqs = torch.einsum("i,j->ij", pos, self.inv_freq)
            elif pos.dim() == 2:
                freqs = torch.einsum("bi,j->bij", pos, self.inv_freq)
            else:
                raise ValueError(f"position_ids must be rank-1 or rank-2, got shape {tuple(pos.shape)}")
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.to(device=device, dtype=dtype)


def apply_rotary(q: torch.Tensor, k: torch.Tensor, freqs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if freqs.dim() == 2:
        cos = freqs.cos()[None, None, :, :]
        sin = freqs.sin()[None, None, :, :]
    elif freqs.dim() == 3:
        cos = freqs.cos()[:, None, :, :]
        sin = freqs.sin()[:, None, :, :]
    else:
        raise ValueError(f"freqs must be rank-2 or rank-3, got shape {tuple(freqs.shape)}")

    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k



class E2ESWIGLULayer(nn.Module):
    """Single public attention forward used by E2E-TTT blocks."""

    def __init__(self, config: E2ETTTConfig, layer_idx: int, is_suffix: bool):
        super().__init__()
        self.config = config
        self.layer_idx = int(layer_idx)
        self.is_suffix = bool(is_suffix)
        self._debug_suffix_logged = False

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.window_size = int(getattr(config, "window_size", 2048))
        self.default_backend = str(getattr(config, "attn_backend", "flash"))
        self.attn_pdrop = float(getattr(config, "attn_pdrop", 0.0))
        self.qk_norm = bool(getattr(config, "qk_norm", True))
        self.pre_norm = bool(getattr(config, "pre_norm", True))
        self.post_norm = bool(getattr(config, "post_norm", True))

        self.qkv = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.resid_dropout = nn.Dropout(float(getattr(config, "resid_pdrop", 0.0)))
        self.rotary = RotaryEmbedding(self.head_dim, base=float(config.rope_theta))

    def _sdpa_local(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q_len = q.size(-2)
        kv_len = k.size(-2)

        q_idx = torch.arange(q_len, device=q.device)[:, None] + (kv_len - q_len)
        k_idx = torch.arange(kv_len, device=k.device)[None, :]

        causal = q_idx >= k_idx
        if self.window_size > 0:
            local = (q_idx - k_idx) <= self.window_size
            causal_local_mask = causal & local
        else:
            causal_local_mask = causal

        if attention_mask is not None:
            if attention_mask.dim() != 2:
                raise ValueError(
                    "attention_mask must have shape [batch_size, seq_len], "
                    f"got {tuple(attention_mask.shape)}"
                )
            kv_mask = attention_mask[:, -kv_len:].to(device=q.device, dtype=torch.bool)
            q_mask = attention_mask[:, -q_len:].to(device=q.device, dtype=torch.bool)
            mask = (
                causal_local_mask[None, None, :, :]
                & kv_mask[:, None, None, :]
                & q_mask[:, None, :, None]
            )
        else:
            mask = causal_local_mask[None, None, :, :]

        # 强制使用 Math 后端，以支持 create_graph=True (二阶导数)
        # 兼容 PyTorch 2.x
        with sdpa_kernel(SDPBackend.MATH):
            return F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=(self.attn_pdrop if self.training else 0.0),
            )

    def _flash_local(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attention_mask is not None:
            return self._sdpa_local(q, k, v, attention_mask=attention_mask)
        try:
            from flash_attn import flash_attn_func  # type: ignore

            q_ = q.transpose(1, 2)
            k_ = k.transpose(1, 2)
            v_ = v.transpose(1, 2)

            window = (self.window_size, 0) if self.window_size > 0 else (-1, -1)
            out = flash_attn_func(
                q_,
                k_,
                v_,
                dropout_p=(self.attn_pdrop if self.training else 0.0),
                causal=True,
                window_size=window,
            )
            return out.transpose(1, 2)
        except Exception:
            return self._sdpa_local(q, k, v, attention_mask=attention_mask)

    def _forward_with_backend(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]],
        attention_mask: Optional[torch.Tensor],
        backend: str,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len, _ = hidden_states.shape
        qkv = self.qkv(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        freqs = self.rotary(seq_len, device=hidden_states.device, dtype=hidden_states.dtype, position_ids=position_ids)
        q, k = apply_rotary(q, k, freqs)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        if self.window_size > 0 and k.size(-2) > self.window_size:
            k = k[:, :, -self.window_size :, :]
            v = v[:, :, -self.window_size :, :]

        next_kv = (k, v)

        if backend == "flash":
            out = self._flash_local(q, k, v, attention_mask=attention_mask)
        else:
            out = self._sdpa_local(q, k, v, attention_mask=attention_mask)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.resid_dropout(self.o_proj(out)), next_kv

    def _forward_prefix_flash(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]],
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self._forward_with_backend(
            hidden_states, position_ids, past_key_value, attention_mask=attention_mask, backend="flash"
        )

    def _forward_suffix_sdpa(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]],
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self._forward_with_backend(
            hidden_states, position_ids, past_key_value, attention_mask=attention_mask, backend="sdpa"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_e2e_ttt_context: bool = False,
        attn_backend_override: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if use_e2e_ttt_context:
            if self.is_suffix:
                if bool(getattr(self.config, "debug_ttt_logs", False)) and not self._debug_suffix_logged:
                    logger.warning(
                        "[TTT-DEBUG] suffix-attn active: layer=%d backend=sdpa use_e2e_ttt_context=%s",
                        self.layer_idx,
                        True,
                    )
                    self._debug_suffix_logged = True
                return self._forward_suffix_sdpa(hidden_states, position_ids, past_key_value, attention_mask)
            return self._forward_prefix_flash(hidden_states, position_ids, past_key_value, attention_mask)

        backend = str(attn_backend_override if attn_backend_override is not None else self.default_backend)
        if backend not in {"flash", "sdpa"}:
            backend = "flash"
        return self._forward_with_backend(
            hidden_states, position_ids, past_key_value, attention_mask=attention_mask, backend=backend
        )
