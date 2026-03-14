# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers.utils import logging

from torch.nn.attention import SDPBackend, sdpa_kernel

from fla.modules import RMSNorm, RotaryEmbedding
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange, repeat

from .configuration_ttt_e2e import E2ETTTConfig

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
except ImportError:
    warnings.warn(
        "Flash Attention is not installed. Please install it via `pip install flash-attn --no-build-isolation`",
        category=ImportWarning,
    )
    flash_attn_func = None

logger = logging.get_logger(__name__)

LayerKV = Tuple[torch.Tensor, torch.Tensor]
LegacyCache = Tuple[Optional[LayerKV], ...]


def inv_softplus(x):
    if isinstance(x, torch.Tensor):
        y = x + torch.log(-torch.expm1(-x))
    else:
        y = x + math.log(-math.expm1(-x))
    return y




class E2ESWIGLULayer(nn.Module):

    def __init__(self, config: E2ETTTConfig, layer_idx: int, is_suffix: bool):
        super().__init__()
        self.config = config
        self.layer_idx = int(layer_idx)
        self.is_suffix = bool(is_suffix)
        self._debug_suffix_logged = False
        self._debug_past_kv_logged = False
        

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.window_size = int(getattr(config, "window_size", 2048))
        self.default_backend = str(getattr(config, "attn_backend", "flash"))

        self.qkv = nn.Linear(self.hidden_size, self.hidden_size * 3, bias=False)
        self.attn_qk_norm = bool(getattr(config, "qk_norm", True))
        if self.attn_qk_norm:
            self.q_norm = RMSNorm(self.hidden_size)
            self.k_norm = RMSNorm(self.hidden_size)

        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.rope_theta = config.rope_theta
        self.rotary = RotaryEmbedding(dim=self.head_dim, base=self.rope_theta)
        self.max_position_embeddings = getattr(config, "max_position_embeddings", None)
        # Compatibility attrs used by shared init path in modeling.
        # self.register_buffer("qk_scale", torch.ones(self.hidden_size, 2), persistent=False)
        # self.register_buffer("qk_offset", torch.zeros(self.hidden_size, 2), persistent=False)
        

        

    # def _rescale_qk(self, q, k):
    #     """
    #     Args:
    #         q: [b, s, d]
    #         k: [b, s, d]
    #     Returns:
    #         q: [b, s, d]
    #         k: [b, s, d]
    #     """
    #     qk_scale = self.qk_scale.view(1, 1, -1, 2)
    #     qk_offset = self.qk_offset.view(1, 1, -1, 2)
    #     q = q * qk_scale[:, :, :, 0] + qk_offset[:, :, :, 0]
    #     k = k * qk_scale[:, :, :, 1] + qk_offset[:, :, :, 1]
    #     return q, k

    
    # ------------------------------------------------------------
    # 1) Mask construction: semantics aligned with LACT (window_size -> (window_size-1, 0))
    # ------------------------------------------------------------
    def _build_causal_local_mask_2d(
        self,
        q_len: int,
        kv_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        return: bool mask [q_len, kv_len], True = allowed
        Aligned with LACT: q is aligned to the tail of kv (kv_len - q_len offset),
        and local uses <= (window_size - 1).
        """
        q_idx = torch.arange(q_len, device=device)[:, None] + (kv_len - q_len)
        k_idx = torch.arange(kv_len, device=device)[None, :]

        causal = q_idx >= k_idx

        # LACT: window_size is None => full causal
        if self.window_size is None:
            return causal

        # window_size <= 0 is also equivalent to full causal (you may assert this never happens).
        if self.window_size <= 0:
            return causal

        # LACT flash: (window_size - 1, 0) => allows looking back by window_size-1.
        left = self.window_size - 1
        local = (q_idx - k_idx) <= left
        return causal & local

    # ------------------------------------------------------------
    # 2) SDPA path: for suffix. Mask/padding aligned with LACT, and force MATH backend (2nd-order grads).
    # ------------------------------------------------------------
    def _sdpa_window(
        self,
        q: torch.Tensor,  # [b, s_q, h, d]
        k: torch.Tensor,  # [b, s_kv, h, d]
        v: torch.Tensor,  # [b, s_kv, h, d]
        attention_mask: Optional[torch.Tensor] = None,  # [b, seq_len_total] or [b, s_kv]
    ) -> torch.Tensor:
        b, s_q, h, d = q.shape
        s_kv = k.shape[1]

        # SDPA expects [b, h, s, d]
        q_t = q.permute(0, 2, 1, 3).contiguous()
        k_t = k.permute(0, 2, 1, 3).contiguous()
        v_t = v.permute(0, 2, 1, 3).contiguous()

        base_2d = self._build_causal_local_mask_2d(s_q, s_kv, device=q.device)  # [s_q, s_kv]
        mask = base_2d[None, None, :, :]  # [1,1,s_q,s_kv] broadcast to [b,h,s_q,s_kv]

        if attention_mask is not None:
            if attention_mask.dim() != 2:
                raise ValueError(
                    "attention_mask must have shape [batch_size, seq_len], "
                    f"got {tuple(attention_mask.shape)}"
                )
            # Match your original behavior: take only the tail aligned to the current q/k context.
            kv_mask = attention_mask[:, -s_kv:].to(device=q.device, dtype=torch.bool)  # [b, s_kv]
            q_mask = attention_mask[:, -s_q:].to(device=q.device, dtype=torch.bool)    # [b, s_q]
            mask = (
                mask
                & kv_mask[:, None, None, :]   # [b,1,1,s_kv]
                & q_mask[:, None, :, None]    # [b,1,s_q,1]
            )

        # Force Math backend: supports create_graph=True / second-order gradients.
        with sdpa_kernel(SDPBackend.MATH):
            out = F.scaled_dot_product_attention(
                q_t, k_t, v_t,
                attn_mask=mask,  # bool mask: True = allowed (PyTorch SDPA semantics)
            )  # [b,h,s_q,d]

        return out.permute(0, 2, 1, 3).contiguous()  # [b,s_q,h,d]

    # ------------------------------------------------------------
    # 3) Flash path: for prefix. Branch structure fully aligned with LACT.
    # ------------------------------------------------------------
    def _flash_window(
        self,
        q: torch.Tensor,  # [b, s_q, h, d]
        k: torch.Tensor,  # [b, s_kv, h, d]
        v: torch.Tensor,  # [b, s_kv, h, d]
        attention_mask: Optional[torch.Tensor],
        cu_seqlens: Optional[torch.Tensor],
        batch_size: int,
        q_len: int,
        max_seqlen: int,
    ) -> torch.Tensor:
        if flash_attn_func is None:
            raise ImportError(
                "Please install Flash Attention via `pip install flash-attn --no-build-isolation` first"
            )

        # LACT window tuple
        window = (-1, -1) if self.window_size is None else (self.window_size - 1, 0)
        causal_flag = True

        if attention_mask is not None:
            # Aligned with LACT: padding -> varlen.
            q_u, k_u, v_u, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                q, k, v, attention_mask, q_len
            )
            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_q, max_seqlen_k = max_seq_lens
            o = flash_attn_varlen_func(
                q_u,
                k_u,
                v_u,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                causal=causal_flag,
                window_size=window,
            )
            o = pad_input(o, indices_q, batch_size, q_len)  # -> [b, q_len, h, d]
            return o

        if cu_seqlens is not None:
            # Aligned with LACT: packed sequences.
            o = flash_attn_varlen_func(
                q.squeeze(0),
                k.squeeze(0),
                v.squeeze(0),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=causal_flag,
                window_size=window,
            ).unsqueeze(0)
            return o

        # Regular flash path.
        return flash_attn_func(
            q, k, v,
            causal=causal_flag,
            window_size=window,
        )

    # ------------------------------------------------------------
    # 4) LACT-style forward: prefix flash / suffix sdpa.
    # ------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,  # [b, s, d]
        attention_mask: Optional[torch.LongTensor] = None,  # [b, seq]
        past_key_values: Optional[LegacyCache] = None,      # per-layer tuple cache
        output_attentions: bool = False,
        use_cache: bool = False,
        attn_backend_override: Optional[str] = None,
        ttt_step_idx: Optional[int] = None,
        ttt_num_steps: Optional[int] = None,
        ttt_chunk_size: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[LegacyCache]]:
        if output_attentions:
            # If you need attention weights, add a dedicated branch (flash path does not expose them directly).
            raise NotImplementedError("output_attentions=True is not supported in this backend-aligned impl.")

        if attention_mask is not None:
            assert attention_mask.dim() == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.size()

        # ---- QKV
        q, k, v = self.qkv(hidden_states).chunk(3, dim=-1)

        # qk norm (naming aligned with LACT).
        if self.attn_qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        # [b, s, h, d] layout aligned with LACT flash input format.
        q = q.view(batch_size, q_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, q_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, q_len, self.num_heads, self.head_dim)

        cu_seqlens = kwargs.get("cu_seqlens", None)
        _ = (attn_backend_override, ttt_step_idx, ttt_num_steps, ttt_chunk_size)
        layer_kv = self._get_layer_cache(past_key_values, self.layer_idx)
        layer_k_past = None if layer_kv is None else layer_kv[0]
        layer_v_past = None if layer_kv is None else layer_kv[1]

        # ---- Rotary offset logic, aligned with LACT.
        seqlen_offset, max_seqlen = 0, q_len
        if layer_k_past is not None:
            seqlen_offset = int(layer_k_past.shape[1])
            max_seqlen = q_len + seqlen_offset

            if attention_mask is not None:
                # Aligned with LACT: adjust padding offset (typically left padding).
                seqlen_offset = seqlen_offset + attention_mask.sum(-1) - attention_mask.shape[-1]
                max_seqlen = q_len + int(torch.max(seqlen_offset).item())

        if self.max_position_embeddings is not None:
            max_seqlen = max(max_seqlen, self.max_position_embeddings)

        # Rotary: aligned with LACT interface.
        q, k = self.rotary(
            q, k,
            seqlen_offset=seqlen_offset,
            max_seqlen=max_seqlen,
            cu_seqlens=cu_seqlens,
        )
        # if (
        #     layer_k_past is not None
        #     and not self._debug_past_kv_logged
        #     and logger.isEnabledFor(20)
        # ):
        #     cache_seq_len = int(layer_k_past.shape[1])
        #     cache_len = 0 if past_key_values is None else len(past_key_values)
        #     key_shape = tuple(layer_k_past.shape)
        #     value_shape = None if layer_v_past is None else tuple(layer_v_past.shape)

        #     logger.info(
        #         "past_key_values detected: type=%s, len=%s, layer_idx=%s, seq_len=%s, key_shape=%s, value_shape=%s",
        #         type(past_key_values).__name__,
        #         cache_len,
        #         self.layer_idx,
        #         cache_seq_len,
        #         key_shape,
        #         value_shape,
        #     )
        #     self._debug_past_kv_logged = True
        # ---- Cache update: tuple cache (overwrite k/v for attention only when cache_has_content is true).
        if past_key_values is not None:
            k_cur = k.flatten(-2, -1)
            v_cur = v.flatten(-2, -1)
            cache_has_content = layer_k_past is not None and int(layer_k_past.shape[1]) > 0

            if layer_k_past is not None:
                k_cached = torch.cat((layer_k_past, k_cur), dim=1)
                v_cached = torch.cat((layer_v_past, v_cur), dim=1)
            else:
                k_cached = k_cur
                v_cached = v_cur

            if self.window_size is not None and self.window_size > 0:
                k_effective = k_cached[:, -self.window_size :, :].contiguous()
                v_effective = v_cached[:, -self.window_size :, :].contiguous()
            else:
                k_effective = k_cached
                v_effective = v_cached

            past_key_values = self._set_layer_cache(
                past_key_values=past_key_values,
                layer_idx=self.layer_idx,
                k=k_effective,
                v=v_effective,
            )

            if cache_has_content:
                # Generation step: use k/v composed from cache.
                k, v = k_effective, v_effective
                k = k.view(batch_size, -1, self.num_heads, self.head_dim)
                v = v.view(batch_size, -1, self.num_heads, self.head_dim)
                # Note: do not overwrite during prefill (empty cache), to avoid truncating current-sequence attention.

        # ---- backend selection：suffix = sdpa, prefix = flash
        # If suffix layers are marked by self.is_suffix, use it directly.
        is_suffix = bool(getattr(self, "is_suffix", False))
        if attn_backend_override is None:
            backend = "sdpa" if is_suffix else "flash"
        else:
            backend = str(attn_backend_override)
        if backend not in {"flash", "sdpa"}:
            backend = "flash"

        if backend == "sdpa":
            # If cu_seqlens is passed in suffix, extra handling is required here (otherwise assert).
            if cu_seqlens is not None:
                raise NotImplementedError("SDPA suffix path does not support cu_seqlens-packed input yet.")
            o = self._sdpa_window(q, k, v, attention_mask=attention_mask)  # [b, q_len, h, d]
        else:
            o = self._flash_window(
                q, k, v,
                attention_mask=attention_mask,
                cu_seqlens=cu_seqlens,
                batch_size=batch_size,
                q_len=q_len,
                max_seqlen=max_seqlen,
            )  # [b, q_len, h, d]

        # ---- proj
        o = o.reshape(batch_size, q_len, -1)
        out = self.o_proj(o)

        # ---- Return (aligned with HF conventions).
        return out, None, (past_key_values if use_cache else None)

    def _upad_input(self, q, k, v, attention_mask, q_len):
        batch_size, seq_len, num_key_value_heads, head_dim = k.shape
        cache_mask = attention_mask[:, -seq_len:]
        seqlens = cache_mask.sum(-1, dtype=torch.int32)
        indices_k = torch.nonzero(cache_mask.flatten(), as_tuple=False).flatten()
        max_seqlen_k = seqlens.max().item()
        cu_seqlens_k = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))

        k = index_first_axis(
            k.reshape(batch_size * seq_len, num_key_value_heads, head_dim), indices_k
        )
        v = index_first_axis(
            v.reshape(batch_size * seq_len, num_key_value_heads, head_dim), indices_k
        )
        if q_len == seq_len:
            q = index_first_axis(
                q.reshape(batch_size * seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_q = max_seqlen_k
            indices_q = indices_k
        elif q_len == 1:
            max_seqlen_q = 1
            # There is a memcpy here, that is very bad.
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=q.device
            )
            indices_q = cu_seqlens_q[:-1]
            q = q.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
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

    @staticmethod
    def _get_layer_cache(
        past_key_values: Optional[LegacyCache],
        layer_idx: int,
    ) -> Optional[LayerKV]:
        if past_key_values is None or layer_idx >= len(past_key_values):
            return None
        kv = past_key_values[layer_idx]
        if kv is None:
            return None
        if not isinstance(kv, (tuple, list)) or len(kv) != 2:
            raise ValueError(f"Invalid cache entry at layer {layer_idx}: expected (k, v), got {type(kv)}")
        k, v = kv
        if not (torch.is_tensor(k) and torch.is_tensor(v)):
            raise ValueError(f"Invalid cache tensors at layer {layer_idx}: got ({type(k)}, {type(v)})")
        return k, v

    @staticmethod
    def _set_layer_cache(
        past_key_values: Optional[LegacyCache],
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> LegacyCache:
        cache_list: List[Optional[LayerKV]] = list(past_key_values) if past_key_values is not None else []
        if len(cache_list) <= layer_idx:
            cache_list.extend([None] * (layer_idx + 1 - len(cache_list)))
        cache_list[layer_idx] = (k, v)
        return tuple(cache_list)
