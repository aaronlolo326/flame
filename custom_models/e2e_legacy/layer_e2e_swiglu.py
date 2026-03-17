# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional, Tuple
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from fla.modules import RMSNorm, RotaryEmbedding
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


# class RotaryEmbedding(nn.Module):
#     def __init__(self, dim: int, base: float = 10000.0):
#         super().__init__()
#         inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
#         self.register_buffer("inv_freq", inv_freq, persistent=False)

#     def forward(
#         self,
#         seq_len: int,
#         device: torch.device,
#         dtype: torch.dtype,
#         position_ids: Optional[torch.Tensor] = None,
#     ) -> torch.Tensor:
#         if position_ids is None:
#             t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
#             freqs = torch.einsum("i,j->ij", t, self.inv_freq)
#         else:
#             pos = position_ids.to(device=device, dtype=self.inv_freq.dtype)
#             if pos.dim() == 1:
#                 freqs = torch.einsum("i,j->ij", pos, self.inv_freq)
#             elif pos.dim() == 2:
#                 freqs = torch.einsum("bi,j->bij", pos, self.inv_freq)
#             else:
#                 raise ValueError(f"position_ids must be rank-1 or rank-2, got shape {tuple(pos.shape)}")
#         emb = torch.cat((freqs, freqs), dim=-1)
#         return emb.to(device=device, dtype=dtype)


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
        self.qk_norm = bool(getattr(config, "qk_norm", True))

        self.qkv = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        if self.qk_norm:
            self.q_norm = RMSNorm(self.hidden_size, eps=config.norm_eps)
            self.k_norm = RMSNorm(self.hidden_size, eps=config.norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None
        self.rotary = RotaryEmbedding(self.head_dim, base=float(config.rope_theta))

    def _local_window_tuple(self) -> Tuple[int, int]:
        if self.window_size > 0:
            return (max(self.window_size - 1, 0), 0)
        return (-1, -1)

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
            left_window = max(self.window_size - 1, 0)
            local = (q_idx - k_idx) <= left_window
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
            )

    def _flash_local(self, q, k, v, attention_mask=None):
        """
        q,k,v: [B, H, S, D]  (your current layout)
        return: [B, H, S, D]
        """
        # logger.warning(f"mask_has_pad={not bool(attention_mask.all().item())}")
        # fast path: no mask or all-valid mask
        if attention_mask is None or bool(attention_mask.all().item()):
            try:
                from flash_attn import flash_attn_func  # type: ignore
                q_ = q.transpose(1, 2)  # [B, S, H, D]
                k_ = k.transpose(1, 2)
                v_ = v.transpose(1, 2)
                window = self._local_window_tuple()
                out = flash_attn_func(q_, k_, v_, causal=True, window_size=window)
                return out.transpose(1, 2)  # [B, H, S, D]
            except Exception as e:
                logger.warning(f"FALL BACK to _sdpa_local, Exception:{e}")
                return self._sdpa_local(q, k, v, attention_mask=attention_mask)

        # varlen path
        try:
            from flash_attn.bert_padding import unpad_input, pad_input  # type: ignore
            try:
                from flash_attn import flash_attn_varlen_qkvpacked_func  # type: ignore
                varlen_kind = "qkvpacked"
            except Exception:
                from flash_attn import flash_attn_varlen_func  # type: ignore
                varlen_kind = "qkv"
            
            B, H, S, D = q.shape
            am = attention_mask.to(device=q.device)
            # unpad_input expects bool/int where 1=valid, 0=pad :contentReference[oaicite:1]{index=1}
            am_int = am.to(torch.int32)

            # guard: if some sample has 0 valid tokens, some flash-attn versions can error in edge cases :contentReference[oaicite:2]{index=2}
            seqlens = am_int.sum(dim=-1)
            if (seqlens == 0).any():
                return self._sdpa_local(q, k, v, attention_mask=attention_mask)

            window = self._local_window_tuple()

            dropout_p = 0.0  # you currently don't have attn dropout; keep 0
            softmax_scale = None
            deterministic = not self.training

            if varlen_kind == "qkvpacked":
                # pack: [B, S, 3, H, D] -> unpad -> [nnz, 3, H, D]
                q_ = q.transpose(1, 2)  # [B, S, H, D]
                k_ = k.transpose(1, 2)
                v_ = v.transpose(1, 2)
                qkv = torch.stack([q_, k_, v_], dim=2).contiguous()  # [B, S, 3, H, D]

                qkv_unpad, indices, cu_seqlens, max_seqlen, _ = unpad_input(qkv, am_int)
                # varlen qkvpacked signature includes window_size (tuple) in recent flash-attn :contentReference[oaicite:3]{index=3}
                sig = inspect.signature(flash_attn_varlen_qkvpacked_func)
                if "window_size" in sig.parameters:
                    out_unpad = flash_attn_varlen_qkvpacked_func(
                        qkv_unpad, cu_seqlens, max_seqlen,
                        dropout_p=dropout_p, softmax_scale=softmax_scale,
                        causal=True, window_size=window,
                        deterministic=deterministic,
                    )
                else:
                    # very old flash-attn: no window_size support
                    out_unpad = flash_attn_varlen_qkvpacked_func(
                        qkv_unpad, cu_seqlens, max_seqlen,
                        dropout_p=dropout_p, softmax_scale=softmax_scale,
                        causal=True,
                        deterministic=deterministic,
                    )

                out = pad_input(out_unpad, indices, B, S)  # [B, S, H, D] :contentReference[oaicite:4]{index=4}
                return out.transpose(1, 2)  # [B, H, S, D]

            else:
                # fallback API: q,k,v unpad separately: [B,S,H,D] -> [nnz,H,D]
                q_ = q.transpose(1, 2).contiguous()
                k_ = k.transpose(1, 2).contiguous()
                v_ = v.transpose(1, 2).contiguous()

                q_unpad, indices, cu_seqlens, max_seqlen, _ = unpad_input(q_, am_int)
                k_unpad, _, _, _, _ = unpad_input(k_, am_int)
                v_unpad, _, _, _, _ = unpad_input(v_, am_int)

                sig = inspect.signature(flash_attn_varlen_func)
                if "window_size" in sig.parameters:
                    out_unpad = flash_attn_varlen_func(
                        q_unpad, k_unpad, v_unpad,
                        cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
                        dropout_p=dropout_p, softmax_scale=softmax_scale,
                        causal=True, window_size=window,
                        deterministic=deterministic,
                    )
                else:
                    out_unpad = flash_attn_varlen_func(
                        q_unpad, k_unpad, v_unpad,
                        cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
                        dropout_p=dropout_p, softmax_scale=softmax_scale,
                        causal=True,
                        deterministic=deterministic,
                    )

                out = pad_input(out_unpad, indices, B, S)  # [B,S,H,D]
                return out.transpose(1, 2)

        except Exception as e:
            logger.warning(f"FALL BACK to _sdpa_local, Exception:{e}")
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

        # freqs = self.rotary(seq_len, device=hidden_states.device, dtype=hidden_states.dtype, position_ids=position_ids)
        # q, k = apply_rotary(q, k, freqs)
        cache_offset = past_key_value[0].size(-2) if past_key_value is not None else None
        position_offset = None
        if position_ids is not None:
            if position_ids.dim() == 1:
                start_positions = position_ids[:1]
            else:
                start_positions = position_ids[:, 0]
            if start_positions.numel() > 1 and not torch.equal(start_positions, start_positions[:1].expand_as(start_positions)):
                raise ValueError(
                    "All samples in a batch must share the same start position for E2E rotary offset."
                )
            position_offset = int(start_positions[0].item())

        if position_offset is None:
            offset = int(cache_offset) if cache_offset is not None else 0
        else:
            offset = position_offset

        max_seqlen = offset + seq_len
        max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
        if max_position_embeddings is not None:
            max_seqlen = max(max_seqlen, int(max_position_embeddings))
        q, k = self.rotary(q, k, seqlen_offset=offset, max_seqlen=max_seqlen)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        # --- 加入当场 Flush 的 Debugger ---
        if self.layer_idx == 0 and not hasattr(self, "_e2e_debug_logged"):
            self._e2e_debug_logged = True
            logger.warning(f"\n" + "="*50)
            logger.warning(f"[E2E FATAL BUG CHECK] Layer 0 Forward")
            logger.warning(f"[TTT-DEBUG]Original Q seq_len: {q.size(-2)}")
            logger.warning(f"[TTT-DEBUG]Original K seq_len: {k.size(-2)}")
            logger.warning(f"[TTT-DEBUG]Window Size Config: {self.window_size}")
            logger.warning("="*50 + "\n")
        # ----------------------------------

        # 计算返回给下一步推理的 Cache，这里可以截断，但不影响当前的 k, v
        if self.window_size > 0 and k.size(-2) > self.window_size:
            next_k = k[:, :, -self.window_size :, :]
            next_v = v[:, :, -self.window_size :, :]
        else:
            next_k, next_v = k, v
            
        next_kv = (next_k, next_v)
        if backend == "flash":
            out = self._flash_local(q, k, v, attention_mask=attention_mask)
        else:
            out = self._sdpa_local(q, k, v, attention_mask=attention_mask)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(out), next_kv

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
