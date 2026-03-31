# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.models.utils import Cache

from .layer_lact_swiglu import LaCTSWIGLULayer
from .ttt_operation import l2_norm
from .ttt_operation_srlact import (
    block_causal_srlact_swiglu,
    prenorm_block_causal_srlact_swiglu,
    topk_sparse_softmax,
)

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input
except ImportError:
    warnings.warn(
        "Flash Attention is not installed. Please install it via `pip install flash-attn --no-build-isolation`",
        category=ImportWarning,
    )
    flash_attn_func = None


class SRLaCTSWIGLULayer(LaCTSWIGLULayer):
    """Write-routed Slot LaCT layered on top of the base LaCT implementation."""

    def __init__(self, num_slots: int = 2, slot_iso_param: bool = True, **kwargs):
        super().__init__(**kwargs)

        self.num_slots = int(num_slots)
        self.slot_iso_param = bool(slot_iso_param)
        self._aux_loss: Optional[torch.Tensor] = None

        if self.num_slots <= 1 or not self.use_ttt:
            return

        if self.w0_w2_low_rank > 0:
            raise NotImplementedError(
                "SR-LaCT currently requires full-rank fast weights; set w0_w2_low_rank=-1."
            )

        if self.slot_iso_param and self.d_h % self.num_slots != 0:
            raise ValueError(
                f"d_h={self.d_h} must be divisible by num_slots={self.num_slots}"
            )

        self.d_slot = self.d_h // self.num_slots if self.slot_iso_param else self.d_h

        del self.w0, self.w1, self.w2

        self.w0s = nn.Parameter(
            torch.randn(
                self.num_fw_heads, self.num_slots, self.d_slot, self.d_in
            ) / math.sqrt(self.d_in)
        )
        self.w1s = nn.Parameter(
            torch.randn(
                self.num_fw_heads, self.num_slots, self.d_out, self.d_slot
            ) / math.sqrt(self.d_slot)
        )
        self.w2s = nn.Parameter(
            torch.randn(
                self.num_fw_heads, self.num_slots, self.d_slot, self.d_in
            ) / math.sqrt(self.d_in)
        )

        self.slot_router = nn.Linear(
            self.hidden_size, self.num_fw_heads * self.num_slots, bias=False
        )
        nn.init.zeros_(self.slot_router.weight)
        self.register_buffer("router_temp", torch.tensor(1.0))

    def _materialize_initial_slot_fast_weights(self, batch_size: int):
        fw_w0s = self.w0s.repeat(batch_size, 1, 1, 1)
        fw_w1s = self.w1s.repeat(batch_size, 1, 1, 1)
        fw_w2s = self.w2s.repeat(batch_size, 1, 1, 1)

        if self.fp32_states:
            fw_w0s = fw_w0s.to(torch.float32)
            fw_w1s = fw_w1s.to(torch.float32)
            fw_w2s = fw_w2s.to(torch.float32)

        return fw_w0s, fw_w1s, fw_w2s

    def _build_router_gates(
        self,
        hidden_states: torch.Tensor,
        batch_size: int,
        q_len: int,
    ) -> torch.Tensor:
        chunk_size = int(self.lact_chunk_size)
        num_update_chunks = max(0, (q_len - 1) // chunk_size)

        if num_update_chunks == 0:
            if self.training:
                self._aux_loss = hidden_states.new_zeros(())
            return hidden_states.new_zeros(
                batch_size * self.num_fw_heads,
                0,
                self.num_slots,
            )

        update_hidden_states = hidden_states[:, : num_update_chunks * chunk_size, :]
        chunk_means = update_hidden_states.reshape(
            batch_size,
            num_update_chunks,
            chunk_size,
            self.hidden_size,
        ).mean(dim=2)

        router_logits = self.slot_router(chunk_means)
        router_logits = rearrange(
            router_logits,
            "b n_chunks (n_h slots) -> (b n_h) n_chunks slots",
            n_h=self.num_fw_heads,
            slots=self.num_slots,
        )

        tau = self.router_temp.to(router_logits.dtype).clamp_min(1e-5)
        gates = topk_sparse_softmax(
            router_logits / tau,
            topk=min(2, self.num_slots),
        )

        if self.training:
            self._aux_loss = self.num_slots * (gates.mean(dim=(0, 1)) ** 2).sum()

        return gates

    def _run_slotted_ttt(
        self,
        hidden_states: torch.Tensor,
        batch_size: int,
        q_len: int,
        fast_q: torch.Tensor,
        fast_k: torch.Tensor,
        fast_v: torch.Tensor,
        fw_lr0: torch.Tensor,
        fw_lr1: torch.Tensor,
        fw_lr2: torch.Tensor,
        momentum: Optional[torch.Tensor],
    ) -> torch.Tensor:
        router_gates = self._build_router_gates(hidden_states, batch_size, q_len)
        fw_w0s, fw_w1s, fw_w2s = self._materialize_initial_slot_fast_weights(batch_size)

        if self.ttt_prenorm:
            return prenorm_block_causal_srlact_swiglu(
                fw_w0s,
                fw_w1s,
                fw_w2s,
                fast_q,
                fast_k,
                fast_v,
                fw_lr0,
                fw_lr1,
                fw_lr2,
                router_gates,
                chunk_size=self.lact_chunk_size,
                use_muon=self.use_muon,
                momentum=momentum,
            )

        return block_causal_srlact_swiglu(
            fw_w0s,
            fw_w1s,
            fw_w2s,
            fast_q,
            fast_k,
            fast_v,
            fw_lr0,
            fw_lr1,
            fw_lr2,
            router_gates,
            chunk_size=self.lact_chunk_size,
            use_muon=self.use_muon,
            momentum=momentum,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if self.num_slots <= 1 or not self.use_ttt:
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )

        self._aux_loss = None

        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        if use_cache:
            raise NotImplementedError(
                "SR-LaCT cached decoding is not implemented yet. Use use_cache=False for now."
            )

        batch_size, q_len, _ = hidden_states.size()

        q, k, v = self.qkv(hidden_states).chunk(3, dim=-1)

        if self.attn_qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        if self.num_fw_heads > 0:
            fast_q, fast_k = self._rescale_qk(q, k)
            fast_v = v

        q = rearrange(q, "... (h d) -> ... h d", d=self.head_dim)
        k = rearrange(k, "... (h d) -> ... h d", d=self.head_dim)
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_dim)

        cu_seqlens = kwargs.get("cu_seqlens", None)

        seqlen_offset, max_seqlen = 0, q_len
        if past_key_values is not None:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            max_seqlen = q.shape[1] + seqlen_offset

            if attention_mask is not None:
                seqlen_offset = (
                    seqlen_offset + attention_mask.sum(-1) - attention_mask.shape[-1]
                )
                max_seqlen = q.shape[1] + max(seqlen_offset)

        if self.max_position_embeddings is not None:
            max_seqlen = max(max_seqlen, self.max_position_embeddings)

        q, k = self.rotary(
            q,
            k,
            seqlen_offset=seqlen_offset,
            max_seqlen=max_seqlen,
            cu_seqlens=cu_seqlens,
        )

        if past_key_values is not None:
            cache_has_content = past_key_values.get_seq_length(self.layer_idx) > 0
            k_cached, v_cached = past_key_values.update(
                attn_state=(k.flatten(-2, -1), v.flatten(-2, -1)),
                layer_idx=self.layer_idx,
                offset=q_len,
                cache_kwargs=dict(window_size=self.window_size),
            )["attn_state"]
            if cache_has_content:
                k, v = k_cached, v_cached
                k = rearrange(k, "... (h d) -> ... h d", d=self.head_dim)
                v = rearrange(v, "... (h d) -> ... h d", d=self.head_dim)

        if flash_attn_func is None:
            raise ImportError(
                "Please install Flash Attention via `pip install flash-attn --no-build-isolation` first"
            )

        if attention_mask is not None:
            q, k, v, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                q, k, v, attention_mask, q_len
            )
            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_q, max_seqlen_k = max_seq_lens
            o = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                causal=(self.window_size is None),
                window_size=(
                    (-1, -1) if self.window_size is None else (self.window_size - 1, 0)
                ),
            )
            o = pad_input(o, indices_q, batch_size, q_len)
        elif cu_seqlens is not None:
            o = flash_attn_varlen_func(
                q.squeeze(0),
                k.squeeze(0),
                v.squeeze(0),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=(self.window_size is None),
                window_size=(
                    (-1, -1) if self.window_size is None else (self.window_size - 1, 0)
                ),
            ).unsqueeze(0)
        else:
            o = flash_attn_func(
                q,
                k,
                v,
                causal=(self.window_size is None),
                window_size=(
                    (-1, -1) if self.window_size is None else (self.window_size - 1, 0)
                ),
            )
        o = o.reshape(batch_size, q_len, -1)

        if self.num_fw_heads > 0:
            fast_q = rearrange(
                fast_q, "b s (n_h d) -> (b n_h) s d", n_h=self.num_fw_heads
            )
            fast_k = rearrange(
                fast_k, "b s (n_h d) -> (b n_h) s d", n_h=self.num_fw_heads
            )
            fast_v = rearrange(
                fast_v, "b s (n_h d) -> (b n_h) s d", n_h=self.num_fw_heads
            )

            if self.qkv_silu:
                if self.no_v_silu:
                    fast_q = F.silu(fast_q)
                    fast_k = F.silu(fast_k)
                else:
                    fast_q = F.silu(fast_q)
                    fast_k = F.silu(fast_k)
                    fast_v = F.silu(fast_v)

            fast_q = l2_norm(fast_q)
            fast_k = l2_norm(fast_k)

            if not self.ttt_nope:
                fast_q = rearrange(
                    fast_q, "(b n_h) s d -> b s (n_h d)", n_h=self.num_fw_heads
                )
                fast_k = rearrange(
                    fast_k, "(b n_h) s d -> b s (n_h d)", n_h=self.num_fw_heads
                )

                fast_q = rearrange(
                    fast_q, "b s (n_h d) -> b s n_h d", n_h=self.num_heads
                )
                fast_k = rearrange(
                    fast_k, "b s (n_h d) -> b s n_h d", n_h=self.num_heads
                )

                fast_q, fast_k = self.rotary(
                    fast_q,
                    fast_k,
                    seqlen_offset=seqlen_offset,
                    max_seqlen=max_seqlen,
                    cu_seqlens=cu_seqlens,
                )

                fast_q = rearrange(
                    fast_q, "b s n_h d -> b s (n_h d)", n_h=self.num_heads
                )
                fast_k = rearrange(
                    fast_k, "b s n_h d -> b s (n_h d)", n_h=self.num_heads
                )

                fast_q = rearrange(
                    fast_q, "b s (n_h d) -> (b n_h) s d", n_h=self.num_fw_heads
                )
                fast_k = rearrange(
                    fast_k, "b s (n_h d) -> (b n_h) s d", n_h=self.num_fw_heads
                )

            self._log_fast_kv_chunk_diversity(fast_k, fast_v)
            fw_lr0, fw_lr1, fw_lr2 = self._build_token_lrs(hidden_states)
            momentum = self._build_token_momentum(hidden_states)

            fw_x = self._run_slotted_ttt(
                hidden_states,
                batch_size,
                q_len,
                fast_q,
                fast_k,
                fast_v,
                fw_lr0,
                fw_lr1,
                fw_lr2,
                momentum,
            )

            ttt_x_normed = self._apply_ttt_postprocess(fw_x, hidden_states)
            o = o + ttt_x_normed

        o = self.o_proj(o)

        attentions = None
        return o, attentions, past_key_values
