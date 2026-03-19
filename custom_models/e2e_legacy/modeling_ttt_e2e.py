# # -*- coding: utf-8 -*-

# from __future__ import annotations

# import math
# from typing import Any, List, Optional, Tuple, Union

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.utils.checkpoint
# from transformers.generation import GenerationMixin
# from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
# from transformers.modeling_utils import PreTrainedModel
# from transformers.utils import logging

# from .configuration_ttt_e2e import TttE2EConfig

# logger = logging.get_logger(__name__)

# try:
#     from torch.func import functional_call as torch_functional_call
# except ImportError:
#     from torch.nn.utils.stateless import functional_call as torch_functional_call


# def _rotate_half(x: torch.Tensor) -> torch.Tensor:
#     x1 = x[..., ::2]
#     x2 = x[..., 1::2]
#     return torch.stack((-x2, x1), dim=-1).flatten(-2)


# def _build_inv_freq(head_dim: int, theta: float, device: torch.device) -> torch.Tensor:
#     return 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))


# def _apply_rope(q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
#     # q, k: [b, h, s, d]
#     head_dim = q.size(-1)
#     inv_freq = _build_inv_freq(head_dim, theta, q.device)
#     freqs = torch.einsum("bs,d->bsd", position_ids.float(), inv_freq)
#     cos = torch.repeat_interleave(freqs.cos(), repeats=2, dim=-1).unsqueeze(1).type_as(q)
#     sin = torch.repeat_interleave(freqs.sin(), repeats=2, dim=-1).unsqueeze(1).type_as(q)
#     return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


# class SwiGLUMLP(nn.Module):
#     def __init__(self, config: TttE2EConfig):
#         super().__init__()
#         self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
#         self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
#         self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
#         self.dropout = nn.Dropout(config.resid_pdrop)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = F.silu(self.w1(x)) * self.w3(x)
#         x = self.w2(x)
#         return self.dropout(x)


# class TttE2EAttention(nn.Module):
#     def __init__(self, config: TttE2EConfig):
#         super().__init__()
#         self.config = config
#         self.num_heads = config.num_attention_heads
#         self.head_dim = config.hidden_size // config.num_attention_heads
#         if self.head_dim * self.num_heads != config.hidden_size:
#             raise ValueError("hidden_size must be divisible by num_attention_heads")

#         self.wq = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
#         self.wk = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
#         self.wv = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
#         self.wo = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
#         self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
#         self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
#         self.resid_dropout = nn.Dropout(config.resid_pdrop)

#     def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
#         b, s, _ = x.shape
#         return x.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

#     def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
#         b, _, s, _ = x.shape
#         return x.transpose(1, 2).contiguous().view(b, s, self.config.hidden_size)

#     def _build_mask(
#         self,
#         q_len: int,
#         k_len: int,
#         q_start: int,
#         k_start: int,
#         device: torch.device,
#         sliding_window_size: Optional[int] = None,
#     ) -> torch.Tensor:
#         q_abs = torch.arange(q_start, q_start + q_len, device=device).unsqueeze(1)
#         k_abs = torch.arange(k_start, k_start + k_len, device=device).unsqueeze(0)
#         disallow = k_abs > q_abs
#         if sliding_window_size is not None:
#             disallow = disallow | (k_abs < (q_abs - sliding_window_size + 1))
#         return disallow.unsqueeze(0).unsqueeze(0)

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         position_ids: torch.Tensor,
#         past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
#         use_cache: bool = False,
#         sliding_window_size: Optional[int] = None,
#     ) -> tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
#         q = self._split_heads(self.wq(hidden_states))
#         k = self._split_heads(self.wk(hidden_states))
#         v = self._split_heads(self.wv(hidden_states))

#         if self.config.qk_norm:
#             q = self.q_norm(q)
#             k = self.k_norm(k)

#         q, k = _apply_rope(q, k, position_ids=position_ids, theta=self.config.rope_theta)

#         past_len = 0
#         if past_key_value is not None:
#             past_k, past_v = past_key_value
#             past_len = past_k.size(-2)
#             k = torch.cat([past_k, k], dim=-2)
#             v = torch.cat([past_v, v], dim=-2)

#         if sliding_window_size is not None and k.size(-2) > sliding_window_size:
#             k = k[:, :, -sliding_window_size:, :]
#             v = v[:, :, -sliding_window_size:, :]
#         q_start = past_len
#         total_kv_len = past_len + q.size(-2)
#         k_start = total_kv_len - k.size(-2)

#         attn_mask = self._build_mask(
#             q_len=q.size(-2),
#             k_len=k.size(-2),
#             q_start=q_start,
#             k_start=k_start,
#             device=hidden_states.device,
#             sliding_window_size=sliding_window_size,
#         )
#         use_eager_attn = self.config.force_eager_attention or (self.config.train_mode == "meta" and self.training)
#         if use_eager_attn:
#             scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
#             scores = scores.masked_fill(attn_mask, torch.finfo(scores.dtype).min)
#             probs = torch.softmax(scores, dim=-1)
#             probs = F.dropout(probs, p=self.config.attn_pdrop if self.training else 0.0, training=self.training)
#             out = torch.matmul(probs, v)
#         else:
#             out = F.scaled_dot_product_attention(
#                 q,
#                 k,
#                 v,
#                 attn_mask=~attn_mask,
#                 dropout_p=self.config.attn_pdrop if self.training else 0.0,
#                 is_causal=False,
#             )
#         out = self._merge_heads(out)
#         out = self.resid_dropout(self.wo(out))
#         new_past = (k, v) if use_cache else None
#         return out, new_past


# class TttE2EBlock(nn.Module):
#     def __init__(self, config: TttE2EConfig):
#         super().__init__()
#         self.config = config
#         self.seq_modeling = TttE2EAttention(config)
#         self.feed_forward = SwiGLUMLP(config)
#         self.seq_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
#         self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
#         self.seq_post_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
#         self.ffn_post_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         position_ids: torch.Tensor,
#         past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
#         use_cache: bool = False,
#     ) -> tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
#         seq_input = self.seq_norm(hidden_states) if self.config.pre_norm else hidden_states
#         sliding_window_size = None
#         if self.config.seq_modeling_block in ("SWA", "SWAFull"):
#             sliding_window_size = self.config.sliding_window_size

#         seq_output, new_past = self.seq_modeling(
#             seq_input,
#             position_ids=position_ids,
#             past_key_value=past_key_value,
#             use_cache=use_cache,
#             sliding_window_size=sliding_window_size,
#         )
#         if self.config.post_norm:
#             seq_output = self.seq_post_norm(seq_output)
#         hidden_states = hidden_states + seq_output

#         ffn_input = self.ffn_norm(hidden_states) if self.config.pre_norm else hidden_states
#         ffn_output = self.feed_forward(ffn_input)
#         if self.config.post_norm:
#             ffn_output = self.ffn_post_norm(ffn_output)
#         hidden_states = hidden_states + ffn_output
#         return hidden_states, new_past


# class TttE2EPreTrainedModel(PreTrainedModel):
#     config_class = TttE2EConfig
#     base_model_prefix = "model"
#     supports_gradient_checkpointing = True
#     _no_split_modules = ["TttE2EBlock"]

#     def _init_weights(self, module: nn.Module):
#         if isinstance(module, (nn.Linear, nn.Embedding)):
#             nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
#             if hasattr(module, "bias") and module.bias is not None:
#                 nn.init.zeros_(module.bias)


# class TttE2EModel(TttE2EPreTrainedModel):
#     def __init__(self, config: TttE2EConfig):
#         super().__init__(config)
#         self.padding_idx = config.pad_token_id
#         self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
#         self.dropout = nn.Dropout(config.embd_pdrop)
#         self.layers = nn.ModuleList([TttE2EBlock(config) for _ in range(config.num_hidden_layers)])
#         self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
#         self.gradient_checkpointing = False
#         self.post_init()

#     def get_input_embeddings(self):
#         return self.embeddings

#     def set_input_embeddings(self, value):
#         self.embeddings = value

#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         **kwargs: Any,
#     ) -> Union[Tuple, BaseModelOutputWithPast]:
#         del attention_mask  # Padding mask is not yet supported in this baseline port.
#         del output_attentions
#         del kwargs

#         output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         if input_ids is not None and inputs_embeds is not None:
#             raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
#         if input_ids is None and inputs_embeds is None:
#             raise ValueError("You have to specify either input_ids or inputs_embeds")

#         if inputs_embeds is None:
#             hidden_states = self.embeddings(input_ids)
#         else:
#             hidden_states = inputs_embeds
#         hidden_states = self.dropout(hidden_states)

#         bsz, seq_len, _ = hidden_states.shape
#         if position_ids is None:
#             past_len = 0
#             if past_key_values and past_key_values[0] is not None:
#                 past_len = past_key_values[0][0].size(-2)
#             position_ids = torch.arange(past_len, past_len + seq_len, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)
#         else:
#             if position_ids.shape != (bsz, seq_len):
#                 raise ValueError(
#                     f"position_ids must have shape {(bsz, seq_len)}, got {tuple(position_ids.shape)}."
#                 )
#             position_ids = position_ids.to(device=hidden_states.device, dtype=torch.long)

#         all_hidden_states = () if output_hidden_states else None
#         next_cache = [] if use_cache else None

#         if past_key_values is None:
#             past_key_values = [None] * len(self.layers)

#         for i, layer in enumerate(self.layers):
#             if output_hidden_states:
#                 all_hidden_states += (hidden_states,)

#             if self.gradient_checkpointing and self.training:
#                 if use_cache:
#                     logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
#                     use_cache = False
#                     next_cache = None
#                 hidden_states, _ = self._gradient_checkpointing_func(
#                     layer.__call__,
#                     hidden_states,
#                     position_ids,
#                     past_key_values[i],
#                     False,
#                 )
#             else:
#                 hidden_states, new_past = layer(
#                     hidden_states=hidden_states,
#                     position_ids=position_ids,
#                     past_key_value=past_key_values[i],
#                     use_cache=use_cache,
#                 )
#                 if use_cache:
#                     next_cache.append(new_past)

#         hidden_states = self.norm(hidden_states)

#         if output_hidden_states:
#             all_hidden_states += (hidden_states,)

#         if not return_dict:
#             return tuple(v for v in (hidden_states, next_cache, all_hidden_states, None) if v is not None)

#         return BaseModelOutputWithPast(
#             last_hidden_state=hidden_states,
#             past_key_values=next_cache,
#             hidden_states=all_hidden_states,
#             attentions=None,
#         )


# class TttE2EForCausalLM(TttE2EPreTrainedModel, GenerationMixin):
#     _tied_weights_keys = {"lm_head.weight": "model.embeddings.weight"}

#     def __init__(self, config: TttE2EConfig):
#         super().__init__(config)
#         self.model = TttE2EModel(config)
#         self.vocab_size = config.vocab_size
#         self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
#         self.post_init()

#     def get_input_embeddings(self):
#         return self.model.embeddings

#     def set_input_embeddings(self, value):
#         self.model.embeddings = value

#     def get_output_embeddings(self):
#         return self.lm_head

#     def set_output_embeddings(self, new_embeddings):
#         self.lm_head = new_embeddings

#     @staticmethod
#     def _shift_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
#         shift_logits = logits[:, :-1, :].contiguous()
#         shift_labels = labels[:, 1:].contiguous()
#         return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

#     @staticmethod
#     def _causal_chunk_loss(
#         logits: torch.Tensor,
#         labels: torch.Tensor,
#         chunk_start: int,
#         chunk_end: int,
#         total_seq_len: int,
#     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         if chunk_end < total_seq_len:
#             target_labels = labels[:, chunk_start + 1 : chunk_end + 1].contiguous()
#             pred_logits = logits
#         else:
#             target_labels = labels[:, chunk_start + 1 : chunk_end].contiguous()
#             pred_logits = logits[:, : target_labels.size(1), :]

#         loss_sum = F.cross_entropy(
#             pred_logits.reshape(-1, pred_logits.size(-1)),
#             target_labels.reshape(-1),
#             reduction="sum",
#             ignore_index=-100,
#         )
#         valid_count = target_labels.ne(-100).sum()
#         denom = valid_count.clamp_min(1).to(loss_sum.dtype)
#         loss_mean = loss_sum / denom
#         return loss_mean, loss_sum, valid_count

#     def _get_inner_param_names(self) -> list[str]:
#         suffix_len = int(self.config.suffix_len)
#         if suffix_len <= 0:
#             return []
#         n_layers = len(self.model.layers)
#         start_idx = max(0, n_layers - suffix_len)
#         names: list[str] = []
#         for name, _ in self.model.named_parameters():
#             if not name.startswith("layers."):
#                 continue
#             layer_idx = int(name.split(".", 2)[1])
#             if layer_idx >= start_idx:
#                 names.append(name)
#         return names

#     def _meta_forward(
#         self,
#         input_ids: torch.LongTensor,
#         labels: torch.LongTensor,
#         attention_mask: Optional[torch.Tensor],
#         output_hidden_states: Optional[bool],
#         return_dict: bool,
#     ) -> Union[Tuple, CausalLMOutputWithPast]:
#         if attention_mask is not None:
#             raise NotImplementedError("Meta slow path currently does not support attention_mask.")
#         if input_ids is None or labels is None:
#             raise ValueError("Meta slow path requires both input_ids and labels.")

#         chunk_size = int(self.config.mini_batch_size)
#         if chunk_size <= 1:
#             raise ValueError("mini_batch_size must be > 1 for meta slow path.")
#         if input_ids.size(1) % chunk_size != 0:
#             raise ValueError("For meta slow path, sequence length must be divisible by mini_batch_size.")

#         inner_names = self._get_inner_param_names()
#         if not inner_names:
#             raise ValueError("Meta slow path requires config.suffix_len > 0 to define inner parameters.")

#         model_param_map = dict(self.model.named_parameters())
#         inner_params = {name: model_param_map[name] for name in inner_names}
#         static_params = {name: p for name, p in model_param_map.items() if name not in inner_params}

#         chunk_logits: list[torch.Tensor] = []
#         hidden_states_out: list[torch.Tensor] = []
#         n_chunks = input_ids.size(1) // chunk_size
#         total_seq_len = input_ids.size(1)
#         meta_past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
#         total_loss_sum: Optional[torch.Tensor] = None
#         total_valid_count = torch.zeros((), device=input_ids.device, dtype=torch.long)

#         for i in range(n_chunks):
#             s = i * chunk_size
#             e = s + chunk_size
#             input_chunk = input_ids[:, s:e]
#             position_chunk = torch.arange(s, e, device=input_ids.device).unsqueeze(0).expand(input_ids.size(0), -1)

#             merged_params = {**static_params, **inner_params}
#             model_outputs = torch_functional_call(
#                 self.model,
#                 merged_params,
#                 kwargs={
#                     "input_ids": input_chunk,
#                     "attention_mask": None,
#                     "past_key_values": meta_past_key_values,
#                     "position_ids": position_chunk,
#                     "inputs_embeds": None,
#                     "use_cache": True,
#                     "output_attentions": False,
#                     "output_hidden_states": output_hidden_states,
#                     "return_dict": True,
#                 },
#             )
#             meta_past_key_values = model_outputs.past_key_values

#             hidden_chunk = model_outputs.last_hidden_state
#             logits_chunk = F.linear(hidden_chunk, self.lm_head.weight, self.lm_head.bias)
#             loss_chunk, loss_chunk_sum, valid_chunk_count = self._causal_chunk_loss(
#                 logits=logits_chunk,
#                 labels=labels,
#                 chunk_start=s,
#                 chunk_end=e,
#                 total_seq_len=total_seq_len,
#             )

#             chunk_logits.append(logits_chunk)
#             hidden_states_out.append(hidden_chunk)
#             if total_loss_sum is None:
#                 total_loss_sum = loss_chunk_sum
#             else:
#                 total_loss_sum = total_loss_sum + loss_chunk_sum
#             total_valid_count = total_valid_count + valid_chunk_count

#             grad_inputs = [inner_params[name] for name in inner_names]
#             grads = torch.autograd.grad(loss_chunk, grad_inputs, create_graph=True)
#             inner_params = {
#                 name: param - self.config.inner_lr * grad
#                 for (name, param), grad in zip(inner_params.items(), grads)
#             }

#         logits = torch.cat(chunk_logits, dim=1)
#         loss = total_loss_sum / total_valid_count.clamp_min(1).to(total_loss_sum.dtype)
#         all_hidden_states = None
#         if output_hidden_states:
#             all_hidden_states = (torch.cat(hidden_states_out, dim=1),)

#         if not return_dict:
#             return tuple(v for v in (loss, logits, None, all_hidden_states, None) if v is not None)
#         return CausalLMOutputWithPast(
#             loss=loss,
#             logits=logits,
#             past_key_values=None,
#             hidden_states=all_hidden_states,
#             attentions=None,
#         )

#     def prepare_inputs_for_generation(
#         self,
#         input_ids: torch.LongTensor = None,
#         past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         inputs_embeds: Optional[torch.Tensor] = None,
#         use_cache: bool = True,
#         **kwargs: Any,
#     ):
#         if past_key_values:
#             input_ids = input_ids[:, -1:]
#         if inputs_embeds is not None and not past_key_values:
#             model_inputs = {"inputs_embeds": inputs_embeds}
#         else:
#             model_inputs = {"input_ids": input_ids.contiguous()}
#         model_inputs.update(
#             {
#                 "past_key_values": past_key_values,
#                 "use_cache": use_cache,
#                 "attention_mask": attention_mask,
#             }
#         )
#         return model_inputs

#     def forward(
#         self,
#         input_ids: torch.LongTensor = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         **kwargs: Any,
#     ) -> Union[Tuple, CausalLMOutputWithPast]:
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         if self.training and self.config.train_mode == "meta":
#             return self._meta_forward(
#                 input_ids=input_ids,
#                 labels=labels,
#                 attention_mask=attention_mask,
#                 output_hidden_states=output_hidden_states,
#                 return_dict=return_dict,
#             )

#         outputs = self.model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             **kwargs,
#         )
#         hidden_states = outputs[0]
#         logits = self.lm_head(hidden_states)

#         loss = None
#         if labels is not None:
#             loss = self._shift_ce_loss(logits, labels)

#         if not return_dict:
#             output = (logits,) + outputs[1:]
#             return (loss,) + output if loss is not None else output

#         return CausalLMOutputWithPast(
#             loss=loss,
#             logits=logits,
#             past_key_values=outputs.past_key_values,
#             hidden_states=outputs.hidden_states,
#             attentions=None,
#         )





# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from .configuration_ttt_e2e import E2ETTTConfig


# -----------------
# Small utilities
# -----------------

def _silu(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b, n, d)
        # var = x.pow(2).mean(dim=-1, keepdim=True)
        # x = x * torch.rsqrt(var + self.eps)
        # return x * self.weight
        var = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (x.to(self.weight.dtype) * self.weight)

class RotaryEmbedding(nn.Module):
    """Minimal RoPE for causal attention.

    This is intentionally small; replace with your project RoPE if you already have one.
    """

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
    ):
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
    # q,k: (b,h,n,dh); freqs: (n,dh) or (b,n,dh)
    if freqs.dim() == 2:
        cos = freqs.cos()[None, None, :, :]
        sin = freqs.sin()[None, None, :, :]
    elif freqs.dim() == 3:
        cos = freqs.cos()[:, None, :, :]
        sin = freqs.sin()[:, None, :, :]
    else:
        raise ValueError(f"freqs must be rank-2 or rank-3, got shape {tuple(freqs.shape)}")

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


# -----------------
# Attention
# -----------------

class SlidingWindowAttention(nn.Module):
    def __init__(self, config: E2ETTTConfig, backend: Optional[str] = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.window_size = int(config.window_size)
        self.backend = str(backend if backend is not None else config.attn_backend)
        self.attn_pdrop = float(getattr(config, "attn_pdrop", 0.0))
        self.qk_norm = bool(getattr(config, "qk_norm", True))

        self.qkv = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=False)
        self.out = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.resid_dropout = nn.Dropout(float(getattr(config, "resid_pdrop", 0.0)))
        self.rotary = RotaryEmbedding(self.head_dim, base=float(config.rope_theta))

    def _sdpa_local(self, q, k, v):
        # # q,k,v: (b,h,n,dh)
        # b, h, n, d = q.shape
        # # build a causal+window mask (n,n) (True = keep)
        # idx = torch.arange(n, device=q.device)
        # q_idx = idx[:, None]
        # k_idx = idx[None, :]
        # causal = q_idx >= k_idx
        # local = (q_idx - k_idx) <= self.window_size
        # mask = causal & local
        # # SDPA expects additive mask or bool mask depending on torch version.
        # # Use bool mask and set is_causal=False (we already include causal).
        # attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
        # return attn



        # q: (b, h, q_len, d), k,v: (b, h, kv_len, d)
        q_len = q.size(-2)
        kv_len = k.size(-2)
        
        # [修复] 构建支持 q_len != kv_len 的 causal + window mask
        q_idx = torch.arange(q_len, device=q.device)[:, None] + (kv_len - q_len)
        k_idx = torch.arange(kv_len, device=k.device)[None, :]
        
        causal = q_idx >= k_idx
        local = (q_idx - k_idx) <= self.window_size
        mask = causal & local
        
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=(self.attn_pdrop if self.training else 0.0),
        )

    def _flash_local(self, q, k, v):
        # Draft: use flash-attn if present; otherwise fallback.
        try:
            from flash_attn import flash_attn_func  # type: ignore

            # flash_attn_func expects (batch, seqlen, nheads, headdim)
            # q: (b, h, q_len, d) -> (b, q_len, h, d)
            q_ = q.transpose(1, 2)
            k_ = k.transpose(1, 2)
            v_ = v.transpose(1, 2)
            
            # Flash Attention 2 完美支持 q_len != kv_len。
            # 当 causal=True 时，它会自动对齐序列末尾，计算正确的因果掩码。
            out = flash_attn_func(
                q_, k_, v_,
                dropout_p=(self.attn_pdrop if self.training else 0.0),
                causal=True,
                window_size=(self.window_size, 0),  # 左侧窗口为 window_size，右侧无
            )
            # 输出转回 (b, h, q_len, d)
            return out.transpose(1, 2)
        except Exception:
            return self._sdpa_local(q, k, v)

    def forward(
        self, 
        x: torch.Tensor, 
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # x: (b,n,d)
        b, n, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # 变为 (b,h,n,dh)
        q = q.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)  
        k = k.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # 1. 旋转位置编码 (RoPE) 仅作用于当前 chunk 的 q 和 k
        freqs = self.rotary(n, device=x.device, dtype=x.dtype, position_ids=position_ids)
        q, k = apply_rotary(q, k, freqs)

        # 2. 拼接上一轮 Chunk 传过来的历史 KV Cache
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        # 3. 截断 KV Cache，物理上强制限制在窗口大小内，防止内存/显存 OOM
        # 这一步也是为了让 SWA 永远只保留 window_size 大小的记忆
        if k.size(-2) > self.window_size:
            k = k[:, :, -self.window_size:, :]
            v = v[:, :, -self.window_size:, :]
            
        # 收集当前层的 KV Cache 用于传给下一个 Chunk
        next_kv = (k, v)

        # 4. 计算注意力 (此时的 k 和 v 包含了历史 token)
        if self.backend == "flash":
            out = self._flash_local(q, k, v)
        else:
            out = self._sdpa_local(q, k, v)

        # 5. 恢复形状输出
        out = out.transpose(1, 2).contiguous().view(b, n, self.hidden_size)
        
        # 注意：此处返回签名变成了 Tuple[Tensor, Tuple]
        return self.resid_dropout(self.out(out)), next_kv


# -----------------
# MLPs
# -----------------

class SwiGLU(nn.Module):
    def __init__(self, config: E2ETTTConfig):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(float(getattr(config, "resid_pdrop", 0.0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(_silu(self.w1(x)) * self.w3(x)))


class PrimeSwiGLU(nn.Module):
    """SwiGLU whose parameters may be overridden by a fast-weight dict."""

    def __init__(self, config: E2ETTTConfig, name_prefix: str):
        super().__init__()
        self.name_prefix = name_prefix
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(float(getattr(config, "resid_pdrop", 0.0)))

    def fast_param_names(self) -> List[str]:
        # names as they appear in model.named_parameters()
        return [
            f"{self.name_prefix}.w1.weight",
            f"{self.name_prefix}.w2.weight",
            f"{self.name_prefix}.w3.weight",
        ]

    def forward(self, x: torch.Tensor, fast: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        if fast is None:
            return self.dropout(self.w2(_silu(self.w1(x)) * self.w3(x)))

        w1 = fast.get(f"{self.name_prefix}.w1.weight", self.w1.weight)
        w2 = fast.get(f"{self.name_prefix}.w2.weight", self.w2.weight)
        w3 = fast.get(f"{self.name_prefix}.w3.weight", self.w3.weight)

        z1 = F.linear(x, w1)
        z3 = F.linear(x, w3)
        x2 = _silu(z1) * z3
        return self.dropout(F.linear(x2, w2))


# -----------------
# Transformer block
# -----------------

class E2ETTTBlock(nn.Module):
    def __init__(self, config: E2ETTTConfig, layer_idx: int, is_suffix: bool):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_suffix = bool(is_suffix)
        self.pre_norm = bool(getattr(config, "pre_norm", True))
        self.post_norm = bool(getattr(config, "post_norm", True))
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        attn_backend = "sdpa" if self.is_suffix else config.attn_backend
        self.attn = SlidingWindowAttention(config, backend=attn_backend)
        self.seq_post_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLU(config)
        self.ffn_post_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.prime_mlp_norm = None
        self.prime_mlp_post_norm = None
        self.prime_mlp = None

        if self.is_suffix and config.two_mlp_per_block:
            # "prime" MLP is the one updated in the inner loop.
            self.prime_mlp_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.prime_mlp_post_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.prime_mlp = PrimeSwiGLU(config, name_prefix=f"layers.{layer_idx}.prime_mlp")

    def prime_param_names(self) -> List[str]:
        if self.prime_mlp is None:
            return []
        return self.prime_mlp.fast_param_names()

    # def forward(
    #     self,
    #     x: torch.Tensor,
    #     fast: Optional[Dict[str, torch.Tensor]] = None,
    #     position_ids: Optional[torch.Tensor] = None,
    # ) -> torch.Tensor:
    #     # attention
    #     a = self.attn(self.attn_norm(x), position_ids=position_ids)
    #     x = x + a

    #     # prime MLP (inner-loop updated)
    #     if self.prime_mlp is not None:
    #         p = self.prime_mlp(self.prime_mlp_norm(x), fast=fast)
    #         x = x + p

    #     # safe MLP (outer-loop only)
    #     m = self.mlp(self.mlp_norm(x))
    #     x = x + m
    #     return x
    def forward(
        self,
        x: torch.Tensor,
        fast: Optional[Dict[str, torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_in = self.attn_norm(x) if self.pre_norm else x
        a, next_kv = self.attn(seq_in, position_ids=position_ids, past_key_value=past_key_value)
        if self.post_norm:
            a = self.seq_post_norm(a)
        x = x + a

        if self.prime_mlp is not None:
            p_in = self.prime_mlp_norm(x) if self.pre_norm else x
            p = self.prime_mlp(p_in, fast=fast)
            if self.post_norm and self.prime_mlp_post_norm is not None:
                p = self.prime_mlp_post_norm(p)
            x = x + p

        m_in = self.mlp_norm(x) if self.pre_norm else x
        m = self.mlp(m_in)
        if self.post_norm:
            m = self.ffn_post_norm(m)
        x = x + m
        
        return x, next_kv

# -----------------
# HF model wrappers
# -----------------

@dataclass
class E2ETTTOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    fast_weights: Optional[Dict[str, torch.Tensor]] = None


class E2ETTTPreTrainedModel(PreTrainedModel):
    config_class = E2ETTTConfig
    base_model_prefix = "model"


class E2ETTTModel(E2ETTTPreTrainedModel):
    def __init__(self, config: E2ETTTConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_dropout = nn.Dropout(float(getattr(config, "embd_pdrop", 0.0)))
        self.layers = nn.ModuleList([])

        suffix_start = config.num_hidden_layers - config.suffix_len

        for i in range(config.num_hidden_layers):
            self.layers.append(E2ETTTBlock(config, layer_idx=i, is_suffix=(i >= suffix_start)))

        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # init
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _collect_prime_params(self) -> List[str]:
        names: List[str] = []
        for layer in self.layers:
            names.extend(layer.prime_param_names())
        return names

    def init_fast_weights(self) -> Dict[str, torch.Tensor]:
        """Return a dict of initial fast weights (references model parameters)."""
        prime_names = set(self._collect_prime_params())
        fast: Dict[str, torch.Tensor] = {}
        for name, p in self.named_parameters():
            if name in prime_names:
                fast[name] = p
        return fast

    def _normalize_position_ids(
        self,
        position_ids: Optional[torch.Tensor],
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        if position_ids is None:
            return torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

        if position_ids.dim() == 1:
            if position_ids.size(0) != seq_len:
                raise ValueError(f"position_ids length {position_ids.size(0)} does not match seq_len {seq_len}")
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        elif position_ids.dim() == 2:
            if position_ids.size(-1) != seq_len:
                raise ValueError(
                    f"position_ids last dim {position_ids.size(-1)} does not match seq_len {seq_len}"
                )
            if position_ids.size(0) == 1 and batch_size > 1:
                position_ids = position_ids.expand(batch_size, -1)
            elif position_ids.size(0) != batch_size:
                raise ValueError(
                    f"position_ids batch dim {position_ids.size(0)} does not match batch_size {batch_size}"
                )
        else:
            raise ValueError(f"position_ids must be rank-1 or rank-2, got shape {tuple(position_ids.shape)}")

        return position_ids.to(device=device, dtype=torch.long)

    # def forward_blocks(
    #     self,
    #     x: torch.Tensor,
    #     fast: Optional[Dict[str, torch.Tensor]] = None,
    #     position_ids: Optional[torch.Tensor] = None,
    # ) -> torch.Tensor:
    #     for blk in self.layers:
    #         x = blk(x, fast=fast, position_ids=position_ids)
    #     return self.final_norm(x)
    def forward_blocks(
        self,
        x: torch.Tensor,
        fast: Optional[Dict[str, torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
    ) -> Tuple[torch.Tensor, Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]:
        
        next_kvs = []
        for i, blk in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, kv = blk(x, fast=fast, position_ids=position_ids, past_key_value=past_kv)
            next_kvs.append(kv)
            
        return self.final_norm(x), tuple(next_kvs)


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        del attention_mask, output_attentions, output_hidden_states, cache_position, kwargs

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = bool(getattr(self.config, "use_cache", True)) if use_cache is None else bool(use_cache)

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            x = self.embed_tokens(input_ids)
        else:
            x = inputs_embeds
        x = self.embed_dropout(x)

        position_ids = self._normalize_position_ids(position_ids, x.size(0), x.size(1), x.device)
        hidden_states, next_past_key_values = self.forward_blocks(
            x,
            fast=None,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        if not use_cache:
            next_past_key_values = None

        if not return_dict:
            return (hidden_states, next_past_key_values)

        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=next_past_key_values)


class E2ETTTForCausalLM(E2ETTTPreTrainedModel):
    def __init__(self, config: E2ETTTConfig):
        super().__init__(config)
        self.model = E2ETTTModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    @staticmethod
    def _causal_lm_loss(logits: torch.Tensor, labels: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if labels is None:
            return None
        if logits.size(1) < 2:
            return logits.new_zeros(())

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
    @staticmethod
    def _chunked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # 此时 logits 和 labels 已经是对齐的，不需要切片
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_e2e_ttt: Optional[bool] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        return_fast_weights: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """Forward.

        - If labels is None: return logits (no inner loop).
        - If labels is not None and use_e2e_ttt=True: run chunked inner-loop updates on prime MLP weights.

        Note: HF Trainer expects (loss, logits, ...). We return CausalLMOutputWithPast.
        """

        del attention_mask, output_attentions, output_hidden_states, cache_position
        del cu_seqlens, return_fast_weights, kwargs

        cfg = self.config
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = bool(getattr(self.config, "use_cache", True)) if use_cache is None else bool(use_cache)
        if use_e2e_ttt is None:
            use_e2e_ttt = bool(cfg.use_e2e_ttt)

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            x = self.model.embed_tokens(input_ids)
        else:
            x = inputs_embeds
        x = self.model.embed_dropout(x)

        position_ids = self.model._normalize_position_ids(position_ids, x.size(0), x.size(1), x.device)

        if labels is None or not use_e2e_ttt:
            h, next_past_key_values = self.model.forward_blocks(
                x,
                fast=None,
                position_ids=position_ids,
                past_key_values=past_key_values,
            )
            if not use_cache:
                next_past_key_values = None
            logits = self.lm_head(h)
            loss = self._causal_lm_loss(logits, labels)
            if not return_dict:
                return ((loss, logits, next_past_key_values) if loss is not None else (logits, next_past_key_values))
            return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=next_past_key_values)

        # -----------------
        # E2E-TTT meta forward
        # -----------------
        x = x[:, :-1, :]          # 模型输入：去掉最后一个 token
        labels = labels[:, 1:] if labels is not None else None    # 预测目标：去掉第一个 token
        position_ids = position_ids[:, :-1]
        bsz, seqlen = x.shape[:2]
        chunk = int(cfg.mini_batch_size)
        # if seqlen % chunk != 0:
        #     raise ValueError(f"seqlen ({seqlen}) must be divisible by mini_batch_size ({chunk}) in this draft")

        fast = self.model.init_fast_weights()
        prime_keys = list(fast.keys())
        chunk_past_key_values = None

        total_loss = x.new_zeros(())
        total_tokens = 0
        import math
        steps = math.ceil(seqlen / chunk)

        # process sequentially over chunks (cannot fully parallelize; matches paper motivation)
        for i in range(steps):
            s = i * chunk
            e = min((i + 1) * chunk, seqlen)  # 防止越界，最后一个 chunk 会是 1536 到 2047
            current_chunk_len = e - s

            x_chunk = x[:, s:e, :]
            y_chunk = labels[:, s:e] if labels is not None else None 
            p_chunk = position_ids[:, s:e]

            # h_chunk = self.model.forward_blocks(x_chunk, fast=fast, position_ids=p_chunk)
            h_chunk, chunk_past_key_values = self.model.forward_blocks(
                x_chunk, fast=fast, position_ids=p_chunk, past_key_values=chunk_past_key_values
            )
            logits = self.lm_head(h_chunk)

            # loss_i = self._chunked_cross_entropy(logits, y_chunk)
            # total_loss = total_loss + loss_i
            loss_i = None
            if y_chunk is not None:
                # 假设你已经把交叉熵改为了不 shift 的版本
                loss_i = self._chunked_cross_entropy(logits, y_chunk)
                # [修复] 按照 chunk 实际长度加权，保证最终 Loss 在 token 级别是平均的
                total_loss = total_loss + loss_i * current_chunk_len
                total_tokens += current_chunk_len

            # inner update: SGD on fast weights only
            if prime_keys:
                for _ in range(int(cfg.inner_steps_per_chunk)):
                    grads = torch.autograd.grad(
                        loss_i,
                        [fast[k] for k in prime_keys],
                        create_graph=(not cfg.detach_fast_weights),
                        retain_graph=True,
                        allow_unused=True,
                    )

                    new_fast: Dict[str, torch.Tensor] = {}
                    for k, w, g in zip(prime_keys, [fast[k] for k in prime_keys], grads):
                        if g is None:
                            new_fast[k] = w
                        else:
                            new_fast[k] = w - cfg.inner_lr * g
                    fast = {**fast, **new_fast}

                    # if cfg.detach_fast_weights:
                    #     fast = {k: v.detach() for k, v in fast.items()}
                    if cfg.detach_fast_weights and chunk_past_key_values is not None:
                        chunk_past_key_values = tuple(
                            tuple(t.detach() for t in kv) for kv in chunk_past_key_values
                        )
        # if labels is not None:
        #     total_loss = total_loss / steps
        if labels is not None and total_tokens > 0:
            total_loss = total_loss / total_tokens

        # For logging convenience, return logits from the last chunk
        if not return_dict:
            return total_loss, logits, None
        return CausalLMOutputWithPast(loss=total_loss, logits=logits, past_key_values=None)
