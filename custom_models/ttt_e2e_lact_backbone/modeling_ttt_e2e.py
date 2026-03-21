# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from dataclasses import dataclass
import warnings
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union, Dict

import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.utils.deprecation import deprecate_kwarg

from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss
from fla.modules import GatedMLP as TransformerMLP
from fla.modules import RMSNorm
import torch.nn as nn

from .layer_e2e_swiglu import E2ESWIGLULayer
from .configuration_ttt_e2e import E2ETTTConfig

import torch.nn.functional as F
logger = logging.get_logger(__name__)

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack


LayerKV = Tuple[torch.Tensor, torch.Tensor]
LegacyCache = Tuple[Optional[LayerKV], ...]


def _normalize_cache_entry(entry: Any) -> Optional[LayerKV]:
    if entry is None:
        return None
    if isinstance(entry, dict):
        entry = entry.get("attn_state", None)
    if isinstance(entry, (tuple, list)) and len(entry) == 2 and torch.is_tensor(entry[0]) and torch.is_tensor(entry[1]):
        return entry[0], entry[1]
    return None


def _to_legacy_cache(past_key_values: Optional[Union[LegacyCache, List[Any], Any]]) -> Optional[LegacyCache]:
    if past_key_values is None:
        return None

    if isinstance(past_key_values, (tuple, list)):
        entries = tuple(past_key_values)
    else:
        to_legacy = getattr(past_key_values, "to_legacy_cache", None)
        if not callable(to_legacy):
            return None
        entries = tuple(to_legacy())

    if len(entries) == 2 and torch.is_tensor(entries[0]) and torch.is_tensor(entries[1]):
        return ((entries[0], entries[1]),)
    return tuple(_normalize_cache_entry(entry) for entry in entries)


def _has_past_tokens(past_key_values: Optional[Union[LegacyCache, List[Any], Any]]) -> bool:
    cache = _to_legacy_cache(past_key_values)
    if cache is None:
        return False
    for kv in cache:
        if kv is not None and int(kv[0].shape[1]) > 0:
            return True
    return False


def _silu(x):
    return F.silu(x)



# === 2. Core Network Components ===

@dataclass
class E2ETTTOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    fast_weights: Optional[Dict[str, torch.Tensor]] = None


class PrimeSwiGLU(nn.Module):
    """SwiGLU branch whose weights can be overridden by fast weights."""

    def __init__(self, config, name_prefix: str):
        super().__init__()
        self.name_prefix = name_prefix
        self.w0 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w1 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.w2 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(float(getattr(config, "resid_pdrop", 0.0)))

    def fast_param_names(self) -> List[str]:
        return [
            f"{self.name_prefix}.w0.weight",
            f"{self.name_prefix}.w1.weight",
            f"{self.name_prefix}.w2.weight",
        ]

    def forward(self, x: torch.Tensor, fast: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        if fast is None:
            h1 = F.linear(x, self.w0.weight)
            h2 = F.linear(x, self.w2.weight)
            out = F.linear(_silu(h1) * h2, self.w1.weight)
            return self.dropout(out)

        w0 = fast.get(f"{self.name_prefix}.w0.weight", self.w0.weight)
        w1 = fast.get(f"{self.name_prefix}.w1.weight", self.w1.weight)
        w2 = fast.get(f"{self.name_prefix}.w2.weight", self.w2.weight)
        
        h1 = F.linear(x, w0)
        h2 = F.linear(x, w2)
        act = _silu(h1) * h2
        out = F.linear(act, w1)

        return self.dropout(out)









class E2EBlock(nn.Module):

    def __init__(self, config, layer_idx: int, is_suffix: bool):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_suffix = bool(is_suffix) and bool(getattr(config, "use_e2e_ttt", False))

        self.attn_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(
            config.hidden_size, eps=config.norm_eps
        )
        self.attn = E2ESWIGLULayer(config=config, layer_idx=layer_idx, is_suffix=is_suffix)

        self.mlp_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(
            config.hidden_size, eps=config.norm_eps
        )
        self.mlp = TransformerMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu,
        )

        self.prime_mlp_norm: Optional[nn.Module] = None
        self.prime_mlp: Optional[PrimeSwiGLU] = None
        if self.is_suffix and bool(getattr(config, "two_mlp_per_block", True)):
            self.prime_mlp_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(
                config.hidden_size, eps=config.norm_eps
            )
            self.prime_mlp = PrimeSwiGLU(config, name_prefix=f"layers.{layer_idx}.prime_mlp")

    def prime_param_names(self) -> List[str]:
        if self.prime_mlp is None:
            return []
        return self.prime_mlp.fast_param_names()

    def forward(
        self,
        hidden_states: torch.Tensor,
        fast: Optional[Dict[str, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[LegacyCache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs: Unpack[Any],
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:

        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )

        if self.prime_mlp is not None:
            if self.config.fuse_norm:
                p_in, _ = self.prime_mlp_norm(hidden_states, residual, True)
            else:
                hidden_states = residual + hidden_states
                residual = hidden_states
                p_in = self.prime_mlp_norm(hidden_states)
            p = self.prime_mlp(p_in, fast=fast)
            hidden_states = residual + p


        if self.config.fuse_norm:
            hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states, **kwargs)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attentions,)

        if use_cache:
            outputs += (past_key_values,)

        return outputs


class E2EPreTrainedModel(PreTrainedModel):

    config_class = E2ETTTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["E2EBlock"]
    _supports_cache_class = False

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(
        self,
        module: nn.Module,
        rescale_prenorm_residual: bool = False,
        num_residuals_per_layer: int = 2,
    ):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif hasattr(module, "reset_parameters"):
            module.reset_parameters()

        if rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            p = None
            if hasattr(module, "o_proj"):
                p = module.o_proj.weight
            elif hasattr(module, "down_proj"):
                p = module.down_proj.weight
            if p is not None:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(
                        num_residuals_per_layer * self.config.num_hidden_layers
                    )

        if isinstance(module, E2ESWIGLULayer):
            # Keep layer-specific init resilient to implementation variants.
            if hasattr(module, "qk_scale"):
                nn.init.ones_(module.qk_scale)
            if hasattr(module, "qk_offset"):
                nn.init.zeros_(module.qk_offset)
            if hasattr(module, "w0") and isinstance(module.w0, torch.Tensor):
                nn.init.normal_(module.w0, mean=0.0, std=0.02)
            if hasattr(module, "w2") and isinstance(module.w2, torch.Tensor):
                nn.init.normal_(module.w2, mean=0.0, std=0.02)
            if hasattr(module, "w1") and isinstance(module.w1, torch.Tensor):
                nn.init.normal_(module.w1, mean=0.0, std=0.02)


class E2EModel(E2EPreTrainedModel):

    def __init__(self, config: E2ETTTConfig) -> E2EModel:
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        suffix_start = int(config.num_hidden_layers - config.suffix_len)
        self.suffix_start = max(0, suffix_start)

        self.embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                E2EBlock(config, layer_idx=i, is_suffix=(i >= self.suffix_start))
                for i in range(config.num_hidden_layers)
            ]
        )
        # for flame, full act_ckpt will throw error if we fuse the last layer norm,
        self.norm = (RMSNorm if config.last_layer_fuse_norm else nn.RMSNorm)(
            config.hidden_size, eps=config.norm_eps
        )

        self.gradient_checkpointing = False
        self.post_init()


    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value
    ### get fast weight #################################################################################
    def _collect_prime_params(self) -> List[str]:
        names: List[str] = []
        for layer in self.layers:
            names.extend(layer.prime_param_names())
        return names

    @staticmethod
    def _canonical_param_name(name: str) -> str:
        wrapper_segments = {"_orig_mod", "_fsdp_wrapped_module"}
        return ".".join(part for part in name.split(".") if part not in wrapper_segments)

    def init_fast_weights(self) -> Dict[str, torch.Tensor]:
        prime_names = set(self._collect_prime_params())
        if not prime_names:
            return {}

        fast: Dict[str, torch.Tensor] = {}
        for name, p in self.named_parameters():
            canonical_name = self._canonical_param_name(name)
            if canonical_name in prime_names and canonical_name not in fast:
                fast[canonical_name] = p
        return fast
        ##############################################################################
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[LegacyCache, List[Any], Any]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Unpack[Any],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if output_attentions:
            warnings.warn(
                "`TransformerModel` does not support output attention weights now, so `output_attentions` is set to `False`."
            )
            output_attentions = False
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = (
            use_cache
            if use_cache is not None
            else (self.config.use_cache if not self.training else False)
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        cache_state = _to_legacy_cache(past_key_values)
        if use_cache and cache_state is None:
            cache_state = tuple()

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        # embed positions
        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None
        next_cache = None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    cache_state,
                    output_attentions,
                    use_cache,
                    **kwargs,
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_values=cache_state,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    **kwargs,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                cache_state = layer_outputs[2 if output_attentions else 1]
                next_cache = cache_state

            if output_attentions:
                all_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_attns]
                if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_attns,
        )
    


    def forward_prefix_blocks(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
    ) -> torch.Tensor:
        if self.suffix_start <= 0:
            return x
        for i in range(self.suffix_start):
            layer = self.layers[i]
            layer_outputs = layer(
                x,
                attention_mask=attention_mask,
                past_key_values=None,
                use_cache=use_cache,
                attn_backend_override="flash",
            )
            x = layer_outputs[0]
        return x


    
    def forward_suffix_blocks(
        self,
        x: torch.Tensor,
        fast: Optional[Dict[str, torch.Tensor]],
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        past_key_values: Optional[LegacyCache] = None,
        *,
        ttt_step_idx: Optional[int] = None,
        ttt_num_steps: Optional[int] = None,
        ttt_chunk_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[LegacyCache]]:
        if self.suffix_start >= len(self.layers):
            return  self.norm(x), tuple()

        cache_state = _to_legacy_cache(past_key_values)
        if use_cache and cache_state is None:
            cache_state = tuple()
        for i in range(self.suffix_start, len(self.layers)):
            layer = self.layers[i]
            layer_outputs = layer(
                x,
                fast=fast,
                attention_mask=attention_mask,
                use_cache=use_cache,
                past_key_values=cache_state,
                attn_backend_override="sdpa",
                ttt_step_idx=ttt_step_idx,
                ttt_num_steps=ttt_num_steps,
                ttt_chunk_size=ttt_chunk_size,
            )
            x = layer_outputs[0]
            if use_cache:
                cache_state = layer_outputs[-1]
        next_kvs = cache_state if use_cache else None
        return  self.norm(x), next_kvs

class E2EForCausalLM(E2EPreTrainedModel, GenerationMixin):

    _tied_weights_keys = {"lm_head.weight": "model.embeddings.weight"} # ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = E2EModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

        # Initialize weights and apply final processing
        self.post_init()
        self.register_buffer("_inner_update_step", torch.zeros((), dtype=torch.long), persistent=False)
        self._inference_ttt_state: Optional[Dict[str, Any]] = None

    @staticmethod
    def _warn_once(msg: str):
        if hasattr(logger, "warning_once"):
            logger.warning_once(msg)
        else:
            logger.warning(msg)

    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @staticmethod
    def _detach_past(
        past_key_values: Optional[LegacyCache],
    ) -> Optional[LegacyCache]:
        if past_key_values is None:
            return None
        detached: List[Optional[LayerKV]] = []
        for kv in past_key_values:
            if kv is None:
                detached.append(None)
            else:
                detached.append((kv[0].detach(), kv[1].detach()))
        return tuple(detached)

    def _resolve_inner_optimizer(self) -> Tuple[str, float, float]:
        cfg = self.config
        optimizer_type = "sgd"
        lr = float(getattr(cfg, "inner_lr", 1e-3))
        clip_gradient = 0.0

        opt_cfg = getattr(cfg, "optimizer_inner", None)
        if opt_cfg is not None:
            if not isinstance(opt_cfg, dict):
                try:
                    opt_cfg = dict(opt_cfg)
                except Exception:
                    opt_cfg = None
                    self._warn_once("optimizer_inner is not a dict-like object; fallback to inner_lr + SGD.")
            if isinstance(opt_cfg, dict):
                optimizer_type = str(opt_cfg.get("optimizer_type", "sgd")).lower()
                lr = float(opt_cfg.get("lr", lr))
                clip_gradient = float(opt_cfg.get("clip_gradient", 0.0))

        if optimizer_type != "sgd":
            self._warn_once(
                f"optimizer_inner.optimizer_type={optimizer_type!r} is not implemented in v4; fallback to 'sgd'."
            )
            optimizer_type = "sgd"

        return optimizer_type, lr, clip_gradient

    def _compute_inner_lr(self, target_lr: float) -> float:
        cfg = self.config
        warmup_steps = int(getattr(cfg, "ilr_warmup_steps", 0))
        if warmup_steps <= 0:
            return target_lr

        ilr_init = float(getattr(cfg, "ilr_init", target_lr))
        step = int(self._inner_update_step.detach().item())
        progress = min(1.0, float(step + 1) / float(warmup_steps))
        return ilr_init + (target_lr - ilr_init) * progress

    @staticmethod
    def _clip_inner_grads(
        grads: Tuple[Optional[torch.Tensor], ...],
        max_norm: float,
    ) -> Tuple[Tuple[Optional[torch.Tensor], ...], Optional[torch.Tensor]]:
        if max_norm <= 0.0:
            return grads, None

        present_grads = [g for g in grads if g is not None]
        if not present_grads:
            return grads, None

        grad_norms = [g.norm(2) for g in present_grads]
        if len(grad_norms) == 1:
            total_norm = grad_norms[0]
        else:
            total_norm = torch.linalg.vector_norm(torch.stack(grad_norms), ord=2)

        scale = torch.clamp(
            torch.as_tensor(max_norm, device=total_norm.device, dtype=total_norm.dtype) / (total_norm + 1e-6),
            max=1.0,
        )
        clipped_grads = tuple(None if g is None else g * scale.to(dtype=g.dtype) for g in grads)
        return clipped_grads, total_norm

    def _causal_lm_loss(self, logits: torch.Tensor, labels: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if labels is None:
            return None
        if logits.size(1) < 2:
            return logits.new_zeros(())

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return self.loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

    def _reset_inference_ttt_state(self):
        self._inference_ttt_state = None

    def _init_inference_ttt_state(self, batch_size: int) -> Dict[str, Any]:
        fast = self.model.init_fast_weights()
        normalized_fast: Dict[str, torch.Tensor] = {}
        for k, w in fast.items():
            if w.requires_grad:
                normalized_fast[k] = w
            else:
                normalized_fast[k] = w.detach().requires_grad_(True)
        return {
            "batch_size": int(batch_size),
            "fast": normalized_fast,
            "prime_keys": list(normalized_fast.keys()),
            "suffix_past": tuple(),
            "pending_input_ids": None,
            "pending_attention_mask": None,
        }

    @staticmethod
    def _slice_new_attention_segment(
        attention_mask: Optional[torch.Tensor], token_len: int
    ) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None
        if token_len <= 0:
            return attention_mask[:, :0]
        return attention_mask[:, -token_len:]

    @staticmethod
    def _make_shift_labels(
        token_ids: torch.LongTensor,
        token_mask: Optional[torch.Tensor] = None,
    ) -> torch.LongTensor:
        labels = token_ids[:, 1:].clone()
        if token_mask is not None:
            valid = token_mask[:, 1:].to(device=labels.device, dtype=torch.bool)
            labels = labels.masked_fill(~valid, -100)
        return labels

    def _ensure_fast_requires_grad(self, fast: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        normalized: Dict[str, torch.Tensor] = {}
        for k, w in fast.items():
            if w.requires_grad:
                normalized[k] = w
            else:
                normalized[k] = w.detach().requires_grad_(True)
        return normalized

    def _inference_teacher_forcing_update(
        self,
        token_window: torch.LongTensor,
        token_window_mask: Optional[torch.Tensor],
        fast: Dict[str, torch.Tensor],
        prime_keys: List[str],
        inner_target_lr: float,
        inner_clip_gradient: float,
    ) -> Dict[str, torch.Tensor]:
        if not prime_keys or token_window.size(1) < 2:
            return fast

        fast = self._ensure_fast_requires_grad(fast)

        with torch.enable_grad():
            x = self.model.embeddings(token_window)
            prefix_output = self.model.forward_prefix_blocks(
                x,
                attention_mask=token_window_mask,
                use_cache=False,
            )
            x_chunk = prefix_output[:, :-1, :]
            mask_chunk = token_window_mask[:, :-1] if token_window_mask is not None else None
            h_chunk, _ = self.model.forward_suffix_blocks(
                x_chunk,
                fast=fast,
                attention_mask=mask_chunk,
                use_cache=False,
                past_key_values=None,
            )
            chunk_logits = self.lm_head(h_chunk)
            chunk_labels = self._make_shift_labels(token_window, token_window_mask)
            valid_count = chunk_labels.ne(-100).sum()
            if int(valid_count.detach().item()) <= 0:
                return fast

            chunk_loss = F.cross_entropy(
                chunk_logits.reshape(-1, chunk_logits.size(-1)),
                chunk_labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            loss_for_inner = chunk_loss / valid_count.clamp(min=1).float()

            fast_params = [fast[k] for k in prime_keys]
            grads = torch.autograd.grad(
                loss_for_inner,
                fast_params,
                create_graph=False,
                retain_graph=False,
                allow_unused=True,
            )
            lr = self._compute_inner_lr(inner_target_lr)
            grads, _ = self._clip_inner_grads(grads, inner_clip_gradient)

            updated: Dict[str, torch.Tensor] = {}
            for k, w, g in zip(prime_keys, fast_params, grads):
                if g is None:
                    w_new = w.detach()
                else:
                    w_new = (w - lr * g).detach()
                updated[k] = w_new.requires_grad_(True)

        with torch.no_grad():
            self._inner_update_step.add_(1)
        return {**fast, **updated}

    def _append_pending_observations(
        self,
        state: Dict[str, Any],
        input_ids: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor],
    ):
        if input_ids is None:
            return

        new_ids = input_ids.detach()
        new_mask = self._slice_new_attention_segment(attention_mask, new_ids.size(1))
        if new_mask is not None:
            new_mask = new_mask.detach()

        if state["pending_input_ids"] is None:
            state["pending_input_ids"] = new_ids
            state["pending_attention_mask"] = new_mask
            return

        state["pending_input_ids"] = torch.cat([state["pending_input_ids"], new_ids], dim=1)
        if state["pending_attention_mask"] is None:
            if new_mask is None:
                return
            old = torch.ones_like(state["pending_input_ids"][:, :-new_ids.size(1)])
            state["pending_attention_mask"] = torch.cat([old, new_mask], dim=1)
        elif new_mask is None:
            ones = torch.ones_like(new_ids, dtype=state["pending_attention_mask"].dtype, device=new_ids.device)
            state["pending_attention_mask"] = torch.cat([state["pending_attention_mask"], ones], dim=1)
        else:
            state["pending_attention_mask"] = torch.cat([state["pending_attention_mask"], new_mask], dim=1)

    def _consume_pending_windows(
        self,
        state: Dict[str, Any],
        chunk_size: int,
        inner_target_lr: float,
        inner_clip_gradient: float,
    ):
        if chunk_size <= 0:
            return

        pending_ids = state.get("pending_input_ids", None)
        pending_mask = state.get("pending_attention_mask", None)
        if pending_ids is None:
            return

        while pending_ids is not None and pending_ids.size(1) >= (chunk_size + 1):
            token_window = pending_ids[:, : chunk_size + 1]
            token_window_mask = (
                pending_mask[:, : chunk_size + 1] if pending_mask is not None else None
            )
            state["fast"] = self._inference_teacher_forcing_update(
                token_window=token_window,
                token_window_mask=token_window_mask,
                fast=state["fast"],
                prime_keys=state["prime_keys"],
                inner_target_lr=inner_target_lr,
                inner_clip_gradient=inner_clip_gradient,
            )
            pending_ids = pending_ids[:, chunk_size:]
            if pending_mask is not None:
                pending_mask = pending_mask[:, chunk_size:]

        state["pending_input_ids"] = pending_ids
        state["pending_attention_mask"] = pending_mask
    
    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor = None,
        past_key_values: Optional[Union[LegacyCache, List[Any], Any]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        logits_to_keep: Optional[int] = None,
        **kwargs,
    ):
        has_past = use_cache and _has_past_tokens(past_key_values)
        if (
            bool(getattr(self.config, "use_e2e_ttt", False))
            and bool(getattr(self.config, "enable_inference_ttt", True))
            and not has_past
        ):
            self._reset_inference_ttt_state()

        # only last token for `inputs_ids` if the `past_key_values` is not empty.
        if has_past:
            input_ids = input_ids[:, -1:]
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and not has_past:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard.
            # Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {"input_ids": input_ids.contiguous()}

        if logits_to_keep is not None:
            model_inputs["logits_to_keep"] = logits_to_keep

        model_inputs.update(
            {
                "past_key_values": _to_legacy_cache(past_key_values),
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "inference_ttt_generation": True,
            }
        )
        return model_inputs

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[LegacyCache, List[Any], Any]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_e2e_ttt: Optional[bool] = None,
        logits_to_keep: Optional[int] = 0,
        **kwargs: Unpack[Any],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        use_cache = (
            use_cache
            if use_cache is not None
            else (self.config.use_cache if not self.training else False)
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        cfg = self.config
        if use_e2e_ttt is None:
            use_e2e_ttt = bool(getattr(cfg, "use_e2e_ttt", False))

        debug_ttt_logs = bool(getattr(cfg, "debug_ttt_logs", False))
        debug_ttt_log_every = max(1, int(getattr(cfg, "debug_ttt_log_every", 50)))
        create_graph_for_inner = not bool(getattr(cfg, "detach_fast_weights", False))
        inner_optimizer_type, inner_target_lr, inner_clip_gradient = self._resolve_inner_optimizer()

        inner_steps = int(getattr(cfg, "inner_steps_per_chunk", 1))
        if inner_steps != 1:
            self._warn_once(
                f"inner_steps_per_chunk={inner_steps} is not implemented in v4; forcing to 1."
            )
            inner_steps = 1

        run_e2e_ttt_train = bool(use_e2e_ttt) and self.training
        run_e2e_ttt_infer = bool(use_e2e_ttt) and (not self.training)

        outputs = None
        if not run_e2e_ttt_train and not run_e2e_ttt_infer:
            self._reset_inference_ttt_state()
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

            hidden_states = outputs[0]
            fuse_linear_and_cross_entropy = self.config.fuse_cross_entropy and self.training
            if fuse_linear_and_cross_entropy:
                logits = None
            else:
                if logits_to_keep is not None and logits_to_keep > 0:
                    logits_input = hidden_states[:, -logits_to_keep:]
                else:
                    logits_input = hidden_states
                logits = self.lm_head(logits_input)

            loss = None
            if labels is not None:
                if getattr(self, "criterion", None) is None:
                    if fuse_linear_and_cross_entropy:
                        criterion = FusedLinearCrossEntropyLoss()
                    elif self.config.fuse_cross_entropy:
                        criterion = FusedCrossEntropyLoss(inplace_backward=True)
                    else:
                        criterion = nn.CrossEntropyLoss()
                else:
                    criterion = self.criterion
                # Enable model parallelism
                labels = labels.to(hidden_states.device)
                labels = torch.cat(
                    (
                        labels[..., 1:],
                        torch.full_like(labels[:, :1], criterion.ignore_index),
                    ),
                    1,
                )
                if fuse_linear_and_cross_entropy:
                    loss = criterion(
                        hidden_states, labels, self.lm_head.weight, self.lm_head.bias
                    )
                else:
                    loss = criterion(logits.view(labels.numel(), -1), labels.view(-1))

        elif run_e2e_ttt_train:
            self._reset_inference_ttt_state()
            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
            if input_ids is None and inputs_embeds is None:
                raise ValueError("You must specify either input_ids or inputs_embeds")

            x = self.model.embeddings(input_ids) if inputs_embeds is None else inputs_embeds
            total_seq_len = x.size(1)
            effective_seq_len = max(total_seq_len - 1, 0)
            bsz = x.size(0)
            labels = labels.to(x.device)
            total_loss_sum = x.new_zeros(())
            total_valid_tokens = x.new_zeros(())
            suffix_past = _to_legacy_cache(past_key_values)
            if use_cache and suffix_past is None:
                suffix_past = tuple()
            logits = x.new_zeros((bsz, 0, self.lm_head.out_features))
            chunk = int(cfg.mini_batch_size)
            if chunk <= 0:
                raise ValueError("mini_batch_size must be > 0")

            fast = self.model.init_fast_weights()
            prime_keys = list(fast.keys())

            steps = math.ceil(effective_seq_len / chunk) if effective_seq_len > 0 else 0


            prefix_output = self.model.forward_prefix_blocks(
                x,
                attention_mask=attention_mask,
                use_cache=use_cache,
            )
            if debug_ttt_logs:
                suffix_layers = len(self.model.layers) - int(self.model.suffix_start)
                suffix_layers_with_prime = sum(
                    int(getattr(layer, "prime_mlp", None) is not None)
                    for layer in self.model.layers[self.model.suffix_start :]
                )
                self._warn_once(
                    "[TTT-DEBUG] e2e_ttt on: "
                    f"suffix_start={self.model.suffix_start}, "
                    f"suffix_layers={suffix_layers}, "
                    f"suffix_layers_with_prime_mlp={suffix_layers_with_prime}, "
                    f"prime_keys={len(prime_keys)}, "
                    f"create_graph={create_graph_for_inner}, "
                    f"detach_fast_weights={bool(getattr(cfg, 'detach_fast_weights', False))}, "
                    f"chunk_size={chunk}, steps={steps}"
                )
                if not prime_keys:
                    logger.warning(
                        "[TTT-DEBUG] prime_keys is empty. Inner-loop updates for prime MLP are skipped."
                    )
                    
            all_logits_list = []
            for i in range(steps):
                s = i * chunk
                e = min((i + 1) * chunk, effective_seq_len)
                if e <= s:
                    continue

                x_chunk = prefix_output[:, s:e, :]
                y_chunk = labels[:, s + 1 : e + 1]
                mask_chunk = attention_mask[:, :e] if attention_mask is not None else None
                h_chunk, suffix_past = self.model.forward_suffix_blocks(
                    x_chunk,
                    fast=fast,
                    attention_mask=mask_chunk,
                    use_cache=use_cache,
                    past_key_values=suffix_past,
                    ttt_step_idx=i,
                    ttt_num_steps=steps,
                    ttt_chunk_size=chunk,
                )
                chunk_logits = self.lm_head(h_chunk)
                    
                if logits_to_keep > 0:
                    all_logits_list.append(chunk_logits.detach().to("cpu", non_blocking=True))

                valid_mask = y_chunk.ne(-100)
                valid_count = valid_mask.sum() 
                total_valid_tokens = total_valid_tokens + valid_count

                safe_valid_count = valid_count.clamp(min=1).float()

                flat_logits = chunk_logits.reshape(-1, chunk_logits.size(-1))
                flat_labels = y_chunk.reshape(-1)
                
                chunk_loss_sum = F.cross_entropy(
                    flat_logits,
                    flat_labels,
                    ignore_index=-100,
                    reduction="sum",
                )

                total_loss_sum = total_loss_sum + chunk_loss_sum

                loss_i = chunk_loss_sum / safe_valid_count

                loss_i = torch.where(valid_count > 0, loss_i, loss_i.new_zeros(()))
                should_log_chunk = debug_ttt_logs and (i % debug_ttt_log_every == 0 or i == (steps - 1))
                if should_log_chunk:
                    logger.warning(
                        "[TTT-DEBUG] chunk=%d/%d token_span=[%d,%d) valid_tokens=%d suffix_past=%s",
                        i + 1,
                        steps,
                        s,
                        e,
                        int(valid_count.detach().item()),
                        "yes" if suffix_past is not None else "no",
                    )

                if not prime_keys:
                    continue
                for _ in range(inner_steps):
                    fast_params = [fast[k] for k in prime_keys]
                    loss_for_inner = loss_i
                    grads = torch.autograd.grad(
                        loss_for_inner,
                        fast_params,
                        create_graph=create_graph_for_inner,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    if should_log_chunk:
                        grad_present = sum(1 for g in grads if g is not None)
                        grad_require_graph = sum(1 for g in grads if (g is not None and g.requires_grad))
                        sample = ", ".join(
                            f"{name.split('.')[-3]}.{name.split('.')[-2]}.{name.split('.')[-1]}="
                            f"{'none' if g is None else ('rg1' if g.requires_grad else 'rg0')}"
                            for name, g in zip(prime_keys[:6], grads[:6])
                        )
                        logger.warning(
                            "[TTT-DEBUG] prime_grad chunk=%d present=%d/%d require_grad=%d/%d "
                            "second_order_ready=%s create_graph=%s sample=[%s]",
                            i + 1,
                            grad_present,
                            len(grads),
                            grad_require_graph,
                            len(grads),
                            grad_require_graph > 0,
                            create_graph_for_inner,
                            sample,
                        )

                    lr = self._compute_inner_lr(inner_target_lr)
                    grads, grad_norm = self._clip_inner_grads(grads, inner_clip_gradient)
                    if should_log_chunk:
                        logger.warning(
                            "[TTT-DEBUG] inner_opt chunk=%d type=%s lr=%.6g ilr_init=%.6g warmup_steps=%d "
                            "clip_gradient=%.6g grad_norm=%s inner_step=%d",
                            i + 1,
                            inner_optimizer_type,
                            lr,
                            float(getattr(cfg, "ilr_init", inner_target_lr)),
                            int(getattr(cfg, "ilr_warmup_steps", 0)),
                            inner_clip_gradient,
                            "none" if grad_norm is None else f"{float(grad_norm.detach().item()):.6g}",
                            int(self._inner_update_step.detach().item()),
                        )

                    updated: Dict[str, torch.Tensor] = {}
                    for k, w, g in zip(prime_keys, fast_params, grads):
                        if g is None:
                            updated[k] = w
                        else:
                            updated[k] = w - lr * g
                    fast = {**fast, **updated}
                    with torch.no_grad():
                        self._inner_update_step.add_(1)

                    if bool(getattr(cfg, "detach_fast_weights", False)):
                        suffix_past = self._detach_past(suffix_past)

            if len(all_logits_list) > 0:
                logits_full = torch.cat(all_logits_list, dim=1)
            else:
                logits_full = logits
                
            if labels is None and logits_to_keep is not None and logits_to_keep > 0:
                logits = logits_full[:, -logits_to_keep:]
            else:
                logits = logits_full

            if total_valid_tokens > 0:
                # Divide by the tensor directly to keep it in the graph/device
                loss = total_loss_sum / total_valid_tokens.float()
            else:
                loss = total_loss_sum.new_zeros(()).requires_grad_(True)
        else:
            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
            if input_ids is None and inputs_embeds is None:
                raise ValueError("You must specify either input_ids or inputs_embeds")

            chunk = int(cfg.mini_batch_size)
            if chunk <= 0:
                raise ValueError("mini_batch_size must be > 0")

            has_past = use_cache and _has_past_tokens(past_key_values)

            # 这里只是内部运行时判断：
            # - eval + use_cache + no labels => 认为是 stateful generation/eval
            # - 否则就是单次 stateless eval
            is_stateful_eval = (not self.training) and bool(use_cache) and (labels is None)

            # 新的一轮 eval/generation 开始
            if not has_past:
                self._reset_inference_ttt_state()

            if is_stateful_eval:
                bsz = int((input_ids if input_ids is not None else inputs_embeds).size(0))
                if (
                    self._inference_ttt_state is None
                    or int(self._inference_ttt_state.get("batch_size", -1)) != bsz
                ):
                    self._inference_ttt_state = self._init_inference_ttt_state(batch_size=bsz)

                state = self._inference_ttt_state

                # 把本步新观察到的 token 追加到 pending buffer
                self._append_pending_observations(
                    state,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                # 每凑够 chunk+1 个 token，就做一次 teacher-forcing inner update
                self._consume_pending_windows(
                    state=state,
                    chunk_size=chunk,
                    inner_target_lr=inner_target_lr,
                    inner_clip_gradient=inner_clip_gradient,
                )

                fast = state["fast"]
                prime_keys = state["prime_keys"]

                suffix_past = state.get("suffix_past", None)
                if suffix_past is None:
                    suffix_past = _to_legacy_cache(past_key_values)
                    if suffix_past is None:
                        suffix_past = tuple()
            else:
                state = None
                fast = self.model.init_fast_weights()
                prime_keys = list(fast.keys())

                suffix_past = _to_legacy_cache(past_key_values)
                if suffix_past is None:
                    suffix_past = tuple()

            x = self.model.embeddings(input_ids) if inputs_embeds is None else inputs_embeds
            total_seq_len = x.size(1)

            prefix_output = self.model.forward_prefix_blocks(
                x,
                attention_mask=attention_mask,
                use_cache=use_cache,
            )

            all_logits_list = []
            for s in range(0, total_seq_len, chunk):
                e = min(s + chunk, total_seq_len)
                x_chunk = prefix_output[:, s:e, :]

                mask_chunk = attention_mask[:, :e] if attention_mask is not None else None

                h_chunk, suffix_past = self.model.forward_suffix_blocks(
                    x_chunk,
                    fast=fast,
                    attention_mask=mask_chunk,
                    use_cache=use_cache,
                    past_key_values=suffix_past,
                )
                chunk_logits = self.lm_head(h_chunk)
                all_logits_list.append(chunk_logits)

                # 对于非 stateful eval（例如一次性整句 eval / ppl eval），
                # 可以在当前 forward 内部按 chunk 做 teacher-forcing 更新
                has_lookahead = e < total_seq_len
                is_full_chunk = (e - s) == chunk
                if (
                    (not is_stateful_eval)
                    and input_ids is not None
                    and is_full_chunk
                    and has_lookahead
                ):
                    token_window = input_ids[:, s : e + 1]
                    token_window_mask = (
                        attention_mask[:, s : e + 1] if attention_mask is not None else None
                    )
                    fast = self._inference_teacher_forcing_update(
                        token_window=token_window,
                        token_window_mask=token_window_mask,
                        fast=fast,
                        prime_keys=prime_keys,
                        inner_target_lr=inner_target_lr,
                        inner_clip_gradient=inner_clip_gradient,
                    )

            logits_full = (
                torch.cat(all_logits_list, dim=1)
                if len(all_logits_list) > 0
                else x.new_zeros((x.size(0), 0, self.lm_head.out_features))
            )

            if logits_to_keep is not None and logits_to_keep > 0:
                logits = logits_full[:, -logits_to_keep:]
            else:
                logits = logits_full

            loss = None
            if labels is not None:
                labels = labels.to(logits_full.device)
                shift_labels = torch.cat(
                    (
                        labels[..., 1:],
                        torch.full_like(labels[:, :1], -100),
                    ),
                    dim=1,
                )
                loss = F.cross_entropy(
                    logits_full.reshape(-1, logits_full.size(-1)),
                    shift_labels.reshape(-1),
                    ignore_index=-100,
                )

            if is_stateful_eval and state is not None:
                state["fast"] = fast
                state["suffix_past"] = suffix_past
                self._inference_ttt_state = state
            else:
                self._reset_inference_ttt_state()

        if not return_dict:
            if outputs is None:
                output = (logits, suffix_past if use_cache else None)
            else:
                output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=(suffix_past if use_cache else None) if outputs is None else outputs.past_key_values,
            hidden_states=None if outputs is None else outputs.hidden_states,
            attentions=None if outputs is None else outputs.attentions,
        )