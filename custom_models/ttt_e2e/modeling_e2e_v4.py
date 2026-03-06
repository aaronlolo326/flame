# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.utils.deprecation import deprecate_kwarg

from fla.modules import GatedMLP as TransformerMLP
from .configuration_ttt_e2e import E2ETTTConfig
from .layer_e2e_swiglu import E2ESWIGLULayer
from fla.modules import RMSNorm

logger = logging.get_logger(__name__)

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


class E2ETTTBlock(nn.Module):
    def __init__(self, config, layer_idx: int, is_suffix: bool):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_suffix = bool(is_suffix)

        self.attn_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(
            config.hidden_size, eps=config.norm_eps
        )
        self.attn = E2ESWIGLULayer(config, layer_idx=layer_idx, is_suffix=is_suffix) 

        mlp_hidden_act = str(getattr(config, "hidden_act", "swish")).lower()
        if mlp_hidden_act == "silu":
            mlp_hidden_act = "swish"

        self.mlp_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(
            config.hidden_size, eps=config.norm_eps
        )
        # Assume TransformerMLP internals are safe.
        self.mlp = TransformerMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=mlp_hidden_act,
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
        x: torch.Tensor,
        fast: Optional[Dict[str, torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_e2e_ttt_context: bool = False,
        attn_backend_override: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        # Align block flow with a standard prenorm transformer:
        # attn_norm -> attn -> residual -> mlp_norm -> mlp -> residual
        residual = x
        x_norm = self.attn_norm(x)
        a, next_kv = self.attn(
            x_norm,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_e2e_ttt_context=use_e2e_ttt_context,
            attn_backend_override=attn_backend_override,
        )

        if self.prime_mlp is not None:
            if self.config.fuse_norm:
                p_in, _ = self.prime_mlp_norm(a, residual, True)
            else:
                x = residual + a
                p_in = self.prime_mlp_norm(x)
            p = self.prime_mlp(p_in, fast=fast)
            a = a + p

        if self.config.fuse_norm:
            m_in, residual = self.mlp_norm(a, residual, True)
        else:
            x = residual + a
            residual = x
            m_in = self.mlp_norm(x)
        m = self.mlp(m_in)
        x = residual + m

        return x, next_kv


class E2ETTTPreTrainedModel(PreTrainedModel):
    config_class = E2ETTTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["E2ETTTBlock"]


class E2ETTTModel(E2ETTTPreTrainedModel):
    def __init__(self, config: E2ETTTConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.embed_dropout = nn.Dropout(float(getattr(config, "embd_pdrop", 0.0)))

        suffix_start = int(config.num_hidden_layers - config.suffix_len)
        self.suffix_start = max(0, suffix_start)

        self.layers = nn.ModuleList(
            [
                E2ETTTBlock(config, layer_idx=i, is_suffix=(i >= self.suffix_start))
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = (RMSNorm if config.last_layer_fuse_norm else nn.RMSNorm)(
            config.hidden_size, eps=config.norm_eps
        )

        self.gradient_checkpointing = False

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value


    ### get fast weight ###
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
        ##########################


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
                raise ValueError(f"position_ids last dim {position_ids.size(-1)} does not match seq_len {seq_len}")
            if position_ids.size(0) == 1 and batch_size > 1:
                position_ids = position_ids.expand(batch_size, -1)
            elif position_ids.size(0) != batch_size:
                raise ValueError(f"position_ids batch dim {position_ids.size(0)} does not match batch_size {batch_size}")
        else:
            raise ValueError(f"position_ids must be rank-1 or rank-2, got shape {tuple(position_ids.shape)}")

        return position_ids.to(device=device, dtype=torch.long)

    def _forward_layer_range(
        self,
        x: torch.Tensor,
        layer_start: int,
        layer_end: int,
        fast: Optional[Dict[str, torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        use_e2e_ttt_context: bool = False,
        attn_backend_override: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]:
        next_kvs: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for local_idx, layer_idx in enumerate(range(layer_start, layer_end)):
            blk = self.layers[layer_idx]
            past_kv = None
            if past_key_values is not None and local_idx < len(past_key_values):
                past_kv = past_key_values[local_idx]
            x, kv = blk(
                x,
                fast=fast,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_value=past_kv,
                use_e2e_ttt_context=use_e2e_ttt_context,
                attn_backend_override=attn_backend_override,
            )
            next_kvs.append(kv)

        return x, tuple(next_kvs)

    def forward_prefix_blocks(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        use_e2e_ttt_context: bool = True,
    ) -> torch.Tensor:
        if self.suffix_start <= 0:
            return x
        x, _ = self._forward_layer_range(
            x,
            layer_start=0,
            layer_end=self.suffix_start,
            fast=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            use_e2e_ttt_context=use_e2e_ttt_context,
            attn_backend_override="flash",
        )
        return x

    def forward_suffix_blocks(
        self,
        x: torch.Tensor,
        fast: Optional[Dict[str, torch.Tensor]],
        position_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        use_e2e_ttt_context: bool = True,
    ) -> Tuple[torch.Tensor, Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]:
        if self.suffix_start >= len(self.layers):
            return  self.norm(x), tuple()

        x, next_kvs = self._forward_layer_range(
            x,
            layer_start=self.suffix_start,
            layer_end=len(self.layers),
            fast=fast,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_e2e_ttt_context=use_e2e_ttt_context,
            attn_backend_override="sdpa" if use_e2e_ttt_context else None,
        )
        return  self.norm(x), next_kvs

    def forward_blocks(
        self,
        x: torch.Tensor,
        fast: Optional[Dict[str, torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        use_e2e_ttt_context: bool = False,
        attn_backend_override: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]:
        x, next_kvs = self._forward_layer_range(
            x,
            layer_start=0,
            layer_end=len(self.layers),
            fast=fast,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_e2e_ttt_context=use_e2e_ttt_context,
            attn_backend_override=attn_backend_override,
        )
        return self.norm(x), next_kvs

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
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        del cache_position, kwargs

        if output_attentions:
            logger.warning_once(
                "`E2ETTTModel` does not return attention weights. Setting `output_attentions=False`."
            )
            output_attentions = False

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = bool(getattr(self.config, "use_cache", True)) if use_cache is None else bool(use_cache)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must specify either input_ids or inputs_embeds")

        x = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        x = self.embed_dropout(x)

        position_ids = self._normalize_position_ids(position_ids, x.size(0), x.size(1), x.device)

        all_hidden_states = () if output_hidden_states else None
        next_cache = None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (x,)

            past_kv = None
            if past_key_values is not None and i < len(past_key_values):
                past_kv = past_key_values[i]

            x, kv = layer(
                x,
                fast=None,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_value=past_kv,
                use_e2e_ttt_context=False,
                attn_backend_override=str(getattr(self.config, "attn_backend", "flash")),
            )
            if use_cache:
                if next_cache is None:
                    next_cache = []
                next_cache.append(kv)

        x = self.norm(x)

        if output_hidden_states:
            all_hidden_states += (x,)

        next_cache_tuple = tuple(next_cache) if (use_cache and next_cache is not None) else None

        if not return_dict:
            return tuple(v for v in [x, next_cache_tuple, all_hidden_states] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=x,
            past_key_values=next_cache_tuple,
            hidden_states=all_hidden_states,
            attentions=None,
        )


class E2ETTTForCausalLM(E2ETTTPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: E2ETTTConfig):
        super().__init__(config)
        self.model = E2ETTTModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        self.post_init()
        # Used for inner-loop LR warmup; non-persistent so checkpoints stay backward compatible.
        self.register_buffer("_inner_update_step", torch.zeros((), dtype=torch.long), persistent=False)

    @staticmethod
    def _warn_once(msg: str):
        if hasattr(logger, "warning_once"):
            logger.warning_once(msg)
        else:
            logger.warning(msg)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

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
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]],
    ) -> Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]:
        if past_key_values is None:
            return None
        return tuple(tuple(t.detach() for t in kv) for kv in past_key_values)

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

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        logits_to_keep: Optional[int] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if past_key_values is not None and len(past_key_values) > 0 and use_cache:
            input_ids = input_ids[:, -1:]
            if position_ids is not None:
                if position_ids.dim() == 2:
                    position_ids = position_ids[:, -1:]
                elif position_ids.dim() == 1:
                    position_ids = position_ids[-1:]

        if inputs_embeds is not None and (past_key_values is None or len(past_key_values) == 0):
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}

        if logits_to_keep is not None:
            model_inputs["logits_to_keep"] = logits_to_keep

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }
        )
        return model_inputs

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
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
        logits_to_keep: Optional[int] = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        del output_attentions, output_hidden_states, cache_position
        del cu_seqlens, kwargs

        cfg = self.config
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = bool(getattr(self.config, "use_cache", True)) if use_cache is None else bool(use_cache)
        if use_e2e_ttt is None:
            use_e2e_ttt = bool(getattr(cfg, "use_e2e_ttt", False))

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must specify either input_ids or inputs_embeds")

        x = self.model.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        x = self.model.embed_dropout(x)
        position_ids = self.model._normalize_position_ids(position_ids, x.size(0), x.size(1), x.device)

        if labels is None or not use_e2e_ttt:
            h, next_past_key_values = self.model.forward_blocks(
                x,
                fast=None,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_e2e_ttt_context=False,
                attn_backend_override=str(getattr(cfg, "attn_backend", "flash")),
            )
            if not use_cache:
                next_past_key_values = None

            logits_full = F.linear(h, self.lm_head.weight, self.lm_head.bias)
            if labels is None and logits_to_keep is not None:
                logits = logits_full[:, -logits_to_keep:]
            else:
                logits = logits_full
            loss = self._causal_lm_loss(logits_full, labels)

            if not return_dict:
                output = (logits, next_past_key_values)
                if return_fast_weights:
                    output = output + (None,)
                return ((loss,) + output) if loss is not None else output

            return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=next_past_key_values)

        if getattr(cfg, "inner_param_filter", "prime_mlp") != "prime_mlp":
            self._warn_once(
                f"inner_param_filter={cfg.inner_param_filter!r} is not implemented in v4; fallback to 'prime_mlp'."
            )

        inner_steps = int(getattr(cfg, "inner_steps_per_chunk", 1))
        if inner_steps != 1:
            self._warn_once(
                f"inner_steps_per_chunk={inner_steps} is not implemented in v4; forcing to 1."
            )
            inner_steps = 1
        debug_ttt_logs = bool(getattr(cfg, "debug_ttt_logs", False))
        debug_ttt_log_every = max(1, int(getattr(cfg, "debug_ttt_log_every", 50)))
        create_graph_for_inner = not bool(getattr(cfg, "detach_fast_weights", False))
        inner_optimizer_type, inner_target_lr, inner_clip_gradient = self._resolve_inner_optimizer()

        x = x[:, :-1, :]
        labels = labels[:, 1:]
        position_ids = position_ids[:, :-1]
        if attention_mask is not None:
            if attention_mask.dim() != 2:
                raise ValueError(
                    "attention_mask must have shape [batch_size, seq_len], "
                    f"got {tuple(attention_mask.shape)}"
                )
            attention_mask = attention_mask[:, :-1]

        bsz, seqlen = x.shape[:2]
        chunk = int(cfg.mini_batch_size)
        if chunk <= 0:
            raise ValueError("mini_batch_size must be > 0")

        fast = self.model.init_fast_weights()
        prime_keys = list(fast.keys())

        prefix_outputs = self.model.forward_prefix_blocks(
            x,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_e2e_ttt_context=True,
        )

        suffix_past: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None
        total_loss_sum = x.new_zeros(())
        total_valid_tokens = x.new_zeros(()) # Initialize as a Tensor instead of int 0
        
        # This initial logits tensor is overwritten later, but keeps a safe fallback when steps == 0.
        logits = x.new_zeros((bsz, 1, self.lm_head.out_features))

        steps = math.ceil(seqlen / chunk)
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
                
        # 1. Initialize the logits collection list.
        all_logits_list = []
        
        for i in range(steps):
            s = i * chunk
            e = min((i + 1) * chunk, seqlen)
            if e <= s:
                continue

            x_chunk = prefix_outputs[:, s:e, :]
            y_chunk = labels[:, s:e]
            p_chunk = position_ids[:, s:e]
            mask_chunk = attention_mask[:, :e] if attention_mask is not None else None
            h_chunk, suffix_past = self.model.forward_suffix_blocks(
                x_chunk,
                fast=fast,
                position_ids=p_chunk,
                attention_mask=mask_chunk,
                past_key_values=suffix_past,
                use_e2e_ttt_context=True,
            )

            chunk_logits = F.linear(h_chunk, self.lm_head.weight, self.lm_head.bias)
            
            all_logits_list.append(chunk_logits)

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
        # ==========================================================

        if total_valid_tokens > 0:
            # Divide by the tensor directly to keep it in the graph/device
            total_loss = total_loss_sum / total_valid_tokens.float()
        else:
            total_loss = total_loss_sum.new_zeros(()).requires_grad_(True)

        if not return_dict:
            output = (total_loss, logits, None)
            if return_fast_weights:
                output = output + (fast,)
            return output

        if return_fast_weights:
            self._warn_once("return_fast_weights=True is only returned in tuple output mode (return_dict=False).")

        return CausalLMOutputWithPast(loss=total_loss, logits=logits, past_key_values=None)
