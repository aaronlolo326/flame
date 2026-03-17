# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from numbers import Integral
from dataclasses import dataclass
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union

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

    def ttt_meta(self) -> Dict[str, Any]:
        return {
            "layer_idx": self.layer_idx,
            "is_suffix": self.is_suffix,
            "has_prime_mlp": self.prime_mlp is not None,
            "prime_keys": self.prime_param_names(),
        }

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

        if self.prime_mlp is not None and fast is not None:
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

    def _default_e2e_ttt_group_specs(self, suffix_layers: int) -> List[Dict[str, Any]]:
        if suffix_layers <= 0:
            return []

        group_count = min(3, suffix_layers)
        base = suffix_layers // group_count
        rem = suffix_layers % group_count
        rel_groups: List[List[int]] = []
        start = 0
        for i in range(group_count):
            span = base + (1 if i < rem else 0)
            end = start + span
            rel_groups.append(list(range(start, end)))
            start = end

        mb = int(getattr(self.config, "mini_batch_size", 1))
        chunk_sizes = [mb, mb * 2, mb * 4]
        names = ["L2", "L1", "L0"]
        specs: List[Dict[str, Any]] = []
        for idx, rel_layers in enumerate(rel_groups):
            specs.append(
                {
                    "name": names[idx] if idx < len(names) else f"group_{idx}",
                    "layers": rel_layers,
                    "chunk_size": chunk_sizes[idx],
                    "lr_scale": 1.0,
                }
            )
        return specs

    def build_e2e_ttt_groups(self) -> List[Dict[str, Any]]:
        suffix_blocks = list(self.layers[self.suffix_start :])
        suffix_layers = len(suffix_blocks)
        if suffix_layers == 0:
            return []

        raw_specs = getattr(self.config, "e2e_ttt_groups", None)
        if raw_specs is None:
            raw_specs = self._default_e2e_ttt_group_specs(suffix_layers)

        if not isinstance(raw_specs, (list, tuple)) or len(raw_specs) == 0:
            raise ValueError("e2e_ttt_groups must be a non-empty list of group specs")

        covered: Set[int] = set()
        groups: List[Dict[str, Any]] = []

        for group_idx, raw_spec in enumerate(raw_specs):
            if isinstance(raw_spec, dict):
                spec = dict(raw_spec)
            else:
                try:
                    spec = dict(raw_spec)
                except Exception as exc:
                    raise ValueError(
                        f"e2e_ttt_groups[{group_idx}] must be dict-like, got {type(raw_spec).__name__}"
                    ) from exc

            name = str(spec.get("name", f"group_{group_idx}"))
            rel_layers_raw = spec.get("layers", None)
            if not isinstance(rel_layers_raw, (list, tuple)) or len(rel_layers_raw) == 0:
                raise ValueError(f"group '{name}' must provide non-empty 'layers'")

            chunk_size_raw = spec.get("chunk_size", None)
            if (
                chunk_size_raw is None
                or isinstance(chunk_size_raw, bool)
                or not isinstance(chunk_size_raw, Integral)
            ):
                raise ValueError(f"group '{name}' has invalid chunk_size={chunk_size_raw!r}")
            chunk_size = int(chunk_size_raw)
            if chunk_size <= 0:
                raise ValueError(f"group '{name}' chunk_size must be > 0, got {chunk_size}")

            try:
                lr_scale = float(spec.get("lr_scale", 1.0))
            except Exception as exc:
                raise ValueError(
                    f"group '{name}' lr_scale must be float-like, got {spec.get('lr_scale')!r}"
                ) from exc

            rel_layers: List[int] = []
            local_seen: Set[int] = set()
            for raw_rel_idx in rel_layers_raw:
                if isinstance(raw_rel_idx, bool) or not isinstance(raw_rel_idx, Integral):
                    raise ValueError(f"group '{name}' has invalid layer index {raw_rel_idx!r}")
                rel_idx = int(raw_rel_idx)
                if rel_idx < 0 or rel_idx >= suffix_layers:
                    raise ValueError(
                        f"group '{name}' layer index out of range: {rel_idx}, expected [0, {suffix_layers - 1}]"
                    )
                if rel_idx in local_seen:
                    raise ValueError(f"group '{name}' has duplicate layer index {rel_idx}")
                if rel_idx in covered:
                    raise ValueError(f"group '{name}' overlaps with previous groups at layer index {rel_idx}")
                local_seen.add(rel_idx)
                covered.add(rel_idx)
                rel_layers.append(rel_idx)

            if not rel_layers:
                raise ValueError(f"group '{name}' cannot be empty")

            abs_layers: List[int] = []
            keys: List[str] = []
            for rel_idx in rel_layers:
                block = suffix_blocks[rel_idx]
                abs_layers.append(int(getattr(block, "layer_idx", self.suffix_start + rel_idx)))
                keys.extend(block.prime_param_names())

            groups.append(
                {
                    "name": name,
                    "rel_layers": rel_layers,
                    "abs_layers": abs_layers,
                    "chunk_size": chunk_size,
                    "lr_scale": lr_scale,
                    "keys": keys,
                }
            )

        missing = sorted(set(range(suffix_layers)) - covered)
        if missing:
            raise ValueError(
                f"e2e_ttt_groups must strictly cover all suffix relative layers; missing indices: {missing}"
            )

        return groups

    def build_suffix_layer_ttt_schedule(self) -> List[Dict[str, Any]]:
        suffix_layers = len(self.layers) - int(self.suffix_start)
        if suffix_layers <= 0:
            return []

        groups = self.build_e2e_ttt_groups()
        schedule: List[Optional[Dict[str, Any]]] = [None] * suffix_layers
        for group in groups:
            chunk_size = int(group["chunk_size"])
            for rel_idx in group["rel_layers"]:
                rel_idx = int(rel_idx)
                if rel_idx < 0 or rel_idx >= suffix_layers:
                    raise ValueError(
                        f"group '{group['name']}' has invalid rel layer index {rel_idx} for suffix_layers={suffix_layers}"
                    )
                schedule[rel_idx] = {
                    "group_name": str(group["name"]),
                    "chunk_size": chunk_size,
                }

        missing = [idx for idx, meta in enumerate(schedule) if meta is None]
        if missing:
            raise ValueError(
                f"suffix layer schedule is incomplete. Missing relative indices: {missing}"
            )
        return [meta for meta in schedule if meta is not None]
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
        global_step_idx: Optional[int] = None,
        global_token_start: Optional[int] = None,
        base_chunk_size: Optional[int] = None,
        total_effective_seq_len: Optional[int] = None,
        ttt_step_idx: Optional[int] = None,
        ttt_num_steps: Optional[int] = None,
        ttt_chunk_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[LegacyCache]]:
        if self.suffix_start >= len(self.layers):
            return  self.norm(x), tuple()

        # Backward-compatible adapter: if callers still use old args, infer the new global clock metadata.
        if base_chunk_size is None and ttt_chunk_size is not None:
            base_chunk_size = int(ttt_chunk_size)
        if global_step_idx is None and ttt_step_idx is not None:
            global_step_idx = int(ttt_step_idx)
        if global_token_start is None:
            if global_step_idx is not None and base_chunk_size is not None:
                global_token_start = int(global_step_idx) * int(base_chunk_size)
            elif ttt_step_idx is not None and ttt_chunk_size is not None:
                global_token_start = int(ttt_step_idx) * int(ttt_chunk_size)
            else:
                global_token_start = 0
        if base_chunk_size is None:
            base_chunk_size = int(x.size(1))
        if global_step_idx is None:
            global_step_idx = int(global_token_start) // max(int(base_chunk_size), 1)

        chunk_len = int(x.size(1))
        is_last_global_step = False
        if total_effective_seq_len is not None:
            is_last_global_step = (
                int(global_token_start) + chunk_len >= int(total_effective_seq_len)
            )
        elif ttt_num_steps is not None:
            is_last_global_step = int(global_step_idx) >= int(ttt_num_steps) - 1

        layer_schedule = self.build_suffix_layer_ttt_schedule()
        cache_state = _to_legacy_cache(past_key_values)
        if use_cache and cache_state is None:
            cache_state = tuple()
        for i in range(self.suffix_start, len(self.layers)):
            layer = self.layers[i]
            rel_idx = i - self.suffix_start
            layer_meta = layer_schedule[rel_idx]
            layer_chunk_size = int(layer_meta["chunk_size"])
            layer_chunk_idx = int(global_token_start) // layer_chunk_size
            layer_chunk_offset = int(global_token_start) % layer_chunk_size
            layer_is_chunk_start = layer_chunk_offset == 0
            layer_is_boundary = (layer_chunk_offset + chunk_len >= layer_chunk_size) or is_last_global_step
            layer_num_steps = None
            if total_effective_seq_len is not None:
                layer_num_steps = math.ceil(int(total_effective_seq_len) / layer_chunk_size)
            elif ttt_num_steps is not None:
                layer_num_steps = int(ttt_num_steps)

            layer_outputs = layer(
                x,
                fast=fast,
                attention_mask=attention_mask,
                use_cache=use_cache,
                past_key_values=cache_state,
                attn_backend_override="sdpa",
                ttt_step_idx=layer_chunk_idx,
                ttt_num_steps=layer_num_steps,
                ttt_chunk_size=layer_chunk_size,
                ttt_chunk_offset=layer_chunk_offset,
                ttt_is_chunk_start=layer_is_chunk_start,
                ttt_is_boundary=layer_is_boundary,
                ttt_base_chunk_size=int(base_chunk_size),
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
        self._inner_update_steps_by_group: Dict[str, torch.Tensor] = {}

        # Initialize weights and apply final processing
        self.post_init()
        # self.register_buffer("_inner_update_step", torch.zeros((), dtype=torch.long), persistent=False)

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

        
    def _get_group_inner_step(self, group_name: str, device: torch.device) -> torch.Tensor:
        step = self._inner_update_steps_by_group.get(group_name, None)
        if step is None:
            step = torch.zeros((), dtype=torch.long, device=device)
            self._inner_update_steps_by_group[group_name] = step
        elif step.device != device:
            step = step.to(device=device)
            self._inner_update_steps_by_group[group_name] = step
        return step
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
    
    def _compute_inner_lr(self, target_lr: float, *, group_name: Optional[str] = None, device: Optional[torch.device] = None) -> float:
        cfg = self.config
        warmup_steps = int(getattr(cfg, "ilr_warmup_steps", 0))
        if warmup_steps <= 0:
            return target_lr

        ilr_init = float(getattr(cfg, "ilr_init", target_lr))

        if group_name is None:
            step = 0
        else:
            if device is None:
                raise ValueError("device must be provided when group_name is used")
            step_tensor = self._get_group_inner_step(group_name, device)
            step = int(step_tensor.detach().item())

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
        past_key_values: Optional[Union[LegacyCache, List[Any], Any]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        logits_to_keep: Optional[int] = None,
        **kwargs,
    ):
        
        # only last token for `inputs_ids` if the `past_key_values` is not empty.
        if use_cache and _has_past_tokens(past_key_values):
            input_ids = input_ids[:, -1:]
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and not _has_past_tokens(past_key_values):
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
        use_cache = True
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

        run_e2e_ttt = bool(use_e2e_ttt) and labels is not None
        outputs = None
        if not run_e2e_ttt:
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
            logits = (
                None
                if fuse_linear_and_cross_entropy
                else self.lm_head(hidden_states[:, -logits_to_keep:])
            )

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

                    
        else:
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
            groups = self.model.build_e2e_ttt_groups()
            if not groups:
                raise ValueError("No suffix group available for e2e_ttt. Check suffix_len and e2e_ttt_groups.")

            chunk_sizes = [int(g["chunk_size"]) for g in groups]
            base_chunk = chunk_sizes[0]
            for chunk_size in chunk_sizes[1:]:
                base_chunk = math.gcd(base_chunk, chunk_size)
            if base_chunk <= 0:
                raise ValueError(f"Invalid base_chunk={base_chunk}, chunk_sizes={chunk_sizes}")

            for g in groups:
                if int(g["chunk_size"]) % base_chunk != 0:
                    raise ValueError(
                        f"group '{g['name']}' chunk_size={g['chunk_size']} is not divisible by base_chunk={base_chunk}"
                    )
                period = int(g["chunk_size"]) // base_chunk
                if period <= 0:
                    raise ValueError(
                        f"group '{g['name']}' has invalid period={period} from chunk_size={g['chunk_size']}"
                    )
                g["period"] = period
                g["loss_sum"] = x.new_zeros(())
                g["valid_tokens"] = x.new_zeros(())
                g["warned_no_keys"] = False

            fast = self.model.init_fast_weights()
            fast_keys = set(fast.keys())
            for g in groups:
                original_keys = list(g["keys"])
                missing_keys = [k for k in original_keys if k not in fast_keys]
                if missing_keys:
                    self._warn_once(
                        f"[TTT-DEBUG] group={g['name']} has {len(missing_keys)} unresolved prime keys."
                    )
                g["keys"] = [k for k in original_keys if k in fast_keys]

            steps = math.ceil(effective_seq_len / base_chunk) if effective_seq_len > 0 else 0


            prefix_output = self.model.forward_prefix_blocks(
                x,
                attention_mask=attention_mask,
                # use_cache=use_cache,
                use_cache=False,
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
                    f"create_graph={create_graph_for_inner}, "
                    f"detach_fast_weights={bool(getattr(cfg, 'detach_fast_weights', False))}, "
                    f"base_chunk={base_chunk}, steps={steps}, groups={len(groups)}"
                )
                for g in groups:
                    logger.warning(
                        "[TTT-DEBUG] group=%s rel_layers=%s abs_layers=%s chunk_size=%d period=%d lr_scale=%.6g keys=%d",
                        g["name"],
                        g["rel_layers"],
                        g["abs_layers"],
                        int(g["chunk_size"]),
                        int(g["period"]),
                        float(g["lr_scale"]),
                        len(g["keys"]),
                    )
                    
            all_logits_list = []
            for i in range(steps):
                s = i * base_chunk
                e = min((i + 1) * base_chunk, effective_seq_len)
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
                    global_step_idx=i,
                    global_token_start=s,
                    base_chunk_size=base_chunk,
                    total_effective_seq_len=effective_seq_len,
                )
                chunk_logits = self.lm_head(h_chunk)
                    
                if logits_to_keep > 0:
                    all_logits_list.append(chunk_logits.detach().to("cpu", non_blocking=True))

                valid_count = y_chunk.ne(-100).sum()
                total_valid_tokens = total_valid_tokens + valid_count

                flat_logits = chunk_logits.reshape(-1, chunk_logits.size(-1))
                flat_labels = y_chunk.reshape(-1)
                
                chunk_loss_sum = F.cross_entropy(
                    flat_logits,
                    flat_labels,
                    ignore_index=-100,
                    reduction="sum",
                )

                total_loss_sum = total_loss_sum + chunk_loss_sum

                for g in groups:
                    g["loss_sum"] = g["loss_sum"] + chunk_loss_sum
                    g["valid_tokens"] = g["valid_tokens"] + valid_count

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

                for g in groups:
                    hit_boundary = ((i + 1) % int(g["period"]) == 0) or (e == effective_seq_len)
                    if not hit_boundary:
                        continue

                    safe_valid_count = g["valid_tokens"].clamp(min=1).float()
                    loss_for_inner = g["loss_sum"] / safe_valid_count
                    loss_for_inner = torch.where(g["valid_tokens"] > 0, loss_for_inner, loss_for_inner.new_zeros(()))

                    if not g["keys"]:
                        if debug_ttt_logs and not bool(g["warned_no_keys"]):
                            logger.warning(
                                "[TTT-DEBUG] group=%s has no prime keys. Inner update is skipped.",
                                g["name"],
                            )
                            g["warned_no_keys"] = True
                        g["loss_sum"] = x.new_zeros(())
                        g["valid_tokens"] = x.new_zeros(())
                        continue

                    for _ in range(inner_steps):
                        fast_params = [fast[k] for k in g["keys"]]
                        grads = torch.autograd.grad(
                            loss_for_inner,
                            fast_params,
                            create_graph=create_graph_for_inner,
                            retain_graph=True,
                            allow_unused=True,
                        )
                        if should_log_chunk:
                            grad_present = sum(1 for grad in grads if grad is not None)
                            grad_require_graph = sum(
                                1 for grad in grads if (grad is not None and grad.requires_grad)
                            )
                            sample = ", ".join(
                                f"{name.split('.')[-3]}.{name.split('.')[-2]}.{name.split('.')[-1]}="
                                f"{'none' if grad is None else ('rg1' if grad.requires_grad else 'rg0')}"
                                for name, grad in zip(g["keys"][:6], grads[:6])
                            )
                            logger.warning(
                                "[TTT-DEBUG] prime_grad chunk=%d group=%s present=%d/%d require_grad=%d/%d "
                                "second_order_ready=%s create_graph=%s sample=[%s] group_valid=%d",
                                i + 1,
                                g["name"],
                                grad_present,
                                len(grads),
                                grad_require_graph,
                                len(grads),
                                grad_require_graph > 0,
                                create_graph_for_inner,
                                sample,
                                int(g["valid_tokens"].detach().item()),
                            )

                        group_step = self._get_group_inner_step(g["name"], x.device)
                        lr = self._compute_inner_lr(
                            inner_target_lr,
                            group_name=g["name"],
                            device=x.device,
                        ) * float(g["lr_scale"])
                        grads, grad_norm = self._clip_inner_grads(grads, inner_clip_gradient)
                        if should_log_chunk:
                            logger.warning(
                                "[TTT-DEBUG] inner_opt chunk=%d group=%s type=%s lr=%.6g ilr_init=%.6g warmup_steps=%d "
                                "clip_gradient=%.6g grad_norm=%s inner_step=%d",
                                i + 1,
                                g["name"],
                                inner_optimizer_type,
                                lr,
                                float(getattr(cfg, "ilr_init", inner_target_lr)),
                                int(getattr(cfg, "ilr_warmup_steps", 0)),
                                inner_clip_gradient,
                                "none" if grad_norm is None else f"{float(grad_norm.detach().item()):.6g}",
                                int(group_step.detach().item()),
                            )

                        updated: Dict[str, torch.Tensor] = {}
                        for k, w, grad in zip(g["keys"], fast_params, grads):
                            if grad is None:
                                updated[k] = w
                            else:
                                updated[k] = w - lr * grad
                        fast = {**fast, **updated}
                        with torch.no_grad():
                            group_step.add_(1)

                        if bool(getattr(cfg, "detach_fast_weights", False)):
                            suffix_past = self._detach_past(suffix_past)

                    g["loss_sum"] = x.new_zeros(())
                    g["valid_tokens"] = x.new_zeros(())

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

        if not return_dict:
            if outputs is None:
                output = (logits, suffix_past)
            else:
                output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=suffix_past if outputs is None else outputs.past_key_values,
            hidden_states=None if outputs is None else outputs.hidden_states,
            attentions=None if outputs is None else outputs.attentions,
        )
