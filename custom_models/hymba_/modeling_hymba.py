# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

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

from fla.models.utils import Cache
from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss
from fla.modules import GatedMLP as TransformerMLP
from fla.modules import RMSNorm
import torch.nn as nn

from .cache_lact import LaCTCache
from .configuration_hymba import HymbaConfig #LaCTSWIGLUConfig
from .modeling_hymba_orig import HybridMambaAttentionDynamicCache, HymbaBlock

logger = logging.get_logger(__name__)

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack


def _cache_length(past_key_values: Optional[Union[LaCTCache, List[Any], Tuple[Any, ...]]]) -> int:
    if past_key_values is None:
        return 0
    if isinstance(past_key_values, LaCTCache):
        return int(past_key_values.get_seq_length())
    if hasattr(past_key_values, "get_seq_length"):
        try:
            return int(past_key_values.get_seq_length())
        except Exception:
            pass
    if isinstance(past_key_values, (list, tuple)):
        return 1 if len(past_key_values) > 0 else 0
    return 0


# def _ensure_hymba_compat_config(config: LaCTSWIGLUConfig) -> LaCTSWIGLUConfig:
#     if getattr(config, "_hymba_compat_initialized", False):
#         return config

#     config.num_attention_heads = getattr(config, "num_attention_heads", config.num_attn_heads)
#     config.num_key_value_heads = getattr(
#         config,
#         "num_key_value_heads",
#         getattr(config, "num_attn_heads", config.num_attention_heads),
#     )
#     config.attn_hidden_size = getattr(config, "attn_hidden_size", 0)
#     config.kq_head_dim = getattr(config, "kq_head_dim", 0)
#     config.v_head_dim = getattr(config, "v_head_dim", 0)
#     config.attention_dropout = getattr(config, "attention_dropout", 0.0)
#     config.kq_norm = getattr(
#         config,
#         "kq_norm",
#         "perhead-rms" if getattr(config, "attn_qk_norm", False) else "none",
#     )
#     config.rope = getattr(config, "rope", True)
#     config.rope_type = getattr(config, "rope_type", "default")
#     config.orig_max_position_embeddings = getattr(
#         config,
#         "orig_max_position_embeddings",
#         config.max_position_embeddings,
#     )
#     config.attn_implementation = getattr(config, "attn_implementation", "flash_attention_2")
#     config.mamba_expand = getattr(config, "mamba_expand", max(1, int(config.inter_multi)))
#     config.mamba_d_state = getattr(config, "mamba_d_state", 16)
#     config.mamba_d_conv = getattr(config, "mamba_d_conv", 4)
#     config.mamba_dt_rank = getattr(
#         config,
#         "mamba_dt_rank",
#         max(1, math.ceil(config.hidden_size / 16)),
#     )
#     config.mamba_conv_bias = getattr(config, "mamba_conv_bias", True)
#     config.mamba_proj_bias = getattr(config, "mamba_proj_bias", False)
#     config.mamba_inner_layernorms = getattr(config, "mamba_inner_layernorms", False)
#     config.use_mamba_kernels = getattr(config, "use_mamba_kernels", True)
#     config.rms_norm_eps = getattr(config, "rms_norm_eps", config.norm_eps)
#     config.num_memory_tokens = getattr(config, "num_memory_tokens", 0)
#     config.global_attn_idx = getattr(config, "global_attn_idx", None)
#     config.sliding_window = getattr(
#         config,
#         "sliding_window",
#         config.window_size if getattr(config, "use_sliding_window", True) else None,
#     )
#     config.layer_type = getattr(config, "layer_type", ["h"] * config.num_hidden_layers)
#     config.conv_dim = getattr(config, "conv_dim", {})
#     config._hymba_compat_initialized = True
#     return config


def _build_position_ids(
    *,
    attention_mask: Optional[torch.Tensor],
    batch_size: int,
    sequence_length: int,
    past_length: int,
    device: torch.device,
) -> torch.LongTensor:
    if attention_mask is not None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        return position_ids[:, -sequence_length:].to(device=device)

    position_ids = torch.arange(
        past_length,
        past_length + sequence_length,
        device=device,
        dtype=torch.long,
    )
    return position_ids.unsqueeze(0).expand(batch_size, -1)


def _has_past_tokens(past_key_values: Optional[Union[LaCTCache, List[Any], Tuple[Any, ...]]]) -> bool:
    return _cache_length(past_key_values) > 0


def _is_padding_free_attention_mask(attention_mask: Optional[torch.Tensor]) -> bool:
    if attention_mask is None:
        return True
    return bool(torch.all(attention_mask.to(dtype=torch.bool)))


def _normalize_cached_attention_mask(
    attention_mask: Optional[torch.Tensor],
    *,
    use_cache: bool,
) -> Optional[torch.Tensor]:
    if not use_cache or attention_mask is None:
        return attention_mask
    if _is_padding_free_attention_mask(attention_mask):
        return None
    return attention_mask


class HymbaBlockWithMLP(nn.Module):

    def __init__(self, config: HymbaConfig, layer_idx: int):
        super().__init__()

        # self.config = _ensure_hymba_compat_config(config)
        self.layer_idx = layer_idx

        self.attn_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(
            config.hidden_size, eps=config.norm_eps
        )
        # Determine the per-layer attention mode, optionally driven by a
        # Qwen3GDN-style `layer_types` list if present on the config.
        # Mapping:
        # - "full_attention"     -> global/full attention   (window_size=None)
        # - "linear_attention"   -> LaCT + sliding window   (window_size=config.window_size)
        # - "sliding_attention"  -> LaCT + sliding window   (window_size=config.window_size)
        # If `layer_types` is not provided, we fall back to the global
        # `use_sliding_window` flag.
        layer_config = None
        if getattr(config, "layer_types", None) is not None:
            if 0 <= layer_idx < len(config.layer_types):
                layer_config = config.layer_types[layer_idx]

        if layer_config is not None:
            if layer_config['attn_type'] == 'full':
                window_size = None
            elif layer_config['attn_type'] in ("linear", "swa"):
                window_size = config.window_size
            else:
                raise ValueError(f"Invalid {layer_config['attn_type']=}")
        else:
            # Fallback: behave like the original global toggle.
            window_size = (
                config.window_size
                if getattr(config, "use_sliding_window", True)
                else None
            )
        self.use_swa = window_size is not None

        self.attn = HymbaBlock(config=self.config, layer_idx=layer_idx, reuse_kv=False)

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

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs: Unpack[Any],
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:

        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, _, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_values,
            position_ids=kwargs.get("position_ids"),
            use_cache=use_cache,
            use_swa=self.use_swa,
            kv_last_layer=None,
        )
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
            outputs += (None,)

        if use_cache:
            outputs += (past_key_values,)

        return outputs


class HymbaPreTrainedModel(PreTrainedModel):

    config_class = HymbaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["HymbaBlockWithMLP"]
    _supports_cache_class = True

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


class HymbaModel(HymbaPreTrainedModel):

    def __init__(self, config: HymbaConfig) -> HymbaModel:
        super().__init__(config)
        # _ensure_hymba_compat_config(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                HymbaBlockWithMLP(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
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

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
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

        if use_cache:
            if past_key_values is None:
                batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
                past_key_values = HybridMambaAttentionDynamicCache(
                    self.config,
                    batch_size=batch_size,
                    dtype=(inputs_embeds.dtype if inputs_embeds is not None else self.embeddings.weight.dtype),
                    device=(inputs_embeds.device if inputs_embeds is not None else input_ids.device),
                    layer_type=self.config.layer_type,
                )
            elif not isinstance(past_key_values, HybridMambaAttentionDynamicCache):
                raise ValueError(
                    "`past_key_values` must be a `HybridMambaAttentionDynamicCache` when "
                    "`HymbaModel` is backed by `HymbaBlock`."
                )

        attention_mask = _normalize_cached_attention_mask(
            attention_mask,
            use_cache=use_cache,
        )

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        batch_size, sequence_length = inputs_embeds.shape[:2]
        past_length = _cache_length(past_key_values) if use_cache else 0

        # embed positions
        hidden_states = inputs_embeds
        position_ids = kwargs.pop("position_ids", None)
        if position_ids is None:
            position_ids = _build_position_ids(
                attention_mask=attention_mask,
                batch_size=batch_size,
                sequence_length=sequence_length,
                past_length=past_length,
                device=hidden_states.device,
            )

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
                    past_key_values,
                    output_attentions,
                    use_cache,
                    position_ids=position_ids,
                    **kwargs,
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    position_ids=position_ids,
                    **kwargs,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_cache = layer_outputs[2 if output_attentions else 1]

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


class HymbaForCausalLM(HymbaPreTrainedModel, GenerationMixin):

    _tied_weights_keys = {"lm_head.weight": "model.embeddings.weight"} # ["lm_head.weight"]

    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        self.model = HymbaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None

        # Initialize weights and apply final processing
        self.post_init()

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

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        logits_to_keep: Optional[int] = None,
        **kwargs,
    ):
        has_past = use_cache and _has_past_tokens(past_key_values)
        attention_mask = _normalize_cached_attention_mask(
            attention_mask,
            use_cache=use_cache,
        )
        if has_past and input_ids is not None:
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
                "past_key_values": past_key_values,
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
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Optional[int] = 0,
        **kwargs: Unpack[Any],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        attention_mask = _normalize_cached_attention_mask(
            attention_mask,
            use_cache=use_cache,
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

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
