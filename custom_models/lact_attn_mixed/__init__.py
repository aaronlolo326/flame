from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_lact_attn_mixed import LaCTAttnMixedConfig
from .modeling_lact_attn_mixed import LaCTAttnMixedForCausalLM, LaCTAttnMixedModel

__all__ = [
    "LaCTAttnMixedConfig",
    "LaCTAttnMixedModel",
    "LaCTAttnMixedForCausalLM",
]

AutoConfig.register("lact_attn_mixed", LaCTAttnMixedConfig)
AutoModel.register(LaCTAttnMixedConfig, LaCTAttnMixedModel)
AutoModelForCausalLM.register(LaCTAttnMixedConfig, LaCTAttnMixedForCausalLM)
