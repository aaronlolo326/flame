from .configuration_ttt_e2e import E2ETTTConfig
from .modeling_e2e_v4 import E2ETTTForCausalLM, E2ETTTModel

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

__all__ = [
    "E2ETTTConfig",
    "E2ETTTModel",
    "E2ETTTForCausalLM",
]

AutoConfig.register("e2e_legacy", E2ETTTConfig)
AutoModel.register(E2ETTTConfig, E2ETTTModel)
AutoModelForCausalLM.register(E2ETTTConfig, E2ETTTForCausalLM)
