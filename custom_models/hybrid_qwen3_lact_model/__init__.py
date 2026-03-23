from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_hybrid_qwen3_lact import HybridQwen3LaCTConfig
from .modeling_hybrid_qwen3_lact import HybridQwen3LaCTForCausalLM, HybridQwen3LaCTModel

__all__ = ["HybridQwen3LaCTConfig", "HybridQwen3LaCTModel", "HybridQwen3LaCTForCausalLM"]

AutoConfig.register("hybrid_qwen3_lact", HybridQwen3LaCTConfig)
AutoModel.register(HybridQwen3LaCTConfig, HybridQwen3LaCTModel)
AutoModelForCausalLM.register(HybridQwen3LaCTConfig, HybridQwen3LaCTForCausalLM)