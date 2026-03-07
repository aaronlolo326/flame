from .configuration_qwen3 import Qwen3GDNConfig
from .modeling_qwen3 import Qwen3GDNModel, Qwen3GDNForCausalLM

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

__all__ = ['Qwen3GDNConfig', 'Qwen3GDNModel', 'Qwen3GDNForCausalLM']

AutoConfig.register("qwen3_gdn", Qwen3GDNConfig)
AutoModel.register(Qwen3GDNConfig, Qwen3GDNModel)
AutoModelForCausalLM.register(Qwen3GDNConfig, Qwen3GDNForCausalLM)
