from .configuration_lact_swiglu import LaCTSWIGLUConfig
from .cache_lact import LaCTCache
from .modeling_lact import LaCTModel, LaCTForCausalLM
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

__all__ = [
    'LaCTCache',
    'LaCTSWIGLUConfig',
    'LaCTModel',
    'LaCTForCausalLM',
]

AutoConfig.register("lact_swiglu", LaCTSWIGLUConfig)
AutoModel.register(LaCTSWIGLUConfig, LaCTModel)
AutoModelForCausalLM.register(LaCTSWIGLUConfig, LaCTForCausalLM)
