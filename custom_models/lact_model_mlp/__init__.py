from .configuration_lact_swiglu import LaCTMLPSWIGLUConfig
from .modeling_lact import LaCTMLPModel, LaCTMLPForCausalLM
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

__all__ = [
    'LaCTMLPSWIGLUConfig',
    'LaCTMLPModel',
    'LaCTMLPForCausalLM',
]

AutoConfig.register("lact_mlp_swiglu", LaCTMLPSWIGLUConfig)
AutoModel.register(LaCTMLPSWIGLUConfig, LaCTMLPModel)
AutoModelForCausalLM.register(LaCTMLPSWIGLUConfig, LaCTMLPForCausalLM)
