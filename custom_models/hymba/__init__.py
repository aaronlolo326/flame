

from .configuration_hymba import MyHymbaConfig
from .modeling_hymba import MyHymbaModel, MyHymbaForCausalLM
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

__all__ = [
    'MyHymbaConfig',
    'MyHymbaModel',
    'MyHymbaForCausalLM',
]

AutoConfig.register("my_hymba", MyHymbaConfig)
AutoModel.register(MyHymbaConfig, MyHymbaModel)
AutoModelForCausalLM.register(MyHymbaConfig, MyHymbaForCausalLM)
