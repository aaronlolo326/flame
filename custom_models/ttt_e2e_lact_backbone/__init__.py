from .configuration_ttt_e2e import E2ETTTV2Config
from .modeling_ttt_e2e import E2EForCausalLM, E2EModel

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

class E2ETTTV2Model(E2EModel):
    config_class = E2ETTTV2Config


class E2ETTTV2ForCausalLM(E2EForCausalLM):
    config_class = E2ETTTV2Config

__all__ = [
    "E2ETTTV2Config",
    "E2ETTTV2Model",
    "E2ETTTV2ForCausalLM",
]

AutoConfig.register("ttt_e2e_v2", E2ETTTV2Config)
AutoModel.register(E2ETTTV2Config, E2ETTTV2Model)
AutoModelForCausalLM.register(E2ETTTV2Config, E2ETTTV2ForCausalLM)
