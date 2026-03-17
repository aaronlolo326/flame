from .configuration_ttt_e2e import E2EMultiLevelConfig
from .modeling_ttt_e2e import E2EForCausalLM, E2EModel

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

class E2EMultiLevelModel(E2EModel):
    config_class = E2EMultiLevelConfig


class E2EMultiLevelForCausalLM(E2EForCausalLM):
    config_class = E2EMultiLevelConfig


# Backward-compatible aliases for existing imports.
E2ETTTV2Config = E2EMultiLevelConfig
E2ETTTV2Model = E2EMultiLevelModel
E2ETTTV2ForCausalLM = E2EMultiLevelForCausalLM

__all__ = [
    "E2EMultiLevelConfig",
    "E2EMultiLevelModel",
    "E2EMultiLevelForCausalLM",
    "E2ETTTV2Config",
    "E2ETTTV2Model",
    "E2ETTTV2ForCausalLM",
]

AutoConfig.register("e2e_multi_level", E2EMultiLevelConfig)
AutoModel.register(E2EMultiLevelConfig, E2EMultiLevelModel)
AutoModelForCausalLM.register(E2EMultiLevelConfig, E2EMultiLevelForCausalLM)
