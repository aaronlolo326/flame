# # Copyright 2024 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# from typing import TYPE_CHECKING

# from ...utils import _LazyModule
# from ...utils.import_utils import define_import_structure


# if TYPE_CHECKING:
#     from .configuration_qwen3 import *
#     from .modeling_qwen3 import *
# else:
#     import sys

#     _file = globals()["__file__"]
#     sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)

from transformers import PreTrainedModel

from .configuration_qwen3 import Qwen3_Config
from . import modeling_qwen3
from .modeling_qwen3 import Qwen3_Model, Qwen3_ForCausalLM

from liger_kernel.transformers.functional import liger_cross_entropy
from liger_kernel.transformers.rms_norm import LigerRMSNorm
from liger_kernel.transformers.rope import liger_rotary_pos_emb
from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
from liger_kernel.transformers.monkey_patch import _patch_rms_norm_module, _patch_swiglu_module
from liger_kernel.transformers.model.qwen3 import lce_forward as qwen3_lce_forward

# from .liger_kernel_sync.transformers.model.qwen3 import lce_forward as qwen3_lce_forward


def apply_liger_kernel_to_qwen3(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Qwen3 models.
    """
    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    if rope:
        modeling_qwen3.apply_rotary_pos_emb = liger_rotary_pos_emb

    if rms_norm:
        modeling_qwen3.Qwen3RMSNorm = LigerRMSNorm

    if cross_entropy:
        from transformers.loss.loss_utils import nn

        nn.functional.cross_entropy = liger_cross_entropy

    if fused_linear_cross_entropy:
        modeling_qwen3.Qwen3_ForCausalLM.forward = qwen3_lce_forward

    if swiglu:
        modeling_qwen3.Qwen3MLP = LigerSwiGLUMLP

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules

        # get the base model from the model instance
        base_model: Qwen3_Model = getattr(model, model.base_model_prefix, model)

        if rms_norm:
            _patch_rms_norm_module(base_model.norm)
        for decoder_layer in base_model.layers:
            if swiglu:
                _patch_swiglu_module(decoder_layer.mlp, LigerSwiGLUMLP)
            if rms_norm:
                _patch_rms_norm_module(decoder_layer.input_layernorm)
                _patch_rms_norm_module(decoder_layer.post_attention_layernorm)


from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

apply_liger_kernel_to_qwen3()

__all__ = ['Qwen3_Config', 'Qwen3_Model', 'Qwen3_ForCausalLM']

AutoConfig.register("qwen3_", Qwen3_Config)
AutoModel.register(Qwen3_Config, Qwen3_Model)
AutoModelForCausalLM.register(Qwen3_Config, Qwen3_ForCausalLM)
