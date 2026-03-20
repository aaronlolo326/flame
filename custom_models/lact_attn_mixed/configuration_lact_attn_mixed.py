# -*- coding: utf-8 -*-

from ..lact_model.configuration_lact_swiglu import LaCTSWIGLUConfig


class LaCTAttnMixedConfig(LaCTSWIGLUConfig):
    model_type = "lact_attn_mixed"

    def __init__(
        self,
        use_ttt_depth_mix: bool = False,
        depth_mix_dim: int = 256,
        depth_mix_block_size: int = 0,
        depth_mix_detach_cache: bool = True,
        depth_mix_use_embedding_slot: bool = False,
        depth_mix_init_gate_bias: float = -4.0,
        **kwargs,
    ):
        self.use_ttt_depth_mix = use_ttt_depth_mix
        self.depth_mix_dim = depth_mix_dim
        self.depth_mix_block_size = depth_mix_block_size
        self.depth_mix_detach_cache = depth_mix_detach_cache
        self.depth_mix_use_embedding_slot = depth_mix_use_embedding_slot
        self.depth_mix_init_gate_bias = depth_mix_init_gate_bias
        super().__init__(**kwargs)
