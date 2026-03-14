from __future__ import annotations

import importlib.util
import json
import math
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, TypedDict


REPO_ROOT = Path(__file__).resolve().parents[3]
FLAME_ROOT = REPO_ROOT / "flame"
HYBRID_ROOT = REPO_ROOT / "hybrid_models"
FLASH_LINEAR_ATTENTION_ROOT = REPO_ROOT / "flash-linear-attention"
DEFAULT_LACT_CONFIG = FLAME_ROOT / "configs" / "qwen3_lact_1B4.json"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    kind: str


MODEL_SPECS = {
    "lact": ModelSpec("lact", "LaCT", "lact"),
    "e2e_ttt": ModelSpec("e2e_ttt", "E2E-TTT", "e2e_ttt"),
    "hybrid_lact": ModelSpec("hybrid_lact", "75% LaCT + 25% FA", "hybrid_lact"),
    "full_attention": ModelSpec("full_attention", "Full Attention (FlashAttention)", "qwen3_swa"),
    "hybrid_swa": ModelSpec("hybrid_swa", "75% SWA + 25% FA", "qwen3_swa"),
    "hybrid_gdn": ModelSpec("hybrid_gdn", "75% GatedDeltaNet + 25% FA", "qwen3_gdn"),
}

KERNEL_MODEL_LABELS = {
    "lact_full_layer": "LaCT Full Layer",
    "lact_ttt_branch_only": "LaCT TTT Branch Only",
    "fa_branch_only": "Full Attention Branch Only",
    "swa_branch_only": "Sliding-Window Attention Branch Only",
    "gdn_branch_only": "GatedDeltaNet Branch Only",
}

QWEN35_2B_BASE_GDN = {
    "num_attention_heads": 16,
    "head_dim": 128,
    "expand_v": 1,
    "mode": "chunk",
    "use_gate": True,
    "use_short_conv": True,
    "conv_size": 4,
    "conv_bias": False,
    "pad_value": 0,
    "selection_window_size": 100,
    "use_qk_norm": True,
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ceil_ratio_count(total: int, ratio: float) -> int:
    return max(1, min(total, int(math.ceil(total * ratio))))


def build_layer_types(num_layers: int, recurrent_label: str, recurrent_ratio: float = 0.75) -> list[str]:
    recurrent_layers = ceil_ratio_count(num_layers, recurrent_ratio)
    full_layers = max(0, num_layers - recurrent_layers)
    return ["full_attention"] * full_layers + [recurrent_label] * recurrent_layers


def resolve_chunk_and_window(
    *,
    seq_len: int,
    base_cfg: dict[str, Any],
    lact_chunk_size: int | None,
    window_size: int | None,
    paper_lm_defaults: bool,
) -> tuple[int, int]:
    chunk = lact_chunk_size or int(base_cfg.get("lact_chunk_size", 2048))
    window = window_size or int(base_cfg.get("window_size", chunk))
    if paper_lm_defaults:
        chunk = min(seq_len, chunk)
        window = max(window, chunk)
    return chunk, window


def lact_config_to_qwen3_dict(
    lact_cfg: dict[str, Any],
    *,
    seq_len: int,
    layer_types: list[str],
    sliding_window: int | None,
    attn_implementation: str,
    gdn_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    num_heads = lact_cfg.get("num_attn_heads", lact_cfg.get("num_heads"))
    num_kv_heads = lact_cfg.get("num_key_value_heads", num_heads)
    hidden_size = lact_cfg["hidden_size"]
    return {
        "vocab_size": lact_cfg["vocab_size"],
        "hidden_size": hidden_size,
        "intermediate_size": lact_cfg.get("intermediate_size", hidden_size * 4),
        "num_hidden_layers": lact_cfg["num_hidden_layers"],
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "head_dim": hidden_size // num_heads,
        "hidden_act": lact_cfg.get("hidden_act", "silu"),
        "max_position_embeddings": max(seq_len, lact_cfg.get("max_position_embeddings", seq_len)),
        "initializer_range": lact_cfg.get("initializer_range", 0.02),
        "rms_norm_eps": lact_cfg.get("norm_eps", 1e-6),
        "use_cache": False,
        "tie_word_embeddings": lact_cfg.get("tie_word_embeddings", False),
        "rope_theta": lact_cfg.get("rope_theta", 10_000.0),
        "rope_scaling": lact_cfg.get("rope_scaling"),
        "attention_bias": lact_cfg.get("attention_bias", lact_cfg.get("qkv_bias", False)),
        "use_sliding_window": sliding_window is not None,
        "sliding_window": sliding_window,
        "max_window_layers": len([name for name in layer_types if name == "full_attention"]),
        "layer_types": layer_types,
        "attention_dropout": lact_cfg.get("attention_dropout", 0.0),
        "bos_token_id": lact_cfg.get("bos_token_id"),
        "eos_token_id": lact_cfg.get("eos_token_id"),
        "pad_token_id": lact_cfg.get("pad_token_id"),
        "torch_dtype": lact_cfg.get("torch_dtype", "bfloat16"),
        "attn_implementation": attn_implementation,
        **(gdn_overrides or {}),
    }


@contextmanager
def prepend_sys_path(*paths: Path):
    inserted = []
    try:
        for path in reversed(paths):
            text = str(path)
            if text not in sys.path:
                sys.path.insert(0, text)
                inserted.append(text)
        yield
    finally:
        for text in inserted:
            if text in sys.path:
                sys.path.remove(text)


def load_module_from_path(module_name: str, file_path: Path, aliases: list[str] | None = None) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    alias_names = aliases or []
    for alias in alias_names:
        sys.modules[alias] = module
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def ensure_transformers_utils_compat() -> None:
    try:
        import transformers.utils as transformers_utils
    except ImportError:
        return

    if not hasattr(transformers_utils, "LossKwargs"):
        class LossKwargs(TypedDict, total=False):  # noqa: N801
            pass

        transformers_utils.LossKwargs = LossKwargs

    if not hasattr(transformers_utils, "auto_docstring"):
        def auto_docstring(*decorator_args, **decorator_kwargs):
            if decorator_args and callable(decorator_args[0]) and len(decorator_args) == 1 and not decorator_kwargs:
                return decorator_args[0]

            def decorator(fn):
                return fn

            return decorator

        transformers_utils.auto_docstring = auto_docstring

    if not hasattr(transformers_utils, "can_return_tuple"):
        def can_return_tuple(fn):
            return fn

        transformers_utils.can_return_tuple = can_return_tuple


def ensure_transformers_cache_compat() -> None:
    try:
        import transformers.cache_utils as cache_utils
    except ImportError:
        return

    if not hasattr(cache_utils, "SlidingWindowCache") and hasattr(cache_utils, "DynamicCache"):
        cache_utils.SlidingWindowCache = cache_utils.DynamicCache


def ensure_transformers_rope_compat() -> None:
    try:
        import transformers.modeling_rope_utils as rope_utils
    except ImportError:
        return

    if "default" in getattr(rope_utils, "ROPE_INIT_FUNCTIONS", {}):
        return

    def _legacy_default_rope_parameters(
        config=None,
        device=None,
        seq_len=None,
        layer_type=None,
    ):
        import torch

        if config is None:
            raise ValueError("config is required for legacy default rope initialization")
        base = getattr(config, "rope_theta", 10_000.0)
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)
        attention_factor = 1.0
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, dim, 2, dtype=torch.int64, device=device).float()
                / dim
            )
        )
        return inv_freq, attention_factor

    rope_utils.ROPE_INIT_FUNCTIONS["default"] = _legacy_default_rope_parameters


def load_qwen3_variant(variant: str) -> tuple[type, type, type, type | None]:
    import torch

    if variant not in {"qwen3_swa", "qwen3_gdn"}:
        raise ValueError(f"Unknown Qwen3 variant: {variant}")
    model_root = HYBRID_ROOT / ("qwen3_swa" if variant == "qwen3_swa" else "qwen3_gdn")
    with prepend_sys_path(HYBRID_ROOT, model_root):
        config_mod = load_module_from_path(
            f"{variant}_configuration_qwen3",
            model_root / "configuration_qwen3.py",
            aliases=["configuration_qwen3"],
        )
        if variant == "qwen3_gdn":
            ensure_transformers_utils_compat()
            ensure_transformers_cache_compat()
            ensure_transformers_rope_compat()
            original_compile = torch.compile
            try:
                torch.compile = lambda fn=None, *args, **kwargs: (lambda inner: inner) if fn is None else fn
                modeling_mod = load_module_from_path(
                    f"{variant}_modeling_qwen3",
                    model_root / "modeling_qwen3.py",
                    aliases=["modeling_qwen3"],
                )
            finally:
                torch.compile = original_compile
        else:
            modeling_mod = load_module_from_path(
                f"{variant}_modeling_qwen3",
                model_root / "modeling_qwen3.py",
                aliases=["modeling_qwen3"],
            )
    gdn_cls = getattr(modeling_mod, "GatedDeltaNet", None)
    return (
        config_mod.Qwen3Config,
        modeling_mod.Qwen3ForCausalLM,
        modeling_mod.Qwen3Model,
        gdn_cls,
    )


def import_lact_modules() -> tuple[type, type, type]:
    with prepend_sys_path(FLAME_ROOT):
        import custom_models  # noqa: F401
        from custom_models.lact_model.configuration_lact_swiglu import LaCTSWIGLUConfig
        from custom_models.lact_model.layer_lact_swiglu import LaCTSWIGLULayer
        from custom_models.lact_model.modeling_lact import LaCTForCausalLM

    return LaCTSWIGLUConfig, LaCTForCausalLM, LaCTSWIGLULayer


def import_e2e_ttt_modules() -> tuple[type, type]:
    with prepend_sys_path(FLAME_ROOT):
        import custom_models  # noqa: F401
        from custom_models.ttt_e2_lact_backbone.configuration_ttt_e2e import E2ETTTConfig
        from custom_models.ttt_e2_lact_backbone.modeling_ttt_e2e import E2EForCausalLM

    return E2ETTTConfig, E2EForCausalLM


def resolve_dtype(torch_module: Any, dtype_name: str):
    normalized = str(dtype_name).replace("torch.", "")
    mapping = {
        "float32": torch_module.float32,
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[normalized]


def canonical_kernel_key(model_key: str) -> str:
    return model_key


def build_whole_model(
    *,
    model_key: str,
    seq_len: int,
    device: str,
    dtype_name: str,
    base_config_path: Path = DEFAULT_LACT_CONFIG,
    sliding_window: int | None = None,
    lact_chunk_size: int | None = None,
    num_attn_heads_override: int | None = None,
    num_lact_heads_override: int | None = None,
    use_fused_kernel: bool | None = None,
    paper_lm_defaults: bool = True,
) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from types import SimpleNamespace

    base_cfg = load_json(base_config_path)
    base_cfg_for_model = dict(base_cfg)
    if num_attn_heads_override is not None:
        base_cfg_for_model["num_attn_heads"] = int(num_attn_heads_override)
        base_cfg_for_model["num_heads"] = int(num_attn_heads_override)
        # Keep KV heads valid and not larger than attention heads when overriding.
        base_cfg_for_model["num_key_value_heads"] = min(
            int(base_cfg_for_model.get("num_key_value_heads", num_attn_heads_override)),
            int(num_attn_heads_override),
        )
    if num_lact_heads_override is not None:
        base_cfg_for_model["num_lact_heads"] = int(num_lact_heads_override)

    dtype = resolve_dtype(torch, dtype_name)
    chunk_size, window_size = resolve_chunk_and_window(
        seq_len=seq_len,
        base_cfg=base_cfg_for_model,
        lact_chunk_size=lact_chunk_size,
        window_size=sliding_window,
        paper_lm_defaults=paper_lm_defaults,
    )

    if model_key == "lact":
        config_cls, model_cls, _ = import_lact_modules()
        config_kwargs = dict(base_cfg_for_model)
        config_kwargs["max_position_embeddings"] = max(seq_len, config_kwargs.get("max_position_embeddings", seq_len))
        config_kwargs["use_cache"] = False
        config_kwargs["lact_chunk_size"] = chunk_size
        config_kwargs["window_size"] = window_size
        if config_kwargs["hidden_size"] % int(config_kwargs["num_attn_heads"]) != 0:
            raise ValueError(
                f"Invalid num_attn_heads={config_kwargs['num_attn_heads']} for hidden_size={config_kwargs['hidden_size']}. "
                "hidden_size must be divisible by num_attn_heads."
            )
        if config_kwargs["hidden_size"] % int(config_kwargs["num_lact_heads"]) != 0:
            raise ValueError(
                f"Invalid num_lact_heads={config_kwargs['num_lact_heads']} for hidden_size={config_kwargs['hidden_size']}. "
                "hidden_size must be divisible by num_lact_heads."
            )
        if use_fused_kernel is not None:
            config_kwargs["use_fused_kernel"] = use_fused_kernel
        config = config_cls(**config_kwargs)
        model = model_cls(config)
    elif model_key == "e2e_ttt":
        config_cls, model_cls = import_e2e_ttt_modules()
        hidden_size = int(base_cfg_for_model["hidden_size"])
        num_heads = int(base_cfg_for_model.get("num_attn_heads", base_cfg_for_model.get("num_heads")))
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"Invalid num_attn_heads={num_heads} for hidden_size={hidden_size}. "
                "hidden_size must be divisible by num_attn_heads."
            )
        config_kwargs = {
            "vocab_size": int(base_cfg_for_model["vocab_size"]),
            "bos_token_id": base_cfg_for_model.get("bos_token_id", 1),
            "eos_token_id": base_cfg_for_model.get("eos_token_id", 2),
            "pad_token_id": base_cfg_for_model.get("pad_token_id"),
            "tie_word_embeddings": bool(base_cfg_for_model.get("tie_word_embeddings", False)),
            "hidden_size": hidden_size,
            "num_hidden_layers": int(base_cfg_for_model["num_hidden_layers"]),
            "num_attention_heads": num_heads,
            "intermediate_size": int(base_cfg_for_model.get("intermediate_size", hidden_size * 4)),
            "hidden_act": str(base_cfg_for_model.get("hidden_act", "swish")),
            "rms_norm_eps": float(base_cfg_for_model.get("norm_eps", 1e-6)),
            "norm_eps": float(base_cfg_for_model.get("norm_eps", 1e-6)),
            "initializer_range": float(base_cfg_for_model.get("initializer_range", 0.02)),
            "max_position_embeddings": max(seq_len, int(base_cfg_for_model.get("max_position_embeddings", seq_len))),
            "rope_theta": float(base_cfg_for_model.get("rope_theta", 10_000.0)),
            "window_size": int(window_size),
            "use_e2e_ttt": True,
            "suffix_frac": 0.25,
            "mini_batch_size": int(chunk_size),
            "optimizer_inner": {
                "optimizer_type": "sgd",
                "lr": 1e-3,
                "clip_gradient": 0.0,
            },
            "inner_steps_per_chunk": 1,
            "ttt_mlp_only": True,
            "two_mlp_per_block": True,
            "detach_fast_weights": False,
            "inner_param_filter": "prime_mlp",
            "attn_backend": "flash",
            "fuse_cross_entropy": True,
            "fuse_norm": True,
            "last_layer_fuse_norm": True,
            "fuse_swiglu": True,
            "hidden_ratio": 4,
            "use_cache": False,
        }
        config = config_cls(**config_kwargs)
        model = model_cls(config)
    elif model_key == "hybrid_lact":
        # Benchmark-only mixed stack:
        # first 25% layers = full-attention Qwen3 decoder layers
        # remaining 75% layers = LaCT blocks
        config_cls_lact, _, _ = import_lact_modules()
        with prepend_sys_path(FLAME_ROOT):
            from custom_models.lact_model.modeling_lact import LaCTBlock

        qwen_config_cls, qwen_model_cls, qwen_backbone_cls, _ = load_qwen3_variant("qwen3_swa")

        lact_config_kwargs = dict(base_cfg_for_model)
        lact_config_kwargs["max_position_embeddings"] = max(
            seq_len, lact_config_kwargs.get("max_position_embeddings", seq_len)
        )
        lact_config_kwargs["use_cache"] = False
        lact_config_kwargs["lact_chunk_size"] = chunk_size
        lact_config_kwargs["window_size"] = window_size
        if lact_config_kwargs["hidden_size"] % int(lact_config_kwargs["num_attn_heads"]) != 0:
            raise ValueError(
                f"Invalid num_attn_heads={lact_config_kwargs['num_attn_heads']} for hidden_size={lact_config_kwargs['hidden_size']}. "
                "hidden_size must be divisible by num_attn_heads."
            )
        if lact_config_kwargs["hidden_size"] % int(lact_config_kwargs["num_lact_heads"]) != 0:
            raise ValueError(
                f"Invalid num_lact_heads={lact_config_kwargs['num_lact_heads']} for hidden_size={lact_config_kwargs['hidden_size']}. "
                "hidden_size must be divisible by num_lact_heads."
            )
        lact_config = config_cls_lact(**lact_config_kwargs)

        num_layers = int(base_cfg_for_model["num_hidden_layers"])
        num_lact_layers = ceil_ratio_count(num_layers, 0.75)
        num_fa_layers = max(0, num_layers - num_lact_layers)
        qwen_layer_types = ["full_attention"] * num_layers
        qwen_cfg_kwargs = lact_config_to_qwen3_dict(
            base_cfg_for_model,
            seq_len=seq_len,
            layer_types=qwen_layer_types,
            sliding_window=None,
            attn_implementation="flash_attention_2",
        )
        qwen_cfg = qwen_config_cls(**qwen_cfg_kwargs)
        qwen_cfg.use_cache = False
        qwen_backbone = qwen_backbone_cls(qwen_cfg)

        class HybridLaCTForCausalLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = lact_config
                self.num_fa_layers = num_fa_layers
                self.num_lact_layers = num_lact_layers
                self.embed_tokens = nn.Embedding(
                    lact_config.vocab_size, lact_config.hidden_size, lact_config.pad_token_id
                )
                self.fa_layers = nn.ModuleList(
                    [qwen_backbone.layers[i] for i in range(self.num_fa_layers)]
                )
                self.lact_layers = nn.ModuleList(
                    [LaCTBlock(lact_config, layer_idx=self.num_fa_layers + i) for i in range(self.num_lact_layers)]
                )
                self.rotary_emb = qwen_backbone.rotary_emb
                self.norm = qwen_backbone.norm
                self.lm_head = nn.Linear(lact_config.hidden_size, lact_config.vocab_size, bias=False)

            def forward(self, input_ids: Any, labels: Any = None, use_cache: bool = False):
                del use_cache
                hidden_states = self.embed_tokens(input_ids)
                batch_size, seq_len_local, _ = hidden_states.shape
                device_local = hidden_states.device
                position_ids = torch.arange(seq_len_local, device=device_local).unsqueeze(0).expand(batch_size, -1)
                position_embeddings = self.rotary_emb(hidden_states, position_ids)

                for layer in self.fa_layers:
                    hidden_states = layer(
                        hidden_states=hidden_states,
                        attention_mask=None,
                        position_ids=position_ids,
                        past_key_value=None,
                        output_attentions=False,
                        use_cache=False,
                        cache_position=None,
                        position_embeddings=position_embeddings,
                    )[0]

                for layer in self.lact_layers:
                    hidden_states = layer(
                        hidden_states=hidden_states,
                        attention_mask=None,
                        past_key_values=None,
                        output_attentions=False,
                        use_cache=False,
                    )[0]

                hidden_states = self.norm(hidden_states)
                logits = self.lm_head(hidden_states)
                loss = None
                if labels is not None:
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = labels[:, 1:].contiguous()
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                    )
                return SimpleNamespace(loss=loss, logits=logits)

        model = HybridLaCTForCausalLM()
    elif model_key in {"full_attention", "hybrid_swa", "hybrid_gdn"}:
        spec = MODEL_SPECS[model_key]
        config_cls, model_cls, _, _ = load_qwen3_variant(spec.kind)
        if model_key == "full_attention":
            layer_types = ["full_attention"] * base_cfg_for_model["num_hidden_layers"]
            window = None
        elif model_key == "hybrid_swa":
            layer_types = build_layer_types(base_cfg_for_model["num_hidden_layers"], "sliding_attention")
            window = window_size
        else:
            layer_types = build_layer_types(base_cfg_for_model["num_hidden_layers"], "linear_attention")
            window = None

        gdn_overrides = None
        if model_key == "hybrid_gdn":
            gdn_overrides = {
                "expand_v": 1,
                "mode": "chunk",
                "use_gate": True,
                "use_short_conv": True,
                "conv_size": 4,
                "conv_bias": False,
                "pad_value": 0,
                "selection_window_size": 100,
                "use_qk_norm": True,
            }
        config_kwargs = lact_config_to_qwen3_dict(
            base_cfg_for_model,
            seq_len=seq_len,
            layer_types=layer_types,
            sliding_window=window,
            attn_implementation="flash_attention_2",
            gdn_overrides=gdn_overrides,
        )
        config = config_cls(**config_kwargs)
        config.use_cache = False
        model = model_cls(config)
    else:
        raise ValueError(f"Unknown model key: {model_key}")

    model.to(device=device, dtype=dtype)
    model.train()
    return model, base_cfg_for_model


def build_kernel_module(
    *,
    model_key: str,
    seq_len: int,
    device: str,
    dtype_name: str,
    batch_size: int,
    base_config_path: Path = DEFAULT_LACT_CONFIG,
    sliding_window: int | None = None,
    lact_chunk_size: int | None = None,
    lact_attn_heads_override: int | None = None,
    lact_ttt_heads_override: int | None = None,
    use_fused_kernel: bool | None = None,
    paper_lm_defaults: bool = True,
) -> tuple[Any, int]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from einops import rearrange

    model_key = canonical_kernel_key(model_key)
    base_cfg = load_json(base_config_path)
    dtype = resolve_dtype(torch, dtype_name)
    hidden_size = base_cfg["hidden_size"]
    chunk_size, window_size = resolve_chunk_and_window(
        seq_len=seq_len,
        base_cfg=base_cfg,
        lact_chunk_size=lact_chunk_size,
        window_size=sliding_window,
        paper_lm_defaults=paper_lm_defaults,
    )

    if model_key in {"lact_full_layer", "lact_ttt_branch_only"}:
        config_cls, _, layer_cls = import_lact_modules()
        with prepend_sys_path(FLAME_ROOT):
            from custom_models.lact_model.layer_lact_swiglu import l2_norm
            from custom_models.lact_model.ttt_operation import (
                block_causal_lact_swiglu,
                prenorm_block_causal_lact_swiglu,
            )
            from custom_models.lact_model.ttt_operation_fused_kernel import (
                postnorm_block_causal_lact_swiglu_fused_kernel_triton,
                prenorm_block_causal_lact_swiglu_fused_kernel_triton,
            )

        config_kwargs = dict(base_cfg)
        config_kwargs["max_position_embeddings"] = max(seq_len, config_kwargs.get("max_position_embeddings", seq_len))
        config_kwargs["lact_chunk_size"] = chunk_size
        config_kwargs["window_size"] = window_size
        if lact_attn_heads_override is not None:
            config_kwargs["num_attn_heads"] = int(lact_attn_heads_override)
        if lact_ttt_heads_override is not None:
            config_kwargs["num_lact_heads"] = int(lact_ttt_heads_override)
        if hidden_size % int(config_kwargs["num_attn_heads"]) != 0:
            raise ValueError(
                f"Invalid num_attn_heads={config_kwargs['num_attn_heads']} for hidden_size={hidden_size}. "
                "hidden_size must be divisible by num_attn_heads."
            )
        if hidden_size % int(config_kwargs["num_lact_heads"]) != 0:
            raise ValueError(
                f"Invalid num_lact_heads={config_kwargs['num_lact_heads']} for hidden_size={hidden_size}. "
                "hidden_size must be divisible by num_lact_heads."
            )
        if use_fused_kernel is not None:
            config_kwargs["use_fused_kernel"] = use_fused_kernel
        config = config_cls(**config_kwargs)
        full_layer = layer_cls(
            hidden_size=config.hidden_size,
            num_attn_heads=config.num_attn_heads,
            num_lact_heads=config.num_lact_heads,
            inter_multi=config.inter_multi,
            window_size=config.window_size,
            lact_chunk_size=config.lact_chunk_size,
            qkv_bias=config.qkv_bias,
            attn_qk_norm=config.attn_qk_norm,
            qkv_silu=config.qkv_silu,
            no_v_silu=config.no_v_silu,
            lr_dim=config.lr_dim,
            use_muon=config.use_muon,
            ttt_prenorm=config.ttt_prenorm,
            ttt_nope=config.ttt_nope,
            lr_parameterization=config.lr_parameterization,
            learnable_ttt_scale=config.learnable_ttt_scale,
            rope_theta=config.rope_theta,
            max_position_embeddings=max(seq_len, config.max_position_embeddings),
            layer_idx=0,
            w0_w2_low_rank=config.w0_w2_low_rank,
            use_momentum=config.use_momentum,
            ttt_loss_type=config.ttt_loss_type,
            fw_init_gain=config.fw_init_gain,
            use_fused_kernel=config.use_fused_kernel,
            fp32_states=config.fp32_states,
        ).to(device=device, dtype=dtype)
        full_layer.train()
        if model_key == "lact_full_layer":
            return full_layer, hidden_size

        class LaCTTTTBranchOnly(nn.Module):
            def __init__(self, layer: Any):
                super().__init__()
                self.layer = layer

            def forward(self, hidden_states: Any) -> Any:
                layer = self.layer
                batch_size_local, q_len, _ = hidden_states.size()
                q, k, v = layer.qkv(hidden_states).chunk(3, dim=-1)
                if layer.attn_qk_norm:
                    q, k = layer.q_norm(q), layer.k_norm(k)

                fast_q, fast_k = layer._rescale_qk(q, k)
                fast_v = v

                fast_q = rearrange(fast_q, "b s (n_h d) -> (b n_h) s d", n_h=layer.num_fw_heads)
                fast_k = rearrange(fast_k, "b s (n_h d) -> (b n_h) s d", n_h=layer.num_fw_heads)
                fast_v = rearrange(fast_v, "b s (n_h d) -> (b n_h) s d", n_h=layer.num_fw_heads)

                if layer.qkv_silu:
                    if layer.no_v_silu:
                        fast_q = F.silu(fast_q)
                        fast_k = F.silu(fast_k)
                    else:
                        fast_q = F.silu(fast_q)
                        fast_k = F.silu(fast_k)
                        fast_v = F.silu(fast_v)

                fast_q = l2_norm(fast_q)
                fast_k = l2_norm(fast_k)

                seqlen_offset = 0
                max_seqlen = max(q_len, layer.max_position_embeddings)
                if not layer.ttt_nope:
                    fast_q = rearrange(fast_q, "(b n_h) s d -> b s (n_h d)", n_h=layer.num_fw_heads)
                    fast_k = rearrange(fast_k, "(b n_h) s d -> b s (n_h d)", n_h=layer.num_fw_heads)
                    fast_q = rearrange(fast_q, "b s (n_h d) -> b s n_h d", n_h=layer.num_heads)
                    fast_k = rearrange(fast_k, "b s (n_h d) -> b s n_h d", n_h=layer.num_heads)
                    fast_q, fast_k = layer.rotary(
                        fast_q,
                        fast_k,
                        seqlen_offset=seqlen_offset,
                        max_seqlen=max_seqlen,
                        cu_seqlens=None,
                    )
                    fast_q = rearrange(fast_q, "b s n_h d -> b s (n_h d)", n_h=layer.num_heads)
                    fast_k = rearrange(fast_k, "b s n_h d -> b s (n_h d)", n_h=layer.num_heads)
                    fast_q = rearrange(fast_q, "b s (n_h d) -> (b n_h) s d", n_h=layer.num_fw_heads)
                    fast_k = rearrange(fast_k, "b s (n_h d) -> (b n_h) s d", n_h=layer.num_fw_heads)

                if layer.w0_w2_low_rank > 0:
                    fw_w0 = layer.w0().repeat(batch_size_local, 1, 1)
                    fw_w2 = layer.w2().repeat(batch_size_local, 1, 1)
                else:
                    fw_w0 = layer.w0.repeat(batch_size_local, 1, 1)
                    fw_w2 = layer.w2.repeat(batch_size_local, 1, 1)
                fw_w1 = layer.w1.repeat(batch_size_local, 1, 1)

                lr = layer.lr_proj(hidden_states)
                if layer.lr_parameterization == "mamba":
                    lr = torch.nn.functional.softplus(lr.float() + layer.base_lr_inv)
                else:
                    raise NotImplementedError(
                        f"LR parameterization {layer.lr_parameterization} not implemented"
                    )
                fw_lr = rearrange(lr, "b s (n_h lr_dim) -> (b n_h) s lr_dim", n_h=layer.num_fw_heads)
                fw_lr1, fw_lr2, fw_lr3 = fw_lr.chunk(3, dim=-1)

                if layer.use_momentum:
                    momentum = layer.momentum_proj(hidden_states).float()
                    momentum = rearrange(momentum, "b s (n_h d) -> (b n_h) s d", n_h=layer.num_fw_heads)
                else:
                    momentum = None

                if layer.fp32_states:
                    fw_w0 = fw_w0.to(torch.float32)
                    fw_w1 = fw_w1.to(torch.float32)
                    fw_w2 = fw_w2.to(torch.float32)

                if layer.ttt_prenorm:
                    if layer.use_fused_kernel:
                        fw_x = prenorm_block_causal_lact_swiglu_fused_kernel_triton(
                            fw_w0, fw_w1, fw_w2, fast_q, fast_k, fast_v, fw_lr1, fw_lr2, fw_lr3,
                            chunk_size=layer.lact_chunk_size, use_muon=layer.use_muon, momentum=momentum
                        )
                    else:
                        fw_x = prenorm_block_causal_lact_swiglu(
                            fw_w0, fw_w1, fw_w2, fast_q, fast_k, fast_v, fw_lr1, fw_lr2, fw_lr3,
                            chunk_size=layer.lact_chunk_size, use_muon=layer.use_muon, momentum=momentum
                        )
                else:
                    if layer.use_fused_kernel:
                        fw_x = postnorm_block_causal_lact_swiglu_fused_kernel_triton(
                            fw_w0, fw_w1, fw_w2, fast_q, fast_k, fast_v, fw_lr1, fw_lr2, fw_lr3,
                            chunk_size=layer.lact_chunk_size, use_muon=layer.use_muon, momentum=momentum
                        )
                    else:
                        fw_x = block_causal_lact_swiglu(
                            fw_w0, fw_w1, fw_w2, fast_q, fast_k, fast_v, fw_lr1, fw_lr2, fw_lr3,
                            chunk_size=layer.lact_chunk_size, use_muon=layer.use_muon, momentum=momentum
                        )

                ttt_x_normed = layer.ttt_norm(fw_x)
                if layer.learnable_ttt_scale:
                    ttt_scale = F.silu(layer.ttt_scale_proj(hidden_states), inplace=False)
                    ttt_scale = rearrange(ttt_scale, "b s (n_h d) -> (b n_h) s d", n_h=layer.num_fw_heads)
                    ttt_x_normed = ttt_x_normed * ttt_scale
                ttt_x_normed = rearrange(ttt_x_normed, "(b n_h) s d -> b s (n_h d)", n_h=layer.num_fw_heads)
                return layer.o_proj(ttt_x_normed)

        ttt_only = LaCTTTTBranchOnly(full_layer).to(device=device, dtype=dtype)
        ttt_only.train()
        return ttt_only, hidden_size

    if model_key in {"fa_branch_only", "swa_branch_only"}:
        config_cls, _, qwen_model_cls, _ = load_qwen3_variant("qwen3_swa")
        layer_types = ["full_attention"] if model_key == "fa_branch_only" else ["sliding_attention"]
        config_kwargs = lact_config_to_qwen3_dict(
            base_cfg,
            seq_len=seq_len,
            layer_types=layer_types,
            sliding_window=window_size if model_key == "swa_branch_only" else None,
            attn_implementation="flash_attention_2",
        )
        config_kwargs["num_hidden_layers"] = 1
        config = config_cls(**config_kwargs)
        support_model = qwen_model_cls(config).to(device=device, dtype=dtype)
        support_model.train()
        attn = support_model.layers[0].self_attn
        # Keep a strong non-module reference for runtime helpers without registering the full support
        # model as a child module of the attention layer, which would pollute parameter counts.
        object.__setattr__(attn, "_benchmark_support_model_holder", [support_model])
        return attn, hidden_size

    if model_key == "gdn_branch_only":
        try:
            config_cls, _, _, gdn_cls = load_qwen3_variant("qwen3_gdn")
            if gdn_cls is None:
                raise RuntimeError("Unable to locate GatedDeltaNet class for qwen3_gdn")
            num_heads = QWEN35_2B_BASE_GDN["num_attention_heads"]
            head_dim = QWEN35_2B_BASE_GDN["head_dim"]
            config_kwargs = {
                "vocab_size": base_cfg["vocab_size"],
                "hidden_size": base_cfg["hidden_size"],
                "intermediate_size": base_cfg.get("intermediate_size", base_cfg["hidden_size"] * 4),
                "num_hidden_layers": 1,
                "num_attention_heads": num_heads,
                "num_key_value_heads": num_heads,
                "head_dim": head_dim,
                "hidden_act": base_cfg.get("hidden_act", "silu"),
                "max_position_embeddings": max(seq_len, base_cfg.get("max_position_embeddings", seq_len)),
                "initializer_range": base_cfg.get("initializer_range", 0.02),
                "rms_norm_eps": base_cfg.get("norm_eps", 1e-6),
                "use_cache": False,
                "tie_word_embeddings": base_cfg.get("tie_word_embeddings", False),
                "rope_theta": base_cfg.get("rope_theta", 1_000_000.0),
                "rope_scaling": base_cfg.get("rope_scaling"),
                "attention_bias": base_cfg.get("attention_bias", base_cfg.get("qkv_bias", False)),
                "use_sliding_window": False,
                "sliding_window": None,
                "max_window_layers": 1,
                "layer_types": ["linear_attention"],
                "attention_dropout": base_cfg.get("attention_dropout", 0.0),
                **QWEN35_2B_BASE_GDN,
            }
            config = config_cls(**config_kwargs)
            layer = gdn_cls(config=config, layer_idx=0).to(device=device, dtype=dtype)
        except (ImportError, AttributeError) as exc:
            qwen_import_error = repr(exc)
            try:
                with prepend_sys_path(FLASH_LINEAR_ATTENTION_ROOT):
                    try:
                        from fla.layers import GatedDeltaNet as FlaGatedDeltaNet
                    except Exception as fla_exc:
                        try:
                            from fla.layers.gated_deltanet import GatedDeltaNet as FlaGatedDeltaNet
                        except Exception as fla_direct_exc:
                            raise RuntimeError(
                                "Unable to import GatedDeltaNet from either qwen3_gdn or fla.layers. "
                                f"qwen3_gdn_error={qwen_import_error}; "
                                f"fla_layers_error={repr(fla_exc)}; "
                                f"fla_direct_error={repr(fla_direct_exc)}; "
                                f"flash_linear_attention_root={FLASH_LINEAR_ATTENTION_ROOT}"
                            ) from fla_direct_exc
            except RuntimeError:
                raise

            layer = FlaGatedDeltaNet(
                hidden_size=base_cfg["hidden_size"],
                expand_v=QWEN35_2B_BASE_GDN["expand_v"],
                head_dim=QWEN35_2B_BASE_GDN["head_dim"],
                num_heads=QWEN35_2B_BASE_GDN["num_attention_heads"],
                mode=QWEN35_2B_BASE_GDN["mode"],
                use_gate=QWEN35_2B_BASE_GDN["use_gate"],
                use_short_conv=QWEN35_2B_BASE_GDN["use_short_conv"],
                conv_size=QWEN35_2B_BASE_GDN["conv_size"],
                conv_bias=QWEN35_2B_BASE_GDN["conv_bias"],
                norm_eps=base_cfg.get("norm_eps", 1e-6),
                layer_idx=0,
            ).to(device=device, dtype=dtype)
            object.__setattr__(layer, "_benchmark_gdn_backend", f"fla.layers fallback: {qwen_import_error}")
        layer.train()
        return layer, hidden_size

    raise ValueError(f"Unknown kernel benchmark key: {model_key}")


def build_kernel_subject(
    *,
    model_key: str,
    seq_len: int,
    device: str,
    dtype_name: str,
    batch_size: int,
    base_config_path: Path = DEFAULT_LACT_CONFIG,
    sliding_window: int | None = None,
    lact_chunk_size: int | None = None,
    lact_attn_heads_override: int | None = None,
    lact_ttt_heads_override: int | None = None,
    use_fused_kernel: bool | None = None,
    paper_lm_defaults: bool = True,
) -> tuple[Callable[[Any], Any], int]:
    import torch

    module, hidden_size = build_kernel_module(
        model_key=model_key,
        seq_len=seq_len,
        device=device,
        dtype_name=dtype_name,
        batch_size=batch_size,
        base_config_path=base_config_path,
        sliding_window=sliding_window,
        lact_chunk_size=lact_chunk_size,
        lact_attn_heads_override=lact_attn_heads_override,
        lact_ttt_heads_override=lact_ttt_heads_override,
        use_fused_kernel=use_fused_kernel,
        paper_lm_defaults=paper_lm_defaults,
    )
    canonical_key = canonical_kernel_key(model_key)

    if canonical_key in {"fa_branch_only", "swa_branch_only"}:
        def runner(hidden_states: Any) -> Any:
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
            support_model_holder = getattr(module, "_benchmark_support_model_holder", None)
            support_model = support_model_holder[0] if support_model_holder else None
            if support_model is None:
                raise RuntimeError("Missing support model for attention benchmark subject")
            position_embeddings = support_model.rotary_emb(hidden_states, position_ids)
            return module(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=None,
                past_key_value=None,
                cache_position=None,
            )[0]
        return runner, hidden_size

    def runner(hidden_states: Any) -> Any:
        if canonical_key == "lact_full_layer":
            return module(hidden_states=hidden_states, attention_mask=None, use_cache=False)[0]
        if canonical_key == "gdn_branch_only":
            return module(
                hidden_states=hidden_states,
                attention_mask=None,
                past_key_values=None,
                use_cache=False,
                output_attentions=False,
            )[0]
        return module(hidden_states)

    return runner, hidden_size
