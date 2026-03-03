from __future__ import annotations

import importlib.util
import json
import math
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[3]
FLAME_ROOT = REPO_ROOT / "flame"
HYBRID_ROOT = REPO_ROOT / "hybrid_models"
DEFAULT_LACT_CONFIG = FLAME_ROOT / "configs" / "qwen3_lact_1B4.json"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    kind: str


MODEL_SPECS = {
    "lact": ModelSpec("lact", "LaCT", "lact"),
    "full_attention": ModelSpec("full_attention", "Full Attention (FlashAttention)", "qwen3_swa"),
    "hybrid_swa": ModelSpec("hybrid_swa", "75% SWA + 25% FA", "qwen3_swa"),
    "hybrid_gdn": ModelSpec("hybrid_gdn", "75% GatedDeltaNet + 25% FA", "qwen3_gdn"),
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


def load_qwen3_variant(variant: str) -> tuple[type, type, type, type | None]:
    if variant not in {"qwen3_swa", "qwen3_gdn"}:
        raise ValueError(f"Unknown Qwen3 variant: {variant}")
    model_root = HYBRID_ROOT / ("qwen3_swa" if variant == "qwen3_swa" else "qwen3_gdn")
    with prepend_sys_path(HYBRID_ROOT, model_root):
        config_mod = load_module_from_path(
            f"{variant}_configuration_qwen3",
            model_root / "configuration_qwen3.py",
            aliases=["configuration_qwen3"],
        )
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


def build_whole_model(
    *,
    model_key: str,
    seq_len: int,
    device: str,
    dtype_name: str,
    base_config_path: Path = DEFAULT_LACT_CONFIG,
    sliding_window: int | None = None,
    use_fused_kernel: bool | None = None,
) -> tuple[Any, dict[str, Any]]:
    import torch

    base_cfg = load_json(base_config_path)
    dtype = resolve_dtype(torch, dtype_name)

    if model_key == "lact":
        config_cls, model_cls, _ = import_lact_modules()
        config_kwargs = dict(base_cfg)
        config_kwargs["max_position_embeddings"] = max(seq_len, config_kwargs.get("max_position_embeddings", seq_len))
        config_kwargs["use_cache"] = False
        if use_fused_kernel is not None:
            config_kwargs["use_fused_kernel"] = use_fused_kernel
        config = config_cls(**config_kwargs)
        model = model_cls(config)
    elif model_key in {"full_attention", "hybrid_swa", "hybrid_gdn"}:
        spec = MODEL_SPECS[model_key]
        config_cls, model_cls, _, _ = load_qwen3_variant(spec.kind)
        if model_key == "full_attention":
            layer_types = ["full_attention"] * base_cfg["num_hidden_layers"]
            window = None
        elif model_key == "hybrid_swa":
            layer_types = build_layer_types(base_cfg["num_hidden_layers"], "sliding_attention")
            window = sliding_window or base_cfg.get("window_size", base_cfg.get("lact_chunk_size", 2048))
        else:
            layer_types = build_layer_types(base_cfg["num_hidden_layers"], "linear_attention")
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
            base_cfg,
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
    return model, base_cfg


def build_kernel_subject(
    *,
    model_key: str,
    seq_len: int,
    device: str,
    dtype_name: str,
    batch_size: int,
    base_config_path: Path = DEFAULT_LACT_CONFIG,
    sliding_window: int | None = None,
    use_fused_kernel: bool | None = None,
) -> tuple[Callable[[Any], Any], int]:
    import torch

    base_cfg = load_json(base_config_path)
    dtype = resolve_dtype(torch, dtype_name)
    hidden_size = base_cfg["hidden_size"]

    if model_key == "lact":
        config_cls, _, layer_cls = import_lact_modules()
        config_kwargs = dict(base_cfg)
        config_kwargs["max_position_embeddings"] = max(seq_len, config_kwargs.get("max_position_embeddings", seq_len))
        if use_fused_kernel is not None:
            config_kwargs["use_fused_kernel"] = use_fused_kernel
        config = config_cls(**config_kwargs)
        layer = layer_cls(
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
        layer.train()

        def runner(hidden_states: Any) -> Any:
            return layer(hidden_states=hidden_states, attention_mask=None, use_cache=False)[0]

        return runner, hidden_size

    if model_key in {"full_attention", "hybrid_swa"}:
        config_cls, _, qwen_model_cls, _ = load_qwen3_variant("qwen3_swa")
        layer_types = ["full_attention"] if model_key == "full_attention" else ["sliding_attention"]
        config_kwargs = lact_config_to_qwen3_dict(
            base_cfg,
            seq_len=seq_len,
            layer_types=layer_types,
            sliding_window=(sliding_window or base_cfg.get("window_size", 2048)) if model_key == "hybrid_swa" else None,
            attn_implementation="flash_attention_2",
        )
        config_kwargs["num_hidden_layers"] = 1
        config = config_cls(**config_kwargs)
        support_model = qwen_model_cls(config).to(device=device, dtype=dtype)
        support_model.train()
        attn = support_model.layers[0].self_attn

        def runner(hidden_states: Any) -> Any:
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
            position_embeddings = support_model.rotary_emb(hidden_states, position_ids)
            return attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=None,
                past_key_value=None,
                cache_position=None,
            )[0]

        return runner, hidden_size

    if model_key == "hybrid_gdn":
        config_cls, _, _, gdn_cls = load_qwen3_variant("qwen3_gdn")
        if gdn_cls is None:
            raise RuntimeError("Unable to locate GatedDeltaNet class for qwen3_gdn")
        config_kwargs = lact_config_to_qwen3_dict(
            base_cfg,
            seq_len=seq_len,
            layer_types=["linear_attention"],
            sliding_window=None,
            attn_implementation="flash_attention_2",
            gdn_overrides={
                "expand_v": 1,
                "mode": "chunk",
                "use_gate": True,
                "use_short_conv": True,
                "conv_size": 4,
                "conv_bias": False,
                "pad_value": 0,
                "selection_window_size": 100,
                "use_qk_norm": True,
            },
        )
        config_kwargs["num_hidden_layers"] = 1
        config = config_cls(**config_kwargs)
        layer = gdn_cls(config=config, layer_idx=0).to(device=device, dtype=dtype)
        layer.train()

        def runner(hidden_states: Any) -> Any:
            return layer(
                hidden_states=hidden_states,
                attention_mask=None,
                past_key_values=None,
                use_cache=False,
                output_attentions=False,
            )[0]

        return runner, hidden_size

    raise ValueError(f"Unknown kernel benchmark key: {model_key}")

