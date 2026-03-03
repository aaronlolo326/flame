from __future__ import annotations

import importlib.util
import json
import math
import sys
import weakref
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
    use_fused_kernel: bool | None = None,
    paper_lm_defaults: bool = True,
) -> tuple[Any, dict[str, Any]]:
    import torch

    base_cfg = load_json(base_config_path)
    dtype = resolve_dtype(torch, dtype_name)
    chunk_size, window_size = resolve_chunk_and_window(
        seq_len=seq_len,
        base_cfg=base_cfg,
        lact_chunk_size=lact_chunk_size,
        window_size=sliding_window,
        paper_lm_defaults=paper_lm_defaults,
    )

    if model_key == "lact":
        config_cls, model_cls, _ = import_lact_modules()
        config_kwargs = dict(base_cfg)
        config_kwargs["max_position_embeddings"] = max(seq_len, config_kwargs.get("max_position_embeddings", seq_len))
        config_kwargs["use_cache"] = False
        config_kwargs["lact_chunk_size"] = chunk_size
        config_kwargs["window_size"] = window_size
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
            window = window_size
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
        # Keep an external reference for runtime helpers without registering the full support model
        # as a child module of the attention layer, which would pollute parameter counts.
        object.__setattr__(attn, "_benchmark_support_model_ref", weakref.ref(support_model))
        return attn, hidden_size

    if model_key == "gdn_branch_only":
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
        use_fused_kernel=use_fused_kernel,
        paper_lm_defaults=paper_lm_defaults,
    )
    canonical_key = canonical_kernel_key(model_key)

    if canonical_key in {"fa_branch_only", "swa_branch_only"}:
        def runner(hidden_states: Any) -> Any:
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
            support_model_ref = getattr(module, "_benchmark_support_model_ref", None)
            support_model = support_model_ref() if support_model_ref is not None else None
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
