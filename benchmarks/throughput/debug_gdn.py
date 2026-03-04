from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from .adapters import DEFAULT_LACT_CONFIG, QWEN35_2B_BASE_GDN, load_json, load_qwen3_variant


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug GatedDeltaNet kernel/runtime issues.")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_LACT_CONFIG)
    parser.add_argument(
        "--disable-import-patch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the benchmark loader as-is by default. If set, import qwen3_gdn directly without the temporary torch.compile patch.",
    )
    return parser.parse_args()


def print_env(torch_module: object) -> None:
    print("Environment", flush=True)
    print(f"  torch={torch_module.__version__}", flush=True)
    print(f"  cuda_available={torch_module.cuda.is_available()}", flush=True)
    if torch_module.cuda.is_available():
        print(f"  device_name={torch_module.cuda.get_device_name(0)}", flush=True)
        print(f"  capability={torch_module.cuda.get_device_capability(0)}", flush=True)
    try:
        import triton

        print(f"  triton={triton.__version__}", flush=True)
    except Exception as exc:  # pragma: no cover - best effort
        print(f"  triton=<unavailable> ({exc})", flush=True)


def import_qwen3_gdn(disable_import_patch: bool):
    if not disable_import_patch:
        return load_qwen3_variant("qwen3_gdn")

    import importlib.util
    import sys

    model_root = Path("/work/yufei/projects/hybrid_models/qwen3_gdn")
    for path in [str(model_root.parent), str(model_root)]:
        if path not in sys.path:
            sys.path.insert(0, path)

    def load(name: str, file_path: Path, alias: str):
        spec = importlib.util.spec_from_file_location(name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to import {file_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        sys.modules[alias] = module
        spec.loader.exec_module(module)
        return module

    config_mod = load("debug_qwen3_gdn_config", model_root / "configuration_qwen3.py", "configuration_qwen3")
    modeling_mod = load("debug_qwen3_gdn_modeling", model_root / "modeling_qwen3.py", "modeling_qwen3")
    return config_mod.Qwen3Config, modeling_mod.Qwen3ForCausalLM, modeling_mod.Qwen3Model, modeling_mod.GatedDeltaNet


def main() -> None:
    import torch
    import torch.nn.functional as F
    from einops import rearrange

    args = parse_args()
    print_env(torch)

    config_cls, _, _, gdn_cls = import_qwen3_gdn(args.disable_import_patch)
    if gdn_cls is None:
        raise RuntimeError("Unable to import GatedDeltaNet")

    base_cfg = load_json(args.base_config)
    dtype = getattr(torch, args.dtype)

    config_kwargs = {
        "vocab_size": base_cfg["vocab_size"],
        "hidden_size": base_cfg["hidden_size"],
        "intermediate_size": base_cfg.get("intermediate_size", base_cfg["hidden_size"] * 4),
        "num_hidden_layers": 1,
        "num_attention_heads": QWEN35_2B_BASE_GDN["num_attention_heads"],
        "num_key_value_heads": QWEN35_2B_BASE_GDN["num_attention_heads"],
        "head_dim": QWEN35_2B_BASE_GDN["head_dim"],
        "hidden_act": base_cfg.get("hidden_act", "silu"),
        "max_position_embeddings": max(args.seq_len, base_cfg.get("max_position_embeddings", args.seq_len)),
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

    print("Config", flush=True)
    for key in [
        "hidden_size",
        "num_attention_heads",
        "head_dim",
        "expand_v",
        "mode",
        "use_gate",
        "use_short_conv",
        "conv_size",
        "use_qk_norm",
    ]:
        print(f"  {key}={getattr(config, key)}", flush=True)

    layer = gdn_cls(config=config, layer_idx=0).to(device=args.device, dtype=dtype)
    layer.train()
    hidden_states = torch.randn(
        args.batch_size,
        args.seq_len,
        config.hidden_size,
        device=args.device,
        dtype=dtype,
        requires_grad=True,
    )

    print("Input", flush=True)
    print(f"  hidden_states={tuple(hidden_states.shape)} dtype={hidden_states.dtype}", flush=True)

    # Stage 1: projections + local preprocessing
    print("\n[stage 1] projections/preprocessing", flush=True)
    try:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.head_dim)
        if layer.use_qk_norm:
            q = layer.q_norm(layer.q_proj(hidden_states).view(hidden_shape))
            k = layer.k_norm(layer.k_proj(hidden_states).view(hidden_shape))
        else:
            q = layer.q_proj(hidden_states)
            k = layer.k_proj(hidden_states)
        v = layer.v_proj(hidden_states)
        print(f"  q={tuple(q.shape)} {q.dtype}", flush=True)
        print(f"  k={tuple(k.shape)} {k.dtype}", flush=True)
        print(f"  v={tuple(v.shape)} {v.dtype}", flush=True)

        q_conv, _ = layer.q_conv1d(x=q.view((*input_shape, -1)), mask=None, cache=None, output_final_state=False, cu_seqlens=None)
        k_conv, _ = layer.k_conv1d(x=k.view((*input_shape, -1)), mask=None, cache=None, output_final_state=False, cu_seqlens=None)
        v_conv, _ = layer.v_conv1d(x=v, mask=None, cache=None, output_final_state=False, cu_seqlens=None)
        q_conv = q_conv.view(hidden_shape)
        k_conv = k_conv.view(hidden_shape)
        q_conv = layer.act_fn(q_conv)
        k_conv = layer.act_fn(k_conv)
        v_conv = layer.act_fn(v_conv)
        v_kernel = rearrange(v_conv, "b t (h d) -> b t h d", d=layer.head_v_dim)
        beta = layer.b_proj(hidden_states).sigmoid()
        g = -layer.A_log.float().exp() * F.softplus(layer.a_proj(hidden_states).float() + layer.dt_bias)
        print(f"  q_conv={tuple(q_conv.shape)} {q_conv.dtype}", flush=True)
        print(f"  k_conv={tuple(k_conv.shape)} {k_conv.dtype}", flush=True)
        print(f"  v_kernel={tuple(v_kernel.shape)} {v_kernel.dtype}", flush=True)
        print(f"  beta={tuple(beta.shape)} {beta.dtype}", flush=True)
        print(f"  g={tuple(g.shape)} {g.dtype}", flush=True)
    except Exception:
        traceback.print_exc()
        return

    # Stage 2: direct kernel call
    print("\n[stage 2] direct chunk_gated_delta_rule", flush=True)
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        o, state = chunk_gated_delta_rule(
            q=q_conv,
            k=k_conv,
            v=v_kernel,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            cu_seqlens=None,
            use_qk_l2norm_in_kernel=True,
        )
        print(f"  kernel_output={tuple(o.shape)} {o.dtype}", flush=True)
        print(f"  final_state={state}", flush=True)
        loss = o.float().square().mean()
        loss.backward(retain_graph=True)
        print("  backward=ok", flush=True)
    except Exception:
        traceback.print_exc()

    # Stage 3: full layer forward
    print("\n[stage 3] full GatedDeltaNet.forward", flush=True)
    try:
        if hidden_states.grad is not None:
            hidden_states.grad = None
        out, _, _ = layer(
            hidden_states=hidden_states,
            attention_mask=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
        )
        print(f"  layer_output={tuple(out.shape)} {out.dtype}", flush=True)
        loss = out.float().square().mean()
        loss.backward()
        print("  backward=ok", flush=True)
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    main()
