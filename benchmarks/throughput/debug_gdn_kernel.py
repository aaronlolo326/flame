from __future__ import annotations

import argparse
import sys
import traceback
from dataclasses import asdict, dataclass
from typing import Any

import torch
from einops import rearrange

from .adapters import DEFAULT_LACT_CONFIG, QWEN35_2B_BASE_GDN, load_json, load_qwen3_variant, resolve_dtype


@dataclass
class StageResult:
    name: str
    status: str
    notes: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug the local GatedDeltaNet path used by the throughput benchmark."
    )
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--attention-mask", action="store_true")
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--head-dim", type=int, default=None)
    parser.add_argument("--expand-v", type=float, default=None)
    parser.add_argument("--disable-short-conv", action="store_true")
    parser.add_argument("--disable-qk-norm", action="store_true")
    parser.add_argument("--disable-qk-l2norm-in-kernel", action="store_true")
    parser.add_argument("--mode", type=str, default="chunk", choices=["chunk", "fused_recurrent"])
    parser.add_argument("--head-first-kwarg", action="store_true")
    return parser.parse_args()


def print_env(device: torch.device, dtype: torch.dtype) -> None:
    print("[env]")
    print(f"  python={sys.version.split()[0]}")
    print(f"  torch={torch.__version__}")
    print(f"  cuda_runtime={torch.version.cuda}")
    try:
        import triton  # type: ignore

        print(f"  triton={triton.__version__}")
    except Exception:
        print("  triton=<unavailable>")
    print(f"  device={device}")
    print(f"  dtype={dtype}")
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        print(f"  gpu_name={props.name}")
        print(f"  capability={props.major}.{props.minor}")
        print(f"  total_memory_gb={props.total_memory / (1024 ** 3):.2f}")


def print_config(config: Any) -> None:
    keys = [
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "expand_v",
        "mode",
        "use_gate",
        "use_short_conv",
        "conv_size",
        "conv_bias",
        "selection_window_size",
        "use_qk_norm",
        "attention_bias",
        "rms_norm_eps",
        "norm_eps",
        "hidden_act",
    ]
    print("[config]")
    for key in keys:
        if hasattr(config, key):
            print(f"  {key}={getattr(config, key)}")
    if hasattr(config, "hidden_size") and hasattr(config, "num_attention_heads") and hasattr(config, "head_dim"):
        key_dim = int(config.num_attention_heads * config.head_dim)
        expected = int(0.75 * config.hidden_size)
        print(f"  derived_key_dim={key_dim}")
        print(f"  expected_key_dim_for_use_gate={expected}")
        print(f"  key_dim_matches_expected={key_dim == expected}")


def print_tensor(label: str, tensor: torch.Tensor) -> None:
    shape = tuple(tensor.shape)
    print(
        f"  {label}: shape={shape} dtype={tensor.dtype} device={tensor.device} "
        f"contiguous={tensor.is_contiguous()} stride={tensor.stride()}"
    )


def summarize_cuda() -> None:
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"  cuda_mem_allocated_gb={allocated:.2f} reserved_gb={reserved:.2f} peak_gb={peak:.2f}")


def fail_result(stage: str) -> StageResult:
    return StageResult(name=stage, status="error", notes=traceback.format_exc())


def build_config(args: argparse.Namespace) -> Any:
    base_cfg = load_json(DEFAULT_LACT_CONFIG)
    config_cls, _, _, _ = load_qwen3_variant("qwen3_gdn")
    hidden_size = int(base_cfg["hidden_size"])
    num_heads = args.num_heads or QWEN35_2B_BASE_GDN["num_attention_heads"]
    head_dim = args.head_dim or QWEN35_2B_BASE_GDN["head_dim"]
    gdn_overrides = {
        key: value
        for key, value in QWEN35_2B_BASE_GDN.items()
        if key not in {"num_attention_heads", "head_dim"}
    }
    if args.expand_v is not None:
        gdn_overrides["expand_v"] = args.expand_v
    gdn_overrides["mode"] = args.mode
    gdn_overrides["use_short_conv"] = not args.disable_short_conv
    gdn_overrides["use_qk_norm"] = not args.disable_qk_norm
    config = config_cls(
        vocab_size=base_cfg["vocab_size"],
        hidden_size=hidden_size,
        intermediate_size=base_cfg.get("intermediate_size", hidden_size * 4),
        num_hidden_layers=1,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        head_dim=head_dim,
        hidden_act=base_cfg.get("hidden_act", "silu"),
        max_position_embeddings=max(args.seq_len, int(base_cfg.get("max_position_embeddings", args.seq_len))),
        initializer_range=base_cfg.get("initializer_range", 0.02),
        rms_norm_eps=base_cfg.get("norm_eps", 1e-6),
        norm_eps=base_cfg.get("norm_eps", 1e-6),
        use_cache=False,
        tie_word_embeddings=base_cfg.get("tie_word_embeddings", False),
        rope_theta=base_cfg.get("rope_theta", 1_000_000.0),
        rope_scaling=base_cfg.get("rope_scaling"),
        attention_bias=base_cfg.get("attention_bias", base_cfg.get("qkv_bias", False)),
        torch_dtype=args.dtype,
        attn_implementation="flash_attention_2",
        **gdn_overrides,
    )
    config.mode = args.mode
    config.use_short_conv = not args.disable_short_conv
    config.use_qk_norm = not args.disable_qk_norm
    return config


def prepare_inputs(
    *,
    module: Any,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    use_cache: bool,
) -> dict[str, torch.Tensor | None]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)

    if module.use_qk_norm:
        q = module.q_norm(module.q_proj(hidden_states).view(hidden_shape))
        k = module.k_norm(module.k_proj(hidden_states).view(hidden_shape))
    else:
        q = module.q_proj(hidden_states).view(hidden_shape)
        k = module.k_proj(hidden_states).view(hidden_shape)

    v = module.v_proj(hidden_states)

    if module.use_short_conv:
        conv_mask = attention_mask[:, -hidden_states.shape[1]:] if attention_mask is not None else None
        q, _ = module.q_conv1d(
            x=q.view((*input_shape, -1)),
            mask=conv_mask,
            cache=None,
            output_final_state=use_cache,
            cu_seqlens=None,
        )
        k, _ = module.k_conv1d(
            x=k.view((*input_shape, -1)),
            mask=conv_mask,
            cache=None,
            output_final_state=use_cache,
            cu_seqlens=None,
        )
        v, _ = module.v_conv1d(
            x=v,
            mask=conv_mask,
            cache=None,
            output_final_state=use_cache,
            cu_seqlens=None,
        )
        q = q.view(hidden_shape)
        k = k.view(hidden_shape)

    q = module.act_fn(q)
    k = module.act_fn(k)
    v = module.act_fn(v)
    v = rearrange(v, "b t (h d) -> b t h d", d=module.head_v_dim)
    beta = module.b_proj(hidden_states).sigmoid()
    g = -module.A_log.float().exp() * torch.nn.functional.softplus(
        module.a_proj(hidden_states).float() + module.dt_bias
    )

    if attention_mask is not None:
        beta = beta.mul(attention_mask[:, -beta.shape[-2] :, None])
        g = g.mul(attention_mask[:, -g.shape[-2] :, None])

    return {
        "q": q,
        "k": k,
        "v": v,
        "beta": beta,
        "g": g,
    }


def run_stage(name: str, fn) -> tuple[StageResult, Any]:
    print(f"[stage] {name}")
    try:
        result = fn()
        print(f"[ok] {name}")
        summarize_cuda()
        return StageResult(name=name, status="ok"), result
    except Exception:
        notes = traceback.format_exc()
        print(f"[error] {name}")
        print(notes)
        summarize_cuda()
        return StageResult(name=name, status="error", notes=notes), None


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    dtype = resolve_dtype(torch, args.dtype)
    print_env(device, dtype)

    config = build_config(args)
    print_config(config)

    _, _, _, gdn_cls = load_qwen3_variant("qwen3_gdn")
    if gdn_cls is None:
        raise RuntimeError("Unable to locate GatedDeltaNet class for qwen3_gdn")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    stage_results: list[StageResult] = []

    build_result, module = run_stage(
        "build_module",
        lambda: gdn_cls(config=config, layer_idx=0).to(device=device, dtype=dtype).train(),
    )
    stage_results.append(build_result)
    if build_result.status != "ok":
        return 1

    hidden_states = torch.randn(
        args.batch_size,
        args.seq_len,
        config.hidden_size,
        device=device,
        dtype=dtype,
        requires_grad=args.backward,
    )
    attention_mask = None
    if args.attention_mask:
        attention_mask = torch.ones(args.batch_size, args.seq_len, device=device, dtype=torch.long)

    print("[inputs]")
    print_tensor("hidden_states", hidden_states)
    if attention_mask is not None:
        print_tensor("attention_mask", attention_mask)

    prep_result, prepared = run_stage(
        "prepare_qkvgb",
        lambda: prepare_inputs(
            module=module,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            use_cache=args.use_cache,
        ),
    )
    stage_results.append(prep_result)
    if prep_result.status == "ok":
        print("[prepared]")
        for key in ("q", "k", "v", "beta", "g"):
            value = prepared[key]
            if isinstance(value, torch.Tensor):
                print_tensor(key, value)

    def run_direct_kernel():
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        kwargs: dict[str, Any] = {}
        if args.head_first_kwarg:
            kwargs["head_first"] = False
        return chunk_gated_delta_rule(
            q=prepared["q"],
            k=prepared["k"],
            v=prepared["v"],
            g=prepared["g"],
            beta=prepared["beta"],
            initial_state=None,
            output_final_state=args.use_cache,
            cu_seqlens=None,
            use_qk_l2norm_in_kernel=not args.disable_qk_l2norm_in_kernel,
            **kwargs,
        )

    if prep_result.status == "ok":
        direct_result, direct_outputs = run_stage("direct_chunk_gated_delta_rule", run_direct_kernel)
        stage_results.append(direct_result)
        if direct_result.status == "ok":
            output, final_state = direct_outputs
            print("[direct_kernel_outputs]")
            print_tensor("o", output)
            if isinstance(final_state, torch.Tensor):
                print_tensor("final_state", final_state)
            if args.backward:
                backward_result, _ = run_stage(
                    "direct_backward",
                    lambda: output.float().square().mean().backward(),
                )
                stage_results.append(backward_result)
                if hidden_states.grad is not None:
                    print_tensor("hidden_states.grad", hidden_states.grad)

    def run_full_forward():
        return module(
            hidden_states,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=args.use_cache,
            output_attentions=False,
        )

    forward_result, forward_outputs = run_stage("module_forward", run_full_forward)
    stage_results.append(forward_result)
    if forward_result.status == "ok":
        output, _, cache = forward_outputs
        print("[module_outputs]")
        print_tensor("o", output)
        if cache is not None:
            print(f"  cache_type={type(cache).__name__}")
        if args.backward:
            backward_result, _ = run_stage(
                "module_backward",
                lambda: output.float().square().mean().backward(),
            )
            stage_results.append(backward_result)
            if hidden_states.grad is not None:
                print_tensor("hidden_states.grad", hidden_states.grad)

    print("[summary]")
    for item in stage_results:
        print(asdict(item))

    return 0 if all(item.status == "ok" for item in stage_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
