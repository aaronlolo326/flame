# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import fla  # noqa: F401
import custom_models  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Layer-wise alignment checker between Qwen3 teacher and hybrid Qwen3+LaCT model."
    )
    parser.add_argument("--teacher", type=str, required=True, help="Teacher model id/path, e.g. Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--hybrid", type=str, required=True, help="Hybrid checkpoint dir")
    parser.add_argument("--text", type=str, default="The quick brown fox jumps over the lazy dog.")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path")
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }
    return mapping[name]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_bshd(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, S, H, D], cos/sin: [B, S, D]
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    return (x * cos) + (rotate_half(x) * sin)


def tensor_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    if a.shape != b.shape:
        return {
            "shape_match": 0.0,
            "max_abs": float("inf"),
            "mean_abs": float("inf"),
            "rmse": float("inf"),
            "finite_count": 0.0,
            "teacher_nan_count": float(torch.isnan(a).sum().item()),
            "teacher_inf_count": float(torch.isinf(a).sum().item()),
            "hybrid_nan_count": float(torch.isnan(b).sum().item()),
            "hybrid_inf_count": float(torch.isinf(b).sum().item()),
        }

    a32 = a.float()
    b32 = b.float()
    a_is_finite = torch.isfinite(a32)
    b_is_finite = torch.isfinite(b32)
    both_finite = a_is_finite & b_is_finite

    if both_finite.any():
        d = a32[both_finite] - b32[both_finite]
        abs_d = d.abs()
        max_abs = float(abs_d.max().item())
        mean_abs = float(abs_d.mean().item())
        rmse = float(torch.sqrt(torch.mean(d * d)).item())
        finite_count = float(both_finite.sum().item())
    else:
        max_abs = float("nan")
        mean_abs = float("nan")
        rmse = float("nan")
        finite_count = 0.0

    return {
        "shape_match": 1.0,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rmse": rmse,
        "finite_count": finite_count,
        "teacher_nan_count": float(torch.isnan(a32).sum().item()),
        "teacher_inf_count": float(torch.isinf(a32).sum().item()),
        "hybrid_nan_count": float(torch.isnan(b32).sum().item()),
        "hybrid_inf_count": float(torch.isinf(b32).sum().item()),
    }


def make_capture_dict() -> dict[str, dict[int, torch.Tensor]]:
    return {
        "layer_in": {},
        "layer_out": {},
        "q_norm": {},
        "k_norm": {},
        "attn_pre_o_proj": {},
    }


def register_alignment_hooks(model) -> tuple[dict[str, dict[int, torch.Tensor]], list[Any]]:
    captures = make_capture_dict()
    handles: list[Any] = []

    layers = model.model.layers

    def make_layer_pre_hook(idx: int):
        def _hook(_module, inputs):
            captures["layer_in"][idx] = inputs[0].detach()

        return _hook

    def make_layer_out_hook(idx: int):
        def _hook(_module, _inputs, output):
            if isinstance(output, tuple):
                output = output[0]
            captures["layer_out"][idx] = output.detach()

        return _hook

    def make_simple_out_hook(name: str, idx: int):
        def _hook(_module, _inputs, output):
            captures[name][idx] = output.detach()

        return _hook

    def make_o_proj_pre_hook(idx: int):
        def _hook(_module, inputs):
            captures["attn_pre_o_proj"][idx] = inputs[0].detach()

        return _hook

    for idx, layer in enumerate(layers):
        attn = layer.self_attn
        handles.append(layer.register_forward_pre_hook(make_layer_pre_hook(idx)))
        handles.append(layer.register_forward_hook(make_layer_out_hook(idx)))
        handles.append(attn.q_norm.register_forward_hook(make_simple_out_hook("q_norm", idx)))
        handles.append(attn.k_norm.register_forward_hook(make_simple_out_hook("k_norm", idx)))
        handles.append(attn.o_proj.register_forward_pre_hook(make_o_proj_pre_hook(idx)))

    return captures, handles


def remove_hooks(handles: list[Any]) -> None:
    for h in handles:
        h.remove()


@torch.no_grad()
def run_forward_with_captures(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[Any, dict[str, dict[int, torch.Tensor]]]:
    captures, handles = register_alignment_hooks(model)
    try:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            logits_to_keep=0,
        )
    finally:
        remove_hooks(handles)
    return outputs, captures


def compute_rope_metrics(
    teacher,
    hybrid,
    teacher_caps: dict[str, dict[int, torch.Tensor]],
    hybrid_caps: dict[str, dict[int, torch.Tensor]],
    position_ids: torch.Tensor,
) -> dict[int, dict[str, dict[str, float]]]:
    rope_metrics: dict[int, dict[str, dict[str, float]]] = {}

    num_layers = min(len(teacher.model.layers), len(hybrid.model.layers))
    for idx in range(num_layers):
        tq = teacher_caps["q_norm"][idx]
        tk = teacher_caps["k_norm"][idx]
        hq = hybrid_caps["q_norm"][idx]
        hk = hybrid_caps["k_norm"][idx]

        t_cos, t_sin = teacher.model.rotary_emb(tq, position_ids)
        h_cos, h_sin = hybrid.model.layers[idx].self_attn.rotary(position_ids, hq.dtype, hq.device)

        tq_rope = apply_rope_bshd(tq, t_cos, t_sin)
        tk_rope = apply_rope_bshd(tk, t_cos, t_sin)
        hq_rope = apply_rope_bshd(hq, h_cos, h_sin)
        hk_rope = apply_rope_bshd(hk, h_cos, h_sin)

        rope_metrics[idx] = {
            "cos": tensor_metrics(t_cos, h_cos),
            "sin": tensor_metrics(t_sin, h_sin),
            "q_after_rope": tensor_metrics(tq_rope, hq_rope),
            "k_after_rope": tensor_metrics(tk_rope, hk_rope),
        }

    return rope_metrics


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    dtype = resolve_dtype(args.dtype)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available but --device cuda was requested.")

    tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    encoded = tokenizer(
        args.text,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0).expand(input_ids.shape[0], -1)

    load_dtype = dtype if dtype != torch.float32 else None
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher,
        trust_remote_code=True,
        torch_dtype=load_dtype,
    ).to(device=device, dtype=dtype).eval()
    hybrid = AutoModelForCausalLM.from_pretrained(
        args.hybrid,
        trust_remote_code=True,
        torch_dtype=load_dtype,
    ).to(device=device, dtype=dtype).eval()

    teacher_out, teacher_caps = run_forward_with_captures(teacher, input_ids, attention_mask)
    hybrid_out, hybrid_caps = run_forward_with_captures(hybrid, input_ids, attention_mask)

    layer0_rotary = hybrid.model.layers[0].self_attn.rotary
    # Recompute theoretical inv_freq instead of reading potentially stale non-persistent buffers.
    layer0_inv = 1.0 / (
        float(layer0_rotary.base)
        ** (
            torch.arange(0, int(layer0_rotary.dim), 2, device=device, dtype=torch.float32)
            / float(layer0_rotary.dim)
        )
    )
    layer0_cos, layer0_sin = layer0_rotary(position_ids, dtype=dtype, device=device)

    num_layers = min(len(teacher.model.layers), len(hybrid.model.layers))
    layer_metrics: dict[int, dict[str, dict[str, float]]] = {}
    for idx in range(num_layers):
        layer_metrics[idx] = {
            "layer_in": tensor_metrics(teacher_caps["layer_in"][idx], hybrid_caps["layer_in"][idx]),
            "layer_out": tensor_metrics(teacher_caps["layer_out"][idx], hybrid_caps["layer_out"][idx]),
            "q_norm": tensor_metrics(teacher_caps["q_norm"][idx], hybrid_caps["q_norm"][idx]),
            "k_norm": tensor_metrics(teacher_caps["k_norm"][idx], hybrid_caps["k_norm"][idx]),
            "attn_pre_o_proj": tensor_metrics(
                teacher_caps["attn_pre_o_proj"][idx],
                hybrid_caps["attn_pre_o_proj"][idx],
            ),
        }

    rope_metrics = compute_rope_metrics(
        teacher=teacher,
        hybrid=hybrid,
        teacher_caps=teacher_caps,
        hybrid_caps=hybrid_caps,
        position_ids=position_ids,
    )

    logits_metrics = tensor_metrics(teacher_out.logits, hybrid_out.logits)

    first_non_finite_layer = None
    for i in range(num_layers):
        m = layer_metrics[i]["layer_out"]
        if (
            m["teacher_nan_count"] > 0
            or m["teacher_inf_count"] > 0
            or m["hybrid_nan_count"] > 0
            or m["hybrid_inf_count"] > 0
        ):
            first_non_finite_layer = i
            break

    summary = {
        "meta": {
            "teacher": args.teacher,
            "hybrid": args.hybrid,
            "device": str(device),
            "dtype": str(dtype),
            "seq_len": int(input_ids.shape[1]),
            "num_layers_compared": num_layers,
            "text": args.text,
            "input_ids": input_ids.detach().cpu().tolist(),
            "attention_mask": attention_mask.detach().cpu().tolist(),
            "position_ids": position_ids.detach().cpu().tolist(),
        },
        "global": {
            "logits": logits_metrics,
            "max_layer_out_abs": max(layer_metrics[i]["layer_out"]["max_abs"] for i in range(num_layers)),
            "max_q_rope_abs": max(rope_metrics[i]["q_after_rope"]["max_abs"] for i in range(num_layers)),
            "max_k_rope_abs": max(rope_metrics[i]["k_after_rope"]["max_abs"] for i in range(num_layers)),
            "first_non_finite_layer_out": -1 if first_non_finite_layer is None else int(first_non_finite_layer),
            "layer0_rotary": {
                "inv_freq_nan_count": float(torch.isnan(layer0_inv).sum().item()),
                "inv_freq_inf_count": float(torch.isinf(layer0_inv).sum().item()),
                "inv_freq_min": float(layer0_inv.min().item()),
                "inv_freq_max": float(layer0_inv.max().item()),
                "cos_nan_count": float(torch.isnan(layer0_cos.float()).sum().item()),
                "cos_inf_count": float(torch.isinf(layer0_cos.float()).sum().item()),
                "sin_nan_count": float(torch.isnan(layer0_sin.float()).sum().item()),
                "sin_inf_count": float(torch.isinf(layer0_sin.float()).sum().item()),
            },
        },
        "per_layer": {
            str(i): {
                **layer_metrics[i],
                "rope": rope_metrics[i],
            }
            for i in range(num_layers)
        },
    }

    print("=== Alignment Summary ===")
    print(json.dumps(summary["meta"], indent=2, ensure_ascii=False))
    print("=== Global Metrics ===")
    print(json.dumps(summary["global"], indent=2, ensure_ascii=False))

    print("=== Layer max_abs snapshot (first 8 layers) ===")
    for i in range(min(8, num_layers)):
        print(
            f"layer {i:02d} | "
            f"in={layer_metrics[i]['layer_in']['max_abs']:.3e} "
            f"out={layer_metrics[i]['layer_out']['max_abs']:.3e} "
            f"q_rope={rope_metrics[i]['q_after_rope']['max_abs']:.3e} "
            f"k_rope={rope_metrics[i]['k_after_rope']['max_abs']:.3e} "
            f"attn_pre_o_proj={layer_metrics[i]['attn_pre_o_proj']['max_abs']:.3e}"
        )

    print("=== Non-finite stats snapshot (first 8 layers) ===")
    for i in range(min(8, num_layers)):
        out_m = layer_metrics[i]["layer_out"]
        qr_m = rope_metrics[i]["q_after_rope"]
        print(
            f"layer {i:02d} | "
            f"layer_out teacher_nan={int(out_m['teacher_nan_count'])} hybrid_nan={int(out_m['hybrid_nan_count'])} "
            f"teacher_inf={int(out_m['teacher_inf_count'])} hybrid_inf={int(out_m['hybrid_inf_count'])} | "
            f"q_rope teacher_nan={int(qr_m['teacher_nan_count'])} hybrid_nan={int(qr_m['hybrid_nan_count'])}"
        )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"saved={args.out}")


if __name__ == "__main__":
    main()
