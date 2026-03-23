# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import fla  # noqa: F401
import custom_models  # noqa: F401
from custom_models.hybrid_qwen3_lact_model.modeling_hybrid_qwen3_lact import (
    apply_rotary_pos_emb,
    l2_norm,
)


def load_text_samples(
    *,
    text: str | None,
    text_file: str | None,
    max_samples: int,
) -> list[str]:
    samples: list[str] = []
    if text_file is not None:
        with open(text_file, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                samples.append(line)
                if len(samples) >= max_samples:
                    break
    elif text is not None:
        samples = [text]
    else:
        raise ValueError("Provide either --text or --text-file.")
    return samples


@torch.no_grad()
def collect_layer_inputs(model, input_ids: torch.Tensor, target_layers: list[int]) -> dict[int, torch.Tensor]:
    captured: dict[int, torch.Tensor] = {}
    hooks = []

    def make_hook(idx: int):
        def _hook(module, inputs):
            captured[idx] = inputs[0].detach()
        return _hook

    for idx in target_layers:
        hooks.append(model.model.layers[idx].register_forward_pre_hook(make_hook(idx)))

    try:
        model(input_ids=input_ids, logits_to_keep=input_ids.shape[1])
    finally:
        for hook in hooks:
            hook.remove()
    return captured


@torch.no_grad()
def compute_branch_features(branch, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len, _ = hidden_states.shape
    q, k, v = branch.qkv(hidden_states).chunk(3, dim=-1)
    if branch.attn_qk_norm:
        q, k = branch.q_norm(q), branch.k_norm(k)
    _, fast_k = branch._rescale_qk(q, k)
    fast_v = v

    fast_k = fast_k.view(batch_size, seq_len, branch.num_fw_heads, branch.fw_head_dim)
    fast_v = fast_v.view(batch_size, seq_len, branch.num_fw_heads, branch.fw_head_dim)

    if branch.qkv_silu:
        fast_k = F.silu(fast_k)
        if not branch.no_v_silu:
            fast_v = F.silu(fast_v)

    fast_k = fast_k.permute(0, 2, 1, 3).reshape(batch_size * branch.num_fw_heads, seq_len, branch.fw_head_dim)
    fast_v = fast_v.permute(0, 2, 1, 3).reshape(batch_size * branch.num_fw_heads, seq_len, branch.fw_head_dim)
    fast_k = l2_norm(fast_k)

    if not branch.ttt_nope:
        fast_k_rope = fast_k.view(batch_size, branch.num_fw_heads, seq_len, branch.fw_head_dim).transpose(1, 2)
        cos, sin = branch.rotary(position_ids, fast_k.dtype, fast_k.device)
        _, fast_k_rope = apply_rotary_pos_emb(fast_k_rope, fast_k_rope, cos, sin)
        fast_k = fast_k_rope.transpose(1, 2).reshape(batch_size * branch.num_fw_heads, seq_len, branch.fw_head_dim)

    return fast_k, fast_v


def low_rank_factorize(target: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    u, s, vh = torch.linalg.svd(target, full_matrices=False)
    rank = min(rank, s.shape[0])
    s_sqrt = s[:rank].sqrt()
    left = u[:, :rank] * s_sqrt.unsqueeze(0)
    right = s_sqrt.unsqueeze(1) * vh[:rank, :]
    return left.contiguous(), right.contiguous()


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-B teacher feature collection for hybrid Qwen3+LaCT initialization.")
    parser.add_argument("--teacher", type=str, required=True, help="Teacher Qwen3 model id/path")
    parser.add_argument("--hybrid-model", type=str, required=True, help="Converted hybrid checkpoint path")
    parser.add_argument("--text", type=str, default=None, help="Single fallback text sample.")
    parser.add_argument("--text-file", type=str, default=None, help="Plain-text file, one sequence per line.")
    parser.add_argument("--max-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--out", type=Path, required=True, help="Output .pt file with collected stats")
    args = parser.parse_args()

    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, trust_remote_code=True).eval()
    hybrid = AutoModelForCausalLM.from_pretrained(args.hybrid_model, trust_remote_code=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)

    samples = load_text_samples(text=args.text, text_file=args.text_file, max_samples=args.max_samples)
    if not samples:
        raise ValueError("No text samples found.")

    target_layers = [i for i, t in enumerate(hybrid.config.hybrid_layer_types) if t == "lact"]
    accumulators: dict[int, list[dict[str, torch.Tensor | int]]] = {}
    for idx in target_layers:
        branch = hybrid.model.layers[idx].self_attn.lact_branch
        accumulators[idx] = [
            {
                "xtx": torch.zeros(branch.fw_head_dim, branch.fw_head_dim, dtype=torch.float64),
                "xty": torch.zeros(branch.fw_head_dim, branch.fw_head_dim, dtype=torch.float64),
                "num_samples": 0,
            }
            for _ in range(branch.num_fw_heads)
        ]

    for start in range(0, len(samples), args.batch_size):
        batch_texts = samples[start : start + args.batch_size]
        encoded = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length)
        input_ids = encoded["input_ids"]
        position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0).expand(input_ids.shape[0], -1)
        teacher_inputs = collect_layer_inputs(teacher, input_ids, target_layers)

        for idx in target_layers:
            branch = hybrid.model.layers[idx].self_attn.lact_branch
            hidden_states = teacher_inputs[idx]
            fast_k, fast_v = compute_branch_features(branch, hidden_states, position_ids)

            for head in range(branch.num_fw_heads):
                hk = fast_k[head::branch.num_fw_heads].reshape(-1, branch.fw_head_dim).double()
                hv = fast_v[head::branch.num_fw_heads].reshape(-1, branch.fw_head_dim).double()
                accumulators[idx][head]["xtx"] += hk.T @ hk
                accumulators[idx][head]["xty"] += hk.T @ hv
                accumulators[idx][head]["num_samples"] += hk.shape[0]

    stats = {
        "layers": {},
        "meta": {
            "teacher": args.teacher,
            "hybrid_model": args.hybrid_model,
            "ridge": args.ridge,
            "num_text_samples": len(samples),
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "intended_mapping": "fast_k -> fast_v",
            "parameterization_note": "The saved ridge_solution is a full matrix candidate for the branch memory map. It is not yet directly written into w0/w1/w2.",
        },
    }

    for idx in target_layers:
        branch = hybrid.model.layers[idx].self_attn.lact_branch
        per_head = []
        for head in range(branch.num_fw_heads):
            xtx = accumulators[idx][head]["xtx"].float()
            xty = accumulators[idx][head]["xty"].float()
            num_samples = accumulators[idx][head]["num_samples"]
            eye = torch.eye(xtx.shape[0], dtype=xtx.dtype)
            ridge_solution = torch.linalg.solve(xtx + args.ridge * eye, xty)
            builtin_identity = 0.5 * eye
            delta_from_builtin = ridge_solution - builtin_identity

            record = {
                "xtx": xtx.cpu(),
                "xty": xty.cpu(),
                "ridge_solution": ridge_solution.cpu(),
                "builtin_identity": builtin_identity.cpu(),
                "delta_from_builtin": delta_from_builtin.cpu(),
                "num_samples": num_samples,
            }
            if branch.w0_w2_low_rank > 0 and branch.d_h == branch.fw_head_dim:
                left, right = low_rank_factorize(delta_from_builtin, branch.w0_w2_low_rank)
                record["candidate_w0_left"] = left.cpu()
                record["candidate_w0_right"] = right.cpu()
            per_head.append(record)
        stats["layers"][idx] = per_head

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, args.out)

    summary = {
        "teacher": args.teacher,
        "hybrid_model": args.hybrid_model,
        "num_lact_layers": len(target_layers),
        "num_lact_heads": hybrid.config.num_lact_heads,
        "num_text_samples": len(samples),
        "note": "This Phase-B scaffold now accumulates K->V ridge statistics across multiple sequences and exports a low-rank candidate factorization for w0 when shapes permit. It still does not write factors back into the checkpoint.",
    }
    print(json.dumps(summary, indent=2))
    print(f"saved_stats={args.out}")


if __name__ == "__main__":
    main()
