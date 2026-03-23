# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import fla  # noqa: F401
import custom_models  # noqa: F401


def init_new_lact_weights(model, config) -> None:
    scale = float(getattr(config, "w0_init_scale", 0.1))
    strategy = str(getattr(config, "w0_init_strategy", "random_small")).lower()
    if strategy not in {"random_small", "zero"}:
        raise ValueError(f"Unsupported w0_init_strategy={strategy}")

    for layer in model.model.layers:
        branch = getattr(layer.self_attn, "lact_branch", None)
        if branch is None:
            continue
        if strategy == "zero":
            if branch.w0_w2_low_rank > 0:
                branch.w0.w_left.data.zero_()
                branch.w0.w_right.data.zero_()
                branch.w2.w_left.data.zero_()
                branch.w2.w_right.data.zero_()
            else:
                branch.w0.data.zero_()
                branch.w2.data.zero_()
            branch.w1.data.zero_()
            branch.lr_proj.weight.data.zero_()
            branch.lr_proj.bias.data.zero_()
        else:
            if branch.w0_w2_low_rank > 0:
                branch.w0.w_left.data.mul_(scale)
                branch.w0.w_right.data.mul_(scale)
                branch.w2.w_left.data.mul_(scale)
                branch.w2.w_right.data.mul_(scale)
            else:
                branch.w0.data.mul_(scale)
                branch.w2.data.mul_(scale)
            branch.w1.data.mul_(scale)
            branch.lr_proj.weight.data.mul_(scale)
            branch.lr_proj.bias.data.zero_()

        branch.qk_scale.data.fill_(1.0)
        branch.qk_offset.data.zero_()
        if branch.learnable_ttt_scale:
            branch.ttt_scale_proj.weight.data.zero_()
            branch.ttt_scale_proj.bias.data.zero_()
        if branch.use_momentum:
            for mod in branch.momentum_proj:
                if isinstance(mod, torch.nn.Linear):
                    mod.weight.data.zero_()
                    mod.bias.data.zero_()


def copy_qwen3_weights(src_model, dst_model) -> None:
    src_sd = src_model.state_dict()
    dst_sd = dst_model.state_dict()

    for key in ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]:
        if key in src_sd and key in dst_sd and src_sd[key].shape == dst_sd[key].shape:
            dst_sd[key] = src_sd[key]
    if "lm_head.weight" in dst_sd and "lm_head.weight" not in src_sd and hasattr(src_model, "lm_head"):
        dst_sd["lm_head.weight"] = src_model.lm_head.weight.detach().cpu().clone()

    for i, layer in enumerate(dst_model.model.layers):
        src_prefix = f"model.layers.{i}"
        dst_prefix = f"model.layers.{i}"
        mappings = [
            ("input_layernorm.weight", "input_layernorm.weight"),
            ("post_attention_layernorm.weight", "post_attention_layernorm.weight"),
            ("self_attn.q_proj.weight", "self_attn.q_proj.weight"),
            ("self_attn.k_proj.weight", "self_attn.k_proj.weight"),
            ("self_attn.v_proj.weight", "self_attn.v_proj.weight"),
            ("self_attn.o_proj.weight", "self_attn.o_proj.weight"),
            ("self_attn.q_norm.weight", "self_attn.q_norm.weight"),
            ("self_attn.k_norm.weight", "self_attn.k_norm.weight"),
            ("mlp.gate_proj.weight", "mlp.gate_proj.weight"),
            ("mlp.up_proj.weight", "mlp.up_proj.weight"),
            ("mlp.down_proj.weight", "mlp.down_proj.weight"),
        ]
        for src_suffix, dst_suffix in mappings:
            src_key = f"{src_prefix}.{src_suffix}"
            dst_key = f"{dst_prefix}.{dst_suffix}"
            if src_key in src_sd and dst_key in dst_sd and src_sd[src_key].shape == dst_sd[dst_key].shape:
                dst_sd[dst_key] = src_sd[src_key]

    dst_model.load_state_dict(dst_sd, strict=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a Qwen3 base checkpoint to a hybrid Qwen3+LaCT checkpoint.")
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--hybrid-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    src_model = AutoModelForCausalLM.from_pretrained(args.src, trust_remote_code=True)
    hybrid_config = AutoConfig.from_pretrained(args.hybrid_config, trust_remote_code=True)
    hybrid_model = AutoModelForCausalLM.from_config(hybrid_config, trust_remote_code=True)

    copy_qwen3_weights(src_model, hybrid_model)
    init_new_lact_weights(hybrid_model, hybrid_config)

    args.out.mkdir(parents=True, exist_ok=True)
    hybrid_config.save_pretrained(args.out)
    hybrid_model.save_pretrained(args.out)
    tokenizer = AutoTokenizer.from_pretrained(args.src, trust_remote_code=True)
    tokenizer.save_pretrained(args.out)
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()
