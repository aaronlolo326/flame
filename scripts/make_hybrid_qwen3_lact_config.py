# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from transformers import AutoConfig


def build_hybrid_layer_types(num_hidden_layers: int, recurrent_ratio: float = 0.75) -> list[str]:
    num_lact_layers = math.ceil(num_hidden_layers * recurrent_ratio)
    num_fa_layers = num_hidden_layers - num_lact_layers
    return ["fa"] * num_fa_layers + ["lact"] * num_lact_layers


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a hybrid Qwen3+LaCT config from a Qwen3 config or model id.")
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-lact-heads", type=int, default=4)
    parser.add_argument("--lact-chunk-size", type=int, default=1024)
    parser.add_argument("--window-size", type=int, default=16834)
    parser.add_argument("--w0-init-strategy", type=str, default="random_small")
    parser.add_argument("--w0-init-scale", type=float, default=0.1)
    args = parser.parse_args()

    src_cfg = AutoConfig.from_pretrained(args.src, trust_remote_code=True)
    num_hidden_layers = int(getattr(src_cfg, "num_hidden_layers"))
    cfg = {
        "architectures": ["HybridQwen3LaCTForCausalLM"],
        "model_type": "hybrid_qwen3_lact",
        "source_model_name_or_path": args.src,
        "source_model_type": getattr(src_cfg, "model_type", None),
        "vocab_size": int(src_cfg.vocab_size),
        "bos_token_id": getattr(src_cfg, "bos_token_id", None),
        "eos_token_id": getattr(src_cfg, "eos_token_id", None),
        "tie_word_embeddings": bool(getattr(src_cfg, "tie_word_embeddings", False)),
        "hidden_size": int(src_cfg.hidden_size),
        "intermediate_size": int(src_cfg.intermediate_size),
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": int(getattr(src_cfg, "num_attention_heads")),
        "num_key_value_heads": int(getattr(src_cfg, "num_key_value_heads", getattr(src_cfg, "num_attention_heads"))),
        "head_dim": int(getattr(src_cfg, "head_dim", src_cfg.hidden_size // src_cfg.num_attention_heads)),
        "hidden_act": str(getattr(src_cfg, "hidden_act", "silu")),
        "max_position_embeddings": int(getattr(src_cfg, "max_position_embeddings", 32768)),
        "initializer_range": float(getattr(src_cfg, "initializer_range", 0.02)),
        "rms_norm_eps": float(getattr(src_cfg, "rms_norm_eps", 1e-6)),
        "use_cache": True,
        "rope_theta": float(getattr(src_cfg, "rope_theta", 1000000.0)),
        "attention_bias": bool(getattr(src_cfg, "attention_bias", False)),
        "attention_dropout": float(getattr(src_cfg, "attention_dropout", 0.0)),
        "use_sliding_window": True,
        "sliding_window": int(args.window_size),
        "hybrid_layer_types": build_hybrid_layer_types(num_hidden_layers),
        "num_lact_heads": int(args.num_lact_heads),
        "inter_multi": 1,
        "qkv_bias": False,
        "attn_qk_norm": False,
        "lact_chunk_size": int(args.lact_chunk_size),
        "use_muon": True,
        "lr_dim": 1,
        "qkv_silu": True,
        "no_v_silu": False,
        "lr_parameterization": "mamba",
        "learnable_ttt_scale": True,
        "use_momentum": True,
        "ttt_loss_type": "dot_product",
        "ttt_prenorm": True,
        "ttt_nope": False,
        "w0_w2_low_rank": 32,
        "fw_init_gain": 0.5,
        "use_fused_kernel": True,
        "fp32_states": False,
        "fuse_cross_entropy": True,
        "w0_init_strategy": args.w0_init_strategy,
        "w0_init_scale": float(args.w0_init_scale),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(cfg, handle, indent=2)
        handle.write("\n")

    print(f"wrote={args.out}")
    print(f"num_hidden_layers={num_hidden_layers}")
    print(f"hybrid_split={cfg['hybrid_layer_types'].count('fa')} fa / {cfg['hybrid_layer_types'].count('lact')} lact")


if __name__ == "__main__":
    main()
