# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from collections import defaultdict

import torch
from transformers import AutoConfig, AutoModelForCausalLM

import custom_models  # noqa: F401


def count_params(model) -> dict[str, int]:
    counts = defaultdict(int)
    for name, param in model.named_parameters():
        n = param.numel()
        counts["total"] += n
        if ".lact_branch." in name:
            counts["lact_branch"] += n
        elif ".self_attn." in name:
            counts["attention_base"] += n
        elif ".mlp." in name:
            counts["mlp"] += n
        elif ".input_layernorm." in name or ".post_attention_layernorm." in name or name == "model.norm.weight":
            counts["norm"] += n
        elif "embed_tokens" in name or name == "lm_head.weight":
            counts["embeddings_lm_head"] += n
        else:
            counts["other"] += n
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Report parameter breakdown for a hybrid Qwen3+LaCT config or model.")
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--meta", action="store_true", help="Instantiate on meta device")
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model_config, trust_remote_code=True)
    with torch.device("meta") if args.meta else torch.device("cpu"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    counts = count_params(model)
    print(f"model_type={config.model_type}")
    print(f"num_layers={config.num_hidden_layers}")
    print(f"fa_layers={config.hybrid_layer_types.count('fa')}")
    print(f"lact_layers={config.hybrid_layer_types.count('lact')}")
    for key in ["total", "embeddings_lm_head", "attention_base", "lact_branch", "mlp", "norm", "other"]:
        print(f"{key}={counts.get(key, 0):,}")


if __name__ == "__main__":
    main()
