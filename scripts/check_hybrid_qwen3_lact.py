# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM

import fla  # noqa: F401
import custom_models  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity check a hybrid Qwen3+LaCT checkpoint.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--seq-len", type=int, default=32)
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)
    model.eval()

    layer_types = [layer.layer_type for layer in model.model.layers]
    print(f"num_layers={len(layer_types)}")
    print(f"fa_layers={sum(t == 'fa' for t in layer_types)}")
    print(f"lact_layers={sum(t == 'lact' for t in layer_types)}")

    input_ids = torch.randint(0, model.config.vocab_size, (1, args.seq_len))
    with torch.no_grad():
        out = model(input_ids=input_ids, logits_to_keep=args.seq_len)
    print(f"logits_shape={tuple(out.logits.shape)}")


if __name__ == "__main__":
    main()
