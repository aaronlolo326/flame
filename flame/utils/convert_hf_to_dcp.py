# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import argparse
from pathlib import Path

import torch
import torch.distributed.checkpoint as DCP
from transformers import AutoModelForCausalLM

import fla  # noqa
from torchtitan.tools.logging import init_logger, logger
import custom_models

@torch.inference_mode()
def convert_hf_weights(model: str, checkpoint: str):
    logger.info(f"Loading model from {model}")
    model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True)
    state_dict = model.state_dict()

    # TorchTitan treats `step-0` checkpoints as model-only checkpoints and loads
    # a flat model state dict. So we must save the flat state dict directly,
    # instead of wrapping it under an extra "model" key.
    if "lm_head.weight" not in state_dict:
        for tied_key in ("model.embed_tokens.weight", "model.embeddings.weight"):
            if tied_key in state_dict:
                logger.warning(
                    f"`lm_head.weight` not found; using tied weight from `{tied_key}`."
                )
                state_dict["lm_head.weight"] = state_dict[tied_key]
                break

    logger.info(f"Writing to DCP at '{checkpoint}'")
    checkpoint.mkdir(parents=True, exist_ok=True)
    storage_writer = DCP.filesystem.FileSystemWriter(checkpoint, thread_count=8)
    DCP.save(state_dict, storage_writer=storage_writer)


if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser(description="Convert huggingface-style model weights to DCP format.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    convert_hf_weights(args.model, args.checkpoint)
