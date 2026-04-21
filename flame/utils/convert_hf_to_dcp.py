# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import argparse
from pathlib import Path

import torch
import torch.distributed.checkpoint as DCP
from transformers import AutoModelForCausalLM

import fla  # noqa
from torchtitan.tools.logging import init_logger, logger


# def _restore_tied_weight_aliases(model, state_dict):
#     # HF may drop duplicate tied-weight aliases from state_dict(); DCP seed
#     # checkpoints need the full key set expected by the training model.
#     for tied_key in getattr(model, "_tied_weights_keys", []):
#         if tied_key in state_dict:
#             continue
#         target = model
#         for attr in tied_key.split("."):
#             target = getattr(target, attr)
#         state_dict[tied_key] = target.detach()


@torch.inference_mode()
def convert_hf_weights(model: str, checkpoint: str):
    logger.info(f"Loading model from {model}")
    model = AutoModelForCausalLM.from_pretrained(model)
    state_dict = model.state_dict()
    # _restore_tied_weight_aliases(model, state_dict)

    logger.info(f"Writing to DCP at '{checkpoint}'")
    checkpoint.mkdir(parents=True, exist_ok=True)
    storage_writer = DCP.filesystem.FileSystemWriter(checkpoint, thread_count=8)
    # DCP.save({"model": state_dict}, storage_writer=storage_writer)
    DCP.save(state_dict, storage_writer=storage_writer)


if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser(description="Convert huggingface-style model weights to DCP format.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    convert_hf_weights(args.model, args.checkpoint)
