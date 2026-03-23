# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import argparse
import os
import tempfile
from pathlib import Path

import torch
import torch.distributed.checkpoint as DCP
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from transformers import AutoModelForCausalLM

import fla  # noqa
import custom_models  # noqa: F401
from torchtitan.tools.logging import init_logger, logger


@torch.inference_mode()
def convert_hf_weights(model: str, checkpoint: str):
    logger.info(f"Loading model from {model}")
    model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True)
    state_dict = model.state_dict()
    logger.info(
        "Loaded HF model_type=%s tie_word_embeddings=%s",
        getattr(model.config, "model_type", None),
        getattr(model.config, "tie_word_embeddings", None),
    )
    logger.info(
        "State dict initially has %d keys; lm_head.weight present=%s; embed key present=%s",
        len(state_dict),
        "lm_head.weight" in state_dict,
        any(k.endswith("embed_tokens.weight") or k.endswith("embeddings.weight") for k in state_dict),
    )

    # Some tied-weight models may not materialize lm_head.weight in the serialized
    # checkpoint even though the training-side model expects it in the DCP seed.
    if "lm_head.weight" not in state_dict and hasattr(model, "lm_head"):
        logger.info("Materializing missing lm_head.weight from tied output embeddings")
        state_dict["lm_head.weight"] = model.lm_head.weight.detach().cpu()

    # DCP may collapse aliased storages for tied embeddings. Force lm_head.weight
    # to be an independent tensor in the saved state dict so both logical keys
    # are present at load time.
    if "lm_head.weight" in state_dict:
        logger.info("Cloning lm_head.weight to break tied-storage alias before DCP save")
        state_dict["lm_head.weight"] = state_dict["lm_head.weight"].detach().cpu().clone()
    logger.info(
        "After materialization: %d keys; lm_head.weight shape=%s",
        len(state_dict),
        tuple(state_dict["lm_head.weight"].shape) if "lm_head.weight" in state_dict else None,
    )
    sample_keys = sorted(state_dict.keys())[:12]
    logger.info("Sample state_dict keys: %s", sample_keys)

    logger.info(f"Writing to DCP at '{checkpoint}'")
    checkpoint.mkdir(parents=True, exist_ok=True)
    storage_writer = DCP.filesystem.FileSystemWriter(checkpoint, thread_count=8)
    DCP.save({"model": state_dict}, storage_writer=storage_writer)

    # Verify what was actually written by round-tripping the DCP checkpoint back
    # to a torch-save file and checking the resulting keys.
    with tempfile.TemporaryDirectory() as tmpdir:
        roundtrip_path = os.path.join(tmpdir, "checkpoint.pt")
        logger.info("Round-tripping DCP checkpoint for verification: %s", roundtrip_path)
        dcp_to_torch_save(str(checkpoint), roundtrip_path)
        loaded = torch.load(roundtrip_path, map_location="cpu")
        loaded_model_sd = loaded["model"]
        logger.info(
            "Round-trip verification: %d keys; lm_head.weight present=%s; sample keys=%s",
            len(loaded_model_sd),
            "lm_head.weight" in loaded_model_sd,
            sorted(list(loaded_model_sd.keys()))[:12],
        )


if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser(description="Convert huggingface-style model weights to DCP format.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    convert_hf_weights(args.model, args.checkpoint)
