# -*- coding: utf-8 -*-

import argparse
import shutil
from pathlib import Path
from typing import Optional

import torch
import torch.distributed.checkpoint as DCP
from torchtitan.tools.logging import init_logger, logger
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


DTYPE_MAP = {
    "auto": None,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _register_custom_models() -> None:
    # Lazy import so `--help` works even in environments without GPU/triton runtime.
    try:
        import fla  # noqa: F401
        import custom_models  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Failed to import local custom model modules (`fla` / `custom_models`). "
            "Please run this script in the same environment used for training."
        ) from e


def _prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output path already exists: {path}. "
                "Use --overwrite to replace it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _log_loading_info(loading_info: dict, max_keys_to_log: int = 40) -> None:
    missing = loading_info.get("missing_keys", [])
    unexpected = loading_info.get("unexpected_keys", [])
    mismatched = loading_info.get("mismatched_keys", [])
    errors = loading_info.get("error_msgs", [])

    # HF may return list/tuple/set depending on version/code path.
    missing = list(missing) if not isinstance(missing, list) else missing
    unexpected = list(unexpected) if not isinstance(unexpected, list) else unexpected
    mismatched = list(mismatched) if not isinstance(mismatched, list) else mismatched
    errors = list(errors) if not isinstance(errors, list) else errors

    logger.info("Loading summary:")
    logger.info("  missing_keys=%d", len(missing))
    logger.info("  unexpected_keys=%d", len(unexpected))
    logger.info("  mismatched_keys=%d", len(mismatched))
    logger.info("  error_msgs=%d", len(errors))

    if missing:
        logger.info("  First missing keys: %s", missing[:max_keys_to_log])
    if unexpected:
        logger.info("  First unexpected keys: %s", unexpected[:max_keys_to_log])
    if mismatched:
        logger.info("  First mismatched keys: %s", mismatched[:max_keys_to_log])
    if errors:
        raise RuntimeError(f"Found error messages during load: {errors}")


@torch.no_grad()
def _zero_init_missing_ttt_scale_proj(model: torch.nn.Module, loading_info: dict) -> None:
    missing = set(loading_info.get("missing_keys", []))
    zeroed = []
    for name, param in model.named_parameters():
        if ".attn.ttt_scale_proj." in name and name in missing:
            param.zero_()
            zeroed.append(name)

    if zeroed:
        logger.info(
            "Zero-initialized missing ttt_scale_proj params (%d): %s",
            len(zeroed),
            zeroed[:40],
        )
    else:
        logger.info("No missing ttt_scale_proj params needed zero initialization.")


@torch.inference_mode()
def convert(
    src_model: str,
    dst_hf: Path,
    num_lact_heads: int,
    dtype: str = "auto",
    dcp_checkpoint: Optional[Path] = None,
    copy_tokenizer: bool = True,
    overwrite: bool = False,
    seed: Optional[int] = None,
    zero_init_missing_ttt_scale_proj: bool = True,
) -> None:
    _register_custom_models()

    if num_lact_heads <= 0:
        raise ValueError("num_lact_heads must be > 0.")

    if dtype not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {dtype}. Choices: {list(DTYPE_MAP.keys())}")

    if seed is not None:
        torch.manual_seed(seed)
        logger.info("Set torch manual seed to %d.", seed)

    logger.info("Loading config from %s", src_model)
    config = AutoConfig.from_pretrained(src_model, trust_remote_code=True)
    old_heads = getattr(config, "num_lact_heads", None)

    if not hasattr(config, "num_lact_heads"):
        raise ValueError(
            "The source config does not have `num_lact_heads`. "
            "Please ensure this is a LaCT config."
        )

    if config.hidden_size % num_lact_heads != 0:
        raise ValueError(
            f"hidden_size ({config.hidden_size}) must be divisible by "
            f"num_lact_heads ({num_lact_heads})."
        )

    config.num_lact_heads = int(num_lact_heads)
    logger.info(
        "Switching num_lact_heads from %s to %d.",
        str(old_heads),
        config.num_lact_heads,
    )

    logger.info("Loading source weights with updated config...")
    model_dtype = DTYPE_MAP[dtype]
    model, loading_info = AutoModelForCausalLM.from_pretrained(
        src_model,
        config=config,
        trust_remote_code=True,
        torch_dtype=model_dtype if model_dtype is not None else "auto",
        output_loading_info=True,
    )
    _log_loading_info(loading_info)
    if zero_init_missing_ttt_scale_proj:
        _zero_init_missing_ttt_scale_proj(model, loading_info)

    logger.info("Saving converted HF checkpoint to %s", dst_hf)
    _prepare_output_dir(dst_hf, overwrite=overwrite)
    model.save_pretrained(dst_hf)
    config.save_pretrained(dst_hf)

    if copy_tokenizer:
        try:
            tokenizer = AutoTokenizer.from_pretrained(src_model, trust_remote_code=True)
            tokenizer.save_pretrained(dst_hf)
            logger.info("Tokenizer copied to %s", dst_hf)
        except Exception as e:
            logger.warning("Tokenizer copy skipped: %s: %s", type(e).__name__, str(e))

    if dcp_checkpoint is not None:
        logger.info("Writing model-only DCP checkpoint to %s", dcp_checkpoint)
        _prepare_output_dir(dcp_checkpoint, overwrite=overwrite)

        state_dict = model.state_dict()
        if "lm_head.weight" not in state_dict:
            for tied_key in ("model.embed_tokens.weight", "model.embeddings.weight"):
                if tied_key in state_dict:
                    logger.warning(
                        "`lm_head.weight` not found; using tied weight from `%s`.",
                        tied_key,
                    )
                    state_dict["lm_head.weight"] = state_dict[tied_key]
                    break

        storage_writer = DCP.filesystem.FileSystemWriter(dcp_checkpoint, thread_count=8)
        DCP.save(state_dict, storage_writer=storage_writer)
        logger.info("DCP checkpoint saved.")

    logger.info("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a FA-style checkpoint (e.g. num_lact_heads=0) into a "
            "new LaCT initialization with target num_lact_heads."
        )
    )
    parser.add_argument("--src_model", type=str, required=True, help="Source HF checkpoint path.")
    parser.add_argument("--dst_hf", type=Path, required=True, help="Output HF checkpoint directory.")
    parser.add_argument("--num_lact_heads", type=int, default=4, help="Target num_lact_heads.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=list(DTYPE_MAP.keys()),
        help="Dtype used while loading source model.",
    )
    parser.add_argument(
        "--dcp_checkpoint",
        type=Path,
        default=None,
        help="Optional output path for model-only DCP checkpoint (typically .../step-0).",
    )
    parser.add_argument(
        "--no_copy_tokenizer",
        action="store_true",
        help="Do not copy tokenizer files to dst_hf.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directories if they already exist.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for deterministic initialization of newly added params.",
    )
    parser.add_argument(
        "--no_zero_init_missing_ttt_scale_proj",
        action="store_true",
        help=(
            "Do not force missing `attn.ttt_scale_proj` weights/bias to zeros. "
            "By default, missing ttt_scale_proj params are zero-initialized."
        ),
    )
    args = parser.parse_args()

    convert(
        src_model=args.src_model,
        dst_hf=args.dst_hf,
        num_lact_heads=args.num_lact_heads,
        dtype=args.dtype,
        dcp_checkpoint=args.dcp_checkpoint,
        copy_tokenizer=not args.no_copy_tokenizer,
        overwrite=args.overwrite,
        seed=args.seed,
        zero_init_missing_ttt_scale_proj=not args.no_zero_init_missing_ttt_scale_proj,
    )


if __name__ == "__main__":
    init_logger()
    main()
