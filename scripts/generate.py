import argparse
import logging
import os
import random
import sys
from pathlib import Path
import importlib

import fla  # noqa: F401
import torch
from torchtitan.tools.logging import init_logger, logger
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# import custom_models  # noqa: F401



DTYPE_MAP = {
    "auto": "auto",
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text with a Hugging Face causal language model."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path or Hugging Face repo id for the model.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Optional tokenizer path or repo id. Defaults to --model.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt text. If omitted, the script reads from --prompt_file or stdin.",
    )
    parser.add_argument(
        "--prompt_file",
        type=Path,
        default=None,
        help="Optional path to a UTF-8 prompt file.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--max_input_len",
        type=int,
        default=None,
        help="Maximum number of prompt tokens to keep. Longer prompts are left-truncated.",
    )
    parser.add_argument(
        "--min_new_tokens",
        type=int,
        default=0,
        help="Minimum number of new tokens to generate.",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Enable sampling. Otherwise greedy decoding is used unless num_beams > 1.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p nucleus sampling threshold.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling threshold.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Allow custom modeling/tokenizer code from HF repos.",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        help="Optional attention implementation override passed to from_pretrained.",
    )
    parser.add_argument(
        "--print_prompt",
        action="store_true",
        help="Print the full decoded text including the prompt.",
    )
    parser.add_argument(
        "--skip_special_tokens",
        action="store_true",
        help="Skip special tokens when decoding generated text.",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="The parameter for repetition penalty. 1.0 means no penalty.",
    )
    parser.add_argument(
        "--use_cache",
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        default=None,
        help="Override generation cache usage. Defaults to the model/generate default.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def maybe_set_seed(seed: int) -> None:
    set_seed(seed)
    # random.seed(seed)
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)


def resolve_prompt(args: argparse.Namespace) -> str:
    if args.prompt is not None:
        return args.prompt
    if args.prompt_file is not None:
        logger.info("Reading prompt from %s", args.prompt_file)
        return args.prompt_file.read_text(encoding="utf-8")
    return sys.stdin.read()

def load_custom_models(model_path: str) -> None:
    model_dir = Path(model_path).resolve()
    package_dir = model_dir / "custom_models"
    package_init = package_dir / "__init__.py"
    if not package_init.is_file():
        logger.warning("No custom_models package found at %s", package_init)
        return

    model_dir_str = str(model_dir)
    if model_dir_str not in sys.path:
        sys.path.insert(0, model_dir_str)

    for module_name in list(sys.modules):
        if module_name == "custom_models" or module_name.startswith("custom_models."):
            del sys.modules[module_name]
    importlib.invalidate_caches()

    spec = importlib.util.spec_from_file_location(
        "custom_models",
        package_init,
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load custom_models from {package_init}")

    custom_models = importlib.util.module_from_spec(spec)
    sys.modules["custom_models"] = custom_models
    spec.loader.exec_module(custom_models)
    print(f"[generate.py] custom_models package: {package_init}", flush=True)

@torch.inference_mode()
def main(args: argparse.Namespace) -> None:
    prompt = resolve_prompt(args)
    tokenizer_path = args.tokenizer or args.model
    device = "cuda"
    # torch_dtype = DTYPE_MAP[args.dtype]

    logger.info(
        "CUDA env: CUDA_VISIBLE_DEVICES=%s cuda_available=%s device_count=%s",
        os.environ.get("CUDA_VISIBLE_DEVICES"),
        torch.cuda.is_available(),
        torch.cuda.device_count(),
    )

    logger.info("Loading tokenizer from %s", tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=args.trust_remote_code,
    )

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        # "torch_dtype": torch_dtype,
    }
    if args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = args.attn_implementation

    logger.info("Loading model from %s", args.model)

    load_custom_models(args.model)


    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.to(device)
    model.eval()
    if torch.cuda.is_available():
        logger.info(
            "Resolved device=%s current_device=%s device_name=%s",
            device,
            torch.cuda.current_device(),
            torch.cuda.get_device_name(torch.cuda.current_device()),
        )

    model_inputs = tokenizer(prompt, return_tensors="pt").input_ids
    if args.max_input_len is not None:
        if args.max_input_len <= 0:
            raise ValueError("--max_input_len must be a positive integer")
        if model_inputs.shape[-1] > args.max_input_len:
            original_input_len = model_inputs.shape[-1]
            model_inputs = model_inputs[:, :args.max_input_len]
            logger.info(
                "Truncated prompt from %s to %s tokens",
                original_input_len,
                model_inputs.shape[-1],
            )

    model_inputs = model_inputs.to(device)
    attention_mask = torch.ones_like(model_inputs, device=device)

    generation_kwargs = {
        "input_ids": model_inputs,
        "attention_mask": attention_mask,
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": args.repetition_penalty,
    }

    if args.use_cache is not None:
        generation_kwargs["use_cache"] = args.use_cache

    if not args.do_sample:
        generation_kwargs.pop("temperature")
        generation_kwargs.pop("top_p")
        generation_kwargs.pop("top_k")

    logger.info(
        "Generating with device=%s use_cache=%s",
        device,
        generation_kwargs.get("use_cache", "default"),
    )
    outputs = model.generate(**generation_kwargs)

    prompt_length = model_inputs.shape[-1]
    sequences = outputs if args.print_prompt else outputs[:, prompt_length:]

    for idx, sequence in enumerate(sequences, start=1):
        text = tokenizer.decode(
            sequence,
            skip_special_tokens=args.skip_special_tokens,
        )
        # if args.num_return_sequences > 1:
        #     print(f"===== Generation {idx} =====")
        print(text)
        print (sequence)


if __name__ == "__main__":
    init_logger()
    formatter = logging.Formatter(
        "[titan] %(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )
    for handler in logger.handlers:
        handler.setFormatter(formatter)

    cli_args = parse_args()
    maybe_set_seed(cli_args.seed)
    main(cli_args)
