#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path
from typing import Optional

import torch

from run_ttt_lora import (
    build_model_and_tokenizer,
    clone_trainable_state,
    get_trainable_lora_parameters,
    resolve_device,
    resolve_dtype,
    set_seed,
)


DEFAULT_SEQ_LENS = [1024, 2048, 4096, 8192, 12288, 16384]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe affordable sequence length for TTT-LoRA prefill.")
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--prompt", type=str, default="Please summarize the following document.")
    parser.add_argument("--prompt-file", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=("auto", "bfloat16", "float16", "float32"))
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seq-lens", type=int, nargs="*", default=None)
    parser.add_argument("--binary-search", action="store_true")
    parser.add_argument("--min-seq-len", type=int, default=1024)
    parser.add_argument("--max-seq-len", type=int, default=16384)
    parser.add_argument("--step", type=int, default=1024)
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def load_base_text(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text()
    return args.prompt


def make_exact_length_prompt(tokenizer, base_text: str, target_len: int) -> str:
    token_ids = tokenizer.encode(base_text, add_special_tokens=False)
    if not token_ids:
        raise ValueError("Base prompt tokenized to zero tokens.")

    repeated = []
    while len(repeated) < target_len:
        repeated.extend(token_ids)
    repeated = repeated[:target_len]
    return tokenizer.decode(repeated, skip_special_tokens=False)


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def run_single_probe(
    model,
    tokenizer,
    prompt_text: str,
    seq_len: int,
    chunk_size: int,
    max_new_tokens: int,
):
    probe_prompt = make_exact_length_prompt(tokenizer, prompt_text, seq_len)
    encoded = tokenizer(probe_prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(next(model.parameters()).device)

    trainable_lora_params = get_trainable_lora_parameters(model)
    initial_lora_state = clone_trainable_state(trainable_lora_params)

    clear_cuda()
    oom = False
    error_message: Optional[str] = None
    peak_gb = None
    generated_tokens = None

    try:
        from run_ttt_lora import adapt_and_generate_for_sample

        result = adapt_and_generate_for_sample(
            model=model,
            tokenizer=tokenizer,
            prompt=probe_prompt,
            prompt_index=0,
            chunk_size=chunk_size,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            initial_lora_state=initial_lora_state,
            trainable_lora_params=trainable_lora_params,
            log_file=None,
        )
        generated_tokens = result["generated_token_count"]
    except torch.OutOfMemoryError as exc:
        oom = True
        error_message = str(exc)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            oom = True
            error_message = str(exc)
        else:
            raise
    finally:
        if torch.cuda.is_available():
            peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        clear_cuda()

    return {
        "seq_len": seq_len,
        "fits": not oom,
        "oom": oom,
        "peak_allocated_gb": peak_gb,
        "generated_tokens": generated_tokens,
        "error": error_message,
    }


def run_binary_search(
    model,
    tokenizer,
    prompt_text: str,
    chunk_size: int,
    max_new_tokens: int,
    min_seq_len: int,
    max_seq_len: int,
    step: int,
):
    low = min_seq_len
    high = max_seq_len
    best = None
    attempts = []

    while low <= high:
        mid = ((low + high) // (2 * step)) * step
        if mid < min_seq_len:
            mid = min_seq_len
        if attempts and mid == attempts[-1]["seq_len"]:
            break

        result = run_single_probe(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            seq_len=mid,
            chunk_size=chunk_size,
            max_new_tokens=max_new_tokens,
        )
        attempts.append(result)

        if result["fits"]:
            best = result
            low = mid + step
        else:
            high = mid - step

    return {"best_fit": best, "attempts": attempts}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)

    device = resolve_device(args.device)
    torch_dtype = resolve_dtype(args.dtype)
    prompt_text = load_base_text(args)

    model, tokenizer = build_model_and_tokenizer(
        model_name_or_path=args.model_name_or_path,
        device=device,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )

    if args.binary_search:
        results = run_binary_search(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            chunk_size=args.chunk_size,
            max_new_tokens=args.max_new_tokens,
            min_seq_len=args.min_seq_len,
            max_seq_len=args.max_seq_len,
            step=args.step,
        )
    else:
        seq_lens = args.seq_lens or DEFAULT_SEQ_LENS
        results = {
            "attempts": [
                run_single_probe(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_text=prompt_text,
                    seq_len=seq_len,
                    chunk_size=args.chunk_size,
                    max_new_tokens=args.max_new_tokens,
                )
                for seq_len in seq_lens
            ]
        }

    output = {
        "model_name_or_path": args.model_name_or_path,
        "device": str(device),
        "dtype": args.dtype,
        "chunk_size": args.chunk_size,
        "max_new_tokens": args.max_new_tokens,
        "results": results,
    }

    print(json.dumps(output, ensure_ascii=True, indent=2))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, ensure_ascii=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
