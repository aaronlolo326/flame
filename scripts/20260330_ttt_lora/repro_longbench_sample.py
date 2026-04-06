#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Optional

import torch

from run_ttt_lora import (
    BETAS,
    CHUNK_SIZE,
    GRAD_CLIP,
    LR,
    WEIGHT_DECAY,
    build_model_and_tokenizer,
    clone_trainable_state,
    compute_delta_norm,
    generate_from_adapted_state,
    get_trainable_lora_parameters,
    LOCAL_TRAIN_WINDOW,
    LOSS_MODE,
    LOSS_TOPK_FRACTION,
    normalize_update_mode,
    rebuild_cache_for_prefix,
    reset_trainable_state,
    resolve_device,
    resolve_dtype,
    run_chunk_update_step,
    set_seed,
    UPDATE_MODE,
)


DEFAULT_SAMPLES_BASE = Path(
    "/work/yufei/projects/flame/results/"
    "20260322_hybrid_qwen3_lact_0p6B_swa_2k_chunk_1k_rerun12_prolong_prolong_from_run12_step9535_v4/"
    "lb/__storage__backup__yufei__ttt__flame__exp__"
    "20260322_hybrid_qwen3_lact_0p6B_swa_2k_chunk_1k_rerun12_prolong_prolong_from_run12_step9535_v4"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce TTT chunked updates on one LongBench sample.")
    parser.add_argument("--model-name-or-path", default="/work/yufei/downloads/Qwen3-0.6B-Base")
    parser.add_argument("--task-name", default="longbench_triviaqa")
    parser.add_argument("--doc-id", type=int, default=126)
    parser.add_argument("--samples-base-dir", type=str, default=str(DEFAULT_SAMPLES_BASE))
    parser.add_argument("--sample-jsonl", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=("auto", "bfloat16", "float16", "float32"))
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--steps-per-chunk", type=int, default=1)
    parser.add_argument(
        "--update-mode",
        type=str,
        default=UPDATE_MODE,
        choices=("full_prefix", "full_prefix_approx", "full_prefix_exact", "local_window"),
    )
    parser.add_argument("--local-train-window", type=int, default=LOCAL_TRAIN_WINDOW)
    parser.add_argument("--loss-mode", type=str, default=LOSS_MODE, choices=("full", "topk_fraction"))
    parser.add_argument("--loss-topk-fraction", type=float, default=LOSS_TOPK_FRACTION)
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument("--max-gen-toks", type=int, default=None)
    parser.add_argument("--raw-chars-per-token", type=float, default=3.0)
    parser.add_argument("--raw-trunc-safety-margin", type=float, default=0.9)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def resolve_sample_path(args: argparse.Namespace) -> Path:
    if args.sample_jsonl:
        return Path(args.sample_jsonl)

    samples_base = Path(args.samples_base_dir)
    candidates = sorted(samples_base.glob(f"samples_{args.task_name}_*.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"No sample file found for task {args.task_name} under {samples_base}")
    return candidates[-1]


def load_sample_prompt(sample_path: Path, doc_id: int) -> tuple[dict, str]:
    with sample_path.open() as f:
        for line in f:
            obj = json.loads(line)
            if int(obj["doc_id"]) == int(doc_id):
                return obj, obj["arguments"]["gen_args_0"]["arg_0"]
    raise ValueError(f"doc_id={doc_id} not found in {sample_path}")


def raw_pretruncate_context(
    context: str,
    max_ctx_len: int,
    raw_chars_per_token: float,
    raw_trunc_safety_margin: float,
) -> tuple[str, int]:
    estimated_char_budget = int(max_ctx_len * raw_chars_per_token * raw_trunc_safety_margin)
    if estimated_char_budget <= 0 or len(context) <= estimated_char_budget:
        return context, estimated_char_budget
    return context[-estimated_char_budget:], estimated_char_budget


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.update_mode = normalize_update_mode(args.update_mode)

    device = resolve_device(args.device)
    torch_dtype = resolve_dtype(args.dtype)
    sample_path = resolve_sample_path(args)
    sample_obj, prompt = load_sample_prompt(sample_path, args.doc_id)
    sample_gen_kwargs = dict(sample_obj["arguments"]["gen_args_0"]["arg_1"])
    max_gen_toks = int(args.max_gen_toks) if args.max_gen_toks is not None else int(sample_gen_kwargs.get("max_gen_toks", 32))
    do_sample = bool(sample_gen_kwargs.get("do_sample", False))
    temperature = float(sample_gen_kwargs.get("temperature", 1.0))
    until = sample_gen_kwargs.get("until", [])
    print(f"[sample] path={sample_path}")
    print(f"[sample] task_name={args.task_name} doc_id={args.doc_id}")
    print(f"[sample] question_preview={sample_obj['doc'].get('question', '')[:200]!r}")
    print(
        "[gen] "
        f"max_gen_toks={max_gen_toks} do_sample={do_sample} "
        f"temperature={temperature} until={until}"
    )

    max_ctx_len = args.max_length - max_gen_toks
    raw_prompt, raw_char_budget = raw_pretruncate_context(
        prompt,
        max_ctx_len=max_ctx_len,
        raw_chars_per_token=args.raw_chars_per_token,
        raw_trunc_safety_margin=args.raw_trunc_safety_margin,
    )
    print(
        "[truncate] "
        f"original_char_len={len(prompt)} "
        f"raw_char_budget={raw_char_budget} "
        f"raw_pretruncated_char_len={len(raw_prompt)}"
    )

    model, tokenizer = build_model_and_tokenizer(
        model_name_or_path=args.model_name_or_path,
        device=device,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )

    encoding = tokenizer(
        [raw_prompt],
        truncation=True,
        padding="longest",
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"][:, -max_ctx_len:].to(device)
    original_token_len = len(tokenizer.encode(prompt, add_special_tokens=False))
    raw_prompt_token_len = len(tokenizer.encode(raw_prompt, add_special_tokens=False))
    print(
        "[tokens] "
        f"original_token_len={original_token_len} "
        f"raw_prompt_token_len={raw_prompt_token_len} "
        f"final_input_token_len={int(input_ids.shape[1])} "
        f"max_ctx_len={max_ctx_len}"
    )

    trainable_lora_params = get_trainable_lora_parameters(model)
    initial_lora_state = clone_trainable_state(trainable_lora_params)
    reset_trainable_state(trainable_lora_params, initial_lora_state)
    optimizer = torch.optim.AdamW(
        [param for _, param in trainable_lora_params],
        lr=LR,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
    )

    clear_cuda()
    logs = []
    status = "ok"
    error_message: Optional[str] = None
    rebuild_result = None
    generated_text = ""
    generated_token_count = 0

    try:
        seq_len = int(input_ids.shape[1])
        num_chunks = (seq_len + args.chunk_size - 1) // args.chunk_size
        print(
            "[ttt] "
            f"seq_len={seq_len} chunk_size={args.chunk_size} steps_per_chunk={args.steps_per_chunk} "
            f"update_mode={args.update_mode} local_train_window={args.local_train_window} "
            f"loss_mode={args.loss_mode} loss_topk_fraction={args.loss_topk_fraction} num_chunks={num_chunks} "
            f"device={device} dtype={args.dtype}"
        )
        for chunk_idx, start in enumerate(range(0, seq_len, args.chunk_size)):
            end = min(start + args.chunk_size, seq_len)
            step_logs = []
            prefix_cache_for_chunk = None if start == 0 or rebuild_result is None else rebuild_result.past_key_values
            for step_idx in range(args.steps_per_chunk):
                prefix_len = end
                if torch.cuda.is_available():
                    allocated_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
                    reserved_gb = torch.cuda.memory_reserved(device) / (1024 ** 3)
                    print(
                        "[chunk:start] "
                        f"idx={chunk_idx} step={step_idx} start={start} end={end} prefix_len={prefix_len} "
                        f"allocated_gb={allocated_gb:.2f} reserved_gb={reserved_gb:.2f}"
                    )
                else:
                    print(f"[chunk:start] idx={chunk_idx} step={step_idx} start={start} end={end} prefix_len={prefix_len}")

                loss, grad_norm = run_chunk_update_step(
                    model=model,
                    optimizer=optimizer,
                    trainable_lora_params=trainable_lora_params,
                    full_input_ids=input_ids,
                    chunk_start=start,
                    chunk_end=end,
                    chunk_size=args.chunk_size,
                    update_mode=args.update_mode,
                    local_train_window=args.local_train_window,
                    loss_mode=args.loss_mode,
                    loss_topk_fraction=args.loss_topk_fraction,
                    base_prefix_cache=prefix_cache_for_chunk,
                )
                lora_norm = compute_delta_norm(trainable_lora_params, initial_lora_state)
                current_allocated_gb = (
                    torch.cuda.memory_allocated(device) / (1024 ** 3) if torch.cuda.is_available() else None
                )
                current_reserved_gb = (
                    torch.cuda.memory_reserved(device) / (1024 ** 3) if torch.cuda.is_available() else None
                )
                print(
                    "[chunk:end] "
                    f"idx={chunk_idx} step={step_idx} loss={None if loss is None else float(loss.detach().cpu().item()):.6f} "
                    f"grad_norm={float(grad_norm.detach().cpu().item()) if torch.is_tensor(grad_norm) else float(grad_norm):.6f} "
                    f"lora_norm={lora_norm:.6f} "
                    f"allocated_gb={current_allocated_gb if current_allocated_gb is None else round(current_allocated_gb, 2)} "
                    f"reserved_gb={current_reserved_gb if current_reserved_gb is None else round(current_reserved_gb, 2)}"
                )

                step_logs.append(
                    {
                        "chunk_idx": chunk_idx,
                        "step_idx": step_idx,
                        "steps_per_chunk": args.steps_per_chunk,
                        "update_mode": args.update_mode,
                        "local_train_window": args.local_train_window if args.update_mode == "local_window" else None,
                        "loss_mode": args.loss_mode,
                        "loss_topk_fraction": args.loss_topk_fraction if args.loss_mode == "topk_fraction" else None,
                        "chunk_start": start,
                        "chunk_end": end,
                        "chunk_tokens": end - start,
                        "loss": None if loss is None else float(loss.detach().cpu().item()),
                        "grad_norm": float(grad_norm.detach().cpu().item()) if torch.is_tensor(grad_norm) else float(grad_norm),
                        "lora_norm": lora_norm,
                        "cuda_allocated_gb": current_allocated_gb,
                        "cuda_reserved_gb": current_reserved_gb,
                    }
                )

            rebuild_result = rebuild_cache_for_prefix(model, input_ids[:, :end], chunk_size=args.chunk_size)
            logs.extend(step_logs)

        if rebuild_result is None:
            raise ValueError("No rebuild result after chunked adaptation.")

        print("[decode:start] generating continuation from adapted cache")
        generated_ids = generate_from_adapted_state(
            model=model,
            prefix_ids=input_ids,
            past_key_values=rebuild_result.past_key_values,
            last_logits=rebuild_result.last_logits,
            max_new_tokens=max_gen_toks,
            do_sample=do_sample,
            temperature=temperature,
            top_p=1.0,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated_token_count = int(generated_ids.shape[1])
        generated_text = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=False)
        print(
            "[decode:end] "
            f"generated_token_count={generated_token_count} "
            f"generated_preview={generated_text[:200]!r}"
        )
    except torch.OutOfMemoryError as exc:
        status = "oom"
        error_message = str(exc)
        print(f"[oom] {error_message}")
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            status = "oom"
            error_message = str(exc)
            print(f"[oom] {error_message}")
        else:
            raise
    finally:
        peak_allocated_gb = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 3) if torch.cuda.is_available() else None
        )
        if peak_allocated_gb is not None:
            print(f"[done] status={status} peak_allocated_gb={peak_allocated_gb:.2f}")
        else:
            print(f"[done] status={status}")
        clear_cuda()

    output = {
        "status": status,
        "error": error_message,
        "task_name": args.task_name,
        "doc_id": args.doc_id,
        "sample_path": str(sample_path),
        "question_preview": sample_obj["doc"].get("question", "")[:300],
        "original_char_len": len(prompt),
        "raw_pretruncated_char_len": len(raw_prompt),
        "raw_char_budget": raw_char_budget,
        "original_token_len": original_token_len,
        "final_input_token_len": int(input_ids.shape[1]),
        "chunk_size": args.chunk_size,
        "steps_per_chunk": args.steps_per_chunk,
        "update_mode": args.update_mode,
        "local_train_window": args.local_train_window if args.update_mode == "local_window" else None,
        "loss_mode": args.loss_mode,
        "loss_topk_fraction": args.loss_topk_fraction if args.loss_mode == "topk_fraction" else None,
        "max_length": args.max_length,
        "max_gen_toks": args.max_gen_toks,
        "peak_allocated_gb": peak_allocated_gb,
        "generated_token_count": generated_token_count,
        "generated_text": generated_text,
        "logs": logs,
    }

    print(json.dumps(output, ensure_ascii=True, indent=2))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, ensure_ascii=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
