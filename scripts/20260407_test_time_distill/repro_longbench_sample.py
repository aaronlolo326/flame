#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from run_ttt_lora import (
    CHUNK_SIZE,
    DEFAULT_FALLBACK_CHUNK_TASK,
    LORA_ALPHA,
    LORA_R,
    NUM_QA_CANDIDATES,
    NUM_JUDGE_CANDIDATES,
    NUM_SELECTED_QA,
    QA_GENERATION_MAX_NEW_TOKENS,
    QA_JUDGE_MAX_NEW_TOKENS,
    SampleSpec,
    adapt_and_generate_for_sample,
    build_model_and_tokenizer,
    clone_trainable_state,
    get_trainable_lora_parameters,
    resolve_device,
    resolve_dtype,
    set_seed,
)


DEFAULT_SAMPLES_BASE = Path(
    "/work/yufei/projects/flame/results/"
    "20260322_hybrid_qwen3_lact_0p6B_swa_2k_chunk_1k_rerun12_prolong_prolong_from_run12_step9535_v4/"
    "lb/__storage__backup__yufei__ttt__flame__exp__"
    "20260322_hybrid_qwen3_lact_0p6B_swa_2k_chunk_1k_rerun12_prolong_prolong_from_run12_step9535_v4"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce QA-distillation updates on one LongBench sample.")
    parser.add_argument("--model-name-or-path", default="/work/yufei/downloads/Qwen3-0.6B-Base")
    parser.add_argument("--task-name", default="longbench_triviaqa")
    parser.add_argument("--doc-id", type=int, default=126)
    parser.add_argument("--samples-base-dir", type=str, default=str(DEFAULT_SAMPLES_BASE))
    parser.add_argument("--sample-jsonl", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=("auto", "bfloat16", "float16", "float32"))
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument("--max-gen-toks", type=int, default=None)
    parser.add_argument("--num-qa-candidates", type=int, default=NUM_QA_CANDIDATES)
    parser.add_argument("--num-judge-candidates", type=int, default=NUM_JUDGE_CANDIDATES)
    parser.add_argument("--num-selected-qa", type=int, default=NUM_SELECTED_QA)
    parser.add_argument("--qa-generation-max-new-tokens", type=int, default=QA_GENERATION_MAX_NEW_TOKENS)
    parser.add_argument("--qa-judge-max-new-tokens", type=int, default=QA_JUDGE_MAX_NEW_TOKENS)
    parser.add_argument("--lr", type=float, default=5e-5)
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


def load_sample(sample_path: Path, doc_id: int) -> dict:
    with sample_path.open() as f:
        for line in f:
            obj = json.loads(line)
            if int(obj["doc_id"]) == int(doc_id):
                return obj
    raise ValueError(f"doc_id={doc_id} not found in {sample_path}")


def raw_pretruncate_text(text: str, max_ctx_len: int, tokenizer) -> str:
    encoded = tokenizer(
        [text],
        truncation=True,
        padding="longest",
        return_tensors="pt",
    )
    kept_ids = encoded["input_ids"][:, -max_ctx_len:]
    return tokenizer.decode(kept_ids[0].tolist(), skip_special_tokens=True)


def build_sample_spec(sample_obj: dict, final_prompt: str, adaptation_text: str) -> SampleSpec:
    doc = sample_obj["doc"]
    task_question = (doc.get("question") or "").strip() or DEFAULT_FALLBACK_CHUNK_TASK
    return SampleSpec(
        final_prompt=final_prompt,
        adaptation_text=adaptation_text,
        task_question=task_question,
        metadata={
            "task_name": doc.get("task"),
            "doc_id": sample_obj.get("doc_id"),
        },
    )


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)
    torch_dtype = resolve_dtype(args.dtype)
    sample_path = resolve_sample_path(args)
    sample_obj = load_sample(sample_path, args.doc_id)
    doc = sample_obj["doc"]
    prompt = sample_obj["arguments"]["gen_args_0"]["arg_0"]
    sample_gen_kwargs = dict(sample_obj["arguments"]["gen_args_0"]["arg_1"])
    max_gen_toks = int(args.max_gen_toks) if args.max_gen_toks is not None else int(sample_gen_kwargs.get("max_gen_toks", 32))
    do_sample = bool(sample_gen_kwargs.get("do_sample", False))
    temperature = float(sample_gen_kwargs.get("temperature", 1.0))

    print(f"[sample] path={sample_path}")
    print(f"[sample] task_name={args.task_name} doc_id={args.doc_id}")
    print(f"[sample] question_preview={doc.get('question', '')[:200]!r}")

    model, tokenizer = build_model_and_tokenizer(
        model_name_or_path=args.model_name_or_path,
        device=device,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
    )

    max_ctx_len = args.max_length - max_gen_toks
    final_prompt = raw_pretruncate_text(prompt, max_ctx_len=max_ctx_len, tokenizer=tokenizer)
    adaptation_text = raw_pretruncate_text(doc.get("context", prompt), max_ctx_len=max_ctx_len, tokenizer=tokenizer)

    print(
        "[tokens] "
        f"final_prompt_tokens={len(tokenizer.encode(final_prompt, add_special_tokens=False))} "
        f"adaptation_tokens={len(tokenizer.encode(adaptation_text, add_special_tokens=False))} "
        f"max_ctx_len={max_ctx_len}"
    )
    print(
        "[distill] "
        f"chunk_size={args.chunk_size} num_qa_candidates={args.num_qa_candidates} "
        f"num_judge_candidates={args.num_judge_candidates} "
        f"num_selected_qa={args.num_selected_qa}"
    )

    trainable_lora_params = get_trainable_lora_parameters(model)
    initial_lora_state = clone_trainable_state(trainable_lora_params)
    sample = build_sample_spec(sample_obj=sample_obj, final_prompt=final_prompt, adaptation_text=adaptation_text)

    clear_cuda()
    status = "ok"
    error_message = None
    result = None

    try:
        result = adapt_and_generate_for_sample(
            model=model,
            tokenizer=tokenizer,
            sample=sample,
            sample_index=0,
            chunk_size=args.chunk_size,
            update_mode="full_prefix_approx",
            lr=args.lr,
            num_qa_candidates=args.num_qa_candidates,
            num_judge_candidates=args.num_judge_candidates,
            num_selected_qa=args.num_selected_qa,
            qa_generation_max_new_tokens=args.qa_generation_max_new_tokens,
            qa_judge_max_new_tokens=args.qa_judge_max_new_tokens,
            max_new_tokens=max_gen_toks,
            do_sample=do_sample,
            temperature=temperature,
            top_p=1.0,
            initial_lora_state=initial_lora_state,
            trainable_lora_params=trainable_lora_params,
            log_file=None,
        )
    except torch.OutOfMemoryError as exc:
        status = "oom"
        error_message = str(exc)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            status = "oom"
            error_message = str(exc)
        else:
            raise
    finally:
        peak_allocated_gb = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 3) if torch.cuda.is_available() else None
        )
        print(
            f"[done] status={status}"
            + ("" if peak_allocated_gb is None else f" peak_allocated_gb={peak_allocated_gb:.2f}")
        )
        clear_cuda()

    output = {
        "status": status,
        "error": error_message,
        "task_name": args.task_name,
        "doc_id": args.doc_id,
        "sample_path": str(sample_path),
        "question": doc.get("question"),
        "context_preview": adaptation_text[:400],
        "result": result,
    }
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, ensure_ascii=True, indent=2) + "\n")
    print(json.dumps(output, ensure_ascii=True))


if __name__ == "__main__":
    main()
