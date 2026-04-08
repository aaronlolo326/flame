#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List, Sequence

from run_ttt_lora import (
    CHUNK_SIZE,
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

DEFAULT_QA_TASKS = (
    "longbench_2wikimqa",
    "longbench_hotpotqa",
    "longbench_musique",
    "longbench_narrativeqa",
    "longbench_qasper",
    "longbench_triviaqa",
    "longbench_multifieldqa_en",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search baseline QA failures and test whether distillation recovers one.")
    parser.add_argument("--model-name-or-path", default="/work/yufei/downloads/Qwen3-0.6B-Base")
    parser.add_argument("--samples-base-dir", type=str, default=str(DEFAULT_SAMPLES_BASE))
    parser.add_argument("--task-name", type=str, default=None, help="Run directly on one sample task name.")
    parser.add_argument("--doc-id", type=int, default=None, help="Run directly on one sample doc id.")
    parser.add_argument("--qa-task", action="append", default=[], help="Optional LongBench QA task name. May be repeated.")
    parser.add_argument("--max-baseline-score", type=float, default=0.0)
    parser.add_argument("--max-trials", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--num-qa-candidates", type=int, default=NUM_QA_CANDIDATES)
    parser.add_argument("--num-judge-candidates", type=int, default=NUM_JUDGE_CANDIDATES)
    parser.add_argument("--num-selected-qa", type=int, default=NUM_SELECTED_QA)
    parser.add_argument("--qa-generation-max-new-tokens", type=int, default=QA_GENERATION_MAX_NEW_TOKENS)
    parser.add_argument("--qa-judge-max-new-tokens", type=int, default=QA_JUDGE_MAX_NEW_TOKENS)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=("auto", "bfloat16", "float16", "float32"))
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = {}
    for token in pred_tokens:
        common[token] = common.get(token, 0) + 1
    overlap = 0
    for token in gold_tokens:
        count = common.get(token, 0)
        if count > 0:
            overlap += 1
            common[token] = count - 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def best_alias_f1(prediction: str, answers: Sequence[str]) -> float:
    return max((token_f1(prediction, answer) for answer in answers), default=0.0)


def first_answer_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return text.strip()


def summarize_baseline_answer(raw_response) -> str:
    if isinstance(raw_response, list) and raw_response:
        return first_answer_line(str(raw_response[0]))
    return first_answer_line(str(raw_response or ""))


def render_truncated_text(tokenizer, text: str, max_ctx_len: int) -> str:
    encoded = tokenizer([text], truncation=True, padding="longest", return_tensors="pt")
    kept_ids = encoded["input_ids"][:, -max_ctx_len:]
    return tokenizer.decode(kept_ids[0].tolist(), skip_special_tokens=True)


def iter_failed_samples(samples_base: Path, task_names: Sequence[str], max_baseline_score: float) -> Iterable[tuple[str, dict]]:
    for task_name in task_names:
        candidates = sorted(samples_base.glob(f"samples_{task_name}_*.jsonl"))
        if not candidates:
            continue
        sample_path = candidates[-1]
        with sample_path.open() as handle:
            for line in handle:
                obj = json.loads(line)
                score = float(obj.get("score", obj.get("qa_f1_score", 0.0)))
                if score <= max_baseline_score:
                    yield task_name, obj


def load_sample_by_task_and_doc_id(samples_base: Path, task_name: str, doc_id: int) -> dict:
    candidates = sorted(samples_base.glob(f"samples_{task_name}_*.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"No sample file found for task {task_name} under {samples_base}")
    sample_path = candidates[-1]
    with sample_path.open() as handle:
        for line in handle:
            obj = json.loads(line)
            if int(obj["doc_id"]) == int(doc_id):
                return obj
    raise ValueError(f"doc_id={doc_id} not found for task {task_name} in {sample_path}")


def build_sample_spec(sample_obj: dict, tokenizer, max_ctx_len: int) -> SampleSpec:
    doc = sample_obj["doc"]
    prompt = sample_obj["arguments"]["gen_args_0"]["arg_0"]
    adaptation_text = render_truncated_text(tokenizer, str(doc.get("context") or prompt), max_ctx_len=max_ctx_len)
    final_prompt = render_truncated_text(tokenizer, prompt, max_ctx_len=max_ctx_len)
    task_question = str(doc.get("question") or "").strip() or "Answer the final benchmark task using the document."
    return SampleSpec(
        final_prompt=final_prompt,
        adaptation_text=adaptation_text,
        task_question=task_question,
        metadata={
            "task_name": doc.get("task"),
            "doc_id": sample_obj.get("doc_id"),
        },
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    samples_base = Path(args.samples_base_dir)
    task_names = tuple(args.qa_task) if args.qa_task else DEFAULT_QA_TASKS
    device = resolve_device(args.device)
    torch_dtype = resolve_dtype(args.dtype)

    model, tokenizer = build_model_and_tokenizer(
        model_name_or_path=args.model_name_or_path,
        device=device,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
    )
    trainable_lora_params = get_trainable_lora_parameters(model)
    initial_lora_state = clone_trainable_state(trainable_lora_params)

    trials = []
    if (args.task_name is None) != (args.doc_id is None):
        raise ValueError("Provide both --task-name and --doc-id together.")

    if args.task_name is not None and args.doc_id is not None:
        sample_stream = [(args.task_name, load_sample_by_task_and_doc_id(samples_base, args.task_name, args.doc_id))]
    else:
        sample_stream = iter_failed_samples(samples_base, task_names, args.max_baseline_score)

    for task_name, sample_obj in sample_stream:
        sample_gen_kwargs = dict(sample_obj["arguments"]["gen_args_0"]["arg_1"])
        max_gen_toks = int(sample_gen_kwargs.get("max_gen_toks", sample_obj["doc"].get("max_new_tokens", 32)))
        max_ctx_len = args.max_length - max_gen_toks
        sample = build_sample_spec(sample_obj, tokenizer=tokenizer, max_ctx_len=max_ctx_len)

        result = adapt_and_generate_for_sample(
            model=model,
            tokenizer=tokenizer,
            sample=sample,
            sample_index=len(trials),
            chunk_size=args.chunk_size,
            update_mode="full_prefix_approx",
            lr=args.lr,
            num_qa_candidates=args.num_qa_candidates,
            num_judge_candidates=args.num_judge_candidates,
            num_selected_qa=args.num_selected_qa,
            qa_generation_max_new_tokens=args.qa_generation_max_new_tokens,
            qa_judge_max_new_tokens=args.qa_judge_max_new_tokens,
            max_new_tokens=max_gen_toks,
            do_sample=bool(sample_gen_kwargs.get("do_sample", False)),
            temperature=float(sample_gen_kwargs.get("temperature", 1.0)),
            top_p=float(sample_gen_kwargs.get("top_p", 1.0)),
            initial_lora_state=initial_lora_state,
            trainable_lora_params=trainable_lora_params,
            log_file=None,
        )

        answers = list(sample_obj["doc"].get("answers") or [])
        prediction = first_answer_line(result["generated_text"])
        distill_f1 = best_alias_f1(prediction, answers)
        baseline_score = float(sample_obj.get("score", sample_obj.get("qa_f1_score", 0.0)))
        trial = {
            "task_name": task_name,
            "doc_id": int(sample_obj["doc_id"]),
            "baseline_score": baseline_score,
            "question": sample_obj["doc"].get("question"),
            "ground_truth_answer": answers[0] if answers else "",
            "baseline_answer": summarize_baseline_answer(sample_obj.get("filtered_resps")),
            "method_answer": prediction,
            "baseline_response": sample_obj.get("filtered_resps"),
            "answers": answers,
            "distill_prediction": prediction,
            "distill_raw_text": result["generated_text"],
            "distill_f1": distill_f1,
        }
        trials.append(trial)
        pretty_trial = {
            "task_name": trial["task_name"],
            "doc_id": trial["doc_id"],
            "baseline_score": trial["baseline_score"],
            "distill_f1": trial["distill_f1"],
            "question": trial["question"],
            "ground_truth_answer": trial["ground_truth_answer"],
            "baseline_answer": trial["baseline_answer"],
            "method_answer": trial["method_answer"],
        }
        print(json.dumps(pretty_trial, ensure_ascii=True, indent=2))

        if distill_f1 > baseline_score:
            summary = {
                "status": "improved",
                "trial_count": len(trials),
                "best_trial": trial,
            }
            if args.output_json:
                Path(args.output_json).write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n")
            return

        if len(trials) >= args.max_trials:
            break

    summary = {
        "status": "no_improvement_found",
        "trial_count": len(trials),
        "trials": trials,
    }
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=True))


if __name__ == "__main__":
    main()
