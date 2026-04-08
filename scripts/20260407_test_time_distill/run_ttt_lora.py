#!/usr/bin/env python3
"""
Question-answer test-time distillation with a temporary LoRA adapter.

Prototype scope:
1. Attach LoRA to q_proj and v_proj in the top 12.5% of layers.
2. Reset the temporary adapter at the start of each sample.
3. Split the adaptation context into contiguous chunks.
4. For each chunk:
   - decode the chunk to text,
   - generate diverse candidate QA pairs conditioned on the task question,
   - optionally filter malformed candidates,
   - let the model score the candidates for utility on the final task,
   - keep the top 2,
   - train LoRA on answer-only causal-LM loss for each selected pair,
   - rebuild cache from the real seen adaptation prefix.
5. After the last chunk, rebuild cache for the final inference prompt and decode.

The base model stays frozen. Only LoRA parameters are updated.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


TARGET_MODULES = ("q_proj", "v_proj")
TOP_LAYER_FRACTION = 0.125
CHUNK_SIZE = 1024
UPDATE_MODE = "full_prefix_approx"
NUM_QA_CANDIDATES = 4
NUM_JUDGE_CANDIDATES = 3
NUM_SELECTED_QA = 2
QA_GENERATION_MAX_NEW_TOKENS = 128
QA_JUDGE_MAX_NEW_TOKENS = 16
LORA_R = 64
LORA_ALPHA = 64
LORA_DROPOUT = 0.0
LR = 5e-5
BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.0
GRAD_CLIP = 1.0
DEFAULT_TASK_DESCRIPTION = "Summarize the full document faithfully."
DEFAULT_FALLBACK_CHUNK_TASK = "Extract task-relevant, factual knowledge from this chunk."
PREFERRED_CANDIDATE_LEN_RANGE = (4, 40)

QA_TASKS = {
    "hotpotqa",
    "2wikimqa",
    "musique",
    "dureader",
    "multifieldqa_en",
    "multifieldqa_zh",
    "narrativeqa",
    "qasper",
    "triviaqa",
}

SUMMARY_TASKS = {
    "gov_report",
    "qmsum",
    "multi_news",
    "vcsum",
    "samsum",
}

CODE_TASKS = {
    "lcc",
    "repobench-p",
}

CLASSIFICATION_TASKS = {
    "trec",
    "lsht",
}

SYNTHETIC_TASKS = {
    "passage_retrieval_en",
    "passage_retrieval_zh",
    "passage_count",
}


@dataclass
class RebuildResult:
    past_key_values: object
    last_logits: torch.Tensor


@dataclass
class SampleSpec:
    final_prompt: str
    adaptation_text: str
    task_question: str
    metadata: Dict[str, object]


@dataclass
class QAPair:
    question: str
    answer: str
    candidate_idx: int
    heuristic_score: float
    candidate_type: str = "qa_pair"
    judge_score: Optional[float] = None
    judge_reason: Optional[str] = None


def should_prefer_flash_attention_2(
    device: torch.device,
    torch_dtype: Optional[torch.dtype],
) -> bool:
    if device.type != "cuda":
        return False
    if torch_dtype not in (torch.bfloat16, torch.float16):
        return False
    return importlib.util.find_spec("flash_attn") is not None


def load_causal_lm_with_preferred_attention(
    model_name_or_path: str,
    torch_dtype: Optional[torch.dtype],
    trust_remote_code: bool,
    device: torch.device,
    **extra_kwargs,
):
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": trust_remote_code,
        **extra_kwargs,
    }
    if should_prefer_flash_attention_2(device=device, torch_dtype=torch_dtype):
        try:
            return AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                attn_implementation="flash_attention_2",
                **model_kwargs,
            )
        except Exception:
            pass
    return AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **model_kwargs,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunked QA-distillation test-time LoRA for a frozen causal LM.")
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen3-0.6B", help="Base model path or HF id.")
    parser.add_argument("--prompt", action="append", default=[], help="Final prompt text. May be passed multiple times.")
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Optional file containing one final prompt per line.",
    )
    parser.add_argument(
        "--task-question",
        action="append",
        default=[],
        help="Optional task question aligned with --prompt entries. Falls back to a task description if omitted.",
    )
    parser.add_argument(
        "--adaptation-text",
        action="append",
        default=[],
        help="Optional adaptation context aligned with --prompt entries. Defaults to the final prompt.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Number of generation tokens.")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Adaptation chunk size.")
    parser.add_argument(
        "--update-mode",
        type=str,
        default=UPDATE_MODE,
        choices=("full_prefix_approx",),
        help="Prototype currently supports only full_prefix_approx.",
    )
    parser.add_argument("--lora-r", type=int, default=LORA_R, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=LORA_ALPHA, help="LoRA alpha.")
    parser.add_argument("--lr", type=float, default=LR, help="AdamW learning rate.")
    parser.add_argument("--num-qa-candidates", type=int, default=NUM_QA_CANDIDATES)
    parser.add_argument("--num-judge-candidates", type=int, default=NUM_JUDGE_CANDIDATES)
    parser.add_argument("--num-selected-qa", type=int, default=NUM_SELECTED_QA)
    parser.add_argument("--qa-generation-max-new-tokens", type=int, default=QA_GENERATION_MAX_NEW_TOKENS)
    parser.add_argument("--qa-judge-max-new-tokens", type=int, default=QA_JUDGE_MAX_NEW_TOKENS)
    parser.add_argument("--device", type=str, default=None, help="Torch device, e.g. cuda:0 or cpu.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=("auto", "bfloat16", "float16", "float32"),
        help="Model dtype.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True.")
    parser.add_argument("--do-sample", action="store_true", help="Use sampling instead of greedy decoding.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling.")
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional JSONL file for per-chunk logs.",
    )
    parser.add_argument(
        "--print-generated-only",
        action="store_true",
        help="Print only generation text instead of structured JSON output.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dtype(dtype_name: str) -> Optional[torch.dtype]:
    if dtype_name == "auto":
        return None
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _load_lines(path: Path) -> List[str]:
    return [line.rstrip("\n") for line in path.read_text().splitlines() if line.strip()]


def load_sample_specs(args: argparse.Namespace) -> List[SampleSpec]:
    prompts = list(args.prompt)
    if args.prompt_file:
        prompts.extend(_load_lines(Path(args.prompt_file)))
    if not prompts:
        raise ValueError("Provide at least one --prompt or a --prompt-file.")

    task_questions = list(args.task_question)
    adaptation_texts = list(args.adaptation_text)

    if task_questions and len(task_questions) != len(prompts):
        raise ValueError("--task-question count must match the number of prompts when provided.")
    if adaptation_texts and len(adaptation_texts) != len(prompts):
        raise ValueError("--adaptation-text count must match the number of prompts when provided.")

    specs = []
    for idx, prompt in enumerate(prompts):
        task_question = task_questions[idx] if idx < len(task_questions) and task_questions[idx].strip() else DEFAULT_TASK_DESCRIPTION
        adaptation_text = adaptation_texts[idx] if idx < len(adaptation_texts) and adaptation_texts[idx].strip() else prompt
        specs.append(
            SampleSpec(
                final_prompt=prompt,
                adaptation_text=adaptation_text,
                task_question=task_question,
                metadata={},
            )
        )
    return specs


def resolve_top_layer_target_modules(
    model,
    module_suffixes: Sequence[str] = TARGET_MODULES,
    top_layer_fraction: float = TOP_LAYER_FRACTION,
) -> List[str]:
    num_layers = int(getattr(model.config, "num_hidden_layers"))
    num_target_layers = max(1, math.ceil(num_layers * top_layer_fraction))
    first_target_layer = num_layers - num_target_layers

    selected = []
    for module_name, _module in model.named_modules():
        if not any(module_name.endswith(suffix) for suffix in module_suffixes):
            continue
        for layer_idx in range(first_target_layer, num_layers):
            if f".layers.{layer_idx}." in module_name:
                selected.append(module_name)
                break

    if not selected:
        raise RuntimeError(
            f"Could not find target modules for top {num_target_layers}/{num_layers} layers "
            f"with suffixes {list(module_suffixes)}"
        )

    return sorted(set(selected))


def attach_ttt_lora(
    model,
    module_suffixes: Sequence[str] = TARGET_MODULES,
    top_layer_fraction: float = TOP_LAYER_FRACTION,
    lora_r: int = LORA_R,
    lora_alpha: int = LORA_ALPHA,
):
    target_modules = resolve_top_layer_target_modules(
        model,
        module_suffixes=module_suffixes,
        top_layer_fraction=top_layer_fraction,
    )

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name

    return model


def build_model_and_tokenizer(
    model_name_or_path: str,
    device: torch.device,
    torch_dtype: Optional[torch.dtype],
    trust_remote_code: bool,
    lora_r: int = LORA_R,
    lora_alpha: int = LORA_ALPHA,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_causal_lm_with_preferred_attention(
        model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        device=device,
    )
    model.eval()
    model = attach_ttt_lora(model, lora_r=lora_r, lora_alpha=lora_alpha)
    model.to(device)

    return model, tokenizer


def get_trainable_lora_parameters(model) -> List[Tuple[str, torch.nn.Parameter]]:
    return [(name, param) for name, param in model.named_parameters() if param.requires_grad]


def clone_trainable_state(named_params: Sequence[Tuple[str, torch.nn.Parameter]]) -> Dict[str, torch.Tensor]:
    return {name: param.detach().clone().cpu() for name, param in named_params}


def reset_trainable_state(
    named_params: Sequence[Tuple[str, torch.nn.Parameter]],
    initial_state: Dict[str, torch.Tensor],
) -> None:
    with torch.no_grad():
        for name, param in named_params:
            param.copy_(initial_state[name].to(device=param.device, dtype=param.dtype))


def compute_delta_norm(
    named_params: Sequence[Tuple[str, torch.nn.Parameter]],
    initial_state: Dict[str, torch.Tensor],
) -> float:
    total = 0.0
    with torch.no_grad():
        for name, param in named_params:
            ref = initial_state[name].to(device=param.device, dtype=param.dtype)
            total += torch.sum((param.detach() - ref) ** 2).item()
    return math.sqrt(total)


def create_optimizer(params: Iterable[torch.nn.Parameter], lr: float = LR) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        params,
        lr=lr,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
    )


def sample_next_token(
    logits: torch.Tensor,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> torch.LongTensor:
    next_token_logits = logits[:, -1, :]
    if not do_sample:
        return torch.argmax(next_token_logits, dim=-1, keepdim=True)

    if temperature <= 0:
        raise ValueError("temperature must be > 0 when do_sample=True")

    next_token_logits = next_token_logits / temperature
    probs = F.softmax(next_token_logits, dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove_mask = cumulative > top_p
        remove_mask[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(remove_mask, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        sampled = torch.multinomial(sorted_probs, num_samples=1)
        return sorted_indices.gather(-1, sampled)

    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def rebuild_cache_for_prefix(
    model,
    prefix_ids: torch.LongTensor,
    chunk_size: int,
) -> RebuildResult:
    past_key_values = None
    last_logits = None
    seq_len = prefix_ids.shape[1]
    device = prefix_ids.device

    with torch.no_grad():
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            chunk_ids = prefix_ids[:, start:end]
            cache_position = torch.arange(start, end, device=device)
            outputs = model(
                input_ids=chunk_ids,
                past_key_values=past_key_values,
                use_cache=True,
                cache_position=cache_position,
            )
            past_key_values = outputs.past_key_values
            last_logits = outputs.logits

    if last_logits is None:
        raise ValueError("Cannot rebuild cache for an empty prefix.")

    return RebuildResult(past_key_values=past_key_values, last_logits=last_logits)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_task_question(task_question: str) -> str:
    task_question = _normalize_whitespace(task_question)
    if task_question:
        return task_question
    return DEFAULT_FALLBACK_CHUNK_TASK


def normalize_task_name(task_name: Optional[str]) -> str:
    return (task_name or "").strip().lower()


def get_task_family(task_name: Optional[str]) -> str:
    normalized = normalize_task_name(task_name)
    if normalized in QA_TASKS:
        return "qa"
    if normalized in SUMMARY_TASKS:
        return "summary"
    if normalized in CODE_TASKS:
        return "code"
    if normalized in CLASSIFICATION_TASKS:
        return "classification"
    if normalized in SYNTHETIC_TASKS:
        return "synthetic"
    return "qa"


def decode_chunk_text(tokenizer, chunk_ids: torch.LongTensor) -> str:
    return tokenizer.decode(chunk_ids[0].tolist(), skip_special_tokens=True).strip()


def build_qa_generation_prompt(chunk_text: str, task_question: str, num_pairs: int) -> str:
    return (
        "You are extracting task-relevant facts from one context chunk.\n\n"
        f"Task question:\n{task_question}\n\n"
        f"Context chunk:\n{chunk_text}\n\n"
        f"Generate {num_pairs} diverse question-answer pairs that capture information from this chunk that is useful "
        "for solving the task question.\n\n"
        "Diversity requirements:\n"
        "- Cover different types of information when possible: entities, numbers, dates, conditions, causes, "
        "definitions, constraints, decisions, conclusions, or code behaviors.\n"
        "- Avoid asking the same thing in different words.\n\n"
        "Faithfulness requirements:\n"
        "- Every question must be answerable using only this chunk.\n"
        "- Answers must be concise and factual.\n"
        "- Avoid vague or generic questions.\n\n"
        "Output format exactly:\n"
        "Q1: ...\n"
        "A1: ...\n"
        "Q2: ...\n"
        "A2: ...\n"
        "...\n"
    )


def build_summary_generation_prompt(chunk_text: str, task_question: str, num_candidates: int) -> str:
    return (
        "You are extracting salient summary units from one chunk.\n\n"
        f"Task:\n{task_question}\n\n"
        f"Context chunk:\n{chunk_text}\n\n"
        f"Generate {num_candidates} diverse summary bullets that would be useful for producing the final summary.\n\n"
        "Requirements:\n"
        "- Each bullet must be faithful to this chunk.\n"
        "- Prefer findings, decisions, recommendations, action items, events, or conclusions.\n"
        "- Avoid generic wording and redundancy.\n\n"
        "Output format exactly:\n"
        "B1: ...\n"
        "B2: ...\n"
        "...\n"
    )


def build_code_generation_prompt(chunk_text: str, task_question: str, num_candidates: int) -> str:
    return (
        "You are extracting reusable code rules from one code chunk.\n\n"
        f"Task:\n{task_question}\n\n"
        f"Code context chunk:\n{chunk_text}\n\n"
        f"Generate {num_candidates} concise code rules that would help predict the correct next code later.\n\n"
        "Requirements:\n"
        "- Focus on identifiers, APIs, control-flow patterns, data-structure usage, or local style rules.\n"
        "- Keep each rule specific and faithful to the chunk.\n"
        "- Avoid generic programming advice.\n\n"
        "Output format exactly:\n"
        "R1: ...\n"
        "R2: ...\n"
        "...\n"
    )


def build_classification_generation_prompt(chunk_text: str, task_question: str, num_candidates: int) -> str:
    return (
        "You are extracting discriminative classification clues from one chunk.\n\n"
        f"Task:\n{task_question}\n\n"
        f"Context chunk:\n{chunk_text}\n\n"
        f"Generate {num_candidates} concise classification clues that would help determine the correct label later.\n\n"
        "Output format exactly:\n"
        "C1: ...\n"
        "C2: ...\n"
        "...\n"
    )


def build_synthetic_generation_prompt(chunk_text: str, task_question: str, num_candidates: int) -> str:
    return (
        "You are extracting exact structured facts from one chunk.\n\n"
        f"Task:\n{task_question}\n\n"
        f"Context chunk:\n{chunk_text}\n\n"
        f"Generate {num_candidates} exact structured facts that would help solve the task later.\n\n"
        "Requirements:\n"
        "- Prefer exact names, counts, positions, retrieval keys, or unique facts.\n"
        "- Avoid vague summaries.\n\n"
        "Output format exactly:\n"
        "F1: ...\n"
        "F2: ...\n"
        "...\n"
    )


def _extract_generated_suffix(prompt_text: str, full_text: str) -> str:
    if full_text.startswith(prompt_text):
        return full_text[len(prompt_text):].strip()
    return full_text.strip()


@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt_text: str,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
) -> str:
    device = next(model.parameters()).device
    encoded = tokenizer(prompt_text, return_tensors="pt").to(device)
    generate_kwargs = dict(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        generate_kwargs["temperature"] = temperature
    generated = model.generate(**generate_kwargs)
    full_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return _extract_generated_suffix(prompt_text, full_text)


def parse_qa_pairs(text: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    lines = text.splitlines()
    current_q: Optional[str] = None
    current_a: Optional[str] = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if re.match(r"^Q\d+\s*:", line):
            parts = line.split(":", 1)
            if len(parts) == 2:
                current_q = _normalize_whitespace(parts[1])
                current_a = None
        elif re.match(r"^A\d+\s*:", line):
            parts = line.split(":", 1)
            if len(parts) == 2:
                current_a = _normalize_whitespace(parts[1])
                if current_q and current_a:
                    pairs.append((current_q, current_a))
                    current_q, current_a = None, None

    return pairs


def parse_prefixed_list(text: str, prefix: str) -> List[str]:
    items: List[str] = []
    pattern = re.compile(rf"^{re.escape(prefix)}\d+\s*:\s*(.*)$")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = pattern.match(line)
        if not match:
            continue
        item = _normalize_whitespace(match.group(1))
        if item:
            items.append(item)
    return items


def lexical_overlap_score(question: str, task_question: str) -> float:
    q_words = set(re.findall(r"\w+", question.lower()))
    task_words = set(re.findall(r"\w+", task_question.lower()))
    if not q_words or not task_words:
        return 0.0
    return float(len(q_words & task_words))


def heuristic_score_pair(question: str, answer: str, task_question: str) -> float:
    answer_words = answer.split()
    score = lexical_overlap_score(question, task_question)
    answer_len = len(answer_words)
    score += min(answer_len, 40) * 0.08
    if 4 <= answer_len <= 40:
        score += 1.0
    if any(ch.isdigit() for ch in answer):
        score += 0.75
    if any(token[:1].isupper() for token in answer_words if token):
        score += 0.4
    if any(ch in answer for ch in "%$:()/-"):
        score += 0.3
    if len(question.split()) < 4:
        score -= 0.75
    return score


def preferred_length_bonus(text: str) -> float:
    low, high = PREFERRED_CANDIDATE_LEN_RANGE
    length = len(text.split())
    if low <= length <= high:
        return 1.0
    if length < low:
        return -0.5
    return max(-0.5, -(length - high) * 0.03)


def dedupe_qa_pairs(pairs: Sequence[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen = set()
    kept = []
    for question, answer in pairs:
        key = (_normalize_whitespace(question).lower(), _normalize_whitespace(answer).lower())
        if key in seen:
            continue
        seen.add(key)
        kept.append((question, answer))
    return kept


def filter_candidate_qa_pairs(
    pairs: Sequence[Tuple[str, str]],
    task_question: str,
) -> List[QAPair]:
    filtered: List[QAPair] = []
    for idx, (question, answer) in enumerate(dedupe_qa_pairs(pairs), start=1):
        q = _normalize_whitespace(question)
        a = _normalize_whitespace(answer)
        if not q or not a:
            continue
        answer_len = len(a.split())
        if answer_len < 2 or answer_len > 80:
            continue
        filtered.append(
            QAPair(
                question=q,
                answer=a,
                candidate_idx=idx,
                heuristic_score=heuristic_score_pair(q, a, task_question),
                candidate_type="qa_pair",
            )
        )
    return filtered


def filter_statement_candidates(
    candidates: Sequence[str],
    task_question: str,
    candidate_type: str,
) -> List[QAPair]:
    filtered: List[QAPair] = []
    seen = set()
    for idx, text in enumerate(candidates, start=1):
        normalized = _normalize_whitespace(text)
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        word_count = len(normalized.split())
        if word_count < 3 or word_count > 80:
            continue
        heuristic = preferred_length_bonus(normalized) + lexical_overlap_score(normalized, task_question)
        filtered.append(
            QAPair(
                question=task_question,
                answer=normalized,
                candidate_idx=idx,
                heuristic_score=heuristic,
                candidate_type=candidate_type,
            )
        )
    return filtered


def build_qa_judge_prompt(
    task_question: str,
    chunk_text: str,
    candidate: QAPair,
) -> str:
    return (
        f"Task question:\n{task_question}\n\n"
        f"Context chunk:\n{chunk_text}\n\n"
        "Candidate QA pair:\n"
        f"Question: {candidate.question}\n"
        f"Answer: {candidate.answer}\n\n"
        "Score this candidate from 1 to 5 for how useful it would be if stored in memory for answering the task "
        "question later, when the original chunk may not be available.\n\n"
        "Scoring criteria:\n"
        "1. Relevance to the task question\n"
        "2. Specificity of entities, numbers, dates, or facts\n"
        "3. Usefulness as direct evidence or a bridge fact\n"
        "4. Non-redundancy\n"
        "5. Faithfulness to the chunk\n\n"
        "Output exactly:\n"
        "Score: <1-5>\n"
        "Reason: <short reason>\n"
    )


def build_summary_judge_prompt(
    task_question: str,
    chunk_text: str,
    candidate: QAPair,
) -> str:
    return (
        f"Task:\n{task_question}\n\n"
        f"Context chunk:\n{chunk_text}\n\n"
        "Candidate summary bullet:\n"
        f"{candidate.answer}\n\n"
        "Score this candidate from 1 to 5 for how useful it would be for producing the final summary later.\n\n"
        "Scoring criteria:\n"
        "1. Salience for the final summary\n"
        "2. Faithfulness to the chunk\n"
        "3. Non-redundancy\n"
        "4. Usefulness for final summary quality\n"
        "5. Specificity (not generic wording)\n\n"
        "Output exactly:\n"
        "Score: <1-5>\n"
        "Reason: <short reason>\n"
    )


def build_code_judge_prompt(
    task_question: str,
    chunk_text: str,
    candidate: QAPair,
) -> str:
    return (
        f"Task:\n{task_question}\n\n"
        f"Code context chunk:\n{chunk_text}\n\n"
        "Candidate code rule:\n"
        f"{candidate.answer}\n\n"
        "Score this candidate from 1 to 5 for how useful it would be for predicting the correct next code later.\n\n"
        "Scoring criteria:\n"
        "1. Relevance to the current completion task\n"
        "2. Specificity of identifiers, method names, APIs, or patterns\n"
        "3. Consistency with local coding style and structure\n"
        "4. Reusability for next-line prediction\n"
        "5. Faithfulness to the code chunk\n\n"
        "Output exactly:\n"
        "Score: <1-5>\n"
        "Reason: <short reason>\n"
    )


def build_classification_judge_prompt(
    task_question: str,
    chunk_text: str,
    candidate: QAPair,
) -> str:
    return (
        f"Task:\n{task_question}\n\n"
        f"Context chunk:\n{chunk_text}\n\n"
        "Candidate classification clue:\n"
        f"{candidate.answer}\n\n"
        "Score this candidate from 1 to 5 for how useful it would be for determining the correct class label later.\n\n"
        "Scoring criteria:\n"
        "1. Discriminative value for the label\n"
        "2. Specificity\n"
        "3. Faithfulness to the chunk\n"
        "4. Relevance to the classification task\n"
        "5. Non-redundancy\n\n"
        "Output exactly:\n"
        "Score: <1-5>\n"
        "Reason: <short reason>\n"
    )


def build_synthetic_judge_prompt(
    task_question: str,
    chunk_text: str,
    candidate: QAPair,
) -> str:
    return (
        f"Final task question:\n{task_question}\n\n"
        f"Context chunk:\n{chunk_text}\n\n"
        "Candidate structured fact:\n"
        f"{candidate.answer}\n\n"
        "Score this candidate from 1 to 5 for how useful it would be for solving the task later.\n\n"
        "Scoring criteria:\n"
        "1. Exactness\n"
        "2. Direct usefulness for retrieval or counting\n"
        "3. Unambiguity\n"
        "4. Faithfulness to the chunk\n"
        "5. Non-redundancy\n\n"
        "Output exactly:\n"
        "Score: <1-5>\n"
        "Reason: <short reason>\n"
    )


def build_judge_prompt(
    family: str,
    task_question: str,
    chunk_text: str,
    candidate: QAPair,
) -> str:
    if family == "summary":
        return build_summary_judge_prompt(task_question=task_question, chunk_text=chunk_text, candidate=candidate)
    if family == "code":
        return build_code_judge_prompt(task_question=task_question, chunk_text=chunk_text, candidate=candidate)
    if family == "classification":
        return build_classification_judge_prompt(task_question=task_question, chunk_text=chunk_text, candidate=candidate)
    if family == "synthetic":
        return build_synthetic_judge_prompt(task_question=task_question, chunk_text=chunk_text, candidate=candidate)
    return build_qa_judge_prompt(task_question=task_question, chunk_text=chunk_text, candidate=candidate)


def parse_judge_score(text: str) -> Tuple[float, Optional[str]]:
    score_match = re.search(r"Score:\s*([1-5])", text)
    reason_match = re.search(r"Reason:\s*(.*)", text)
    score = float(score_match.group(1)) if score_match else 0.0
    reason = _normalize_whitespace(reason_match.group(1)) if reason_match else ""
    return score, reason or None


def judge_qa_pairs_with_model(
    model,
    tokenizer,
    family: str,
    chunk_text: str,
    task_question: str,
    qa_pairs: Sequence[QAPair],
    max_new_tokens: int,
) -> List[QAPair]:
    if not qa_pairs:
        return []
    judged: List[QAPair] = []
    for pair in qa_pairs:
        prompt = build_judge_prompt(
            family=family,
            task_question=task_question,
            chunk_text=chunk_text,
            candidate=pair,
        )
        raw_judgment = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
        )
        judge_score, judge_reason = parse_judge_score(raw_judgment)
        pair.judge_score = judge_score
        pair.judge_reason = judge_reason
        judged.append(pair)
    return judged


def select_top_pairs(
    qa_pairs: Sequence[QAPair],
    task_question: str,
    k: int,
) -> List[QAPair]:
    ranked = sorted(
        qa_pairs,
        key=lambda pair: (
            -(pair.judge_score or 0.0),
            -preferred_length_bonus(pair.answer),
            -lexical_overlap_score(f"{pair.question} {pair.answer}", task_question),
            -pair.heuristic_score,
            pair.candidate_idx,
        ),
    )
    return list(ranked[:k])


def preselect_for_judging(
    qa_pairs: Sequence[QAPair],
    task_question: str,
    k: int,
) -> List[QAPair]:
    if k <= 0:
        return []
    ranked = sorted(
        qa_pairs,
        key=lambda pair: (
            -pair.heuristic_score,
            -preferred_length_bonus(pair.answer),
            -lexical_overlap_score(f"{pair.question} {pair.answer}", task_question),
            pair.candidate_idx,
        ),
    )
    return list(ranked[:k])


def generate_candidate_qa_pairs(
    model,
    tokenizer,
    family: str,
    chunk_text: str,
    task_question: str,
    num_pairs: int,
    max_new_tokens: int,
) -> Tuple[str, List[QAPair]]:
    if family == "summary":
        prompt = build_summary_generation_prompt(chunk_text=chunk_text, task_question=task_question, num_candidates=num_pairs)
    elif family == "code":
        prompt = build_code_generation_prompt(chunk_text=chunk_text, task_question=task_question, num_candidates=num_pairs)
    elif family == "classification":
        prompt = build_classification_generation_prompt(chunk_text=chunk_text, task_question=task_question, num_candidates=num_pairs)
    elif family == "synthetic":
        prompt = build_synthetic_generation_prompt(chunk_text=chunk_text, task_question=task_question, num_candidates=num_pairs)
    else:
        prompt = build_qa_generation_prompt(chunk_text=chunk_text, task_question=task_question, num_pairs=num_pairs)
    raw_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.3,
        do_sample=False,
    )
    if family == "summary":
        return raw_text, filter_statement_candidates(parse_prefixed_list(raw_text, "B"), task_question, "summary_bullet")
    if family == "code":
        return raw_text, filter_statement_candidates(parse_prefixed_list(raw_text, "R"), task_question, "code_rule")
    if family == "classification":
        return raw_text, filter_statement_candidates(parse_prefixed_list(raw_text, "C"), task_question, "classification_clue")
    if family == "synthetic":
        return raw_text, filter_statement_candidates(parse_prefixed_list(raw_text, "F"), task_question, "structured_fact")
    parsed_pairs = parse_qa_pairs(raw_text)
    return raw_text, filter_candidate_qa_pairs(parsed_pairs, task_question=task_question)


def build_qa_training_example(
    tokenizer,
    prompt_prefix: str,
    question: str,
    answer: str,
    device: torch.device,
) -> Tuple[torch.LongTensor, int]:
    prompt_text = f"{prompt_prefix}: {question}\nAnswer:"
    full_text = f"{prompt_text} {answer}"
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    answer_start = int(prompt_ids.shape[1])
    return full_ids, answer_start


def compute_answer_only_loss(
    logits: torch.Tensor,
    input_ids: torch.LongTensor,
    answer_start: int,
) -> Optional[torch.Tensor]:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    if shift_logits.numel() == 0 or shift_labels.numel() == 0:
        return None

    token_losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    ).view_as(shift_labels)

    keep_from = max(answer_start - 1, 0)
    mask = torch.zeros_like(token_losses, dtype=torch.bool)
    mask[:, keep_from:] = True
    kept_losses = token_losses[mask]
    if kept_losses.numel() == 0:
        return None
    return kept_losses.mean()


def train_lora_on_qa_pair(
    model,
    optimizer,
    tokenizer,
    candidate_type: str,
    question: str,
    answer: str,
    trainable_lora_params: Sequence[Tuple[str, torch.nn.Parameter]],
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    device = next(model.parameters()).device
    optimizer.zero_grad(set_to_none=True)
    prompt_prefix = {
        "summary_bullet": "Summary bullet",
        "code_rule": "Code rule",
        "classification_clue": "Classification clue",
        "structured_fact": "Structured fact",
    }.get(candidate_type, "Question")
    full_ids, answer_start = build_qa_training_example(
        tokenizer=tokenizer,
        prompt_prefix=prompt_prefix,
        question=question,
        answer=answer,
        device=device,
    )
    outputs = model(input_ids=full_ids, use_cache=False)
    loss = compute_answer_only_loss(outputs.logits, full_ids, answer_start=answer_start)

    if loss is not None:
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [param for _, param in trainable_lora_params],
            max_norm=GRAD_CLIP,
        )
        optimizer.step()
    else:
        grad_norm = torch.tensor(0.0, device=device)

    return loss, grad_norm


def format_pair_for_log(pair: QAPair) -> Dict[str, object]:
    return {
        "candidate_idx": pair.candidate_idx,
        "question": pair.question,
        "answer": pair.answer,
        "answer_len_tokens": len(pair.answer.split()),
        "heuristic_score": pair.heuristic_score,
        "judge_score": pair.judge_score,
        "judge_reason": pair.judge_reason,
    }


def run_chunk_distillation_step(
    model,
    tokenizer,
    optimizer,
    trainable_lora_params: Sequence[Tuple[str, torch.nn.Parameter]],
    task_name: Optional[str],
    chunk_text: str,
    task_question: str,
    num_qa_candidates: int,
    num_judge_candidates: int,
    num_selected_qa: int,
    qa_generation_max_new_tokens: int,
    qa_judge_max_new_tokens: int,
) -> Dict[str, object]:
    family = get_task_family(task_name)
    raw_qa_text, parsed_pairs = generate_candidate_qa_pairs(
        model=model,
        tokenizer=tokenizer,
        family=family,
        chunk_text=chunk_text,
        task_question=task_question,
        num_pairs=num_qa_candidates,
        max_new_tokens=qa_generation_max_new_tokens,
    )
    preselected_pairs = preselect_for_judging(parsed_pairs, task_question=task_question, k=num_judge_candidates)
    judged_pairs = judge_qa_pairs_with_model(
        model=model,
        tokenizer=tokenizer,
        family=family,
        chunk_text=chunk_text,
        task_question=task_question,
        qa_pairs=preselected_pairs,
        max_new_tokens=qa_judge_max_new_tokens,
    )
    selected_pairs = select_top_pairs(judged_pairs, task_question=task_question, k=num_selected_qa)

    selected_logs = []
    for pair in selected_pairs:
        loss, grad_norm = train_lora_on_qa_pair(
            model=model,
            optimizer=optimizer,
            tokenizer=tokenizer,
            candidate_type=pair.candidate_type,
            question=pair.question,
            answer=pair.answer,
            trainable_lora_params=trainable_lora_params,
        )
        selected_logs.append(
            {
                **format_pair_for_log(pair),
                "loss": None if loss is None else float(loss.detach().cpu().item()),
                "grad_norm": float(grad_norm.detach().cpu().item()) if torch.is_tensor(grad_norm) else float(grad_norm),
            }
        )

    return {
        "task_family": family,
        "raw_qa_text": raw_qa_text,
        "parsed_pairs": [format_pair_for_log(pair) for pair in parsed_pairs],
        "filtered_pairs": [format_pair_for_log(pair) for pair in parsed_pairs],
        "judge_pairs": [format_pair_for_log(pair) for pair in preselected_pairs],
        "selected_pairs": selected_logs,
        "num_selected_pairs": len(selected_logs),
    }


def generate_from_adapted_state(
    model,
    prefix_ids: torch.LongTensor,
    past_key_values,
    last_logits: torch.Tensor,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    eos_token_id: Optional[int],
) -> torch.LongTensor:
    device = prefix_ids.device
    generated: List[torch.LongTensor] = []
    current_past = past_key_values
    current_len = prefix_ids.shape[1]
    logits = last_logits

    with torch.no_grad():
        for _ in range(max_new_tokens):
            next_token = sample_next_token(logits, do_sample=do_sample, temperature=temperature, top_p=top_p)
            generated.append(next_token)

            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break

            cache_position = torch.arange(current_len, current_len + 1, device=device)
            outputs = model(
                input_ids=next_token,
                past_key_values=current_past,
                use_cache=True,
                cache_position=cache_position,
            )
            current_past = outputs.past_key_values
            logits = outputs.logits
            current_len += 1

    if not generated:
        return torch.empty((prefix_ids.shape[0], 0), device=device, dtype=prefix_ids.dtype)

    return torch.cat(generated, dim=1)


def log_chunk(log_file, payload: Dict[str, object]) -> None:
    line = json.dumps(payload, ensure_ascii=True)
    if log_file is not None:
        log_file.write(line + "\n")
        log_file.flush()


def adapt_and_generate_for_sample(
    model,
    tokenizer,
    sample: SampleSpec,
    sample_index: int,
    chunk_size: int,
    update_mode: str,
    lr: float,
    num_qa_candidates: int,
    num_judge_candidates: int,
    num_selected_qa: int,
    qa_generation_max_new_tokens: int,
    qa_judge_max_new_tokens: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    initial_lora_state: Dict[str, torch.Tensor],
    trainable_lora_params: Sequence[Tuple[str, torch.nn.Parameter]],
    log_file,
) -> Dict[str, object]:
    if update_mode != "full_prefix_approx":
        raise ValueError(f"Unsupported update_mode for prototype: {update_mode}")

    device = next(model.parameters()).device
    adaptation_ids = tokenizer(sample.adaptation_text, return_tensors="pt")["input_ids"].to(device)
    final_prompt_ids = tokenizer(sample.final_prompt, return_tensors="pt")["input_ids"].to(device)
    task_question = normalize_task_question(sample.task_question)

    if adaptation_ids.shape[1] == 0:
        raise ValueError("Empty adaptation text after tokenization.")
    if final_prompt_ids.shape[1] == 0:
        raise ValueError("Empty final prompt after tokenization.")

    reset_trainable_state(trainable_lora_params, initial_lora_state)
    optimizer = create_optimizer([param for _, param in trainable_lora_params], lr=lr)

    for chunk_idx, start in enumerate(range(0, adaptation_ids.shape[1], chunk_size)):
        end = min(start + chunk_size, adaptation_ids.shape[1])
        chunk_ids = adaptation_ids[:, start:end]
        chunk_text = decode_chunk_text(tokenizer, chunk_ids)
        if not chunk_text:
            chunk_log = {
                "event": "chunk_update",
                "sample_idx": sample_index,
                "chunk_idx": chunk_idx,
                "chunk_start": start,
                "chunk_end": end,
                "chunk_tokens": end - start,
                "task_question": task_question,
                "skip_reason": "empty_chunk_text",
                "lora_norm": compute_delta_norm(trainable_lora_params, initial_lora_state),
            }
            log_chunk(log_file, chunk_log)
            continue

        chunk_result = run_chunk_distillation_step(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            trainable_lora_params=trainable_lora_params,
            task_name=str(sample.metadata.get("task_name") or ""),
            chunk_text=chunk_text,
            task_question=task_question,
            num_qa_candidates=num_qa_candidates,
            num_judge_candidates=num_judge_candidates,
            num_selected_qa=num_selected_qa,
            qa_generation_max_new_tokens=qa_generation_max_new_tokens,
            qa_judge_max_new_tokens=qa_judge_max_new_tokens,
        )
        seen_prefix = adaptation_ids[:, :end]
        rebuild_cache_for_prefix(model, seen_prefix, chunk_size=chunk_size)

        chunk_log = {
            "event": "chunk_update",
            "sample_idx": sample_index,
            "chunk_idx": chunk_idx,
            "chunk_start": start,
            "chunk_end": end,
            "chunk_tokens": end - start,
            "task_question": task_question,
            "task_family": chunk_result["task_family"],
            "chunk_text_preview": chunk_text[:400],
            "raw_qa_text": chunk_result["raw_qa_text"],
            "parsed_pairs": chunk_result["parsed_pairs"],
            "filtered_pairs": chunk_result["filtered_pairs"],
            "judge_pairs": chunk_result["judge_pairs"],
            "selected_pairs": chunk_result["selected_pairs"],
            "num_selected_pairs": chunk_result["num_selected_pairs"],
            "lora_norm": compute_delta_norm(trainable_lora_params, initial_lora_state),
            **sample.metadata,
        }
        log_chunk(log_file, chunk_log)

    final_rebuild = rebuild_cache_for_prefix(model, final_prompt_ids, chunk_size=chunk_size)
    generated_ids = generate_from_adapted_state(
        model=model,
        prefix_ids=final_prompt_ids,
        past_key_values=final_rebuild.past_key_values,
        last_logits=final_rebuild.last_logits,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
    )

    full_ids = torch.cat([final_prompt_ids, generated_ids], dim=1)
    return {
        "sample_idx": sample_index,
        "prompt": sample.final_prompt,
        "adaptation_text": sample.adaptation_text,
        "task_question": task_question,
        "adaptation_token_count": int(adaptation_ids.shape[1]),
        "prompt_token_count": int(final_prompt_ids.shape[1]),
        "generated_token_count": int(generated_ids.shape[1]),
        "generated_text": tokenizer.decode(generated_ids[0], skip_special_tokens=True),
        "full_text": tokenizer.decode(full_ids[0], skip_special_tokens=True),
        "final_lora_norm": compute_delta_norm(trainable_lora_params, initial_lora_state),
        **sample.metadata,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)
    torch_dtype = resolve_dtype(args.dtype)

    model, tokenizer = build_model_and_tokenizer(
        model_name_or_path=args.model_name_or_path,
        device=device,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )

    trainable_lora_params = get_trainable_lora_parameters(model)
    if not trainable_lora_params:
        raise RuntimeError("No trainable LoRA parameters were found.")

    initial_lora_state = clone_trainable_state(trainable_lora_params)
    samples = load_sample_specs(args)

    log_file = None
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w", encoding="utf-8")

    try:
        for sample_index, sample in enumerate(samples):
            result = adapt_and_generate_for_sample(
                model=model,
                tokenizer=tokenizer,
                sample=sample,
                sample_index=sample_index,
                chunk_size=args.chunk_size,
                update_mode=args.update_mode,
                lr=args.lr,
                num_qa_candidates=args.num_qa_candidates,
                num_judge_candidates=args.num_judge_candidates,
                num_selected_qa=args.num_selected_qa,
                qa_generation_max_new_tokens=args.qa_generation_max_new_tokens,
                qa_judge_max_new_tokens=args.qa_judge_max_new_tokens,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                initial_lora_state=initial_lora_state,
                trainable_lora_params=trainable_lora_params,
                log_file=log_file,
            )
            if args.print_generated_only:
                print(result["generated_text"])
            else:
                print(json.dumps(result, ensure_ascii=True))
    finally:
        if log_file is not None:
            log_file.close()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
