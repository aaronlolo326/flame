#!/usr/bin/env python3
"""
Temporary LoRA test-time training during prefilling for a frozen causal LM.

This script implements a correctness-first online adaptation loop:
1. Attach LoRA to q_proj and v_proj only.
2. Reset the temporary adapter at the start of each sample.
3. Split the prompt into contiguous chunks.
4. For each chunk:
   - run forward on the current chunk with cache,
   - compute causal LM loss on that chunk only,
   - update only LoRA parameters,
   - invalidate the cache,
   - rebuild cache for all seen tokens under no_grad().
5. After the last chunk, keep the final adapter and rebuilt cache for generation.

Note on "zero initialization":
Literal all-zero initialization of both LoRA factors prevents gradients from
flowing through the LoRA branch. To preserve zero adapter output while still
allowing the first update step to learn, this script resets the LoRA weights to
their PEFT initialization state, which has zero initial adapter output.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
import random
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
STEPS_PER_CHUNK = 1
UPDATE_MODE = "full_prefix"
LOCAL_TRAIN_WINDOW = 2048
LORA_R = 64
LORA_ALPHA = 64
LORA_DROPOUT = 0.0
LR = 5e-5
BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.0
GRAD_CLIP = 1.0


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
    parser = argparse.ArgumentParser(description="Chunked TTT-LoRA prefill for a frozen causal LM.")
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen3-0.6B", help="Base model path or HF id.")
    parser.add_argument("--prompt", action="append", default=[], help="Prompt text. May be passed multiple times.")
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Optional file containing one prompt per line.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Number of generation tokens.")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Prefill chunk size.")
    parser.add_argument("--steps-per-chunk", type=int, default=STEPS_PER_CHUNK, help="Optimizer steps per chunk.")
    parser.add_argument(
        "--update-mode",
        type=str,
        default=UPDATE_MODE,
        choices=("full_prefix", "full_prefix_approx", "full_prefix_exact", "local_window"),
    )
    parser.add_argument("--local-train-window", type=int, default=LOCAL_TRAIN_WINDOW)
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


def load_prompts(args: argparse.Namespace) -> List[str]:
    prompts = list(args.prompt)
    if args.prompt_file:
        prompt_path = Path(args.prompt_file)
        prompts.extend([line.rstrip("\n") for line in prompt_path.read_text().splitlines() if line.strip()])
    if not prompts:
        raise ValueError("Provide at least one --prompt or a --prompt-file.")
    return prompts


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
):
    target_modules = resolve_top_layer_target_modules(
        model,
        module_suffixes=module_suffixes,
        top_layer_fraction=top_layer_fraction,
    )

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
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
    model = attach_ttt_lora(model)
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


def create_optimizer(params: Iterable[torch.nn.Parameter]) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        params,
        lr=LR,
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


@dataclass
class RebuildResult:
    past_key_values: object
    last_logits: torch.Tensor


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


def forward_chunk_with_cache(
    model,
    chunk_ids: torch.LongTensor,
    past_key_values,
    start_pos: int,
) -> Tuple[torch.Tensor, object]:
    cache_position = torch.arange(
        start_pos,
        start_pos + chunk_ids.shape[1],
        device=chunk_ids.device,
    )
    outputs = model(
        input_ids=chunk_ids,
        past_key_values=past_key_values,
        use_cache=True,
        cache_position=cache_position,
    )
    return outputs.logits, outputs.past_key_values


def compute_chunk_loss(
    logits: torch.Tensor,
    full_input_ids: torch.LongTensor,
    chunk_start: int,
    chunk_end: int,
) -> Optional[torch.Tensor]:
    valid_len = min(chunk_end, full_input_ids.shape[1] - 1) - chunk_start
    if valid_len <= 0:
        return None

    logits_for_loss = logits[:, :valid_len, :].contiguous()
    labels = full_input_ids[:, chunk_start + 1: chunk_start + 1 + valid_len].contiguous()
    return F.cross_entropy(
        logits_for_loss.view(-1, logits_for_loss.size(-1)),
        labels.view(-1),
    )


def compute_loss_on_segment(
    logits: torch.Tensor,
    segment_input_ids: torch.LongTensor,
    loss_start: int,
    loss_end: int,
) -> Optional[torch.Tensor]:
    valid_len = min(loss_end, segment_input_ids.shape[1] - 1) - loss_start
    if valid_len <= 0:
        return None

    logits_for_loss = logits[:, loss_start: loss_start + valid_len, :].contiguous()
    labels = segment_input_ids[:, loss_start + 1: loss_start + 1 + valid_len].contiguous()
    return F.cross_entropy(
        logits_for_loss.view(-1, logits_for_loss.size(-1)),
        labels.view(-1),
    )


def normalize_update_mode(update_mode: str) -> str:
    if update_mode == "full_prefix":
        return "full_prefix_exact"
    return update_mode


def clone_past_key_values(past_key_values):
    if past_key_values is None:
        return None
    return copy.deepcopy(past_key_values)


def forward_local_window_update(
    model,
    full_input_ids: torch.LongTensor,
    chunk_start: int,
    chunk_end: int,
    local_train_window: int,
) -> Optional[torch.Tensor]:
    window_start = max(0, chunk_start - local_train_window)
    window_ids = full_input_ids[:, window_start:chunk_end]
    cache_position = torch.arange(window_start, chunk_end, device=full_input_ids.device)
    outputs = model(
        input_ids=window_ids,
        use_cache=False,
        cache_position=cache_position,
    )
    loss_start = chunk_start - window_start
    loss_end = chunk_end - window_start
    return compute_loss_on_segment(outputs.logits, window_ids, loss_start=loss_start, loss_end=loss_end)


def forward_full_prefix_update(
    model,
    full_input_ids: torch.LongTensor,
    chunk_start: int,
    chunk_end: int,
    chunk_size: int,
) -> Optional[torch.Tensor]:
    prefix_cache = None
    if chunk_start > 0:
        prefix_cache = rebuild_cache_for_prefix(
            model,
            full_input_ids[:, :chunk_start],
            chunk_size=chunk_size,
        ).past_key_values

    logits, _ = forward_chunk_with_cache(
        model=model,
        chunk_ids=full_input_ids[:, chunk_start:chunk_end],
        past_key_values=prefix_cache,
        start_pos=chunk_start,
    )
    return compute_chunk_loss(
        logits=logits,
        full_input_ids=full_input_ids,
        chunk_start=chunk_start,
        chunk_end=chunk_end,
    )


def forward_full_prefix_approx_update(
    model,
    full_input_ids: torch.LongTensor,
    chunk_start: int,
    chunk_end: int,
    base_prefix_cache,
) -> Optional[torch.Tensor]:
    logits, _ = forward_chunk_with_cache(
        model=model,
        chunk_ids=full_input_ids[:, chunk_start:chunk_end],
        past_key_values=clone_past_key_values(base_prefix_cache),
        start_pos=chunk_start,
    )
    return compute_chunk_loss(
        logits=logits,
        full_input_ids=full_input_ids,
        chunk_start=chunk_start,
        chunk_end=chunk_end,
    )


def run_chunk_update_step(
    model,
    optimizer,
    trainable_lora_params: Sequence[Tuple[str, torch.nn.Parameter]],
    full_input_ids: torch.LongTensor,
    chunk_start: int,
    chunk_end: int,
    chunk_size: int,
    update_mode: str,
    local_train_window: int,
    base_prefix_cache=None,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    optimizer.zero_grad(set_to_none=True)
    resolved_update_mode = normalize_update_mode(update_mode)

    if resolved_update_mode == "local_window":
        loss = forward_local_window_update(
            model=model,
            full_input_ids=full_input_ids,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            local_train_window=local_train_window,
        )
    elif resolved_update_mode == "full_prefix_exact":
        loss = forward_full_prefix_update(
            model=model,
            full_input_ids=full_input_ids,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            chunk_size=chunk_size,
        )
    elif resolved_update_mode == "full_prefix_approx":
        loss = forward_full_prefix_approx_update(
            model=model,
            full_input_ids=full_input_ids,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            base_prefix_cache=base_prefix_cache,
        )
    else:
        raise ValueError(f"Unsupported update_mode: {update_mode}")

    if loss is not None:
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [param for _, param in trainable_lora_params],
            max_norm=GRAD_CLIP,
        )
        optimizer.step()
    else:
        grad_norm = torch.tensor(0.0, device=full_input_ids.device)

    return loss, grad_norm


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
    print(line)
    if log_file is not None:
        log_file.write(line + "\n")
        log_file.flush()


def adapt_and_generate_for_sample(
    model,
    tokenizer,
    prompt: str,
    prompt_index: int,
    chunk_size: int,
    steps_per_chunk: int,
    update_mode: str,
    local_train_window: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    initial_lora_state: Dict[str, torch.Tensor],
    trainable_lora_params: Sequence[Tuple[str, torch.nn.Parameter]],
    log_file,
) -> Dict[str, object]:
    update_mode = normalize_update_mode(update_mode)
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    seq_len = input_ids.shape[1]

    if seq_len == 0:
        raise ValueError("Empty prompt after tokenization.")

    reset_trainable_state(trainable_lora_params, initial_lora_state)
    optimizer = create_optimizer([param for _, param in trainable_lora_params])

    last_rebuild = None

    for chunk_idx, start in enumerate(range(0, seq_len, chunk_size)):
        end = min(start + chunk_size, seq_len)
        step_logs = []
        prefix_cache_for_chunk = None if start == 0 or last_rebuild is None else last_rebuild.past_key_values
        for step_idx in range(steps_per_chunk):
            loss, grad_norm = run_chunk_update_step(
                model=model,
                optimizer=optimizer,
                trainable_lora_params=trainable_lora_params,
                full_input_ids=input_ids,
                chunk_start=start,
                chunk_end=end,
                chunk_size=chunk_size,
                update_mode=update_mode,
                local_train_window=local_train_window,
                base_prefix_cache=prefix_cache_for_chunk,
            )
            step_logs.append(
                {
                "event": "chunk_update",
                "sample_idx": prompt_index,
                "chunk_idx": chunk_idx,
                "step_idx": step_idx,
                "steps_per_chunk": steps_per_chunk,
                "update_mode": normalize_update_mode(update_mode),
                "local_train_window": local_train_window if normalize_update_mode(update_mode) == "local_window" else None,
                "chunk_start": start,
                "chunk_end": end,
                "chunk_tokens": end - start,
                "loss": None if loss is None else float(loss.detach().cpu().item()),
                "grad_norm": float(grad_norm.detach().cpu().item()) if torch.is_tensor(grad_norm) else float(grad_norm),
                "lora_norm": compute_delta_norm(trainable_lora_params, initial_lora_state),
                }
            )

        seen_prefix = input_ids[:, :end]
        last_rebuild = rebuild_cache_for_prefix(model, seen_prefix, chunk_size=chunk_size)
        for chunk_log in step_logs:
            log_chunk(log_file, chunk_log)

    if last_rebuild is None:
        last_rebuild = rebuild_cache_for_prefix(model, input_ids, chunk_size=chunk_size)

    generated_ids = generate_from_adapted_state(
        model=model,
        prefix_ids=input_ids,
        past_key_values=last_rebuild.past_key_values,
        last_logits=last_rebuild.last_logits,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
    )

    full_ids = torch.cat([input_ids, generated_ids], dim=1)
    return {
        "sample_idx": prompt_index,
        "prompt": prompt,
        "prompt_token_count": seq_len,
        "generated_token_count": int(generated_ids.shape[1]),
        "generated_text": tokenizer.decode(generated_ids[0], skip_special_tokens=True),
        "full_text": tokenizer.decode(full_ids[0], skip_special_tokens=True),
        "final_lora_norm": compute_delta_norm(trainable_lora_params, initial_lora_state),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.update_mode = normalize_update_mode(args.update_mode)

    device = resolve_device(args.device)
    torch_dtype = resolve_dtype(args.dtype)

    model, tokenizer = build_model_and_tokenizer(
        model_name_or_path=args.model_name_or_path,
        device=device,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )

    trainable_lora_params = get_trainable_lora_parameters(model)
    if not trainable_lora_params:
        raise RuntimeError("No trainable LoRA parameters were found.")

    initial_lora_state = clone_trainable_state(trainable_lora_params)
    prompts = load_prompts(args)

    log_file = None
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w", encoding="utf-8")

    try:
        for prompt_index, prompt in enumerate(prompts):
            result = adapt_and_generate_for_sample(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                prompt_index=prompt_index,
                chunk_size=args.chunk_size,
                steps_per_chunk=args.steps_per_chunk,
                update_mode=args.update_mode,
                local_train_window=args.local_train_window,
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
