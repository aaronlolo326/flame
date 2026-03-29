#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from datasets import Features, Sequence as HFSequence, Value, load_dataset
from transformers import AutoTokenizer


IGNORE_INDEX = -100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess open-r1/Mixture-of-Thoughts math/train into assistant-only SFT Arrow data."
    )
    parser.add_argument("--dataset", type=str, default="open-r1/Mixture-of-Thoughts")
    parser.add_argument("--subset", type=str, default="math")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="Tokenizer name/path. You can pass a local tokenizer path.",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_proc", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--no_append_final_eos",
        dest="append_final_eos",
        action="store_false",
        help="Disable appending tokenizer eos_token_id at sequence end if missing.",
    )
    parser.set_defaults(append_final_eos=True)
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Optional hard truncation length (keep left-most tokens).",
    )
    parser.add_argument(
        "--keep_only_supervised",
        action="store_true",
        help="Drop samples with zero supervised (assistant) tokens.",
    )
    return parser.parse_args()


def _normalize_messages(messages: Any) -> List[Dict[str, str]]:
    if isinstance(messages, str):
        messages = json.loads(messages)
    if not isinstance(messages, list):
        raise ValueError(f"`messages` must be a list, got: {type(messages)}")

    out: List[Dict[str, str]] = []
    for m in messages:
        if not isinstance(m, dict):
            raise ValueError(f"Each message must be dict, got: {type(m)}")
        role = str(m.get("role", "")).strip()
        content = m.get("content", "")
        if content is None:
            content = ""
        if role == "":
            raise ValueError(f"Message missing role: {m}")
        out.append({"role": role, "content": str(content)})
    return out


def _render_chatml_segment(role: str, content: str) -> str:
    return f"<|im_start|>{role}\n{content}<|im_end|>\n"


def _encode_with_assistant_mask(
    messages: Sequence[Dict[str, str]],
    tokenizer,
    append_final_eos: bool,
    max_length: Optional[int],
) -> Tuple[List[int], List[int], int]:
    input_ids: List[int] = []
    labels: List[int] = []

    for msg in messages:
        segment = _render_chatml_segment(msg["role"], msg["content"])
        seg_ids = tokenizer.encode(segment, add_special_tokens=False)
        input_ids.extend(seg_ids)
        if msg["role"] == "assistant":
            labels.extend(seg_ids)
        else:
            labels.extend([IGNORE_INDEX] * len(seg_ids))

    eos_id = tokenizer.eos_token_id
    if append_final_eos and eos_id is not None:
        if len(input_ids) == 0 or input_ids[-1] != eos_id:
            input_ids.append(eos_id)
            if len(messages) > 0 and messages[-1]["role"] == "assistant":
                labels.append(eos_id)
            else:
                labels.append(IGNORE_INDEX)

    if max_length is not None and max_length > 0 and len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

    num_supervised = sum(1 for x in labels if x != IGNORE_INDEX)
    return input_ids, labels, num_supervised


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        use_fast=True,
        trust_remote_code=True,
    )

    ds = load_dataset(args.dataset, args.subset, split=args.split)
    if "messages" not in ds.column_names:
        raise ValueError(f"`messages` column not found. Columns: {ds.column_names}")

    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    def _map_fn(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        all_input_ids: List[List[int]] = []
        all_labels: List[List[int]] = []
        all_num_tokens: List[int] = []
        all_num_supervised_tokens: List[int] = []

        for raw_messages in batch["messages"]:
            msgs = _normalize_messages(raw_messages)
            input_ids, labels, num_supervised = _encode_with_assistant_mask(
                msgs,
                tokenizer=tokenizer,
                append_final_eos=args.append_final_eos,
                max_length=args.max_length,
            )
            all_input_ids.append(input_ids)
            all_labels.append(labels)
            all_num_tokens.append(len(input_ids))
            all_num_supervised_tokens.append(num_supervised)

        return {
            "input_ids": all_input_ids,
            "labels": all_labels,
            "num_tokens": all_num_tokens,
            "num_supervised_tokens": all_num_supervised_tokens,
        }

    ds = ds.map(
        _map_fn,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        remove_columns=ds.column_names,
        desc="Tokenizing + building assistant-only labels",
    )

    if args.keep_only_supervised:
        ds = ds.filter(
            lambda x: x["num_supervised_tokens"] > 0,
            num_proc=args.num_proc,
            desc="Filtering samples with no supervised tokens",
        )

    target_features = Features(
        {
            "input_ids": HFSequence(Value("int32")),
            "labels": HFSequence(Value("int32")),
            "num_tokens": Value("int32"),
            "num_supervised_tokens": Value("int32"),
        }
    )
    ds = ds.cast(target_features)

    ds.save_to_disk(args.output_dir)

    print(f"Saved: {args.output_dir}")
    print(f"Rows: {len(ds)}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"eos_token_id: {tokenizer.eos_token_id}, eos_token: {tokenizer.eos_token}")
    print("Done.")


if __name__ == "__main__":
    main()
