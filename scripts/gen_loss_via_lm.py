import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any
import random

import fla  # noqa: F401
import numpy as np
import torch
import torch.multiprocessing as mp
from datasets import Dataset, load_dataset, load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import custom_models  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample prefixes from a dataset, generate continuations with a model, "
            "and compute per-token losses with a reference LM."
        )
    )
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--ref_lm", type=str, default="Qwen/Qwen3-8B-Base")
    parser.add_argument("--prefill_len", type=int, default=8192)
    parser.add_argument("--decode_len", type=int, default=8192)
    parser.add_argument("--num_samples", type=int, default=512)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--do_sample", type=int, default=1)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--use_cache", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def _normalize_ref_lm_name(name: str) -> str:
    if "/" in name:
        return name
    return f"Qwen/{name}"


def _load_tokenizer(path: str, trust_remote_code: bool) -> Any:
    return AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code)


def load_any_dataset(data_path: str) -> Dataset:
    path = Path(data_path)
    if path.is_dir():
        try:
            return load_from_disk(str(path))
        except Exception:
            pass

    parquet_suffixes = {".parquet", ".pq"}
    if path.is_file() and path.suffix in parquet_suffixes:
        ds = load_dataset("parquet", data_files=str(path))
        return ds["train"]

    ds = load_dataset(data_path)
    if "train" in ds:
        return ds["train"]
    split_names = list(ds.keys())
    if not split_names:
        raise ValueError(f"Could not infer a split from dataset at {data_path}")
    return ds[split_names[0]]


def get_text_field(sample: dict[str, Any]) -> str | None:
    for key in ("text", "content"):
        value = sample.get(key)
        if isinstance(value, str):
            return value
    return None


def get_token_field(sample: dict[str, Any]) -> list[int] | None:
    for key in ("input_ids", "tokens"):
        value = sample.get(key)
        if value is None:
            continue
        return list(value)
    return None


def sample_prefixes(
    dataset: Dataset,
    tokenizer: Any,
    prefill_len: int,
    num_samples: int,
    seed: int,
) -> np.ndarray:
    if len(dataset) == 0:
        raise ValueError("Dataset is empty.")

    rng = np.random.default_rng(seed)
    prefixes: list[np.ndarray] = []
    attempts = 0
    max_attempts = max(num_samples * 50, 10_000)

    while len(prefixes) < num_samples and attempts < max_attempts:
        attempts += 1
        idx = int(rng.integers(0, len(dataset)))
        sample = dataset[idx]

        token_ids = get_token_field(sample)
        if token_ids is None:
            text = get_text_field(sample)
            if text is None:
                continue
            token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]

        if len(token_ids) < prefill_len:
            continue

        max_start = len(token_ids) - prefill_len
        start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        prefix = np.asarray(token_ids[start : start + prefill_len], dtype=np.int64)
        prefixes.append(prefix)

    if len(prefixes) < num_samples:
        raise RuntimeError(
            f"Could only sample {len(prefixes)} prefixes of length {prefill_len} "
            f"after {attempts} attempts."
        )

    return np.stack(prefixes, axis=0)


def split_indices(num_items: int, num_parts: int) -> list[list[int]]:
    parts: list[list[int]] = [[] for _ in range(num_parts)]
    for i in range(num_items):
        parts[i % num_parts].append(i)
    return parts


def get_dtype(device: str) -> torch.dtype:
    if device.startswith("cuda"):
        return torch.bfloat16
    return torch.float32


@torch.no_grad()
def generate_continuation(
    model: Any,
    prefix_ids: torch.Tensor,
    decode_len: int,
    pad_token_id: int | None,
    eos_token_id: int | None,
    do_sample: int | None,
    temperature: float | None,
    top_p: float | None,
    repetition_penalty: int | None,
    top_k: int | None = None,
    use_cache: bool = False,
    sample_seed: int | None = None,
) -> torch.Tensor:
    if sample_seed is not None:
        torch.manual_seed(sample_seed)
        if prefix_ids.is_cuda:
            torch.cuda.manual_seed(sample_seed)
            torch.cuda.manual_seed_all(sample_seed)

    outputs = model.generate(
        input_ids=prefix_ids.unsqueeze(0),
        attention_mask=torch.ones_like(prefix_ids, dtype=torch.long).unsqueeze(0),
        max_new_tokens=decode_len,
        do_sample=(do_sample==1),
        use_cache=use_cache,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )
    continuation = outputs[0, prefix_ids.shape[0] :]
    if continuation.shape[0] != decode_len:
        if continuation.shape[0] > decode_len:
            continuation = continuation[:decode_len]
        else:
            fill_value = eos_token_id if eos_token_id is not None else 0
            pad = torch.full(
                (decode_len - continuation.shape[0],),
                fill_value=fill_value,
                dtype=continuation.dtype,
                device=continuation.device,
            )
            continuation = torch.cat([continuation, pad], dim=0)
    return continuation

@torch.inference_mode()
def score_with_ref_lm_(
    ref_model: Any,
    prefix_ids: torch.Tensor,
    continuation_ids: torch.Tensor,
) -> np.ndarray:

    losses = torch.empty(
        prefix_ids.shape[0] + continuation_ids.shape[0],
        dtype=torch.float32,
        device=prefix_ids.device,
    )
    
    all_ids = torch.concat([prefix_ids, continuation_ids], dim=0)
    
    outputs = ref_model(
        input_ids=all_ids.unsqueeze(0),
        attention_mask=torch.ones_like(all_ids, dtype=torch.long).unsqueeze(0),
        use_cache=True,
    )
    logits = outputs.logits.float()
    losses = torch.nn.functional.cross_entropy(
        logits.transpose(1, 2)[:,:,:-1], all_ids[1:].unsqueeze(0), reduction="none"
    )
    return losses.squeeze().cpu().numpy()
    
### This method uses ref_lm to generate and use continuation as labels for ce loss
# @torch.inference_mode()
# def score_with_ref_lm(
#     ref_model: Any,
#     prefix_ids: torch.Tensor,
#     continuation_ids: torch.Tensor,
# ) -> np.ndarray:
#     losses = torch.empty(
#         continuation_ids.shape[0],
#         dtype=torch.float32,
#         device=prefix_ids.device,
#     )

#     outputs = ref_model(
#         input_ids=prefix_ids.unsqueeze(0),
#         attention_mask=torch.ones_like(prefix_ids, dtype=torch.long).unsqueeze(0),
#         use_cache=True,
#     )
#     logits = outputs.logits[:, -1, :].float()
#     losses[0] = torch.nn.functional.cross_entropy(
#         logits, continuation_ids[:1], reduction="none"
#     )[0]
#     past_key_values = outputs.past_key_values

#     if continuation_ids.shape[0] == 1:
#         return losses.cpu().numpy()

#     prev_token = continuation_ids[0:1].view(1, 1)
#     for step in range(1, continuation_ids.shape[0]):
#         outputs = ref_model(
#             input_ids=prev_token,
#             use_cache=True,
#             past_key_values=past_key_values,
#         )
#         logits = outputs.logits[:, -1, :].float()
#         losses[step] = torch.nn.functional.cross_entropy(
#             logits, continuation_ids[step : step + 1], reduction="none"
#         )[0]
#         past_key_values = outputs.past_key_values
#         prev_token = continuation_ids[step : step + 1].view(1, 1)

#     return losses.cpu().numpy()


def worker_main(
    worker_id: int,
    device: str,
    sample_indices: list[int],
    prefixes: np.ndarray,
    args: argparse.Namespace,
    tmpdir: str,
) -> None:
    if not sample_indices:
        np.save(
            os.path.join(tmpdir, f"worker_{worker_id}.npy"),
            np.empty((0, args.decode_len), dtype=np.float32),
        )
        np.save(os.path.join(tmpdir, f"worker_{worker_id}_idx.npy"), np.empty((0,), dtype=np.int64))
        return

    torch.set_grad_enabled(False)
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
    maybe_set_seed(args.seed)

    gen_tokenizer = _load_tokenizer(args.model_path, args.trust_remote_code)
    gen_tokenizer.eos_token_id = gen_tokenizer.pad_token_id # for qwen3 base
    # if gen_tokenizer.pad_token_id is None:
    #     if gen_tokenizer.eos_token_id is not None:
    #         gen_tokenizer.pad_token = gen_tokenizer.eos_token

    torch_dtype = get_dtype(device)
    common_model_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": torch_dtype,
    }

    model = AutoModelForCausalLM.from_pretrained(args.model_path, **common_model_kwargs)
    model.to(device)
    model.eval()

    ref_model = AutoModelForCausalLM.from_pretrained(args.ref_lm, **common_model_kwargs)
    ref_model.to(device)
    ref_model.eval()

    losses = np.empty((len(sample_indices),  args.prefill_len + args.decode_len - 1), dtype=np.float32)
    continuations = np.empty((len(sample_indices), args.decode_len), dtype=np.int64)
    progress = tqdm(sample_indices, position=worker_id, desc=f"gpu:{device}", leave=True)
    for local_row, sample_idx in enumerate(progress):
        prefix_ids = torch.as_tensor(prefixes[sample_idx], dtype=torch.long, device=device)
        continuation = generate_continuation(
            model=model,
            prefix_ids=prefix_ids,
            decode_len= args.decode_len,
            pad_token_id=gen_tokenizer.pad_token_id,
            eos_token_id=gen_tokenizer.eos_token_id,
            do_sample=args.do_sample,
            temperature= args.temperature,
            top_p= args.top_p,
            top_k= args.top_k,
            repetition_penalty= args.repetition_penalty,
            use_cache = False if args.use_cache == 0 else True,
            sample_seed=args.seed,
        )
        # losses[local_row] = score_with_ref_lm(
        #     ref_model=ref_model,
        #     prefix_ids=prefix_ids,
        #     continuation_ids=continuation,
        # )
        losses[local_row] = score_with_ref_lm_(
            ref_model=ref_model,
            prefix_ids=prefix_ids,
            continuation_ids=continuation,
        )
        continuations[local_row] = continuation.detach().cpu().numpy().astype(np.int64)

    np.save(os.path.join(tmpdir, f"worker_{worker_id}.npy"), losses)
    np.save(os.path.join(tmpdir, f"worker_{worker_id}_cont.npy"), continuations)
    np.save(
        os.path.join(tmpdir, f"worker_{worker_id}_idx.npy"),
        np.asarray(sample_indices, dtype=np.int64),
    )

def maybe_set_seed(seed: int) -> None:
    set_seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main() -> None:
    args = parse_args()
    
    maybe_set_seed(args.seed)
    args.ref_lm = _normalize_ref_lm_name(args.ref_lm)

    os.makedirs(args.output_dir, exist_ok=True)

    ref_tokenizer = _load_tokenizer(args.ref_lm, args.trust_remote_code)
    dataset = load_any_dataset(args.data_path)
    prefixes = sample_prefixes(
        dataset=dataset,
        tokenizer=ref_tokenizer,
        prefill_len=args.prefill_len,
        num_samples=args.num_samples,
        seed=args.seed,
    )

    if torch.cuda.is_available():
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    else:
        devices = ["cpu"]

    num_workers = min(len(devices), args.num_samples)
    devices = devices[:num_workers]
    shards = split_indices(args.num_samples, num_workers)

    with tempfile.TemporaryDirectory(prefix="gen_loss_via_lm_") as tmpdir:
        processes: list[mp.Process] = []
        for worker_id, device in enumerate(devices):
            proc = mp.get_context("spawn").Process(
                target=worker_main,
                args=(worker_id, device, shards[worker_id], prefixes, args, tmpdir),
            )
            proc.start()
            processes.append(proc)

        for proc in processes:
            proc.join()
            if proc.exitcode != 0:
                raise RuntimeError(f"Worker process failed with exit code {proc.exitcode}")

        all_losses = np.empty((args.num_samples, args.prefill_len + args.decode_len - 1), dtype=np.float32)
        all_continuations = np.empty((args.num_samples, args.decode_len), dtype=np.int64)
        for worker_id in range(num_workers):
            worker_losses = np.load(os.path.join(tmpdir, f"worker_{worker_id}.npy"))
            worker_continuations = np.load(os.path.join(tmpdir, f"worker_{worker_id}_cont.npy"))
            worker_indices = np.load(os.path.join(tmpdir, f"worker_{worker_id}_idx.npy"))
            if len(worker_indices) > 0:
                all_losses[worker_indices] = worker_losses
                all_continuations[worker_indices] = worker_continuations

    loss_data_path = f"{args.output_dir}/loss_qwen3.npy"
    np.save(loss_data_path, all_losses)
    print (f"np saved at {loss_data_path}")

    samples_path = f"{args.output_dir}/samples.json"
    samples: list[dict[str, Any]] = []
    for i in range(args.num_samples):
        prefix_ids = prefixes[i].tolist()
        continuation_ids = all_continuations[i].tolist()
        samples.append(
            {
                "sample_idx": i,
                "prefix_ids": prefix_ids,
                "continuation_ids": continuation_ids,
                "prefix_text": ref_tokenizer.decode(prefix_ids, skip_special_tokens=False),
                "continuation_text": ref_tokenizer.decode(
                    continuation_ids, skip_special_tokens=False
                ),
            }
        )
    with open(samples_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
