import argparse
import os
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset, load_dataset, load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer


arrow_dir = "/storage/backup/mingze/data/prolong1_arrow/train/"
tokenizer = "Qwen/Qwen3-4B-Base"

# See samples of data
# Plot data statistics of sequence length distribution
# Check presence of eos_token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect tokenized Arrow/parquet data.")
    parser.add_argument("--data_dir", type=str, default=arrow_dir)
    parser.add_argument("--tokenizer", type=str, default=tokenizer)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--sample_tokens", type=int, default=256)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--plot_path", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=max(1, (os.cpu_count() or 1) - 8))
    parser.add_argument("--task_rows", type=int, default=50_000)
    return parser.parse_args()


def load_any_dataset(data_path: str, split: str | None) -> Dataset:
    path = Path(data_path)

    if path.is_dir():
        try:
            loaded = load_from_disk(str(path))
            if isinstance(loaded, Dataset):
                return loaded
            if split and split in loaded:
                return loaded[split]
            first_split = next(iter(loaded.keys()))
            return loaded[first_split]
        except Exception:
            pass

        train_dir = path / "train"
        if train_dir.is_dir():
            try:
                loaded = load_from_disk(str(train_dir))
                if isinstance(loaded, Dataset):
                    return loaded
                if split and split in loaded:
                    return loaded[split]
                first_split = next(iter(loaded.keys()))
                return loaded[first_split]
            except Exception:
                pass

        dataset_dict = load_dataset("parquet", data_dir=str(path))
        if split and split in dataset_dict:
            return dataset_dict[split]
        if "train" in dataset_dict:
            return dataset_dict["train"]
        first_split = next(iter(dataset_dict.keys()))
        return dataset_dict[first_split]

    if path.is_file() and path.suffix in {".parquet", ".pq"}:
        dataset_dict = load_dataset("parquet", data_files=str(path))
        return dataset_dict["train"]

    loaded = load_dataset(data_path)
    if isinstance(loaded, Dataset):
        return loaded
    if split and split in loaded:
        return loaded[split]
    if "train" in loaded:
        return loaded["train"]
    first_split = next(iter(loaded.keys()))
    return loaded[first_split]


def detect_token_column(dataset: Dataset) -> str:
    sample = dataset[0]
    for key in ("input_ids", "tokens"):
        if key in sample:
            return key
    raise ValueError(f"Could not find token column in dataset fields: {list(sample.keys())}")


def decode_preview(tokenizer_obj: Any, token_ids: list[int], max_tokens: int) -> str:
    preview_ids = token_ids[:max_tokens]
    return tokenizer_obj.decode(preview_ids, skip_special_tokens=False)


def maybe_plot_hist(lengths: np.ndarray, plot_path: str) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.hist(lengths, bins=min(80, max(10, int(np.sqrt(len(lengths))))))
    plt.xlabel("Sequence length")
    plt.ylabel("Count")
    plt.title("Sequence length distribution")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()


WORKER_DATASET: Dataset | None = None
WORKER_TOKEN_COL: str | None = None
WORKER_EOS_TOKEN_ID: int | None = None


def init_worker(dataset: Dataset, token_col: str, eos_token_id: int | None) -> None:
    global WORKER_DATASET, WORKER_TOKEN_COL, WORKER_EOS_TOKEN_ID
    WORKER_DATASET = dataset
    WORKER_TOKEN_COL = token_col
    WORKER_EOS_TOKEN_ID = eos_token_id


def scan_range(task: tuple[int, int]) -> tuple[np.ndarray, int, int, int]:
    start, end = task
    assert WORKER_DATASET is not None
    assert WORKER_TOKEN_COL is not None

    shard = WORKER_DATASET.select(range(start, end))
    lengths = np.empty(end - start, dtype=np.int64)
    eos_present = 0
    eos_at_end = 0
    eos_total = 0

    for local_idx, row in enumerate(shard):
        token_ids = list(row[WORKER_TOKEN_COL])
        lengths[local_idx] = len(token_ids)
        if WORKER_EOS_TOKEN_ID is None:
            continue
        eos_count = token_ids.count(WORKER_EOS_TOKEN_ID)
        eos_total += eos_count
        if eos_count > 0:
            eos_present += 1
        if token_ids and token_ids[-1] == WORKER_EOS_TOKEN_ID:
            eos_at_end += 1

    return lengths, eos_present, eos_at_end, eos_total


def build_tasks(rows_to_scan: int, task_rows: int) -> list[tuple[int, int]]:
    return [
        (start, min(start + task_rows, rows_to_scan))
        for start in range(0, rows_to_scan, task_rows)
    ]


def scan_dataset(
    dataset: Dataset,
    token_col: str,
    eos_token_id: int | None,
    rows_to_scan: int,
    num_workers: int,
    task_rows: int,
) -> tuple[np.ndarray, int, int, int]:
    if rows_to_scan == 0:
        return np.empty(0, dtype=np.int64), 0, 0, 0

    tasks = build_tasks(rows_to_scan, task_rows)
    num_workers = max(1, min(num_workers, len(tasks)))

    if num_workers == 1:
        init_worker(dataset, token_col, eos_token_id)
        results = [scan_range(task) for task in tqdm(tasks, desc="Scanning", unit="task")]
    else:
        with Pool(
            processes=num_workers,
            initializer=init_worker,
            initargs=(dataset, token_col, eos_token_id),
        ) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(scan_range, tasks),
                    total=len(tasks),
                    desc="Scanning",
                    unit="task",
                )
            )

    lengths = np.concatenate([result[0] for result in results], axis=0)
    eos_present = sum(result[1] for result in results)
    eos_at_end = sum(result[2] for result in results)
    eos_total = sum(result[3] for result in results)
    return lengths, eos_present, eos_at_end, eos_total


def main() -> None:
    args = parse_args()

    dataset = load_any_dataset(args.data_dir, args.split)
    if len(dataset) == 0:
        raise ValueError("Dataset is empty.")

    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    eos_token_id = tok.eos_token_id

    token_col = detect_token_column(dataset)
    rows_to_scan = len(dataset) if args.max_rows is None else min(len(dataset), args.max_rows)

    print(f"Loaded dataset from: {args.data_dir}")
    print(f"Rows in selected split: {len(dataset):,}")
    print(f"Scanning rows: {rows_to_scan:,}")
    print(f"Token column: {token_col}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"EOS token: {tok.eos_token!r}")
    print(f"EOS token id: {eos_token_id}")
    print(f"Workers: {args.num_workers}")
    print(f"Task rows: {args.task_rows:,}")
    print("")

    print("Sample decodes:")
    for idx in range(min(args.num_samples, len(dataset))):
        token_ids = list(dataset[idx][token_col])
        print(f"[sample {idx}] length={len(token_ids)}")
        print(decode_preview(tok, token_ids, args.sample_tokens))
        print("-" * 80)

    lengths, eos_present, eos_at_end, eos_total = scan_dataset(
        dataset=dataset,
        token_col=token_col,
        eos_token_id=eos_token_id,
        rows_to_scan=rows_to_scan,
        num_workers=args.num_workers,
        task_rows=args.task_rows,
    )

    print("")
    print("Length statistics:")
    print(f"min={int(lengths.min())}")
    print(f"p50={float(np.percentile(lengths, 50)):.1f}")
    print(f"p90={float(np.percentile(lengths, 90)):.1f}")
    print(f"p95={float(np.percentile(lengths, 95)):.1f}")
    print(f"p99={float(np.percentile(lengths, 99)):.1f}")
    print(f"max={int(lengths.max())}")
    print(f"mean={float(lengths.mean()):.2f}")

    print("")
    print("EOS statistics:")
    if eos_token_id is None:
        print("Tokenizer has no eos_token_id; skipping EOS checks.")
    else:
        print(f"rows with EOS anywhere: {eos_present:,}/{rows_to_scan:,} ({eos_present / rows_to_scan:.2%})")
        print(f"rows ending with EOS: {eos_at_end:,}/{rows_to_scan:,} ({eos_at_end / rows_to_scan:.2%})")
        print(f"total EOS count: {eos_total:,}")
        print(f"avg EOS per row: {eos_total / rows_to_scan:.4f}")

    if args.plot_path:
        maybe_plot_hist(lengths, args.plot_path)
        print("")
        print(f"Saved histogram to: {args.plot_path}")


if __name__ == "__main__":
    main()
