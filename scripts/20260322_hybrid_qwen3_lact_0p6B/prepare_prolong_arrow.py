# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import concatenate_datasets, load_dataset


def load_parquet_repo(repo_dir: Path):
    data_dir = repo_dir / "data"
    parquet_files = sorted(str(p) for p in data_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {data_dir}")
    return load_dataset("parquet", data_files={"train": parquet_files}, split="train")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert downloaded prolong parquet repos to datasets.save_to_disk Arrow dirs, optionally merged.")
    parser.add_argument(
        "--src1",
        type=Path,
        default=Path("/storage/backup/yufei/data/prolong1-qwen3-8b-tokenized-32768"),
    )
    parser.add_argument(
        "--src2",
        type=Path,
        default=Path("/storage/backup/yufei/data/prolong2-qwen3-8b-tokenized-32768"),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("/storage/backup/yufei/ttt/data"),
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Also concatenate the two datasets and save a merged Arrow dataset.",
    )
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)

    ds1 = load_parquet_repo(args.src1)
    out1 = args.out_root / "prolong1-qwen3-8b-tokenized-32768-arrow"
    ds1.save_to_disk(str(out1))
    print(f"saved={out1} rows={len(ds1)}")

    ds2 = load_parquet_repo(args.src2)
    out2 = args.out_root / "prolong2-qwen3-8b-tokenized-32768-arrow"
    ds2.save_to_disk(str(out2))
    print(f"saved={out2} rows={len(ds2)}")

    if args.merge:
        merged = concatenate_datasets([ds1, ds2])
        outm = args.out_root / "prolong-merged-qwen3-8b-tokenized-32768-arrow"
        merged.save_to_disk(str(outm))
        print(f"saved={outm} rows={len(merged)}")


if __name__ == "__main__":
    main()
