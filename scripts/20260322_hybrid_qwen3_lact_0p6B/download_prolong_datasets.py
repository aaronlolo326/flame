# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the prolong tokenized datasets from Hugging Face to local disk.")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("/storage/backup/yufei/data"),
        help="Root directory to place the downloaded dataset repos.",
    )
    args = parser.parse_args()

    repos = [
        "xfxcwynlc/prolong1-qwen3-8b-tokenized-32768",
        "xfxcwynlc/prolong2-qwen3-8b-tokenized-32768",
    ]

    args.out_root.mkdir(parents=True, exist_ok=True)
    for repo_id in repos:
        local_dir = args.out_root / repo_id.split("/")[-1]
        print(f"downloading {repo_id} -> {local_dir}")
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )


if __name__ == "__main__":
    main()
