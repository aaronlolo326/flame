from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .adapters import (
    DEFAULT_LACT_CONFIG,
    KERNEL_MODEL_LABELS,
    build_kernel_module,
    canonical_kernel_key,
)


def count_params(module: Any) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def print_module_breakdown(module: Any, prefix: str = "") -> None:
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        total, trainable = count_params(child)
        print(f"  {full_name:<32} total={total:>12,d} trainable={trainable:>12,d}")
        print_module_breakdown(child, full_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count parameters for single-layer benchmark subjects.")
    parser.add_argument("--seq-len", type=int, default=32768)
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_LACT_CONFIG)
    parser.add_argument("--lact-chunk-size", type=int, default=None)
    parser.add_argument("--sliding-window", type=int, default=None)
    parser.add_argument("--use-fused-lact-kernel", action="store_true")
    parser.add_argument(
        "--paper-lm-defaults",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "lact_full_layer",
            "lact_ttt_branch_only",
            "fa_branch_only",
            "swa_branch_only",
            "gdn_branch_only",
        ],
    )
    parser.add_argument("--breakdown", action="store_true", help="Print per-submodule parameter counts too.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(
        f"Single-layer parameter counts for seq_len={args.seq_len}, dtype={args.dtype}, "
        f"chunk={args.lact_chunk_size or 'config'}, window={args.sliding_window or 'config'}"
    )
    for name in args.models:
        module, _ = build_kernel_module(
            model_key=name,
            seq_len=args.seq_len,
            device=args.device,
            dtype_name=args.dtype,
            batch_size=1,
            base_config_path=args.base_config,
            lact_chunk_size=args.lact_chunk_size,
            sliding_window=args.sliding_window,
            use_fused_kernel=args.use_fused_lact_kernel,
            paper_lm_defaults=args.paper_lm_defaults,
        )
        total, trainable = count_params(module)
        canonical = canonical_kernel_key(name)
        print(
            f"{canonical:<22} total={total:>12,d} trainable={trainable:>12,d} "
            f"# {KERNEL_MODEL_LABELS[canonical]}"
        )
        if args.breakdown:
            print_module_breakdown(module)


if __name__ == "__main__":
    main()
