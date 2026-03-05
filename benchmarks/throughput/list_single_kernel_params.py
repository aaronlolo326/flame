from __future__ import annotations

import argparse
from pathlib import Path

from .adapters import DEFAULT_LACT_CONFIG, KERNEL_MODEL_LABELS, build_kernel_module, canonical_kernel_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List parameter components (names, shapes, counts) for single-kernel benchmark subjects."
    )
    parser.add_argument("--seq-len", type=int, default=32768)
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_LACT_CONFIG)
    parser.add_argument("--lact-chunk-size", type=int, default=None)
    parser.add_argument("--sliding-window", type=int, default=None)
    parser.add_argument("--lact-attn-heads", type=int, default=8)
    parser.add_argument("--lact-ttt-heads", type=int, default=8)
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
    return parser.parse_args()


def fmt_shape(shape) -> str:
    return "x".join(str(d) for d in shape)


def main() -> None:
    args = parse_args()
    print(
        f"# single-kernel parameter listing "
        f"(seq_len={args.seq_len}, dtype={args.dtype}, device={args.device}, "
        f"lact_attn_heads={args.lact_attn_heads}, lact_ttt_heads={args.lact_ttt_heads})"
    )
    for model_key in args.models:
        module, _ = build_kernel_module(
            model_key=model_key,
            seq_len=args.seq_len,
            device=args.device,
            dtype_name=args.dtype,
            batch_size=1,
            base_config_path=args.base_config,
            lact_chunk_size=args.lact_chunk_size,
            sliding_window=args.sliding_window,
            lact_attn_heads_override=args.lact_attn_heads,
            lact_ttt_heads_override=args.lact_ttt_heads,
            use_fused_kernel=args.use_fused_lact_kernel,
            paper_lm_defaults=args.paper_lm_defaults,
        )
        canonical = canonical_kernel_key(model_key)
        print(f"\n## {canonical}  # {KERNEL_MODEL_LABELS.get(canonical, canonical)}")
        total = 0
        trainable = 0
        for name, param in module.named_parameters():
            n = int(param.numel())
            total += n
            if param.requires_grad:
                trainable += n
            print(
                f"{name:<70} shape={fmt_shape(param.shape):<20} "
                f"dtype={str(param.dtype):<15} numel={n}"
            )
        print(f"-- total={total} trainable={trainable}")


if __name__ == "__main__":
    main()
