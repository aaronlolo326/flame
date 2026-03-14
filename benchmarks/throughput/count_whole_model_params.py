from __future__ import annotations

import argparse
from pathlib import Path

from .adapters import DEFAULT_LACT_CONFIG, MODEL_SPECS, build_whole_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count whole-model parameters for the throughput benchmark models."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_SPECS.keys()),
        choices=list(MODEL_SPECS.keys()),
        help="Whole-model benchmark subjects to instantiate and count.",
    )
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--base-config", type=Path, default=DEFAULT_LACT_CONFIG)
    parser.add_argument("--sliding-window", type=int, default=2048)
    parser.add_argument("--lact-chunk-size", type=int, default=2048)
    parser.add_argument("--num-attn-heads", type=int, default=8)
    parser.add_argument("--num-lact-heads", type=int, default=8)
    parser.add_argument(
        "--paper-lm-defaults",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--use-fused-lact-kernel", action="store_true")
    return parser.parse_args()


def count_params(module) -> tuple[int, int]:
    total = 0
    trainable = 0
    for param in module.parameters():
        numel = int(param.numel())
        total += numel
        if param.requires_grad:
            trainable += numel
    return total, trainable


def main() -> None:
    args = parse_args()
    label_width = max(len(model_key) for model_key in args.models)

    for model_key in args.models:
        model, _ = build_whole_model(
            model_key=model_key,
            seq_len=args.seq_len,
            device=args.device,
            dtype_name=args.dtype,
            base_config_path=args.base_config,
            sliding_window=args.sliding_window,
            lact_chunk_size=args.lact_chunk_size,
            num_attn_heads_override=args.num_attn_heads,
            num_lact_heads_override=args.num_lact_heads,
            use_fused_kernel=args.use_fused_lact_kernel,
            paper_lm_defaults=args.paper_lm_defaults,
        )
        total, trainable = count_params(model)
        print(
            f"{model_key.ljust(label_width)}  "
            f"total={total:>12,}  trainable={trainable:>12,}"
        )


if __name__ == "__main__":
    main()
