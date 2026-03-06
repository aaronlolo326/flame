from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapters import (
    DEFAULT_LACT_CONFIG,
    QWEN35_2B_BASE_GDN,
    build_layer_types,
    ceil_ratio_count,
    lact_config_to_qwen3_dict,
    load_json,
    resolve_chunk_and_window,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export resolved JSON configs for whole-model throughput benchmarks.")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_LACT_CONFIG)
    parser.add_argument("--seq-len", type=int, default=32768)
    parser.add_argument("--sliding-window", type=int, default=None)
    parser.add_argument("--lact-chunk-size", type=int, default=None)
    parser.add_argument("--num-attn-heads", type=int, default=8)
    parser.add_argument("--num-lact-heads", type=int, default=8)
    parser.add_argument("--use-fused-lact-kernel", action="store_true")
    parser.add_argument(
        "--paper-lm-defaults",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "configs",
    )
    return parser.parse_args()


def dump_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=False)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    base_cfg = load_json(args.base_config)

    cfg = dict(base_cfg)
    cfg["num_attn_heads"] = int(args.num_attn_heads)
    cfg["num_heads"] = int(args.num_attn_heads)
    cfg["num_key_value_heads"] = min(int(cfg.get("num_key_value_heads", args.num_attn_heads)), int(args.num_attn_heads))
    cfg["num_lact_heads"] = int(args.num_lact_heads)

    hidden_size = int(cfg["hidden_size"])
    if hidden_size % int(cfg["num_attn_heads"]) != 0:
        raise ValueError("hidden_size must be divisible by num_attn_heads")
    if hidden_size % int(cfg["num_lact_heads"]) != 0:
        raise ValueError("hidden_size must be divisible by num_lact_heads")

    chunk_size, window_size = resolve_chunk_and_window(
        seq_len=args.seq_len,
        base_cfg=cfg,
        lact_chunk_size=args.lact_chunk_size,
        window_size=args.sliding_window,
        paper_lm_defaults=args.paper_lm_defaults,
    )

    num_layers = int(cfg["num_hidden_layers"])
    num_lact_layers = ceil_ratio_count(num_layers, 0.75)
    num_fa_layers = max(0, num_layers - num_lact_layers)

    lact_cfg = dict(cfg)
    lact_cfg["max_position_embeddings"] = max(args.seq_len, int(lact_cfg.get("max_position_embeddings", args.seq_len)))
    lact_cfg["use_cache"] = False
    lact_cfg["lact_chunk_size"] = chunk_size
    lact_cfg["window_size"] = window_size
    lact_cfg["use_fused_kernel"] = bool(args.use_fused_lact_kernel)

    full_attention_cfg = lact_config_to_qwen3_dict(
        cfg,
        seq_len=args.seq_len,
        layer_types=["full_attention"] * num_layers,
        sliding_window=None,
        attn_implementation="flash_attention_2",
    )

    hybrid_swa_cfg = lact_config_to_qwen3_dict(
        cfg,
        seq_len=args.seq_len,
        layer_types=build_layer_types(num_layers, "sliding_attention"),
        sliding_window=window_size,
        attn_implementation="flash_attention_2",
    )

    hybrid_gdn_cfg = lact_config_to_qwen3_dict(
        cfg,
        seq_len=args.seq_len,
        layer_types=build_layer_types(num_layers, "linear_attention"),
        sliding_window=None,
        attn_implementation="flash_attention_2",
        gdn_overrides={
            "expand_v": 1,
            "mode": "chunk",
            "use_gate": True,
            "use_short_conv": True,
            "conv_size": 4,
            "conv_bias": False,
            "pad_value": 0,
            "selection_window_size": 100,
            "use_qk_norm": True,
        },
    )

    hybrid_lact_cfg = {
        "model_type": "hybrid_lact_benchmark",
        "label": "75% LaCT + 25% FA",
        "num_hidden_layers": num_layers,
        "num_full_attention_layers": num_fa_layers,
        "num_lact_layers": num_lact_layers,
        "split": {
            "full_attention_ratio": 0.25,
            "lact_ratio": 0.75,
        },
        "full_attention_subconfig": lact_config_to_qwen3_dict(
            cfg,
            seq_len=args.seq_len,
            layer_types=["full_attention"] * num_layers,
            sliding_window=None,
            attn_implementation="flash_attention_2",
        ),
        "lact_subconfig": lact_cfg,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dump_json(args.output_dir / "lact.json", lact_cfg)
    dump_json(args.output_dir / "hybrid_lact.json", hybrid_lact_cfg)
    dump_json(args.output_dir / "full_attention.json", full_attention_cfg)
    dump_json(args.output_dir / "hybrid_swa.json", hybrid_swa_cfg)
    dump_json(args.output_dir / "hybrid_gdn.json", hybrid_gdn_cfg)

    summary = {
        "base_config_path": str(args.base_config),
        "seq_len": args.seq_len,
        "runtime_overrides": {
            "num_attn_heads": args.num_attn_heads,
            "num_lact_heads": args.num_lact_heads,
            "lact_chunk_size": args.lact_chunk_size,
            "sliding_window": args.sliding_window,
            "paper_lm_defaults": args.paper_lm_defaults,
            "use_fused_lact_kernel": args.use_fused_lact_kernel,
        },
        "gdn_reference": QWEN35_2B_BASE_GDN,
        "written_files": [
            "lact.json",
            "hybrid_lact.json",
            "full_attention.json",
            "hybrid_swa.json",
            "hybrid_gdn.json",
        ],
    }
    dump_json(args.output_dir / "_summary.json", summary)
    print(f"Wrote config snapshots to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
