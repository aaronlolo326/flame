from __future__ import annotations

import argparse
from pathlib import Path

from .adapters import DEFAULT_LACT_CONFIG, MODEL_SPECS, build_kernel_subject
from .common import (
    BenchmarkRow,
    DEFAULT_SEQ_LENS,
    barrier,
    benchmark_timer,
    destroy_distributed,
    get_local_rank,
    get_world_size,
    init_distributed,
    is_main_process,
    now_timestamp,
    reduce_scalar,
    write_rows_csv,
    write_rows_jsonl,
)


KERNEL_MODEL_KEYS = ["lact", "full_attention", "hybrid_swa", "hybrid_gdn"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-layer throughput benchmark scaffold.")
    parser.add_argument("--models", nargs="+", default=KERNEL_MODEL_KEYS, choices=KERNEL_MODEL_KEYS)
    parser.add_argument("--seq-lens", nargs="+", type=int, default=DEFAULT_SEQ_LENS)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_LACT_CONFIG)
    parser.add_argument("--sliding-window", type=int, default=None)
    parser.add_argument("--lact-chunk-size", type=int, default=None)
    parser.add_argument(
        "--paper-lm-defaults",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply paper-aligned LM defaults: treat sliding-window size as at least the LaCT chunk size.",
    )
    parser.add_argument("--use-fused-lact-kernel", action="store_true")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / f"single-kernel-{now_timestamp()}",
    )
    return parser.parse_args()


def main() -> None:
    import torch

    args = parse_args()
    rows: list[BenchmarkRow] = []
    _, local_rank, world_size = init_distributed(torch)
    device = f"cuda:{local_rank}" if args.device == "cuda" else args.device

    try:
        for model_key in args.models:
            for seq_len in args.seq_lens:
                try:
                    runner, hidden_size = build_kernel_subject(
                        model_key=model_key,
                        seq_len=seq_len,
                        device=device,
                        dtype_name=args.dtype,
                        batch_size=args.batch_size,
                        base_config_path=args.base_config,
                        sliding_window=args.sliding_window,
                        lact_chunk_size=args.lact_chunk_size,
                        use_fused_kernel=args.use_fused_lact_kernel,
                        paper_lm_defaults=args.paper_lm_defaults,
                    )
                    hidden_states = torch.randn(
                        args.batch_size,
                        seq_len,
                        hidden_size,
                        device=device,
                        dtype=getattr(torch, args.dtype),
                        requires_grad=True,
                    )

                    def sync() -> None:
                        if "cuda" in device:
                            torch.cuda.synchronize(device=device)
                        barrier(torch)

                    def step_fn() -> tuple[float, float, float]:
                        if hidden_states.grad is not None:
                            hidden_states.grad = None
                        if "cuda" in device:
                            torch.cuda.reset_peak_memory_stats(device=device)

                        start = torch.cuda.Event(enable_timing=True) if "cuda" in device else None
                        mid1 = torch.cuda.Event(enable_timing=True) if "cuda" in device else None
                        mid2 = torch.cuda.Event(enable_timing=True) if "cuda" in device else None

                        if start is not None:
                            start.record()
                        output = runner(hidden_states)
                        if mid1 is not None:
                            mid1.record()
                        loss = output.float().square().mean()
                        loss.backward()
                        if mid2 is not None:
                            mid2.record()
                            torch.cuda.synchronize(device=device)
                            return (
                                start.elapsed_time(mid1),
                                mid1.elapsed_time(mid2),
                                0.0,
                            )
                        raise RuntimeError("CPU timing is not implemented in this scaffold; run on CUDA.")

                    barrier(torch)
                    forward_ms, backward_ms, optimizer_ms, step_ms = benchmark_timer(
                        warmup_steps=args.warmup_steps,
                        measured_steps=args.steps,
                        sync_fn=sync,
                        step_fn=step_fn,
                    )
                    peak_gb = (
                        torch.cuda.max_memory_allocated(device=device) / (1024 ** 3)
                        if "cuda" in device
                        else 0.0
                    )
                    global_forward_ms = reduce_scalar(torch, forward_ms, "max")
                    global_backward_ms = reduce_scalar(torch, backward_ms, "max")
                    global_step_ms = reduce_scalar(torch, step_ms, "max")
                    global_peak_gb = reduce_scalar(torch, peak_gb, "max")
                    if is_main_process():
                        rows.append(
                            BenchmarkRow(
                                benchmark="single_kernel_train_multi_gpu" if world_size > 1 else "single_kernel_train",
                                model=model_key,
                                seq_len=seq_len,
                                batch_size=args.batch_size,
                                world_size=world_size,
                                warmup_steps=args.warmup_steps,
                                measured_steps=args.steps,
                                forward_ms=global_forward_ms,
                                backward_ms=global_backward_ms,
                                optimizer_ms=optimizer_ms,
                                step_ms=global_step_ms,
                                steps_per_second=1000.0 / global_step_ms,
                                tokens_per_second=(args.batch_size * seq_len * world_size) / (global_step_ms / 1000.0),
                                peak_memory_gb=global_peak_gb,
                                status="ok",
                                notes=(
                                    f"{MODEL_SPECS[model_key].label}; "
                                    f"embarrassingly_parallel={world_size > 1}; "
                                    f"lact_chunk_size={args.lact_chunk_size or 'config'}; "
                                    f"sliding_window={args.sliding_window or 'config'}; "
                                    f"paper_lm_defaults={args.paper_lm_defaults}"
                                ),
                            )
                        )
                    del hidden_states
                    if "cuda" in device:
                        torch.cuda.empty_cache()
                except RuntimeError as exc:
                    status = "oom" if "out of memory" in str(exc).lower() else "error"
                    if is_main_process():
                        rows.append(
                            BenchmarkRow(
                                benchmark="single_kernel_train_multi_gpu" if world_size > 1 else "single_kernel_train",
                                model=model_key,
                                seq_len=seq_len,
                                batch_size=args.batch_size,
                                world_size=world_size,
                                warmup_steps=args.warmup_steps,
                                measured_steps=args.steps,
                                forward_ms=0.0,
                                backward_ms=0.0,
                                optimizer_ms=0.0,
                                step_ms=0.0,
                                steps_per_second=0.0,
                                tokens_per_second=0.0,
                                peak_memory_gb=0.0,
                                status=status,
                                notes=str(exc),
                            )
                        )
                    if "cuda" in device:
                        torch.cuda.empty_cache()
                barrier(torch)
    finally:
        if is_main_process():
            write_rows_csv(args.output_prefix.with_suffix(".csv"), rows)
            write_rows_jsonl(args.output_prefix.with_suffix(".jsonl"), rows)
        destroy_distributed(torch)


if __name__ == "__main__":
    main()
