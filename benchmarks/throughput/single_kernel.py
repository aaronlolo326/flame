from __future__ import annotations

import argparse
from pathlib import Path

from .adapters import DEFAULT_LACT_CONFIG, KERNEL_MODEL_LABELS, build_kernel_subject, canonical_kernel_key
from .common import (
    append_row_jsonl,
    BenchmarkRow,
    DEFAULT_SEQ_LENS,
    benchmark_timer,
    log,
    now_timestamp,
    write_rows_csv,
)


KERNEL_MODEL_KEYS = [
    "lact_full_layer",
    "lact_ttt_branch_only",
    "fa_branch_only",
    "swa_branch_only",
    "gdn_branch_only",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-layer throughput benchmark scaffold.")
    parser.add_argument("--models", nargs="+", default=KERNEL_MODEL_KEYS, choices=KERNEL_MODEL_KEYS)
    parser.add_argument("--seq-lens", nargs="+", type=int, default=DEFAULT_SEQ_LENS)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--runtime-env",
        default="unknown",
        help="Logical environment label recorded in the result files, e.g. 'nm-dev' or 'fla'.",
    )
    parser.add_argument("--base-config", type=Path, default=DEFAULT_LACT_CONFIG)
    parser.add_argument("--sliding-window", type=int, default=None)
    parser.add_argument("--lact-chunk-size", type=int, default=None)
    parser.add_argument(
        "--lact-attn-heads",
        type=int,
        default=8,
        help="Override LaCT attention heads for lact_* single-kernel subjects.",
    )
    parser.add_argument(
        "--lact-ttt-heads",
        type=int,
        default=8,
        help="Override LaCT TTT heads (num_lact_heads) for lact_* single-kernel subjects.",
    )
    parser.add_argument(
        "--paper-lm-defaults",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply paper-aligned LM defaults: treat sliding-window size as at least the LaCT chunk size.",
    )
    parser.add_argument("--use-fused-lact-kernel", action="store_true")
    parser.add_argument(
        "--stop-after-oom",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If a model OOMs at one seq_len, skip larger seq_lens for that model in this sweep.",
    )
    parser.add_argument(
        "--stop-after-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If a model hits a persistent backend/runtime error at one seq_len, skip larger seq_lens for that model.",
    )
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
    device = args.device
    output_csv = args.output_prefix.with_suffix(".csv")
    output_jsonl = args.output_prefix.with_suffix(".jsonl")

    log(
        "Starting single-kernel throughput benchmark "
        f"device={device} dtype={args.dtype} runtime_env={args.runtime_env} "
        f"models={args.models} seq_lens={args.seq_lens}"
    )
    if args.use_fused_lact_kernel:
        log(
            "LaCT fused Triton kernel enabled. First warmup steps may include Triton compilation/autotune "
            "before steady-state timing."
        )

    try:
        for model_key in args.models:
            stop_model = False
            stop_reason = "previous_oom"
            for seq_len in args.seq_lens:
                if stop_model:
                    row = BenchmarkRow(
                        benchmark="single_kernel_train",
                        model=model_key,
                        runtime_env=args.runtime_env,
                        seq_len=seq_len,
                        batch_size=args.batch_size,
                        warmup_steps=args.warmup_steps,
                        measured_steps=args.steps,
                        forward_ms=0.0,
                        backward_ms=0.0,
                        optimizer_ms=0.0,
                        step_ms=0.0,
                        steps_per_second=0.0,
                        tokens_per_second=0.0,
                        peak_memory_gb=0.0,
                        status="skipped_after_failure",
                        notes=f"Skipped because a smaller seq_len for this model already failed: {stop_reason}.",
                    )
                    rows.append(row)
                    append_row_jsonl(output_jsonl, row)
                    write_rows_csv(output_csv, rows)
                    log(f"[skip] model={model_key} seq_len={seq_len} reason={stop_reason}")
                    continue
                log(
                    f"[build] model={model_key} seq_len={seq_len} local_batch={args.batch_size} "
                    f"chunk={args.lact_chunk_size or 'config'} window={args.sliding_window or 'config'}"
                )
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
                        lact_attn_heads_override=args.lact_attn_heads,
                        lact_ttt_heads_override=args.lact_ttt_heads,
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
                    log(
                        f"[warmup] model={model_key} seq_len={seq_len} "
                        f"warmup_steps={args.warmup_steps} timed_steps={args.steps}"
                    )

                    def sync() -> None:
                        if "cuda" in device:
                            torch.cuda.synchronize(device=device)

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
                    row = BenchmarkRow(
                        benchmark="single_kernel_train",
                        model=model_key,
                        runtime_env=args.runtime_env,
                        seq_len=seq_len,
                        batch_size=args.batch_size,
                        warmup_steps=args.warmup_steps,
                        measured_steps=args.steps,
                        forward_ms=forward_ms,
                        backward_ms=backward_ms,
                        optimizer_ms=optimizer_ms,
                        step_ms=step_ms,
                        steps_per_second=1000.0 / step_ms,
                        tokens_per_second=(args.batch_size * seq_len) / (step_ms / 1000.0),
                        peak_memory_gb=peak_gb,
                        status="ok",
                        notes=(
                            f"{KERNEL_MODEL_LABELS[canonical_kernel_key(model_key)]}; "
                            f"lact_chunk_size={args.lact_chunk_size or 'config'}; "
                            f"lact_attn_heads={args.lact_attn_heads or 'config'}; "
                            f"lact_ttt_heads={args.lact_ttt_heads or 'config'}; "
                            f"sliding_window={args.sliding_window or 'config'}; "
                            f"paper_lm_defaults={args.paper_lm_defaults}"
                        ),
                    )
                    rows.append(row)
                    append_row_jsonl(output_jsonl, row)
                    write_rows_csv(output_csv, rows)
                    log(
                        f"[done] model={model_key} seq_len={seq_len} status=ok "
                        f"step_ms={step_ms:.2f} tokens/s={row.tokens_per_second:.2f} "
                        f"peak_gb={peak_gb:.2f}"
                    )
                    del hidden_states
                    if "cuda" in device:
                        torch.cuda.empty_cache()
                except RuntimeError as exc:
                    status = "oom" if "out of memory" in str(exc).lower() else "error"
                    notes = str(exc)
                    if "PassManager::run failed" in notes:
                        status = "unsupported_backend"
                    row = BenchmarkRow(
                        benchmark="single_kernel_train",
                        model=model_key,
                        runtime_env=args.runtime_env,
                        seq_len=seq_len,
                        batch_size=args.batch_size,
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
                        notes=notes,
                    )
                    rows.append(row)
                    append_row_jsonl(output_jsonl, row)
                    write_rows_csv(output_csv, rows)
                    log(f"[done] model={model_key} seq_len={seq_len} status={status} error={exc}")
                    if status == "oom" and args.stop_after_oom:
                        stop_model = True
                        stop_reason = "previous_oom"
                    elif status in {"error", "unsupported_backend"} and args.stop_after_error:
                        stop_model = True
                        stop_reason = status
                    if "cuda" in device:
                        torch.cuda.empty_cache()
    finally:
        write_rows_csv(output_csv, rows)
        log(f"Finished single-kernel benchmark. Results written to {output_csv} and {output_jsonl}")


if __name__ == "__main__":
    main()
