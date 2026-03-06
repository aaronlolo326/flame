from __future__ import annotations

import argparse
from pathlib import Path

from .adapters import DEFAULT_LACT_CONFIG, MODEL_SPECS, build_whole_model
from .common import (
    append_row_jsonl,
    BenchmarkRow,
    DEFAULT_SEQ_LENS,
    benchmark_timer,
    log,
    now_timestamp,
    write_rows_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Whole-model training throughput benchmark scaffold.")
    parser.add_argument("--models", nargs="+", default=list(MODEL_SPECS.keys()), choices=list(MODEL_SPECS.keys()))
    parser.add_argument("--seq-lens", nargs="+", type=int, default=DEFAULT_SEQ_LENS)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
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
        "--num-attn-heads",
        type=int,
        default=8,
        help=(
            "Override attention head count for all whole-model baselines to keep head geometry aligned. "
            "Default is 8 for fairer comparison with 8/8 LaCT split."
        ),
    )
    parser.add_argument(
        "--num-lact-heads",
        type=int,
        default=8,
        help="Override LaCT TTT head count (num_lact_heads) for the lact model.",
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
        "--output-prefix",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / f"whole-model-{now_timestamp()}",
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
        "Starting whole-model throughput benchmark "
        f"device={device} dtype={args.dtype} runtime_env={args.runtime_env} "
        f"num_attn_heads={args.num_attn_heads} num_lact_heads={args.num_lact_heads} "
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
            for seq_len in args.seq_lens:
                if stop_model:
                    row = BenchmarkRow(
                        benchmark="whole_model_train",
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
                        status="skipped_after_oom",
                        notes="Skipped because a smaller seq_len for this model already OOMed.",
                    )
                    rows.append(row)
                    append_row_jsonl(output_jsonl, row)
                    write_rows_csv(output_csv, rows)
                    log(f"[skip] model={model_key} seq_len={seq_len} reason=previous_oom")
                    continue
                log(
                    f"[build] model={model_key} seq_len={seq_len} local_batch={args.batch_size} "
                    f"chunk={args.lact_chunk_size or 'config'} window={args.sliding_window or 'config'}"
                )
                try:
                    model, base_cfg = build_whole_model(
                        model_key=model_key,
                        seq_len=seq_len,
                        device=device,
                        dtype_name=args.dtype,
                        base_config_path=args.base_config,
                        sliding_window=args.sliding_window,
                        lact_chunk_size=args.lact_chunk_size,
                        num_attn_heads_override=args.num_attn_heads,
                        num_lact_heads_override=args.num_lact_heads,
                        use_fused_kernel=args.use_fused_lact_kernel,
                        paper_lm_defaults=args.paper_lm_defaults,
                    )
                    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
                    vocab_size = int(base_cfg["vocab_size"])
                    input_ids = torch.randint(vocab_size, (args.batch_size, seq_len), device=device)
                    labels = input_ids.clone()
                    log(
                        f"[warmup] model={model_key} seq_len={seq_len} "
                        f"warmup_steps={args.warmup_steps} timed_steps={args.steps}"
                    )

                    def sync() -> None:
                        if "cuda" in device:
                            torch.cuda.synchronize(device=device)

                    def step_fn() -> tuple[float, float, float]:
                        optimizer.zero_grad(set_to_none=True)
                        if "cuda" in device:
                            torch.cuda.reset_peak_memory_stats(device=device)

                        start = torch.cuda.Event(enable_timing=True) if "cuda" in device else None
                        mid1 = torch.cuda.Event(enable_timing=True) if "cuda" in device else None
                        mid2 = torch.cuda.Event(enable_timing=True) if "cuda" in device else None
                        end = torch.cuda.Event(enable_timing=True) if "cuda" in device else None

                        if start is not None:
                            start.record()
                        outputs = model(input_ids=input_ids, labels=labels, use_cache=False)
                        if mid1 is not None:
                            mid1.record()
                        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                        loss.backward()
                        if mid2 is not None:
                            mid2.record()
                        optimizer.step()
                        if end is not None:
                            end.record()
                            torch.cuda.synchronize(device=device)
                            return (
                                start.elapsed_time(mid1),
                                mid1.elapsed_time(mid2),
                                mid2.elapsed_time(end),
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
                        benchmark="whole_model_train",
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
                            f"local_batch_size={args.batch_size}; "
                            f"num_attn_heads={args.num_attn_heads}; "
                            f"num_lact_heads={args.num_lact_heads}; "
                            f"lact_chunk_size={args.lact_chunk_size or 'config'}; "
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
                    del model
                    del optimizer
                    if "cuda" in device:
                        torch.cuda.empty_cache()
                except RuntimeError as exc:
                    status = "oom" if "out of memory" in str(exc).lower() else "error"
                    row = BenchmarkRow(
                        benchmark="whole_model_train",
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
                        notes=str(exc),
                    )
                    rows.append(row)
                    append_row_jsonl(output_jsonl, row)
                    write_rows_csv(output_csv, rows)
                    log(f"[done] model={model_key} seq_len={seq_len} status={status} error={exc}")
                    if status == "oom" and args.stop_after_oom:
                        stop_model = True
                    if "cuda" in device:
                        torch.cuda.empty_cache()
    finally:
        write_rows_csv(output_csv, rows)
        log(f"Finished whole-model benchmark. Results written to {output_csv} and {output_jsonl}")


if __name__ == "__main__":
    main()
