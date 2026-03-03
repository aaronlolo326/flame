from __future__ import annotations

import argparse
from pathlib import Path

from .adapters import DEFAULT_LACT_CONFIG, MODEL_SPECS, build_whole_model
from .common import BenchmarkRow, DEFAULT_SEQ_LENS, benchmark_timer, now_timestamp, write_rows_csv, write_rows_jsonl


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
    parser.add_argument("--base-config", type=Path, default=DEFAULT_LACT_CONFIG)
    parser.add_argument("--sliding-window", type=int, default=None)
    parser.add_argument("--use-fused-lact-kernel", action="store_true")
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

    for model_key in args.models:
        for seq_len in args.seq_lens:
            try:
                model, base_cfg = build_whole_model(
                    model_key=model_key,
                    seq_len=seq_len,
                    device=args.device,
                    dtype_name=args.dtype,
                    base_config_path=args.base_config,
                    sliding_window=args.sliding_window,
                    use_fused_kernel=args.use_fused_lact_kernel,
                )
                optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
                vocab_size = int(base_cfg["vocab_size"])
                input_ids = torch.randint(vocab_size, (args.batch_size, seq_len), device=args.device)
                labels = input_ids.clone()

                def sync() -> None:
                    if "cuda" in args.device:
                        torch.cuda.synchronize(device=args.device)

                def step_fn() -> tuple[float, float, float]:
                    optimizer.zero_grad(set_to_none=True)
                    if "cuda" in args.device:
                        torch.cuda.reset_peak_memory_stats(device=args.device)

                    start = torch.cuda.Event(enable_timing=True) if "cuda" in args.device else None
                    mid1 = torch.cuda.Event(enable_timing=True) if "cuda" in args.device else None
                    mid2 = torch.cuda.Event(enable_timing=True) if "cuda" in args.device else None
                    end = torch.cuda.Event(enable_timing=True) if "cuda" in args.device else None

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
                        torch.cuda.synchronize(device=args.device)
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
                    torch.cuda.max_memory_allocated(device=args.device) / (1024 ** 3)
                    if "cuda" in args.device
                    else 0.0
                )
                rows.append(
                    BenchmarkRow(
                        benchmark="whole_model_train",
                        model=model_key,
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
                        notes="",
                    )
                )
                del model
                del optimizer
                if "cuda" in args.device:
                    torch.cuda.empty_cache()
            except RuntimeError as exc:
                status = "oom" if "out of memory" in str(exc).lower() else "error"
                rows.append(
                    BenchmarkRow(
                        benchmark="whole_model_train",
                        model=model_key,
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
                )
                if "cuda" in args.device:
                    torch.cuda.empty_cache()

    write_rows_csv(args.output_prefix.with_suffix(".csv"), rows)
    write_rows_jsonl(args.output_prefix.with_suffix(".jsonl"), rows)


if __name__ == "__main__":
    main()

