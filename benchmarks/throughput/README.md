# LaCT Throughput Benchmark Scaffold

This folder is a draft benchmark framework for comparing LaCT against three baselines:

- `lact`: LaCT layer + sliding-window flash attention from `flame/custom_models/lact_model`
- `full_attention`: full-attention Qwen3 with `flash_attention_2`
- `hybrid_swa`: 75% sliding-window attention + 25% full attention
- `hybrid_gdn`: 75% GatedDeltaNet + 25% full attention

The scaffold is aimed at two benchmark modes:

- Whole-network training throughput across sequence lengths: `4k, 8k, 16k, 32k, 64k, 128k, 256k, 512k, 1M`
- Single-layer throughput across the same sequence lengths for LaCT, full attention, SWA, and GDN

## Layout

- `whole_model.py`: synthetic end-to-end train-step throughput
- `single_kernel.py`: isolated layer forward/backward throughput
- `adapters.py`: model/layer builders and config translation
- `common.py`: timing and result export helpers

## Assumptions

- The baselines are normalized to the LaCT config in `configs/qwen3_lact_1B4.json` so hidden size, layer count, MLP size, and vocab size stay aligned unless you override them.
- `hybrid_swa` uses the local implementation in `/work/yufei/projects/hybrid_models/qwen3_swa`.
- `hybrid_gdn` uses the local implementation in `/work/yufei/projects/hybrid_models/qwen3_gdn`.
- The benchmark is intended for a CUDA machine with `torch`, `transformers`, `flash-attn`, and the FLA dependencies installed.

## Usage

From `/work/yufei/projects/flame`:

```bash
python -m benchmarks.throughput.whole_model \
  --models lact full_attention hybrid_swa hybrid_gdn \
  --seq-lens 4096 8192 16384 32768 65536 131072 262144 524288 1048576 \
  --batch-size 1 \
  --steps 10 \
  --warmup-steps 3 \
  --dtype bfloat16 \
  --device cuda \
  --use-fused-lact-kernel
```

```bash
python -m benchmarks.throughput.single_kernel \
  --models lact full_attention hybrid_swa hybrid_gdn \
  --seq-lens 4096 8192 16384 32768 65536 131072 262144 524288 1048576 \
  --batch-size 1 \
  --steps 20 \
  --warmup-steps 5 \
  --dtype bfloat16 \
  --device cuda \
  --use-fused-lact-kernel
```

Outputs are written as both `.csv` and `.jsonl` under `benchmarks/throughput/results/`.

## What The Metrics Mean

- `forward_ms`: average forward-pass time
- `backward_ms`: average backward-pass time
- `optimizer_ms`: optimizer step time, only for whole-model runs
- `step_ms`: wall-clock average for one measured step
- `tokens_per_second`: `batch_size * seq_len / step_time`
- `peak_memory_gb`: `torch.cuda.max_memory_allocated()`
- `status`: `ok`, `oom`, or `error`

## Additional Comparisons Worth Running

These are likely useful for improving the LaCT work beyond the two core throughput plots:

1. Break down LaCT forward time into attention-path vs TTT-path. This will show whether the bottleneck is still in flash attention, the fast-weight update, or the fused Triton kernels.
2. Compare fused vs unfused LaCT kernels with the same config. This isolates whether performance gains come from the algorithm or just from kernel engineering.
3. Plot peak activation memory vs sequence length for all four models. Throughput without memory context is hard to interpret at `256k+`.
4. Sweep `window_size` and `lact_chunk_size` together. The crossover point where LaCT beats the baselines may be sensitive to chunking, not just model class.
5. Measure forward-only and backward-only tokens/s separately. Some recurrent alternatives look good in forward but lose their advantage in backward.
6. Record the OOM boundary for each model under a fixed GPU budget. This is often more actionable than raw throughput at `1M`.

## Limitations In This Draft

- It uses synthetic token inputs, not dataloader throughput.
- CPU timing is not implemented; this scaffold is meant for CUDA benchmarking.
- The layer benchmark is really a layer-level train-step benchmark, not a raw Triton microkernel profiler.
- The LaCT paper PDF at `/work/yufei/projects/paper_list/tttdr.pdf` was not parsed in this environment, so the implementation choices here are grounded in the local code rather than paper text extraction.
