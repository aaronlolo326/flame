# LaCT Throughput Benchmark Scaffold

This folder is a draft benchmark framework for comparing LaCT against three baselines on either one GPU or a single node with 8 GPUs such as `8 x H100`.

- `lact`: LaCT layer + sliding-window flash attention from `flame/custom_models/lact_model`
- `full_attention`: full-attention Qwen3 with `flash_attention_2`
- `hybrid_swa`: 75% sliding-window attention + 25% full attention
- `hybrid_gdn`: 75% GatedDeltaNet + 25% full attention

The scaffold is aimed at two benchmark modes:

- Whole-network training throughput across sequence lengths: `4k, 8k, 16k, 32k, 64k, 128k, 256k, 512k, 1M`
- Single-layer throughput across the same sequence lengths for LaCT, full attention, SWA, and GDN

Paper-grounded notes from `Test-Time Training Done Right`:

- LaCT uses large chunk updates, roughly from `2K` up to `1M` tokens across tasks.
- In the language-model setup, LaCT combines a shifted block-causal TTT mask with causal sliding-window attention.
- For token-level causality without gaps, the paper requires `window_size >= chunk_size`; the implementation notes use `window_size == chunk_size` in practice.
- The LM architecture shares `Q/K/V` between the SWA branch and the LaCT branch.
- The paper’s throughput table is reported as tokens/s per GPU at `32K` sequence length; the distributed benchmark here also reports a global node-level tokens/s number.

## Layout

- `whole_model.py`: synthetic end-to-end train-step throughput, with optional DDP under `torchrun`
- `single_kernel.py`: isolated layer forward/backward throughput, with optional multi-process aggregation under `torchrun`
- `adapters.py`: model/layer builders and config translation
- `common.py`: timing and result export helpers
- `run_whole_model_8xh100.sh`: example launcher for single-node 8-GPU whole-model runs
- `run_single_kernel_8xh100.sh`: example launcher for single-node 8-GPU layer runs

## Assumptions

- The baselines are normalized to the LaCT config in `configs/qwen3_lact_1B4.json` so hidden size, layer count, MLP size, and vocab size stay aligned unless you override them.
- `hybrid_swa` uses the local implementation in `/work/yufei/projects/hybrid_models/qwen3_swa`.
- `hybrid_gdn` uses the local implementation in `/work/yufei/projects/hybrid_models/qwen3_gdn`.
- The benchmark is intended for a CUDA machine with `torch`, `transformers`, `flash-attn`, and the FLA dependencies installed.
- For multi-GPU whole-model throughput, launch with `torchrun`; `whole_model.py` will use DDP if `--ddp` is set.
- For multi-GPU single-kernel throughput, launch with `torchrun`; each rank runs the same layer benchmark on one GPU and rank 0 reports node-level aggregate throughput.
- The benchmark now exposes `--lact-chunk-size` and `--sliding-window` explicitly. With `--paper-lm-defaults` enabled, the harness enforces the paper’s LM condition `window_size >= chunk_size`.

## Usage

From `/work/yufei/projects/flame`:

```bash
python -m benchmarks.throughput.whole_model \
  --models lact full_attention hybrid_swa hybrid_gdn \
  --seq-lens 4096 8192 16384 32768 65536 131072 262144 524288 1048576 \
  --lact-chunk-size 2048 \
  --sliding-window 2048 \
  --batch-size 1 \
  --steps 10 \
  --warmup-steps 3 \
  --dtype bfloat16 \
  --device cuda \
  --use-fused-lact-kernel
```

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  -m benchmarks.throughput.whole_model \
  --ddp \
  --models lact full_attention hybrid_swa hybrid_gdn \
  --seq-lens 32768 \
  --lact-chunk-size 4096 \
  --sliding-window 4096 \
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
  --lact-chunk-size 2048 \
  --sliding-window 2048 \
  --batch-size 1 \
  --steps 20 \
  --warmup-steps 5 \
  --dtype bfloat16 \
  --device cuda \
  --use-fused-lact-kernel
```

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  -m benchmarks.throughput.single_kernel \
  --models lact full_attention hybrid_swa hybrid_gdn \
  --seq-lens 32768 \
  --lact-chunk-size 4096 \
  --sliding-window 4096 \
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
- `batch_size`: local per-GPU batch size
- `world_size`: number of GPUs/processes participating in the run
- `tokens_per_second`: global throughput, `batch_size * world_size * seq_len / step_time`
- `peak_memory_gb`: `torch.cuda.max_memory_allocated()`
- `status`: `ok`, `oom`, or `error`

For multi-GPU runs, rank 0 reports:

- the max per-rank `forward_ms`, `backward_ms`, `optimizer_ms`, and `step_ms`
- the max per-rank `peak_memory_gb`
- global tokens/s based on all GPUs

If you want to mirror the language-model paper setting more closely, start with:

- `seq_len=32768`
- `lact_chunk_size=4096`
- `sliding_window=4096`
- `torch_dtype=bfloat16`
- multi-GPU launch via `torchrun`

## Additional Comparisons Worth Running

These are likely useful for improving the LaCT work beyond the two core throughput plots:

1. Break down LaCT forward time into the SWA branch versus the TTT branch. The paper’s core claim is that large-chunk TTT improves utilization; this split will show whether that is true in your implementation.
2. Sweep `lact_chunk_size` and `sliding_window` together, especially with `window_size == chunk_size` versus `window_size > chunk_size`. The paper explicitly ties causal correctness and utilization to this relationship.
3. Compare LaCT update variants analogous to the paper’s `GD`, `Momentum`, and `Muon` settings. That is likely more informative for improving LaCT than another arbitrary baseline.
4. Compare fused vs unfused LaCT kernels with the same chunk/window settings. This isolates algorithmic gains from kernel-engineering gains.
5. Plot peak activation memory vs sequence length for all four models. Throughput without memory context is hard to interpret at `256k+`.
6. Measure forward-only and backward-only tokens/s separately. Some recurrent alternatives look good in forward but lose their advantage in backward.
7. Record the OOM boundary for each model under a fixed GPU budget. This is often more actionable than raw throughput at `1M`.
8. Add a state-size sweep by varying `num_lact_heads`, head dimension, or low-rank initialization. The paper argues that LaCT’s large chunking is what enables much larger nonlinear state sizes.

## Limitations In This Draft

- It uses synthetic token inputs, not dataloader throughput.
- CPU timing is not implemented; this scaffold is meant for CUDA benchmarking.
- Whole-model multi-GPU mode uses DDP, so it measures train-step throughput including gradient synchronization.
- Single-kernel multi-GPU mode is aggregated per-GPU execution, not a communication benchmark.
- The layer benchmark is really a layer-level train-step benchmark, not a raw Triton microkernel profiler.
- The LaCT paper PDF at `/work/yufei/projects/paper_list/tttdr.pdf` was not parsed in this environment, so the implementation choices here are grounded in the local code rather than paper text extraction.
