# LaCT Throughput Benchmark Scaffold

This folder is a draft benchmark framework for comparing LaCT against three user-requested baselines on a single GPU.

- `lact`: LaCT layer + sliding-window flash attention from `flame/custom_models/lact_model`
- `full_attention`: full-attention Qwen3 with `flash_attention_2`
- `hybrid_swa`: 75% sliding-window attention + 25% full attention
- `hybrid_gdn`: 75% GatedDeltaNet + 25% full attention

The scaffold is aimed at two benchmark modes:

- Whole-network training throughput across sequence lengths: `4k, 8k, 16k, 32k, 64k, 128k, 256k, 512k, 1M`
- Single-layer throughput across the same sequence lengths for the five meaningful layer/branch subjects listed below

Paper-grounded notes from `Test-Time Training Done Right`:

- LaCT uses large chunk updates, roughly from `2K` up to `1M` tokens across tasks.
- In the language-model setup, LaCT combines a shifted block-causal TTT mask with causal sliding-window attention.
- For token-level causality without gaps, the paper requires `window_size >= chunk_size`; the implementation notes use `window_size == chunk_size` in practice.
- The LM architecture shares `Q/K/V` between the SWA branch and the LaCT branch.
- The paper’s throughput table is reported as tokens/s per GPU at `32K` sequence length.

The benchmark targets in this folder are not identical to the paper’s LM baseline table. The paper compares LaCT against full attention, `GLA + SWA`, and `DeltaNet + SWA`, while this scaffold follows your requested comparison set:

- full attention
- `75% SWA + 25% FA`
- `75% GatedDeltaNet + 25% FA`

## Layout

- `whole_model.py`: synthetic end-to-end single-GPU train-step throughput
- `single_kernel.py`: isolated single-GPU layer forward/backward throughput
- `count_single_kernel_params.py`: parameter counter for the single-layer benchmark subjects
- `adapters.py`: model/layer builders and config translation
- `common.py`: timing and result export helpers

## Assumptions

- The baselines are normalized to the LaCT config in `configs/qwen3_lact_1B4.json` so hidden size, layer count, MLP size, and vocab size stay aligned unless you override them.
- `hybrid_swa` uses the local implementation in `/work/yufei/projects/hybrid_models/qwen3_swa`.
- `hybrid_gdn` uses the local implementation in `/work/yufei/projects/hybrid_models/qwen3_gdn`.
- The benchmark is intended for a CUDA machine with `torch`, `transformers`, `flash-attn`, and the FLA dependencies installed.
- The benchmark now exposes `--lact-chunk-size` and `--sliding-window` explicitly. With `--paper-lm-defaults` enabled, the harness enforces the paper’s LM condition `window_size >= chunk_size`.
- No context parallel, tensor parallel, or DDP path is implemented in this scaffold. Cases that do not fit on one GPU will be recorded as `oom`.
- By default, if one model OOMs at a given `seq_len`, the harness skips larger `seq_len` values for that same model in the current sweep.

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
python -m benchmarks.throughput.single_kernel \
  --models lact_full_layer lact_ttt_branch_only \
           fa_branch_only swa_branch_only gdn_branch_only \
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

The current single-layer benchmark subjects are:

- `lact_full_layer`
- `lact_ttt_branch_only`
- `fa_branch_only`
- `swa_branch_only`
- `gdn_branch_only`

Interpretation:

- `lact_full_layer`: the full LaCT token-mixing layer, including both SWA and TTT paths
- `lact_ttt_branch_only`: the LaCT TTT path in isolation
- `fa_branch_only`: the full-attention token-mixing layer
- `swa_branch_only`: the sliding-window attention token-mixing layer
- `gdn_branch_only`: the GatedDeltaNet token-mixing layer

Outputs are written as both `.csv` and `.jsonl` under `benchmarks/throughput/results/`.

## What The Metrics Mean

- `forward_ms`: average forward-pass time
- `backward_ms`: average backward-pass time
- `optimizer_ms`: optimizer step time, only for whole-model runs
- `step_ms`: wall-clock average for one measured step
- `batch_size`: single-GPU batch size
- `tokens_per_second`: single-GPU throughput, `batch_size * seq_len / step_time`
- `peak_memory_gb`: `torch.cuda.max_memory_allocated()`
- `status`: `ok`, `oom`, or `error`

If you want to mirror the language-model paper setting more closely, start with:

- `seq_len=32768`
- `lact_chunk_size=4096`
- `sliding_window=4096`
- `torch_dtype=bfloat16`

That still will not exactly reproduce the paper table, because the benchmarked baseline set here is your requested hybrid set rather than the paper’s `GLA + SWA` and `DeltaNet + SWA` baselines.

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
- It is single-GPU only. It does not implement context parallel, tensor parallel, or DDP.
- The layer benchmark is really a layer-level train-step benchmark, not a raw Triton microkernel profiler.
- The LaCT paper PDF at `/work/yufei/projects/paper_list/tttdr.pdf` is now parseable in this environment, and the chunk/window guidance in this README has been updated from it.
- The current scaffold still does not encode every paper ablation or exact paper baseline; it mainly uses the paper to set correct LaCT benchmarking assumptions while preserving your requested model comparisons.
