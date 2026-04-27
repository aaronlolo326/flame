# Test-Time Distill LongBench Runner

Date: 2026-04-07

This folder contains a prototype test-time distillation variant of the
temporary LoRA method in
[../20260330_ttt_lora](/work/yufei/projects/flame/scripts/20260330_ttt_lora).

## What It Does

For each sample:

1. load a frozen base causal LM
2. attach temporary LoRA adapters on `q_proj` and `v_proj`
3. reset LoRA to its initial zero-output state at sample start
4. split the adaptation context into chunks
5. for each chunk:
   - decode chunk text
   - generate diverse candidate QA pairs conditioned on the real task question
   - judge the candidates with the same model
   - keep the top 2
   - update LoRA on `Question -> Answer` with loss only on answer tokens
   - rebuild full-prefix KV on the real seen adaptation prefix
6. after all chunks, rebuild KV for the final benchmark prompt
7. decode autoregressively with `torch.no_grad()`

The goal is to store task-relevant knowledge units in the temporary LoRA rather
than optimizing for raw next-token continuation of the context itself.

## Current Prototype Scope

- update mode: `full_prefix_approx` only
- chunk size default: `1024`
- LoRA config:
  - top 12.5% of layers
  - `q_proj`, `v_proj`
  - `r=64`
  - `alpha=64`
  - `dropout=0.0`
- QA pipeline:
  - generate `8` candidates
  - select top `2`
  - `1` answer-only update per selected pair
- LongBench path:
  - prefers structured `doc["context"]` and `doc["question"]`
  - falls back to the rendered prompt if sample metadata is unavailable

## Files

- [run_ttt_lora.py](/work/yufei/projects/flame/scripts/20260407_test_time_distill/run_ttt_lora.py)
  Core QA-distillation runner.
- [lm_eval_ttt_lora.py](/work/yufei/projects/flame/scripts/20260407_test_time_distill/lm_eval_ttt_lora.py)
  `lm-eval` backend for LongBench.
- [lm_eval_mp.sh](/work/yufei/projects/flame/scripts/20260407_test_time_distill/lm_eval_mp.sh)
  Main launcher.
- [repro_longbench_sample.py](/work/yufei/projects/flame/scripts/20260407_test_time_distill/repro_longbench_sample.py)
  Single-sample repro aligned with the eval flow.

## Logging

Each chunk log includes:

- chunk index and token span
- task question
- chunk text preview
- raw generated QA text
- parsed QA pairs
- filtered pairs
- selected pairs with judge scores, losses, and grad norms
- LoRA norm after the chunk

This is the main debugging surface for checking whether the selected pairs are
actually useful for the final task.

## Single-Sample Repro

```bash
python /work/yufei/projects/flame/scripts/20260407_test_time_distill/repro_longbench_sample.py \
  --task-name longbench_gov_report \
  --doc-id 127 \
  --device cuda:0 \
  --dtype bfloat16 \
  --max-length 16384 \
  --chunk-size 1024
```

## LongBench Launcher

Default:

```bash
bash /work/yufei/projects/flame/scripts/20260407_test_time_distill/lm_eval_mp.sh
```

Specific task group:

```bash
CUDA_VISIBLE_DEVICES=0 \
TASKS=summarization \
TTT_CHUNK_SIZE=1024 \
bash /work/yufei/projects/flame/scripts/20260407_test_time_distill/lm_eval_mp.sh
```

Plain base-model eval with the same loader:

```bash
USE_TTT_DISTILL=0 \
TASKS=longbench \
bash /work/yufei/projects/flame/scripts/20260407_test_time_distill/lm_eval_mp.sh
```
