# Hybrid Test-Time Adaptation Runner

Date: 2026-04-07

This folder contains a hybrid test-time adaptation setup built on top of the
temporary LoRA method in
[../20260330_ttt_lora](/work/yufei/projects/flame/scripts/20260330_ttt_lora).

## What It Does

For each LongBench sample:

1. load a frozen base causal LM
2. attach temporary LoRA adapters on `q_proj` and `v_proj`
3. reset LoRA to its initial zero-output state at sample start
4. route by task family:
   - QA tasks:
     use test-time distillation
   - non-QA tasks:
     use legacy chunk next-token TTT-LoRA from
     [../20260330_ttt_lora](/work/yufei/projects/flame/scripts/20260330_ttt_lora)
5. decode autoregressively with `torch.no_grad()`

So the intended behavior is:

- QA:
  generate task-conditioned candidates, judge them, keep the top few, and
  train temporary LoRA on those memory units
- summarization, code, synthetic, classification, few-shot:
  reuse the old chunk next-token temporary-LoRA adaptation path

## Current Scope

- update mode: `full_prefix_approx` only
- chunk size default: `1024`
- LoRA config:
  - top 12.5% of layers
  - `q_proj`, `v_proj`
  - `r=64`
  - `alpha=64`
  - `dropout=0.0`
- QA distillation pipeline defaults:
  - generate `4` candidates
  - heuristic-preselect top `3` for judging
  - select top `2`
  - `1` answer-only update per selected pair
- LongBench path:
  - prefers structured `doc["context"]` and `doc["question"]`
  - falls back to the rendered prompt if sample metadata is unavailable
- Hybrid routing:
  - `qa` family -> distillation
  - non-`qa` families -> legacy chunk-NTP TTT-LoRA

## Files

- [run_ttt_lora.py](/work/yufei/projects/flame/scripts/20260407_test_time_distill/run_ttt_lora.py)
  Core QA-distillation runner and task-family helpers.
- [lm_eval_ttt_lora.py](/work/yufei/projects/flame/scripts/20260407_test_time_distill/lm_eval_ttt_lora.py)
  Hybrid `lm-eval` backend for LongBench.
- [lm_eval_mp.sh](/work/yufei/projects/flame/scripts/20260407_test_time_distill/lm_eval_mp.sh)
  Main launcher.
- [repro_longbench_sample.py](/work/yufei/projects/flame/scripts/20260407_test_time_distill/repro_longbench_sample.py)
  Single-sample repro aligned with the hybrid eval flow.
- [test_failed_qa_recovery.py](/work/yufei/projects/flame/scripts/20260407_test_time_distill/test_failed_qa_recovery.py)
  QA-only recovery harness for failed baseline QA samples.
- [probe_seq_len.py](/work/yufei/projects/flame/scripts/20260407_test_time_distill/probe_seq_len.py)
  Sequence-length probe that follows the same hybrid routing by task family.

## Logging

For QA-distillation chunks, logs include:

- chunk index and token span
- task question
- chunk text preview
- raw generated QA text
- parsed QA pairs
- filtered pairs
- selected pairs with judge scores, losses, and grad norms
- LoRA norm after the chunk

For non-QA tasks routed to legacy TTT-LoRA, logs include standard chunk NTP
loss, grad norm, and LoRA norm.

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

This repro now follows the same hybrid routing as `lm-eval`:

- QA sample -> distillation
- non-QA sample -> legacy chunk-NTP TTT-LoRA

## QA Recovery Harness

```bash
python /work/yufei/projects/flame/scripts/20260407_test_time_distill/test_failed_qa_recovery.py \
  --task-name longbench_multifieldqa_en \
  --doc-id 48 \
  --device cuda:0 \
  --dtype bfloat16
```

This harness is QA-only and is meant for checking whether the distillation path
recovers baseline QA failures.

## Sequence-Length Probe

```bash
python /work/yufei/projects/flame/scripts/20260407_test_time_distill/probe_seq_len.py \
  --task-name gov_report \
  --device cuda:0 \
  --dtype bfloat16 \
  --binary-search
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

## Routing Summary

- `single_document_qa` and `multi_document_qa`:
  distillation
- `summarization`:
  legacy TTT-LoRA next-token adaptation
- `code_completion`:
  legacy TTT-LoRA next-token adaptation
- `synthetic_tasks`:
  legacy TTT-LoRA next-token adaptation
- `classification`-style tasks such as `TREC` and `LSHT`:
  legacy TTT-LoRA next-token adaptation
