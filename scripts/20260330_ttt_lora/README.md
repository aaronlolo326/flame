# TTT-LoRA LongBench Runner

Date: 2026-04-03

This folder contains the current temporary LoRA test-time training
implementation for Qwen-style causal LMs, plus the LongBench evaluation wrapper
around it.

## What It Does

For each sample:

1. load a frozen base causal LM
2. attach temporary LoRA adapters on `q_proj` and `v_proj`
3. reset LoRA to its initial zero-output state at sample start
4. split the prompt into chunks
5. for each chunk:
   - run one or more LoRA update steps on that chunk
   - rebuild full-prefix KV cache once after the last step on the chunk
6. after the final chunk, keep the adapted LoRA state and rebuilt KV cache
7. decode autoregressively with `torch.no_grad()`

The base model stays frozen. Only LoRA parameters are updated.

## Files

- [run_ttt_lora.py](/work/yufei/projects/flame/scripts/20260330_ttt_lora/run_ttt_lora.py)
  Standalone runner for prompt-level TTT-LoRA.
- [lm_eval_ttt_lora.py](/work/yufei/projects/flame/scripts/20260330_ttt_lora/lm_eval_ttt_lora.py)
  `lm-eval` backend used for LongBench.
- [lm_eval_mp.sh](/work/yufei/projects/flame/scripts/20260330_ttt_lora/lm_eval_mp.sh)
  Main launcher for LongBench and LongBench task groups.
- [repro_longbench_sample.py](/work/yufei/projects/flame/scripts/20260330_ttt_lora/repro_longbench_sample.py)
  Single-sample repro script aligned with the eval flow.
- [probe_seq_len.py](/work/yufei/projects/flame/scripts/20260330_ttt_lora/probe_seq_len.py)
  Single-GPU affordability probe for sequence length.

## Update Modes

`TTT_UPDATE_MODE` / `--update-mode` supports:

- `full_prefix_exact`
  Legitimate full-prefix update pass. For chunk `[s:e]`, the update step
  rebuilds the exact prefix cache for `[:s]`, then trains on `[s:e]`.
- `full_prefix_approx`
  Approximate faster mode. Reuses the carried prefix cache from previous chunks
  as the base for the current chunk, makes a safe per-step copy, and rebuilds
  full-prefix KV only once after the last step on the chunk.
- `full_prefix`
  Alias for `full_prefix_exact`.
- `local_window`
  Bounded training-view mode. For chunk `[s:e]`, the update pass sees only
  `tokens[max(0, s-LT):e]`, computes loss only on the current chunk portion, and
  does not reuse full-history KV inside the update pass.

The final inference path always uses full-prefix KV after rebuild, regardless
of update mode.

## Loss Modes

`TTT_LOSS_MODE` / `--loss-mode` is independent of update mode and composes with
all update-pass variants.

Supported options:

- `full`
  Standard next-token CE averaged over all scored tokens in the current update
  span.
- `topk_fraction`
  Computes per-token CE, keeps only the highest-loss fraction of tokens, and
  averages over that subset.

`TTT_LOSS_TOPK_FRACTION` / `--loss-topk-fraction` controls the kept fraction
when `loss_mode=topk_fraction`.

So the current implementation has two orthogonal knobs:

1. update-pass context mode
   `full_prefix_exact`, `full_prefix_approx`, or `local_window`
2. loss selection mode
   `full` or `topk_fraction`

Examples of valid combinations:

- `full_prefix_exact + full`
- `full_prefix_exact + topk_fraction`
- `full_prefix_approx + topk_fraction`
- `local_window + topk_fraction`

## Important Notes

### Zero initialization

Literal all-zero initialization of both LoRA factors blocks learning through the
LoRA branch. This implementation resets LoRA to the PEFT zero-output
initialization state instead, so the adapter starts with zero contribution but
still receives gradients.

### Rebuild and generation

- Chunk update steps run with gradients enabled.
- Full-prefix rebuild runs under `torch.no_grad()`.
- Final decoding also runs under `torch.no_grad()`.

### Approximate vs exact full-prefix

`full_prefix_approx` is a speed-oriented approximation. `full_prefix_exact` is
the correctness-first version.

## Current Practical Caveats

- Long full LongBench runs can still hit OOM because of long-lived process
  memory state, even when isolated single-sample repros pass.
- Summarization tasks are the hardest because they combine long inputs with
  `max_gen_toks=512`.
- `full_prefix_exact` is slower than the earlier approximate behavior because
  it reconstructs the exact prefix view for the update pass.
- `full_prefix_approx` relies on safe copying of the carried cache object.

## Standalone Example

```bash
python /work/yufei/projects/flame/scripts/20260330_ttt_lora/run_ttt_lora.py \
  --model-name-or-path Qwen/Qwen3-0.6B \
  --prompt "Write a short summary of transformers and attention." \
  --chunk-size 512 \
  --steps-per-chunk 1 \
  --update-mode full_prefix_exact \
  --max-new-tokens 64 \
  --device cuda:0 \
  --dtype bfloat16 \
  --log-file /work/yufei/projects/flame/results/20260330_ttt_lora/sample.jsonl
```

## LongBench Launcher

Default:

```bash
bash /work/yufei/projects/flame/scripts/20260330_ttt_lora/lm_eval_mp.sh
```

TTT-LoRA on a specific task group:

```bash
CUDA_VISIBLE_DEVICES=0 \
TASKS=summarization \
TTT_UPDATE_MODE=full_prefix_approx \
TTT_CHUNK_SIZE=512 \
bash /work/yufei/projects/flame/scripts/20260330_ttt_lora/lm_eval_mp.sh
```

Plain base-model eval with the same custom loader:

```bash
USE_TTT_LORA=0 \
TASKS=longbench \
bash /work/yufei/projects/flame/scripts/20260330_ttt_lora/lm_eval_mp.sh
```

Supported task aliases in the launcher:

- `code_completion`
- `few_shot_learning`
- `multi_document_qa`
- `single_document_qa`
- `summarization`
- `synthetic_tasks`
- `longbench`

## Important Launcher Knobs

- `USE_TTT_LORA=0|1`
- `eval_hf_path=<model path or HF id>`
- `MAX_LENGTH=<context + generation budget>`
- `TTT_CHUNK_SIZE=<chunk size>`
- `TTT_STEPS_PER_CHUNK=<optimizer steps per chunk>`
- `TTT_UPDATE_MODE=full_prefix_exact|full_prefix_approx|local_window`
- `TTT_LOCAL_TRAIN_WINDOW=<LT>`
- `TTT_RAW_CHARS_PER_TOKEN=<raw pretruncate heuristic>`
- `TTT_RAW_TRUNC_SAFETY_MARGIN=<raw pretruncate heuristic>`

Note:
- [lm_eval_mp.sh](/work/yufei/projects/flame/scripts/20260330_ttt_lora/lm_eval_mp.sh)
  passes explicit LongBench eval-time hyperparameters through `--model_args`.
- Those launcher-time values can override the standalone defaults in
  [run_ttt_lora.py](/work/yufei/projects/flame/scripts/20260330_ttt_lora/run_ttt_lora.py).

## Single-Sample Repro

```bash
python /work/yufei/projects/flame/scripts/20260330_ttt_lora/repro_longbench_sample.py \
  --task-name longbench_gov_report \
  --doc-id 127 \
  --device cuda:0 \
  --dtype bfloat16 \
  --max-length 16384 \
  --chunk-size 512 \
  --steps-per-chunk 1 \
  --update-mode full_prefix_approx
```

This repro script:

- loads a saved LongBench sample
- applies raw-text pretruncation and exact token truncation
- runs chunked TTT updates
- rebuilds full-prefix KV
- decodes using the sample’s generation settings

## Sequence-Length Probe

```bash
python /work/yufei/projects/flame/scripts/20260330_ttt_lora/probe_seq_len.py \
  --model-name-or-path /work/yufei/downloads/Qwen3-0.6B-Base \
  --device cuda:0 \
  --dtype bfloat16 \
  --binary-search \
  --min-seq-len 1024 \
  --max-seq-len 16384 \
  --step 1024 \
  --output-json /work/yufei/projects/flame/results/20260330_ttt_lora/seq_probe.json
```
