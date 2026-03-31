# Hybrid Qwen3-LaCT Pipeline

This folder documents the current pipeline for:

- converting `Qwen/Qwen3-0.6B-Base` into a Qwen3-native hybrid model
- continual pretraining the hybrid model
- exporting checkpoints for evaluation
- evaluating checkpoints with `lm-eval`
- the deferred Phase B plan for FA-informed TTT initialization

## Goal

We want a hybrid decoder stack where:

- `25%` of layers remain standard full-attention (`fa`)
- `75%` of layers become hybrid recurrent layers (`lact`)

For `Qwen3-0.6B-Base`:

- total layers: `28`
- `fa` layers: `7`
- `lact` layers: `21`

The current split is encoded in the generated config via `hybrid_layer_types`.

## Model Overview

Relevant files:

- [configuration_hybrid_qwen3_lact.py](/work/yufei/projects/flame/custom_models/hybrid_qwen3_lact_model/configuration_hybrid_qwen3_lact.py)
- [modeling_hybrid_qwen3_lact.py](/work/yufei/projects/flame/custom_models/hybrid_qwen3_lact_model/modeling_hybrid_qwen3_lact.py)
- [custom_models/__init__.py](/work/yufei/projects/flame/custom_models/__init__.py)

### Why this is Qwen3-native

The hybrid model keeps the native Qwen3 decoder structure:

- Qwen3 attention projections and GQA layout
- Qwen3 MLP
- Qwen3 RMSNorm structure
- Qwen3 embeddings and LM head

It does **not** try to force Qwen3 into the older packed-`qkv` LaCT attention implementation.

### Attention path

Each layer uses `HybridQwen3Attention`.

- `fa` layers:
  - standard full causal attention
- `lact` layers:
  - local/sliding causal attention
  - plus a separate LaCT branch

FlashAttention is used only when:

- tensors are on CUDA
- the fast path is available
- no explicit external attention mask blocks that path

Otherwise it falls back to plain PyTorch attention.

### LaCT branch

Each `lact` layer has a `HybridQwen3LaCTBranch`.

The branch builds a TTT-style memory path from the Qwen3 hidden states using:

- `qkv`
- optional q/k norm
- q/k affine rescaling (`qk_scale`, `qk_offset`)
- SiLU feature transform
- L2 normalization
- optional RoPE on TTT-side q/k
- fast-weight memory update/readout

Fast-weight parameters:

- `w0`
- `w2`
- `w1`
- `lr_proj`
- `ttt_scale_proj`
- `ttt_norm`
- optional `momentum_proj`

Conceptually:

- `w0`: gate-side input projection
- `w2`: parallel hidden/value-side input projection
- `w1`: output projection

The branch output is added to the attention output.

### Learnable TTT scale

The branch has a learned gate:

`ttt_scale = silu(ttt_scale_proj(hidden_states))`

This is the `alpha`-like factor controlling how strongly the TTT branch contributes.

At conversion time:

- `ttt_scale_proj.weight = 0`
- `ttt_scale_proj.bias = 0`

So at step `0` the effective branch contribution is `0`, even though the branch parameters exist.

This was chosen to preserve the pretrained Qwen3 behavior at initialization.

### Runtime TTT diagnostics

The model logs TTT activity metrics during training:

- `ttt/global/scale_mean`
- `ttt/global/scale_abs_mean`
- `ttt/global/scale_max`
- `ttt/global/to_attn_rms_ratio`
- per-layer versions of the same family

Interpretation:

- `scale_abs_mean` tells whether the TTT gate is open
- `to_attn_rms_ratio` tells how large TTT output is relative to attention output

## Conversion Pipeline

Relevant scripts:

- [make_hybrid_qwen3_lact_config.py](/work/yufei/projects/flame/scripts/make_hybrid_qwen3_lact_config.py)
- [convert_qwen3_to_hybrid_lact.py](/work/yufei/projects/flame/scripts/convert_qwen3_to_hybrid_lact.py)
- [check_hybrid_qwen3_lact.py](/work/yufei/projects/flame/scripts/check_hybrid_qwen3_lact.py)
- [init_ckpt.sh](/work/yufei/projects/flame/scripts/20260322_hybrid_qwen3_lact_0p6B/init_ckpt.sh)

### Step 1: make the hybrid config

Generate the config from `Qwen/Qwen3-0.6B-Base`.

Key knobs:

- `num_lact_heads`
- `lact_chunk_size`
- `window_size`
- `hybrid_layer_types`

Current choice:

- sliding window for `lact` layers: `2048`

### Step 2: convert the HF checkpoint

`convert_qwen3_to_hybrid_lact.py`:

- loads Qwen3 base weights
- instantiates the hybrid model
- copies shared Qwen3 weights into the hybrid model
- initializes new LaCT-only parameters conservatively
- saves a hybrid HF checkpoint

### Step 3: convert HF to DCP seed checkpoint

`flame` training uses DCP checkpoints, not raw HF checkpoints.

`init_ckpt.sh` converts the hybrid HF checkpoint into an external seed checkpoint:

- seed root:
  - `/storage/backup/${USERNAME}/ttt/flame/seeds/qwen3_hybrid_qwen3_lact_0p6B/step-0`

This seed is intentionally kept **outside** the run’s normal checkpoint folder.

That separation matters because `flame` treats:

- external initial checkpoint
- run-local resumable checkpoint

as different load modes.

## Continual Pretraining Pipeline

Relevant scripts:

- [vars.sh](/work/yufei/projects/flame/scripts/20260322_hybrid_qwen3_lact_0p6B/vars.sh)
- [train.sh](/work/yufei/projects/flame/scripts/20260322_hybrid_qwen3_lact_0p6B/train.sh)

### Current training load mode

Training is launched with:

- `--checkpoint.initial_load_path ${seed_checkpoint_dir}`
- `--checkpoint.initial_load_model_weights_only`
- `--checkpoint.load_step -1`

This means:

- initialize model weights from the external seed
- do **not** resume as if this were a previous training run

### Current scheduler

The launcher uses:

- optimizer: `AdamW`
- base LR: `3e-4`
- warmup: `1024` steps
- scheduler: cosine
- LR minimum: `0.1 * base_lr`

This is **not** a step scheduler.

### WandB

WandB is enabled in the launcher via:

- `--metrics.enable_wandb`

and the run metadata is set in `vars.sh`:

- `WANDB_PROJECT`
- `WANDB_NAME`
- `WANDB_RUN_ID`
- `WANDB_RESUME`

### Current data choice

FineWeb overlap with Qwen3 may make train loss overly optimistic.

Recommended DCLM path found on disk:

- `/storage/backup/hei/data/qwen3-dclm-filter-16k_train`

Other nearby candidates:

- `/storage/backup/hei/data/qwen3-dclm-filter-8k_train`
- `/storage/backup/hei/data/fineweb100bt-qwen3-tokenized-packed-16384_arrow`

### Important caution about sequence length

Changing the dataset to a `16k` dataset does **not** by itself make training a `16k` run.

That also depends on:

- `seq_len`
- `context_len`
- batch size / grad accumulation

## Evaluation Pipeline

Relevant scripts:

- [lm_eval_mp.sh](/work/yufei/projects/flame/scripts/20260322_hybrid_qwen3_lact_0p6B/lm_eval_mp.sh)
- [lm_eval_with_custom_models.py](/work/yufei/projects/flame/scripts/20260322_hybrid_qwen3_lact_0p6B/lm_eval_with_custom_models.py)
- [gen_results_csv.sh](/work/yufei/projects/flame/scripts/20260322_hybrid_qwen3_lact_0p6B/gen_results_csv.sh)
- [gen_results_csv.py](/work/yufei/projects/flame/gen_results_csv.py)

### Exporting a training checkpoint to HF

`flame.utils.convert_dcp_to_hf` uses:

- `--path`: run root
- `--step`: numeric step
- `--config`
- `--tokenizer`

Important behavior:

- it writes the HF export directly into `--path`
- therefore each new export overwrites the previous HF export in that run folder

### Running lm-eval

The eval script loads from:

- `eval_hf_path=${dump_folder}`

which is correct **after** converting the desired DCP step into HF at the run root.

### Current caveat

Multiple eval runs under the same `RUN_NAME` will produce multiple `results*.json` files in the same results subfolder.

`gen_results_csv.py` expects exactly one file and will fail if there are two or more.

So either:

- remove/move older result files before summarizing
- or use a separate run/output folder per checkpoint eval

## What We Learned From The First Continual-Pretraining Run

The first FineWeb-based run showed:

- train loss dropped very aggressively
- downstream eval was near-random on many tasks
- TTT contribution metrics became large

This strongly suggests train loss alone is not a reliable health signal here.

Plausible causes:

- overlap between the continued-pretraining corpus and the original Qwen3 pretraining data
- TTT branch becoming too strong relative to the pretrained attention path
- optimization dynamics drifting into a bad solution

Recommended response:

- prefer DCLM over FineWeb-like data
- evaluate the seed checkpoint and very early checkpoints
- do not trust late low train loss by itself

## Suggested Checkpoints To Compare

For a fresh run, compare:

- seed hybrid checkpoint
- `step-1`
- `step-1907`
- `step-3814`
- `step-5721`

The earlier experiments suggested later checkpoints may become partially or fully collapsed.

## Phase B Plan: FA-Informed TTT Initialization

Phase B is **not implemented end-to-end yet**, but the scaffold exists.

Relevant script:

- [fit_hybrid_qwen3_lact_phase_b.py](/work/yufei/projects/flame/scripts/fit_hybrid_qwen3_lact_phase_b.py)

### Goal

Use a teacher model to build a better initializer for the new TTT memory branch, instead of relying only on small random initialization.

### Current concept

For each converted `lact` layer:

1. run the teacher model on a corpus sample
2. capture the hidden states entering the target layer
3. reconstruct the exact LaCT branch-side features using the hybrid branch preprocessing
4. collect:
   - `fast_k`
   - `fast_v`
5. fit a per-layer, per-head ridge regression:
   - `fast_k -> fast_v`

### Current script behavior

The script now supports:

- `--text-file`
- `--max-samples`
- `--batch-size`

and accumulates:

- `X^T X`
- `X^T Y`

across many samples, instead of using a single sentence.

It currently exports:

- `xtx`
- `xty`
- `ridge_solution`
- `builtin_identity`
- `delta_from_builtin`
- candidate low-rank factors when shape-compatible

### What Phase B still does not do

- it does not write fitted factors back into a hybrid checkpoint
- it does not yet define the final mapping from the fitted matrix into the full nonlinear `(w0, w2, w1)` branch
- it does not yet create a production-ready offline initializer pipeline

### Intended next steps for Phase B

1. collect a larger teacher corpus sample
2. fit per-layer/per-head regressions in the exact runtime feature space
3. convert the fitted full map into low-rank factors
4. write those factors into a new hybrid checkpoint
5. compare:
   - random-small init
   - FA-informed init

### Important conceptual note

The fitted regression target is **not literally equal to just `w0`**.

The LaCT branch memory function is nonlinear and involves:

- `w0`
- `w2`
- `w1`

So the current Phase B fit should be viewed as a principled initializer proxy for the branch, not a perfect one-to-one transplant.

## Practical Recommendations

- use DCLM rather than FineWeb-like data for the next continual-pretraining attempt
- evaluate the seed and early checkpoints, not only late ones
- watch:
  - `ttt/global/scale_abs_mean`
  - `ttt/global/to_attn_rms_ratio`
- if TTT becomes too dominant too quickly, reduce aggressiveness before the next run:
  - lower branch init scale
  - lower LR
  - consider a more conservative gate strategy

## Files In This Folder

- [vars.sh](/work/yufei/projects/flame/scripts/20260322_hybrid_qwen3_lact_0p6B/vars.sh): run-specific variables
- [init_ckpt.sh](/work/yufei/projects/flame/scripts/20260322_hybrid_qwen3_lact_0p6B/init_ckpt.sh): config + HF conversion + seed creation
- [train.sh](/work/yufei/projects/flame/scripts/20260322_hybrid_qwen3_lact_0p6B/train.sh): continual-pretraining launcher
- [lm_eval_mp.sh](/work/yufei/projects/flame/scripts/20260322_hybrid_qwen3_lact_0p6B/lm_eval_mp.sh): `lm-eval` launcher
- [lm_eval_with_custom_models.py](/work/yufei/projects/flame/scripts/20260322_hybrid_qwen3_lact_0p6B/lm_eval_with_custom_models.py): ensures `custom_models` are registered during eval
- [gen_results_csv.sh](/work/yufei/projects/flame/scripts/20260322_hybrid_qwen3_lact_0p6B/gen_results_csv.sh): summarize eval results
- [param_cnt.sh](/work/yufei/projects/flame/scripts/20260322_hybrid_qwen3_lact_0p6B/param_cnt.sh): parameter count
- [report.sh](/work/yufei/projects/flame/scripts/20260322_hybrid_qwen3_lact_0p6B/report.sh): report launcher
- [phase_b_collect.sh](/work/yufei/projects/flame/scripts/20260322_hybrid_qwen3_lact_0p6B/phase_b_collect.sh): Phase B collection helper

