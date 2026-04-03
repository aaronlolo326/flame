source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/vars.sh"

set -euo pipefail

debug=false
profile=false
no_tokens=$(( 20 * 10**9 ))
seed=42

BASE_RUN_NAME="${RUN_NAME}"
RUN_NAME="${BASE_RUN_NAME}_prolong_from_run12_step9535_v4"
export WANDB_NAME="${RUN_NAME}"
export WANDB_RUN_ID="${RUN_NAME}"

dump_folder=/storage/backup/${USERNAME}/ttt/flame/exp/${RUN_NAME}
checkpoint_folder=${dump_folder}/checkpoint

initial_checkpoint=/storage/backup/${USERNAME}/ttt/flame/exp/20260322_hybrid_qwen3_lact_0p6B_swa_2k_chunk_1k_rerun12/checkpoint/step-9535

batch_size=1
grad_accum=16

if [ ! -d "${initial_checkpoint}" ]; then
  echo "Initial checkpoint missing at ${initial_checkpoint}"
  exit 1
fi

if [ -e "${checkpoint_folder}" ]; then
  echo "Checkpoint folder already exists at ${checkpoint_folder}"
  echo "Remove it or change RUN_NAME before launching so flame will honor --checkpoint.initial_load_path"
  exit 1
fi

if [ "$debug" = true ]; then
  CUDA_VISIBLE_DEVICES=0
  batch_size=1
  grad_accum=1
fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  NGPU=1
else
  NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
fi

initial_step=9535
seq_len=32768
extra_steps=$(( no_tokens / NGPU / seq_len / batch_size / grad_accum ))
steps=$((initial_step + extra_steps))
interval=$(( steps / 20 ))
if [ ${interval} -lt 1 ]; then
  interval=1
fi
if $profile; then
  steps=10
fi

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
NNODE=1 NGPU=${NGPU} LOG_RANK=0 bash train.sh \
  --job.dump_folder ${dump_folder} \
  --job.print_args \
  --comm.init_timeout_seconds 3600 \
  --comm.train_timeout_seconds 600 \
  --model.config ${MODEL_CONFIG_PATH} \
  --model.tokenizer_path ${TOKENIZER_PATH} \
  --optimizer.name AdamW \
  --optimizer.eps 1e-15 \
  --optimizer.lr 3e-5 \
  --lr_scheduler.warmup_steps 1024 \
  --lr_scheduler.lr_min 0.1 \
  --lr_scheduler.decay_type cosine \
  --training.batch_size ${batch_size} \
  --training.seq_len ${seq_len} \
  --training.context_len ${seq_len} \
  --training.gradient_accumulation_steps ${grad_accum} \
  --training.steps ${steps} \
  --training.max_norm 1.0 \
  --training.skip_nan_inf \
  --training.tokenized_dataset_dir /storage/backup/yufei/ttt/data/prolong-merged-qwen3-8b-tokenized-32768-arrow \
  --training.num_workers 32 \
  --training.prefetch_factor 2 \
  --training.seed ${seed} \
  --training.compile \
  --training.disable_loss_parallel \
  --training.data_parallel_replicate_degree 1 \
  --training.data_parallel_shard_degree -1 \
  --training.tensor_parallel_degree 1 \
  --checkpoint.enable_checkpoint \
  --checkpoint.folder checkpoint \
  --checkpoint.interval ${interval} \
  --checkpoint.export_dtype float32 \
  --checkpoint.async_mode disabled \
  --checkpoint.initial_load_path ${initial_checkpoint} \
  --checkpoint.no_initial_load_model_weights_only \
  --checkpoint.load_step -1 \
  --metrics.enable_wandb \
  --metrics.log_freq 10 \
  --experimental.context_parallel_degree 1 \
  --experimental.pipeline_parallel_degree 1 \
  --activation_checkpoint.mode selective \
  --activation_checkpoint.selective_ac_option 1