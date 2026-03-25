source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/vars.sh"
umask 000

set -euo pipefail

debug=false
profile=false



available_nodes=(6 9) # first is master node
# based on env var $THIS_NODE, set local rank according to the order, e.g., THIS_NODE is 0 means local_rank = 0
local_rank=$(echo ${available_nodes[@]} | tr ' ' '\n' | grep -n "^${THIS_NODE}$" | cut -d: -f1)
local_rank=$((local_rank - 1))
echo "THIS_NODE=${THIS_NODE}; local_rank=$local_rank"

MASTER_ADDR=192.168.241.229
MASTER_PORT=29500

NNODE=${#available_nodes[@]}
if [ ${NNODE} -eq 1 ]; then
  MASTER_ADDR=localhost
  MASTER_PORT=29500
fi


echo $RUN_NAME


no_tokens=$(( 20 * 10**9 ))
seed=42

if [ ! -d "${seed_checkpoint_dir}" ]; then
  echo "Seed checkpoint missing at ${seed_checkpoint_dir}"
  echo "Run: bash $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/init_ckpt.sh"
  exit 1
fi

mkdir -p "${checkpoint_folder}"
chmod 777 "${checkpoint_folder}" || true
step0_checkpoint_dir="${checkpoint_folder}/step-0"
if [ ! -d "${step0_checkpoint_dir}" ]; then
  mkdir -p "${step0_checkpoint_dir}"
  cp -a "${seed_checkpoint_dir}/." "${step0_checkpoint_dir}/"
  chmod -R 777 "${step0_checkpoint_dir}"
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

steps=$(( no_tokens / NGPU / seq_len / batch_size / grad_accum ))
interval=$(( steps / 20 ))
if [ ${interval} -lt 1 ]; then
  interval=1
fi
if $profile; then
  steps=10
fi

# CUDA_LAUNCH_BLOCKING=1 TORCH_LOGS="+dynamo" TORCHDYNAMO_VERBOSE=1 \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} \
NNODE=${NNODE} NGPU=${NGPU} LOG_RANK=${local_rank} bash train.sh \
  --job.dump_folder ${dump_folder} \
  --job.print_args \
  --comm.init_timeout_seconds 3600 \
  --comm.train_timeout_seconds 600 \
  --model.config ${MODEL_CONFIG_PATH} \
  --model.tokenizer_path ${TOKENIZER_PATH} \
  --optimizer.name AdamW \
  --optimizer.eps 1e-15 \
  --optimizer.lr 1e-5 \
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
  --training.tokenized_dataset_dir /storage/backup/hei/data/HuggingFaceFW___fineweb-edu___sample-350BT \
  --training.num_workers 32 \
  --training.prefetch_factor 2 \
  --training.seed ${seed} \
  --training.compile \
  --training.disable_loss_parallel \
  --training.data_parallel_replicate_degree 1 \
  --training.data_parallel_shard_degree -1 \
  --training.tensor_parallel_degree 1 \
  --checkpoint.interval ${interval} \
  --checkpoint.enable_checkpoint \
  --checkpoint.folder checkpoint \
  --checkpoint.interval ${interval} \
  --checkpoint.export_dtype float32 \
  --checkpoint.async_mode disabled \
  --checkpoint.load_step -1 \
  --metrics.enable_wandb \
  --metrics.log_freq 1 \
  --experimental.context_parallel_degree 1 \
  --experimental.pipeline_parallel_degree 1 \
  --activation_checkpoint.mode selective \
  --activation_checkpoint.selective_ac_option 1
