source "$(dirname "$0")/vars.sh"

debug=false
profile=false

available_nodes=(9 6) # first is master node
# based on env var $THIS_NODE, set local rank according to the order, e.g., THIS_NODE is 0 means local_rank = 0
local_rank=$(echo ${available_nodes[@]} | tr ' ' '\n' | grep -n "^${THIS_NODE}$" | cut -d: -f1)
local_rank=$((local_rank - 1))
echo "THIS_NODE=${THIS_NODE}; local_rank=$local_rank"

MASTER_ADDR=192.168.240.118 # 192.168.240.118 for node 9; 192.168.240.169  for node 1
MASTER_PORT=29501

NNODE=${#available_nodes[@]}
if [ ${NNODE} -eq 1 ]; then
  MASTER_ADDR=localhost
  MASTER_PORT=29500
fi


echo $RUN_NAME


batch_size=4
grad_accum=1
no_tokens=$(( 20 * 10**9 ))
seed=42

if [[ $local_rank == 0 ]]; then
  if [ -z "${seed_checkpoint_dir}" ]; then
    echo "seed_checkpoint_dir is not set. Training from scratch ..."
  elif [ ! -d "${seed_checkpoint_dir}" ]; then
    echo "Warning: seed_checkpoint_dir is not set or does not exist: ${seed_checkpoint_dir}"
    echo "Proceeding with training from scratch ..."
  else
    mkdir -p "${checkpoint_folder}"
    chmod 777 "${checkpoint_folder}" || true
    step0_checkpoint_dir="${checkpoint_folder}/step-0"
    mkdir -p "${step0_checkpoint_dir}"
    cp -a "${seed_checkpoint_dir}/." "${step0_checkpoint_dir}/"
    chmod -R 777 "${step0_checkpoint_dir}"
    chmod -R 777 "${dump_folder}"
  fi
fi

if [ "$debug" = true ]; then
  CUDA_VISIBLE_DEVICES=0
  batch_size=1
fi

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
  NGPU=1
else
  NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
fi

NGPU_TOTAL=$(( NNODE * NGPU ))

data_parallel_replicate_degree=1
if [ "${MODEL_TYPE}" = "lact" ]; then
  data_parallel_replicate_degree=1
elif [ "${MODEL_TYPE}" = "e2e" ]; then
  data_parallel_replicate_degree=${NGPU_TOTAL}
fi


steps=$(( no_tokens / NGPU_TOTAL / seq_len / batch_size / grad_accum ))
warmup_steps=$(( steps / 50 )) # 1024
interval=$(( steps / 4 ))
echo steps=$steps
if $profile; then
  steps=10
fi

### Data; Make sure to check if an EOS token needs to be appended to each sample in the targetted datasets ###
add_eos_datasets=(
  "/storage/backup/hei/data/HuggingFaceFW___fineweb-edu___sample-350BT"
) # tokenized_dataset_dir that requires eos append; Add more dataset paths to this array if needed

tokenized_dataset_dir="/storage/backup/hei/data/xfxcwynlc__prolong1-qwen3-8b-tokenized-32768_arrow/train,/storage/backup/hei/data/xfxcwynlc__prolong2-qwen3-8b-tokenized-32768_arrow/train"
add_eos_token_to_sample=0
for dir in "${add_eos_datasets[@]}"; do
  if [[ "$tokenized_dataset_dir" == "$dir" ]]; then
    add_eos_token_to_sample=1
    break
  fi
done
###

umask 0000

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
  --optimizer.lr 5e-6 \
  --optimizer.weight_decay 0.1 \
  --lr_scheduler.warmup_steps ${warmup_steps} \
  --lr_scheduler.decay_ratio 0.90 \
  --training.batch_size ${batch_size} \
  --training.seq_len ${seq_len} \
  --training.context_len ${seq_len} \
  --training.sample_trunc_seq 1 \
  --training.add_eos_token_to_sample ${add_eos_token_to_sample} \
  --training.gradient_accumulation_steps ${grad_accum} \
  --training.steps $steps \
  --training.max_norm 1.0 \
  --training.skip_nan_inf \
  --training.tokenized_dataset_dir ${tokenized_dataset_dir} \
  --training.data_probs 0.5,0.5 \
  --training.data_mix_stopping_strategy first_exhausted \
  --training.data_format arrow \
  --training.dataset_split train \
  --training.num_workers 32 \
  --training.prefetch_factor 2 \
  --training.seed ${seed} \
  --training.compile \
  --training.disable_loss_parallel \
  --training.data_parallel_replicate_degree ${data_parallel_replicate_degree} \
  --training.data_parallel_shard_degree -1 \
  --training.tensor_parallel_degree 1 \
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

  

  # --job.config_file flame/models/fla_20260212.toml \

  # --profiling.enable_profiling \
  # --profiling.save_traces_folder = "profile_trace" \
  # --profiling.profile_freq 512
  # --profiling.enable_memory_snapshot \

  # --training.streaming \
  # HuggingFaceFW/fineweb-edu \
  # --training.dataset_name sample-350BT \
  #  --training.num_workers 192 \

   # --training.dataset arrow \
  # --training.dataset_split train \
  # --training.data_dir /nfs-export/hei/.cache/huggingface/datasets/HuggingFaceFW___fineweb-edu/sample-350BT \
