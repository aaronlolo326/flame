source "$(dirname "$0")/vars.sh"
echo $RUN_NAME
local_rank=$(echo ${available_nodes[@]} | tr ' ' '\n' | grep -n "^${THIS_NODE}$" | cut -d: -f1)
local_rank=$((local_rank - 1))
echo "THIS_NODE=${THIS_NODE}; local_rank=$local_rank"

MASTER_ADDR=192.168.241.41
MASTER_PORT=29500

NNODE=${#available_nodes[@]}
if [ ${NNODE} -eq 1 ]; then
  MASTER_ADDR=localhost
  MASTER_PORT=29500
fi

debug=false
profile=false

batch_size=1
seq_len=4096
grad_accum=16
no_tokens=$(( 15 * 10**9 ))

if [ "$debug" = true ]; then
  CUDA_VISIBLE_DEVICES=0
  batch_size=8
fi

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
  NGPU=1
else
  NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
fi

steps=$(( no_tokens / NGPU / seq_len / batch_size / grad_accum ))
echo steps=$steps
if $profile; then
  steps=10
fi


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
  --optimizer.name AdamW \
  --optimizer.eps 1e-15 \
  --optimizer.lr 3e-4 \
  --lr_scheduler.warmup_steps 1024 \
  --lr_scheduler.lr_min 0.1 \
  --lr_scheduler.decay_type cosine \
  --training.batch_size ${batch_size} \
  --training.seq_len ${seq_len} \
  --training.varlen \
  --training.context_len ${seq_len} \
  --training.gradient_accumulation_steps ${grad_accum} \
  --training.steps $steps \
  --training.max_norm 1.0 \
  --training.skip_nan_inf \
  --training.tokenized_dataset_dir /storage/backup/hei/data/HuggingFaceFW___fineweb-edu___sample-350BT \
  --training.num_workers 32 \
  --training.prefetch_factor 2 \
  --training.seed 42 \
  --training.data_parallel_shard_degree 1 \
  --training.data_parallel_replicate_degree 8 \
  --training.disable_loss_parallel \
  --checkpoint.interval 2048 \
  --checkpoint.load_step -1 \
  --metrics.log_freq 1
  # --profiling.enable_memory_snapshot \
  # --profiling.profile_freq 4

  # --training.streaming \
  # HuggingFaceFW/fineweb-edu \
  # --training.dataset_name sample-350BT \
  #  --training.num_workers 192 \

   # --training.dataset arrow \
  # --training.dataset_split train \
  # --training.data_dir /nfs-export/hei/.cache/huggingface/datasets/HuggingFaceFW___fineweb-edu/sample-350BT \