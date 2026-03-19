source "$(dirname "$0")/vars.sh"
echo $RUN_NAME

debug=true
profile=false

batch_size=2
seq_len=2048
grad_accum=8
no_tokens=$(( 50 * 10**9 ))

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


NCCL_DEBUG=INFO CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
NNODE=1 NGPU=${NGPU} LOG_RANK=0 bash train.sh \
  --job.config_file flame/models/fla_20260212.toml \
  --job.dump_folder ${dump_folder} \
  --comm.init_timeout_seconds 3600 \
  --comm.train_timeout_seconds 600 \
  --model.config ${MODEL_CONFIG_PATH} \
  --model.tokenizer_path ${TOKENIZER_PATH} \
  --optimizer.name AdamW \
  --optimizer.eps 1e-15 \
  --optimizer.lr 3e-4 \
  --lr_scheduler.warmup_steps 1024 \
  --lr_scheduler.lr_min 0.1 \
  --lr_scheduler.decay_type cosine \
  --training.batch_size ${batch_size} \
  --training.seq_len ${seq_len} \
  --training.context_len ${seq_len} \
  --training.gradient_accumulation_steps ${grad_accum} \
  --training.steps $steps \
  --training.max_norm 1.0 \
  --training.skip_nan_inf \
  --training.tokenized_dataset_dir /storage/backup/hei/data/HuggingFaceFW___fineweb-edu___sample-350BT \
  --training.num_workers 32 \
  --training.prefetch_factor 2 \
  --training.seed 42 \
  --training.compile \
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