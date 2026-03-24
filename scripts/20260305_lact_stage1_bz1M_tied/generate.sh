source "$(dirname "$0")/vars.sh"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
python scripts/generate.py \
  --model ${dump_folder} \
  --use_cache \
  --max_new_tokens 3072 \
  --repetition_penalty 1.1 \
  --do_sample \
  --top_p 0.95 \
  --temperature 1 \
  --prompt "Once upon a time"

#   --skip_special_tokens
