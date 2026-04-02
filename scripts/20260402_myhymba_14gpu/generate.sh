source "$(dirname "$0")/vars.sh"

python scripts/generate.py \
  --model ${dump_folder} \
  --max_new_tokens 64 \
  --do_sample \
  --temperature 1 \
  --repetition_penalty 1.1 \
  --top_p 0.95 \
  --top_k 50 \
  --use_cache false \
  --prompt "Once upon a time"

#   --skip_special_tokens
python scripts/generate.py \
  --model ${dump_folder} \
  --max_new_tokens 64 \
  --do_sample \
  --temperature 1 \
  --repetition_penalty 1.1 \
  --top_p 0.95 \
  --top_k 50 \
  --use_cache true \
  --prompt "Once upon a time"