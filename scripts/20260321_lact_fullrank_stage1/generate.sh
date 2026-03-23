source "$(dirname "$0")/vars.sh"

python scripts/generate.py \
  --model ${dump_folder} \
  --max_new_tokens 64 \
  --repetition_penalty 1.1 \
  --do_sample \
  --top_p 0.95 \
  --temperature 1 \
  --prompt "Once upon a time"

#   --skip_special_tokens
