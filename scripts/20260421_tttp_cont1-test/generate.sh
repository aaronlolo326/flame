source "$(dirname "$0")/vars.sh"

prompt_file=scripts/${RUN_NAME}/prompt.txt
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
python scripts/generate.py \
  --model ${dump_folder} \
  --use_cache yes \
  --max_input_len 3000 \
  --max_new_tokens 1000 \
  --prompt_file ${prompt_file} \
  --repetition_penalty 1.1 \
  --do_sample \
  --top_p 0.95 \
  --temperature 1

#   --skip_special_tokens
