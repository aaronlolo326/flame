source "$(dirname "$0")/vars.sh"

umask 0000

python scripts/gen_loss_via_lm.py \
  --data_path "/storage/backup/hei/data/qwen3-dclm-filter-16k_train" \
  --model_path "$dump_folder" \
  --ref_lm "Qwen3-8B-Base" \
  --prefill_len 8192 \
  --decode_len 8192 \
  --num_samples 20 \
  --do_sample 1 \
  --repetition_penalty 1.1 \
  --top_p 0.95 \
  --temperature 1 \
  --seed 42 \
  --trust_remote_code \
  --output_dir "/storage/backup/${USERNAME}/ttt/flame/results/${RUN_NAME}/loss"

# python scripts/gen_loss_via_lm.py \
#   --data_path "/storage/backup/hei/data/qwen3-dclm-filter-16k_train" \
#   --model_path "$dump_folder" \
#   --ref_lm "Qwen3-8B-Base" \
#   --prefill_len 8192 \
#   --decode_len 8192 \
#   --num_samples 512 \
#   --do_sample \
#   --repetition_penalty 1.1 \
#   --top_p 0.95 \
#   --temperature 1 \
#   --seed 42 \
#   --trust_remote_code \
#   --output_path "/storage/backup/${USERNAME}/ttt/flame/results/${RUN_NAME}/gen_loss.npy"
