base_model_hfac="Qwen"
base_model_name="Qwen3-1.7B-Base"
TOKENIZER_PATH="${base_model_hfac}/${base_model_name}"

# python flame/flame/utils/preprocess.py \
#   --dataset arrow \
#   --data_dir /nfs-export/hei/.cache/huggingface/datasets/HuggingFaceFW___fineweb-edu/sample-350BT \
#   --num_workers 256 \
#   --path /nfs-export/hei/data/HuggingFaceFW___fineweb-edu___sample-350BT \
#   --tokenizer ${TOKENIZER_PATH}

python flame/flame/utils/preprocess.py \
  --dataset HuggingFaceFW/fineweb-edu \
  --dataset_name sample-350BT \
  --num_workers 192 \
  --path /storage/backup/hei/data/HuggingFaceFW___fineweb-edu___sample-350BT \
  --tokenizer ${TOKENIZER_PATH}
