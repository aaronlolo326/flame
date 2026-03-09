CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 #0,1,3,4,5,6,7

export USERNAME=$(whoami)

flame_dir="."

RUN_NAME=$(basename "$(dirname "$0")")



cd ${flame_dir}

MODEL_CONFIGS_DIR=configs
MODEL_NAME=qwen3_lact_1B4_nolact-fa-swa
MODEL_CONFIG_PATH=${MODEL_CONFIGS_DIR}/${MODEL_NAME}.json

base_model_hfac="Qwen"
base_model_name="Qwen3-1.7B-Base"
TOKENIZER_PATH="${base_model_hfac}/${base_model_name}"

lm_eval_output_path="/storage/backup/${USERNAME}}/ttt/flame/results/${RUN_NAME}"
dump_folder=/storage/backup/${USERNAME}/ttt/flame/exp/${RUN_NAME}

seq_len=16384