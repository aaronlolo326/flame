CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 #0,1,3,4,5,6,7

export USERNAME=$(whoami)

# export LACT_LOG_VARLEN_SEG_STATS=1

# export LACT_LOG_VARLEN_SEG_STATS_EVERY=10
flame_dir="/work/mingze/flame"

RUN_NAME=$(basename "$(dirname "$0")")_4k_data_with_sep_culen



cd ${flame_dir}

MODEL_CONFIGS_DIR=configs
MODEL_NAME=qwen3_lact_1B4
MODEL_CONFIG_PATH=${MODEL_CONFIGS_DIR}/${MODEL_NAME}.json

base_model_hfac="Qwen"
base_model_name="Qwen3-1.7B-Base"
TOKENIZER_PATH="${base_model_hfac}/${base_model_name}"

lm_eval_output_path="results/${RUN_NAME}"
dump_folder=/work/mingze/flame/exp/${RUN_NAME}