CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 #0,1,3,4,5,6,7

export USERNAME=$(whoami)

flame_dir="."

# RUN_SUFFIX=${RUN_SUFFIX:-}
RUN_SUFFIX=${RUN_SUFFIX:-_ttt_chunk_1k}
RUN_NAME="$(basename "$(dirname "$0")")${RUN_SUFFIX}"



cd ${flame_dir}

MODEL_CONFIGS_DIR=configs
MODEL_NAME=${MODEL_NAME:-qwen3_tttip_0.6}
MODEL_CONFIG_PATH=${MODEL_CONFIGS_DIR}/${MODEL_NAME}.json
MODEL_TYPE=${MODEL_TYPE:-"tttip"}

base_model_hfac=${base_model_hfac:-"Qwen"}
base_model_name=${base_model_name:-"Qwen3-0.6B-Base"}
TOKENIZER_PATH="${base_model_hfac}/${base_model_name}"

lm_eval_output_path="/storage/backup/${USERNAME}/ttt/flame/results/${RUN_NAME}"
dump_folder=/storage/backup/${USERNAME}/ttt/flame/exp/${RUN_NAME}
checkpoint_folder=${dump_folder}/checkpoint

# Provide the following variables for continual pretraining over existing checkpoints; Don't provide them if you want to train from scratch
# These will be used to convert the HuggingFace checkpoint to the DCP format required for training
cont_pretrain_from=${cont_pretrain_from:-"Qwen/Qwen3-0.6B-Base"}
seed_root=${seed_root:-/storage/backup/hei/ttt/flame/seeds/qwen3_tttip_0B6}
seed_checkpoint_dir=${seed_checkpoint_dir:-${seed_root}/step-0}

seq_len=${seq_len:-32768}

