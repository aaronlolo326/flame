source "$(dirname "$0")/vars.sh"


MODEL_CONFIGS_DIR=configs
# MODEL_NAME=qwen3_lact_1B3
MODEL_NAME=qwen3_lact_1B3
MODEL_CONFIG_PATH=${MODEL_CONFIGS_DIR}/${MODEL_NAME}.json

python scripts/param_cnt.py --model.config ${MODEL_CONFIG_PATH}