source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/vars.sh"

PYTHONPATH=${flame_dir} python scripts/param_cnt.py --model.config ${MODEL_CONFIG_PATH}
