source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/vars.sh"

PYTHONPATH=${flame_dir} python scripts/report_hybrid_qwen3_lact.py \
  --model-config ${MODEL_CONFIG_PATH} \
  --meta
