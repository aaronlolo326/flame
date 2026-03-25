source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/vars.sh"

set -euo pipefail

# PYTHONPATH=${flame_dir} python scripts/make_hybrid_qwen3_lact_config.py \
#   --src ${BASE_MODEL_ID} \
#   --out ${MODEL_CONFIG_PATH} \
#   --num-lact-heads 4 \
#   --lact-chunk-size 1024 \
#   --window-size 2048

# PYTHONPATH=${flame_dir} python scripts/convert_qwen3_to_hybrid_lact.py \
#   --src ${BASE_MODEL_ID} \
#   --hybrid-config ${MODEL_CONFIG_PATH} \
#   --out ${HYBRID_HF_DIR}

PYTHONPATH=${flame_dir} python -m flame.utils.convert_hf_to_dcp \
  --model ${HYBRID_HF_DIR} \
  --checkpoint ${seed_checkpoint_dir}

echo "Created hybrid HF checkpoint at ${HYBRID_HF_DIR}"
echo "Created DCP seed checkpoint at ${seed_checkpoint_dir}"
