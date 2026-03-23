source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/vars.sh"

mkdir -p ${dump_folder}/phase_b

PYTHONPATH=${flame_dir} python scripts/fit_hybrid_qwen3_lact_phase_b.py \
  --teacher ${BASE_MODEL_ID} \
  --hybrid-model ${HYBRID_HF_DIR} \
  --max-length 256 \
  --out ${dump_folder}/phase_b/kv_ridge_stats.pt
