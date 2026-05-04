source "$(dirname "$0")/vars.sh"
PYTHONPATH=${flame_dir} python -m flame.utils.convert_hf_to_dcp \
  --model ${cont_pretrain_from} \
  --checkpoint ${seed_checkpoint_dir}