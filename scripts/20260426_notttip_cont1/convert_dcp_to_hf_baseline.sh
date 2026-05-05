source "$(dirname "$0")/vars.sh"

# if [ -z "${seed_checkpoint_dir}" ]; then
#     echo "seed_checkpoint_dir is not set. Training from scratch ..."
# elif [ ! -d "${seed_checkpoint_dir}" ]; then
#     echo "Warning: seed_checkpoint_dir is not set or does not exist: ${seed_checkpoint_dir}"
#     echo "Proceeding with training from scratch ..."
# else
#     mkdir -p "${checkpoint_folder}"
#     chmod 777 "${checkpoint_folder}" || true
#     step0_checkpoint_dir="${checkpoint_folder}/step-0"
#     mkdir -p "${step0_checkpoint_dir}"
#     cp -a "${seed_checkpoint_dir}/." "${step0_checkpoint_dir}/"
#     chmod -R 777 "${step0_checkpoint_dir}"
#     chmod -R 777 "${dump_folder}"
# fi

dump_folder_baseline=/storage/backup/${USERNAME}/ttt/flame/exp/20260426_notttip_untrained

cp -r $dump_folder $dump_folder_baseline

python -m flame.utils.convert_dcp_to_hf \
    --path ${dump_folder_baseline} \
    --step 0 \
    --config ${MODEL_CONFIG_PATH} \
    --tokenizer ${TOKENIZER_PATH}