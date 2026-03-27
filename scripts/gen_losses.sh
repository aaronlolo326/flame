export USERNAME=$(whoami)
runs=(
    # "20260305_lact_stage1_bz1M_tied"
    # "20260307_lact_nolact-fa-swa_stage1"
    # "20260309_lact_nolact-fa_stage1"
    # "20260320_lact_75lact25fa_stage1"
    # "20260321_lact_fullrank_stage1"
    "20260325_lact_nolact-swa"
)
umask=0000
prefill_len=256
for run in "${runs[@]}"; do
    dump_folder=/storage/backup/hei/ttt/flame/exp/${run}
    echo ${run}
    echo ${dump_folder} 
    python scripts/gen_loss_via_lm.py \
      --data_path "/storage/backup/hei/data/qwen3-dclm-filter-16k_train" \
      --model_path "$dump_folder" \
      --ref_lm "Qwen3-8B-Base" \
      --prefill_len ${prefill_len} \
      --decode_len 8192 \
      --num_samples 32 \
      --do_sample 1 \
      --repetition_penalty 1.1 \
      --top_p 0.95 \
      --top_k 50 \
      --temperature 1 \
      --seed 42 \
      --trust_remote_code \
      --output_dir "/storage/backup/${USERNAME}/ttt/flame/results/${run}/loss_${prefill_len}_20260326"

done