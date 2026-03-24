runs=(20260305_lact_stage1_bz1M_tied 20260307_lact_nolact-fa-swa_stage1 20260307_lact_nolact-fa_stage1)
for run in "${runs[@]}"; do
  bash scripts/${run}/gen_loss_via_lm.sh
done