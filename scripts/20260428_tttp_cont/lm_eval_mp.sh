#!/bin/bash
source "$(dirname "$0")/vars.sh"
echo $RUN_NAME
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 #1,3,5 #0,1,2,3,4,5,6,7 #0,1,3,4,5,6,7
# export BASE="${1:-/work/${USERNAME}}/radlads"
# export HF_CACHE_DIR="${BASE}/.cache/huggingface/hub"
export MAIN_PROCESS_PORT=29503






checkpoints=()
# checkpoints+=('init')
# start=0
# end=20
# stride=1
# for i in $(seq $start $stride $end); do
#     checkpoints+=("$i")
# done
# checkpoints+=('final')

# tasks=gsm8k,mmlu,lambada_openai,hellaswag

# tasks=gsm8k,winogrande,arc_easy,arc_challenge,hellaswag,piqa,openbookqa,lambada_openai,mmlu,race
tasks=winogrande,arc_easy,arc_challenge,hellaswag,piqa,openbookqa,lambada_openai,mmlu,race,wikitext #social_iqa


eval_hf_path=${dump_folder}

if [ ! -f "${eval_hf_path}/model.safetensors" ] && [ ! -f "${eval_hf_path}/pytorch_model.bin" ] && ! compgen -G "${eval_hf_path}/model-*.safetensors" > /dev/null; then
   echo "ERROR: ${eval_hf_path} does not look like an exported HuggingFace checkpoint." >&2
   echo "Expected model.safetensors, model-*.safetensors, or pytorch_model.bin." >&2
   exit 1
fi



accelerate launch --main_process_port ${MAIN_PROCESS_PORT} -m lm_eval \
   --model hf-custom \
   --model_args pretrained=${eval_hf_path},trust_remote_code=True,dtype=bfloat16,torch_dtype=bfloat16 \
   --tasks ${tasks} \
   --device cuda \
   --trust_remote_code \
   --batch_size 1 \
   --output_path $lm_eval_output_path/lm \
   --log_samples \
   --trust_remote_code

# # Below requires hugginface_custom.py to be updated to support yarn_max_seq_length args.
# accelerate launch --main_process_port ${MAIN_PROCESS_PORT} -m lm_eval \
#    --model hf-custom \
#    --model_args pretrained=${eval_hf_path},trust_remote_code=True,dtype=bfloat16,torch_dtype=bfloat16,max_length=131072,yarn_max_seq_length=65536 \
#    --tasks niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3 \
#    --metadata='{"max_seq_lengths":[65536]}' \
#    --device cuda \
#    --trust_remote_code \
#    --batch_size 1 \
#    --output_path $lm_eval_output_path/niah \
#    --log_samples \
#    --seed 1234
   
# accelerate launch --main_process_port ${MAIN_PROCESS_PORT} -m lm_eval \
#    --model hf-custom \
#    --model_args pretrained=${eval_hf_path},trust_remote_code=True,dtype=bfloat16,torch_dtype=bfloat16,max_length=131072,yarn_max_seq_length=131072 \
#    --tasks niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3 \
#    --metadata='{"max_seq_lengths":[131072]}' \
#    --device cuda \
#    --trust_remote_code \
#    --batch_size 1 \
#    --output_path $lm_eval_output_path/niah \
#    --log_samples \
#    --seed 1234





# python ~/exp.py --gpus ${CUDA_VISIBLE_DEVICES}

