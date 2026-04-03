#!/bin/bash
source "$(dirname "$0")/vars.sh"
echo $RUN_NAME
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 #1,3,5 #0,1,2,3,4,5,6,7 #0,1,3,4,5,6,7
export PYTHONPATH="/work/yufei/projects/flame:${PYTHONPATH}"
# export PREFIX_LINEAR_ATTENTION_DIR="/work/yufei/projects/prefix-linear-attention"
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

# tasks=gsm8k,winogrande,arc_easy,arc_challenge,hellaswag,piqa,openbookqa,lambada_openai,mmlu,mathqa,race
tasks=winogrande,arc_easy,arc_challenge,hellaswag,piqa,openbookqa,lambada_openai,mmlu,race,wikitext #social_iqa


eval_hf_path=${dump_folder}


# tasks=winogrande,arc_easy,piqa,arc_challenge,mmlu,gsm8k,race #mathqa
accelerate launch --main_process_port ${MAIN_PROCESS_PORT} "$(dirname "$0")/lm_eval_with_custom_models.py" \
   --model hf \
   --model_args pretrained=${eval_hf_path},trust_remote_code=True,dtype=bfloat16,torch_dtype=bfloat16,max_length=16384 \
   --tasks ${tasks} \
   --device cuda \
   --trust_remote_code \
   --batch_size 1 \
   --output_path $lm_eval_output_path/lm \
   --log_samples


# jrt_tasks=based_swde,based_swde_twice,based_fda,based_fda_twice,based_squad,based_squad_twice,based_drop,based_drop_twice,based_nq_1024,based_nq_1024_twice,based_triviaqa,based_triviaqa_twice

# PYTHONPATH="${PREFIX_LINEAR_ATTENTION_DIR}/lm-eval-harness:${PYTHONPATH}" \
# accelerate launch --main_process_port $((MAIN_PROCESS_PORT + 1)) "$(dirname "$0")/lm_eval_jrt_with_custom_models.py" \
#    --model hf \
#    --model_args pretrained=${eval_hf_path},trust_remote_code=True,dtype=bfloat16,torch_dtype=bfloat16,max_length=16384 \
#    --tasks ${jrt_tasks} \
#    --device cuda \
#    --trust_remote_code \
#    --num_fewshot 0 \
#    --batch_size 1 \
#    --output_path $lm_eval_output_path/jrt \
#    --log_samples \
#    --context_length 1000 \
#    --answer_length 50 \
#    --cutting_context \
#    --seed 1234

# accelerate launch --main_process_port ${MAIN_PROCESS_PORT} -m lm_eval \
#    --model hf-custom \
#    --model_args pretrained=${eval_hf_path},trust_remote_code=True,dtype=bfloat16,torch_dtype=bfloat16,max_length=32768 \
#    --tasks longbench \
#    --trust_remote_code \
#    --device cuda \
#    --num_fewshot 0 \
#    --batch_size 1 \
#    --output_path $lm_eval_output_path \
#    --log_samples \
#    --seed 1234

# accelerate launch --main_process_port ${MAIN_PROCESS_PORT} -m lm_eval \
#    --model hf-custom \
#    --model_args pretrained=${eval_hf_path},trust_remote_code=True,dtype=bfloat16,torch_dtype=bfloat16,max_length=32768 \
#    --tasks niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3 \
#    --metadata='{"max_seq_lengths":[4096,8192,16384,32768]}' \
#    --device cuda \
#    --trust_remote_code \
#    --batch_size 1 \
#    --output_path $lm_eval_output_path \
#    --log_samples \
#    --seed 1234
#    # --gen_kwargs '{"max_new_tokens": 128}' \




# python ~/exp.py --gpus ${CUDA_VISIBLE_DEVICES}
