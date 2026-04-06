#!/bin/bash
set -euo pipefail

source "$(dirname "$0")/vars.sh"
echo "${RUN_NAME}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

export PYTHONPATH="/work/yufei/projects/flame:/work/yufei/projects/flame/scripts/20260330_ttt_lora:${PYTHONPATH:-}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29503}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

eval_hf_path="${eval_hf_path:-${BASE_MODEL_ID}}"
mkdir -p "${lm_eval_output_path}"
USE_TTT_LORA="${USE_TTT_LORA:-1}"

TASKS="${TASKS:-longbench}"
MAX_LENGTH="${MAX_LENGTH:-16384}"
TTT_CHUNK_SIZE="${TTT_CHUNK_SIZE:-512}"
TTT_STEPS_PER_CHUNK="${TTT_STEPS_PER_CHUNK:-1}"
TTT_UPDATE_MODE="${TTT_UPDATE_MODE:-local_window}"
TTT_LOCAL_TRAIN_WINDOW="${TTT_LOCAL_TRAIN_WINDOW:-2048}"
TTT_LORA_R="${TTT_LORA_R:-64}"
TTT_LORA_ALPHA="${TTT_LORA_ALPHA:-${TTT_LORA_R}}"
TTT_LR="${TTT_LR:-1e-5}"
TTT_LOSS_MODE="${TTT_LOSS_MODE:-topk_fraction}"
TTT_LOSS_TOPK_FRACTION="${TTT_LOSS_TOPK_FRACTION:-0.2}"
TTT_RAW_CHARS_PER_TOKEN="${TTT_RAW_CHARS_PER_TOKEN:-4.0}"
TTT_RAW_TRUNC_SAFETY_MARGIN="${TTT_RAW_TRUNC_SAFETY_MARGIN:-1.1}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEED="${SEED:-1234}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-lb}"
OUTPUT_TAG="${OUTPUT_TAG:-${TASKS}_chunk${TTT_CHUNK_SIZE}_max${MAX_LENGTH}}"
SKIP_SUMMARIZATION="${SKIP_SUMMARIZATION:-0}"

case "${TASKS}" in
  code_completion)
    TASKS="longbench_code"
    OUTPUT_TAG="${OUTPUT_TAG:-longbench_code_chunk${TTT_CHUNK_SIZE}_max${MAX_LENGTH}}"
    ;;
  few_shot_learning)
    TASKS="longbench_fewshot"
    OUTPUT_TAG="${OUTPUT_TAG:-longbench_fewshot_chunk${TTT_CHUNK_SIZE}_max${MAX_LENGTH}}"
    ;;
  multi_document_qa)
    TASKS="longbench_multi"
    OUTPUT_TAG="${OUTPUT_TAG:-longbench_multi_chunk${TTT_CHUNK_SIZE}_max${MAX_LENGTH}}"
    ;;
  single_document_qa)
    TASKS="longbench_single"
    OUTPUT_TAG="${OUTPUT_TAG:-longbench_single_chunk${TTT_CHUNK_SIZE}_max${MAX_LENGTH}}"
    ;;
  summarization)
    TASKS="longbench_summarization"
    OUTPUT_TAG="${OUTPUT_TAG:-longbench_summarization_chunk${TTT_CHUNK_SIZE}_max${MAX_LENGTH}}"
    ;;
  synthetic_tasks)
    TASKS="longbench_synthetic"
    OUTPUT_TAG="${OUTPUT_TAG:-longbench_synthetic_chunk${TTT_CHUNK_SIZE}_max${MAX_LENGTH}}"
    ;;
esac

if [[ "${TASKS}" == "longbench" && "${SKIP_SUMMARIZATION}" == "1" ]]; then
  TASKS="longbench_2wikimqa,longbench_dureader,longbench_hotpotqa,longbench_lcc,longbench_lsht,longbench_multifieldqa_en,longbench_multifieldqa_zh,longbench_musique,longbench_narrativeqa,longbench_passage_count,longbench_passage_retrieval_en,longbench_passage_retrieval_zh,longbench_qasper,longbench_repobench-p,longbench_samsum,longbench_trec,longbench_triviaqa"
  OUTPUT_TAG="${OUTPUT_TAG:-longbench_no_summarization_chunk${TTT_CHUNK_SIZE}_max${MAX_LENGTH}}"
fi

mkdir -p "${lm_eval_output_path}/${OUTPUT_SUBDIR}"
echo "${OUTPUT_TAG}" > "${lm_eval_output_path}/${OUTPUT_SUBDIR}/run_tag.txt"

MODEL_SCRIPT="$(dirname "$0")/lm_eval_ttt_lora.py"
MODEL_NAME="hf-ttt-lora"
MODEL_ARGS="pretrained=${eval_hf_path},trust_remote_code=True,dtype=bfloat16,torch_dtype=bfloat16,max_length=${MAX_LENGTH},ttt_chunk_size=${TTT_CHUNK_SIZE},ttt_steps_per_chunk=${TTT_STEPS_PER_CHUNK},ttt_update_mode=${TTT_UPDATE_MODE},ttt_local_train_window=${TTT_LOCAL_TRAIN_WINDOW},ttt_lora_r=${TTT_LORA_R},ttt_lora_alpha=${TTT_LORA_ALPHA},ttt_loss_mode=${TTT_LOSS_MODE},ttt_loss_topk_fraction=${TTT_LOSS_TOPK_FRACTION},ttt_lr=${TTT_LR},ttt_beta1=0.9,ttt_beta2=0.95,ttt_weight_decay=0.0,ttt_grad_clip=1.0,ttt_raw_chars_per_token=${TTT_RAW_CHARS_PER_TOKEN},ttt_raw_trunc_safety_margin=${TTT_RAW_TRUNC_SAFETY_MARGIN},ttt_log_path=${lm_eval_output_path}/longbench_ttt_lora_chunks.jsonl"

if [[ "${USE_TTT_LORA}" == "0" ]]; then
  MODEL_ARGS="pretrained=${eval_hf_path},trust_remote_code=True,dtype=bfloat16,torch_dtype=bfloat16,max_length=${MAX_LENGTH},ttt_enable=0"
fi

echo "Resolved model: ${eval_hf_path}"
echo "Use TTT LoRA: ${USE_TTT_LORA}"
echo "Resolved tasks: ${TASKS}"
echo "Resolved TTT LoRA rank/alpha/lr: ${TTT_LORA_R}/${TTT_LORA_ALPHA}/${TTT_LR}"

accelerate launch --main_process_port "${MAIN_PROCESS_PORT}" "${MODEL_SCRIPT}" \
   --model "${MODEL_NAME}" \
   --model_args "${MODEL_ARGS}" \
   --tasks "${TASKS}" \
   --trust_remote_code \
   --device cuda \
   --num_fewshot "${NUM_FEWSHOT}" \
   --batch_size "${BATCH_SIZE}" \
   --output_path "${lm_eval_output_path}/${OUTPUT_SUBDIR}" \
   --log_samples \
   --seed "${SEED}"
