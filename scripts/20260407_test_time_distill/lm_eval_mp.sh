#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/vars.sh"
echo "${RUN_NAME}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

export PYTHONPATH="/work/yufei/projects/flame:/work/yufei/projects/flame/scripts/20260407_test_time_distill:${PYTHONPATH:-}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29503}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

eval_hf_path="${eval_hf_path:-${BASE_MODEL_ID}}"
mkdir -p "${lm_eval_output_path}"
USE_TTT_DISTILL="${USE_TTT_DISTILL:-1}"

TASKS="${TASKS:-longbench}"
MAX_LENGTH="${MAX_LENGTH:-16384}"
TTT_CHUNK_SIZE="${TTT_CHUNK_SIZE:-1024}"
TTT_UPDATE_MODE="${TTT_UPDATE_MODE:-full_prefix_approx}"
TTT_LORA_R="${TTT_LORA_R:-64}"
TTT_LORA_ALPHA="${TTT_LORA_ALPHA:-${TTT_LORA_R}}"
TTT_LR="${TTT_LR:-3e-4}"
TTT_NUM_QA_CANDIDATES="${TTT_NUM_QA_CANDIDATES:-4}"
TTT_NUM_JUDGE_CANDIDATES="${TTT_NUM_JUDGE_CANDIDATES:-3}"
TTT_NUM_SELECTED_QA="${TTT_NUM_SELECTED_QA:-2}"
TTT_QA_GENERATION_MAX_NEW_TOKENS="${TTT_QA_GENERATION_MAX_NEW_TOKENS:-128}"
TTT_QA_JUDGE_MAX_NEW_TOKENS="${TTT_QA_JUDGE_MAX_NEW_TOKENS:-16}"
TTT_LONGBENCH_SAMPLES_BASE="${TTT_LONGBENCH_SAMPLES_BASE:-/work/yufei/projects/flame/results/20260322_hybrid_qwen3_lact_0p6B_swa_2k_chunk_1k_rerun12_prolong_prolong_from_run12_step9535_v4/lb/__storage__backup__yufei__ttt__flame__exp__20260322_hybrid_qwen3_lact_0p6B_swa_2k_chunk_1k_rerun12_prolong_prolong_from_run12_step9535_v4}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEED="${SEED:-1234}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-lb}"
OUTPUT_TAG="${OUTPUT_TAG:-${TASKS}_distill_chunk${TTT_CHUNK_SIZE}_max${MAX_LENGTH}}"
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

MODEL_SCRIPT="${SCRIPT_DIR}/lm_eval_ttt_lora.py"
MODEL_NAME="hf-test-time-distill"
MODEL_ARGS="pretrained=${eval_hf_path},trust_remote_code=True,dtype=bfloat16,torch_dtype=bfloat16,max_length=${MAX_LENGTH},ttt_chunk_size=${TTT_CHUNK_SIZE},ttt_update_mode=${TTT_UPDATE_MODE},ttt_lora_r=${TTT_LORA_R},ttt_lora_alpha=${TTT_LORA_ALPHA},ttt_lr=${TTT_LR},ttt_beta1=0.9,ttt_beta2=0.95,ttt_weight_decay=0.0,ttt_grad_clip=1.0,ttt_num_qa_candidates=${TTT_NUM_QA_CANDIDATES},ttt_num_judge_candidates=${TTT_NUM_JUDGE_CANDIDATES},ttt_num_selected_qa=${TTT_NUM_SELECTED_QA},ttt_qa_generation_max_new_tokens=${TTT_QA_GENERATION_MAX_NEW_TOKENS},ttt_qa_judge_max_new_tokens=${TTT_QA_JUDGE_MAX_NEW_TOKENS},ttt_longbench_samples_base=${TTT_LONGBENCH_SAMPLES_BASE},ttt_log_path=${lm_eval_output_path}/longbench_test_time_distill_chunks.jsonl"

if [[ "${USE_TTT_DISTILL}" == "0" ]]; then
  MODEL_ARGS="pretrained=${eval_hf_path},trust_remote_code=True,dtype=bfloat16,torch_dtype=bfloat16,max_length=${MAX_LENGTH},ttt_enable=0"
fi

echo "Resolved model: ${eval_hf_path}"
echo "Use test-time distill: ${USE_TTT_DISTILL}"
echo "Resolved tasks: ${TASKS}"
echo "Resolved distill rank/alpha/lr: ${TTT_LORA_R}/${TTT_LORA_ALPHA}/${TTT_LR}"
echo "Resolved distill QA setup: candidates=${TTT_NUM_QA_CANDIDATES} judge_candidates=${TTT_NUM_JUDGE_CANDIDATES} selected=${TTT_NUM_SELECTED_QA}"

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
