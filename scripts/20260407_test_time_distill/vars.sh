#!/bin/bash

export USERNAME="$(whoami)"

flame_dir="/work/yufei/projects/flame"
cd "${flame_dir}"

SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
RUN_NAME="$(basename "${SCRIPT_DIR}")"

BASE_MODEL_ID="/work/yufei/downloads/Qwen3-0.6B-Base"
eval_hf_path="${eval_hf_path:-${BASE_MODEL_ID}}"
lm_eval_output_path="/work/yufei/projects/flame/results/${RUN_NAME}"
