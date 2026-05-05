#!/bin/bash
source "$(dirname "$0")/vars.sh"
# source $(conda info --base)/etc/profile.d/conda.sh
# conda activate base+

RUN_NAME=Qwen__Qwen3-4B-Base
RUN_NAME=20260426_notttip_untrained
python gen_results_csv.py \
    --data_set lm \
    --run_name ${RUN_NAME}
python gen_results_csv.py \
    --data_set lb \
    --run_name ${RUN_NAME}
python gen_results_csv.py \
    --data_set niah \
    --run_name ${RUN_NAME}
