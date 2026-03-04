#!/bin/bash
source "$(dirname "$0")/vars.sh"
# source $(conda info --base)/etc/profile.d/conda.sh
# conda activate base+

python plot.py \
    --data_set lm \
    --run_name ${RUN_NAME}
python plot.py \
    --data_set lb \
    --run_name ${RUN_NAME}
python plot.py \
    --data_set niah \
    --run_name ${RUN_NAME}
