CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export USERNAME=$(whoami)

flame_dir="/work/yufei/projects/flame"
cd ${flame_dir}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_NAME=$(basename "${SCRIPT_DIR}")_swa_2k_chunk_1k_rerun9
export WANDB_PROJECT=fla
export WANDB_NAME=${RUN_NAME}
export WANDB_RUN_ID=${RUN_NAME}
export WANDB_RESUME=allow

MODEL_CONFIG_PATH=configs/qwen3_hybrid_qwen3_lact_0p6B.json
BASE_MODEL_ID=Qwen/Qwen3-0.6B-Base
HYBRID_HF_DIR=${flame_dir}/checkpoints/qwen3_hybrid_qwen3_lact_0p6B_init

TOKENIZER_PATH=${HYBRID_HF_DIR}

dump_folder=/storage/backup/${USERNAME}/ttt/flame/exp/${RUN_NAME}
checkpoint_folder=${dump_folder}/checkpoint
seed_root=/storage/backup/${USERNAME}/ttt/flame/seeds/qwen3_hybrid_qwen3_lact_0p6B
seed_checkpoint_dir=${seed_root}/step-0
lm_eval_output_path=/work/yufei/projects/flame/results/${RUN_NAME}

seq_len=4096
batch_size=4
grad_accum=4
