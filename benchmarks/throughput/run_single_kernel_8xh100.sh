#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  -m benchmarks.throughput.single_kernel \
  --models lact full_attention hybrid_swa hybrid_gdn \
  --seq-lens 4096 8192 16384 32768 65536 131072 262144 524288 1048576 \
  --batch-size 1 \
  --warmup-steps 5 \
  --steps 20 \
  --dtype bfloat16 \
  --device cuda \
  --use-fused-lact-kernel

