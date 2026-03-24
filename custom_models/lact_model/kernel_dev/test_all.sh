#!/bin/bash
export CUDA_VISIBLE_DEVICES=6
cd "$(dirname "$0")"

python grouped_gemm.py
python triton_swiglu_kernels.py
python triton_swiglu_bwd_with_lr.py
python triton_pointwise_kernels.py
python triton_fused_matmul_kernels.py
python lact_swiglu_ffn.py
python lact_fw_grad.py
