"""Minimal script for ncu profiling of batched vs varlen swiglu kernel.

Usage:
    CUDA_VISIBLE_DEVICES=7 ncu --launch-skip-before-match 0 --kernel-name regex:"_fused_two_mm_swiglu" --launch-count 2 -o swiglu_profile python ncu_swiglu.py
"""
import torch
from triton_swiglu_kernels import fused_two_mm_swiglu_triton, fused_two_mm_swiglu_varlen_triton
from grouped_gemm import _pack_to_padded

device = "cuda"
num_docs = 4
doc_lens = [4096, 3072, 2048, 1024]
d, dh = 512, 512
cu_seqlens = torch.tensor(
    [0] + list(torch.cumsum(torch.tensor(doc_lens), 0).tolist()),
    dtype=torch.int32, device=device,
)
packed_len = cu_seqlens[-1].item()

W0_W2 = torch.randn(num_docs, 2 * dh, d, device=device, dtype=torch.bfloat16)
X = torch.randn(packed_len, d, device=device, dtype=torch.bfloat16)
X_pad = _pack_to_padded(X, cu_seqlens, max(doc_lens))

# Warmup (triggers autotune)
for _ in range(200):
    fused_two_mm_swiglu_varlen_triton(W0_W2, X, cu_seqlens)
    fused_two_mm_swiglu_triton(W0_W2, X_pad)
torch.cuda.synchronize()

# These are the two runs ncu will capture
print(">>> varlen")
out_varlen = fused_two_mm_swiglu_varlen_triton(W0_W2, X, cu_seqlens)
torch.cuda.synchronize()

print(">>> batched")
out_batched = fused_two_mm_swiglu_triton(W0_W2, X_pad)
torch.cuda.synchronize()
