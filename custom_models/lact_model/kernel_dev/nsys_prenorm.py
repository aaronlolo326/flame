"""Minimal script for nsys profiling of varlen vs padded prenorm chunk loop (fwd+bwd).

Usage:
    cd custom_models/lact_model
    CUDA_VISIBLE_DEVICES=7 nsys profile -o prenorm_fwdbwd python kernel_dev/nsys_prenorm.py
"""
import torch
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ttt_operation_fused_kernel_varlen import (
    prenorm_block_causal_lact_swiglu_fused_kernel_triton as varlen_fn,
    _unpack_pad,
)
from custom_models.lact_model.kernel_dev.ttt_operation_fused_kernel_padded import (
    prenorm_block_causal_lact_swiglu_fused_kernel_triton as padded_fn,
)

device = "cuda"
nh, d, dh, chunk_size = 4, 512, 512, 2048
doc_lens = [4096, 3072, 2048, 1024]
num_docs = len(doc_lens)
cu_seqlens = torch.tensor(
    [0] + list(torch.cumsum(torch.tensor(doc_lens), dim=0).tolist()),
    dtype=torch.int32, device=device,
)
packed_len = cu_seqlens[-1].item()
max_doc_len = max(doc_lens)
B = nh * num_docs

# Inputs
w0 = torch.randn(nh, dh, d, device=device, dtype=torch.float32) * (d ** -0.5)
w1 = torch.randn(nh, d, dh, device=device, dtype=torch.float32) * (dh ** -0.5)
w2 = torch.randn(nh, dh, d, device=device, dtype=torch.float32) * (d ** -0.5)
q = torch.randn(nh, packed_len, d, device=device, dtype=torch.bfloat16)
k = torch.randn(nh, packed_len, d, device=device, dtype=torch.bfloat16)
v = torch.randn(nh, packed_len, d, device=device, dtype=torch.bfloat16)
lr0 = torch.sigmoid(torch.randn(nh, packed_len, 1, device=device)) * 0.01
lr1 = torch.sigmoid(torch.randn(nh, packed_len, 1, device=device)) * 0.01
lr2 = torch.sigmoid(torch.randn(nh, packed_len, 1, device=device)) * 0.01
momentum = torch.sigmoid(torch.randn(nh, packed_len, 1, device=device)) * 0.9
kwargs = dict(chunk_size=chunk_size, use_muon=False)

up = lambda t: _unpack_pad(t, cu_seqlens, max_doc_len)

# Pre-compute padded inputs
q_pad = up(q).reshape(B, max_doc_len, d)
k_pad = up(k).reshape(B, max_doc_len, d)
v_pad = up(v).reshape(B, max_doc_len, d)
lr0_pad = up(lr0).reshape(B, max_doc_len, 1)
lr1_pad = up(lr1).reshape(B, max_doc_len, 1)
lr2_pad = up(lr2).reshape(B, max_doc_len, 1)
m_pad = up(momentum).reshape(B, max_doc_len, 1)

grad_varlen = torch.randn(nh, packed_len, d, device=device, dtype=torch.bfloat16)
grad_padded = torch.randn(B, max_doc_len, d, device=device, dtype=torch.bfloat16)

# Warmup
for _ in range(5):
    o = varlen_fn(w0.clone().requires_grad_(), w1.clone().requires_grad_(), w2.clone().requires_grad_(),
                  q, k, v, lr0, lr1, lr2, **kwargs, momentum=momentum, cu_seqlens=cu_seqlens)
    o.backward(grad_varlen)
    o = padded_fn(w0.repeat_interleave(num_docs, 0).requires_grad_(),
                  w1.repeat_interleave(num_docs, 0).requires_grad_(),
                  w2.repeat_interleave(num_docs, 0).requires_grad_(),
                  up(q).reshape(B, max_doc_len, d), up(k).reshape(B, max_doc_len, d),
                  up(v).reshape(B, max_doc_len, d),
                  up(lr0).reshape(B, max_doc_len, 1), up(lr1).reshape(B, max_doc_len, 1),
                  up(lr2).reshape(B, max_doc_len, 1), **kwargs, momentum=up(momentum).reshape(B, max_doc_len, 1))
    o.backward(grad_padded)
torch.cuda.synchronize()

# Profiled runs — only this section is captured
torch.cuda.cudart().cudaProfilerStart()
sync = torch.cuda.synchronize

print(">>> varlen fwd+bwd")
sync()
torch.cuda.nvtx.range_push("varlen_fwd")
o = varlen_fn(w0.clone().requires_grad_(), w1.clone().requires_grad_(), w2.clone().requires_grad_(),
              q, k, v, lr0, lr1, lr2, **kwargs, momentum=momentum, cu_seqlens=cu_seqlens)
sync()
torch.cuda.nvtx.range_pop()
torch.cuda.nvtx.range_push("varlen_bwd")
o.backward(grad_varlen)
sync()
torch.cuda.nvtx.range_pop()

print(">>> padded fwd+bwd")
sync()
torch.cuda.nvtx.range_push("padded_fwd")
_q_pad = up(q).reshape(B, max_doc_len, d)
_k_pad = up(k).reshape(B, max_doc_len, d)
_v_pad = up(v).reshape(B, max_doc_len, d)
_lr0_pad = up(lr0).reshape(B, max_doc_len, 1)
_lr1_pad = up(lr1).reshape(B, max_doc_len, 1)
_lr2_pad = up(lr2).reshape(B, max_doc_len, 1)
_m_pad = up(momentum).reshape(B, max_doc_len, 1)
o = padded_fn(w0.repeat_interleave(num_docs, 0).requires_grad_(),
              w1.repeat_interleave(num_docs, 0).requires_grad_(),
              w2.repeat_interleave(num_docs, 0).requires_grad_(),
              _q_pad, _k_pad, _v_pad, _lr0_pad, _lr1_pad, _lr2_pad, **kwargs, momentum=_m_pad)
sync()
torch.cuda.nvtx.range_pop()
torch.cuda.nvtx.range_push("padded_bwd")
o.backward(grad_padded)
sync()
torch.cuda.nvtx.range_pop()

torch.cuda.cudart().cudaProfilerStop()
print("done")
