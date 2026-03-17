import os
import torch
import torch.nn.functional as F
from einops import rearrange

try:

    from .kernel_dev.lact_swiglu_ffn import fused_swiglu_ffn_fwd
    from .kernel_dev.lact_fw_grad import (
        fused_lact_swiglu_ffn_fast_weight_grads,
    )
    from .kernel_dev.triton_prenorm_update_with_momentum import (
        fused_prenorm_update_with_momentum_and_l2_norm,
    )

    from .kernel_dev.l2norm_triton_kernels import l2_norm_add_fused
    from .profiling_utils import profile_range
except ImportError:

    from kernel_dev.lact_swiglu_ffn import fused_swiglu_ffn_fwd
    from kernel_dev.lact_fw_grad import fused_lact_swiglu_ffn_fast_weight_grads
    from kernel_dev.triton_prenorm_update_with_momentum import (
        fused_prenorm_update_with_momentum_and_l2_norm,
    )

    from kernel_dev.l2norm_triton_kernels import l2_norm_add_fused
    from profiling_utils import profile_range

_PROFILING = bool(os.environ.get("PROFILE_MODE", ""))
_maybe_compile = (lambda fn: fn) if _PROFILING else torch.compile()


@_maybe_compile
def zeropower_via_newtonschulz5(G):
    """
    This is an updated version of the zeropower_via_newtonschulz5 function in here:
    https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt_medium.py#L26
    The code is modified from https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py#L49, which contains the original muon implementation.
    Major change: G is [b, d, d] rather than [d, d]
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    Args:
        G: [b, d, d']
    Returns:
        X: [b, d, d']
    FLOPS:  When d=d', Total FLOPS=30 * b * d^3
    """
    assert len(G.shape) == 3
    X = G.bfloat16()
    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(1, 2), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for a, b, c in [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]:
        # ## Original ###
        # A = X @ X.transpose(1, 2)
        # B = (
        #     b * A + c * A @ A
        # )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        # X = a * X + B @ X
        # ##
        
        ### Ali's suggestion ### OOM
        A = torch.bmm(X, X.transpose(1, 2))
        B = torch.baddbmm(A, A, A, beta=b, alpha=c)
        X = torch.baddbmm(X, B, X, beta=a, alpha=1.0)
        ###


    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    return X


@_maybe_compile
def postnorm_block_causal_lact_swiglu_fused_kernel_triton(
    w0: torch.Tensor,  # [b, dh, d], fp32 or b16
    w1: torch.Tensor,  # [b, d, dh], fp32 or b16
    w2: torch.Tensor,  # [b, dh, d], fp32 or b16
    q: torch.Tensor,  # [b, l, d], bf16
    k: torch.Tensor,  # [b, l, d], bf16
    v: torch.Tensor,  # [b, l, d], bf16
    lr0: torch.Tensor,  # [b, l, 1], fp32
    lr1: torch.Tensor,  # [b, l, 1], fp32
    lr2: torch.Tensor,  # [b, l, 1], fp32
    chunk_size: int = 2048,  # test-time training chunk size
    use_muon: bool = False,
    momentum: torch.Tensor = None,  # [b, s, 1], fp32 or bf16
):
    """
    Block causal LaCT with SwiGLU fast weight function.
        Apply then Update => Shifted Block Causal LaCT
    w0, w1, w2 are the fast weights. f(x) =  w1 @ (silu(w0 @ x) * (w2 @ x))

    About precision:
        w0, w1, w2 are recommended to be fp32.
        q, k, v are fp16.
        lr0, lr1, lr2 are fp32.
        The forward, backward produce bf16 gradients, updated fast weights are fp32.
        The final output are bf16.

    FLOPS:
        (assume dk=dv denoted as D, hidden dimension of swiglu-mlp is H, ignore muon, ignore last chunk)
        Forward pass with key: 4 * D * H * L * B
        Backward pass: 8 * D * H * L * B
        Forward with Query: 6 * D * H * L * B
        Total: 18 * D * H * L * B
    Outputs:
        o: [b, l, dv]
    """

    w0_w2 = torch.cat([w0, w2], dim=1).contiguous()
    w0_w2_norm = w0_w2.norm(dim=2, keepdim=False)
    w1_norm = w1.norm(dim=2, keepdim=False)

    if momentum is not None:
        # same dtype as w0_w2_main and w1_main, recommended to be fp32
        dw0_dw2_momentum = torch.zeros_like(w0_w2)
        dw1_momentum = torch.zeros_like(w1)

    q_original_length = q.shape[1]
    ### Padding the inputs to make the length a multiple of chunk_size
    k = F.pad(k, (0, 0, 0, -k.shape[1] % chunk_size))
    v = F.pad(v, (0, 0, 0, -v.shape[1] % chunk_size))
    q = F.pad(q, (0, 0, 0, -q.shape[1] % chunk_size))
    lr0 = F.pad(lr0, (0, 0, 0, -lr0.shape[1] % chunk_size))
    lr1 = F.pad(lr1, (0, 0, 0, -lr1.shape[1] % chunk_size))
    lr2 = F.pad(lr2, (0, 0, 0, -lr2.shape[1] % chunk_size))
    if momentum is not None:
        momentum = F.pad(momentum, (0, 0, 0, -momentum.shape[1] % chunk_size))
    num_chunks = (q.shape[1] + chunk_size - 1) // chunk_size

    k = rearrange(k, "b (n c) d -> n b c d", n=num_chunks)
    v = rearrange(v, "b (n c) d -> n b c d", n=num_chunks)
    q = rearrange(q, "b (n c) d -> n b c d", n=num_chunks)
    lr0 = rearrange(lr0, "b (n c) d -> n b (c d)", n=num_chunks, d=1)
    lr1 = rearrange(lr1, "b (n c) d -> n b (c d)", n=num_chunks, d=1)
    lr2 = rearrange(lr2, "b (n c) d -> n b (c d)", n=num_chunks, d=1)
    if momentum is not None:
        momentum = rearrange(momentum, "b (n c) 1 -> n b c 1", n=num_chunks)

    output = torch.zeros_like(q)

    e_index = 0
    seq_len = k.shape[1]
    for chunk_idx in range(num_chunks - 1):

        # [b, l, dk]
        ki = k[chunk_idx].contiguous()  # bf16
        # [b, l, dv]
        vi = v[chunk_idx].contiguous()  # bf16
        # [b, dh, l]
        qi = q[chunk_idx].contiguous()
        # [b, l, d/1] fp32
        lr1i = lr1[chunk_idx].contiguous()  # [b, l, d/1] fp32
        lr2i = lr2[chunk_idx].contiguous()  # [b, l, d/1] fp32
        lr0i = lr0[chunk_idx].contiguous()  # [b, l, d/1] fp32

        # apply first, perform swiglu ffn forward pass with the qi.
        with profile_range("ttt_apply"):
            w0_w2_bf16 = w0_w2.to(torch.bfloat16)
            w1_bf16 = w1.to(torch.bfloat16)
            output[chunk_idx] = fused_swiglu_ffn_fwd(w0_w2_bf16, w1_bf16, qi)

        # then, compute test-time training gradients for w0, w1, w2. under negative dot product loss.
        with profile_range("ttt_grad"):
            dw0_w2, dw1 = fused_lact_swiglu_ffn_fast_weight_grads(
                w0_w2_bf16, w1_bf16, ki, vi, lr0i, lr1i, lr2i
            )

        if momentum is not None:
          with profile_range("ttt_momentum"):
            m_i = momentum[chunk_idx].contiguous()
            m_i = m_i.mean(dim=1, keepdim=True)  # [b, 1, 1]

            dw0_w2 = dw0_w2 + dw0_dw2_momentum * m_i
            dw1 = dw1 + dw1_momentum * m_i
            dw0_dw2_momentum = dw0_w2
            dw1_momentum = dw1

        if use_muon:
          with profile_range("ttt_muon"):
            dw1 = zeropower_via_newtonschulz5(dw1)
            dw0_w2 = zeropower_via_newtonschulz5(dw0_w2)

        with profile_range("ttt_update"):
            w0_w2 = l2_norm_add_fused(w0_w2, dw0_w2, w0_w2_norm, eps=1e-5)
            w1 = l2_norm_add_fused(w1, dw1, w1_norm, eps=1e-5)

    # for the last chunk, don't update the fast weights, directly apply the fast weights to the query.

    qi = q[-1].contiguous()

    with profile_range("ttt_apply_last"):
        output[-1] = fused_swiglu_ffn_fwd(
            w0_w2.to(torch.bfloat16), w1.to(torch.bfloat16), qi
        )

    output = rearrange(output, "n b c d -> b (n c) d")

    return output[:, :q_original_length]
def prenorm_block_causal_lact_swiglu_fused_kernel_triton(
    w0: torch.Tensor,  # [nh, dh, d], fp32 or b16
    w1: torch.Tensor,  # [nh, d, dh], fp32 or b16
    w2: torch.Tensor,  # [nh, dh, d], fp32 or b16
    q: torch.Tensor,  # [nh, l, d], bf16
    k: torch.Tensor,  # [nh, l, d], bf16
    v: torch.Tensor,  # [nh, l, d], bf16
    lr0: torch.Tensor,  # [nh, l, 1], fp32
    lr1: torch.Tensor,  # [nh, l, 1], fp32
    lr2: torch.Tensor,  # [nh, l, 1], fp32
    chunk_size: int = 2048,  # test-time training chunk size
    use_muon: bool = False,
    momentum: torch.Tensor = None,  # [nh, l, 1], fp32 or bf16
    cu_seqlens: torch.Tensor = None,  # [num_docs + 1], int32
):
    """
    Block causal LaCT with SwiGLU fast weight function.
        Apply then Update => Shifted Block Causal LaCT
    w0, w1, w2 are the fast weights. f(x) =  w1 @ (silu(w0 @ x) * (w2 @ x))

    About precision:
        w0, w1, w2 are recommended to be fp32.
        q, k, v are fp16.
        lr0, lr1, lr2 are fp32.
        The forward, backward produce bf16 gradients, updated fast weights are fp32.
        The final output are bf16.

    FLOPS:
        (assume dk=dv denoted as D, hidden dimension of swiglu-mlp is H, ignore muon, ignore last chunk)
        Forward pass with key: 4 * D * H * L * B
        Backward pass: 8 * D * H * L * B
        Forward with Query: 6 * D * H * L * B
        Total: 18 * D * H * L * B

    cu_seqlens: optional 1D int32 tensor of shape [num_sequences + 1], cumulative
        sequence lengths following the flash_attn_varlen_func convention.
        E.g. for 3 sequences of lengths 3, 4, 3: cu_seqlens = [0, 3, 7, 10].
        When provided, each sequence gets fresh fast weights and padding beyond
        cu_seqlens[-1] is skipped. If None, the entire input is one sequence.

    Outputs:
        o: [b, l, d]
    """
    
    w0_w2 = torch.cat([w0, w2], dim=1).contiguous()
    w0_w2_norm = w0_w2.norm(dim=2, keepdim=True)
    w1_norm = w1.norm(dim=2, keepdim=True)

    w0_w2_main = w0_w2
    w1_main = w1
    w0_w2 = w0_w2.to(torch.bfloat16)
    w1 = w1.to(torch.bfloat16)

    if momentum is not None:
        # same dtype as w0_w2_main and w1_main, recommended to be fp32
        dw0_dw2_momentum = torch.zeros_like(w0_w2_main)
        dw1_momentum = torch.zeros_like(w1_main)

    q_original_length = q.shape[1]
    ### Padding the inputs to make the length a multiple of chunk_size
    k = F.pad(k, (0, 0, 0, -k.shape[1] % chunk_size))
    v = F.pad(v, (0, 0, 0, -v.shape[1] % chunk_size))
    q = F.pad(q, (0, 0, 0, -q.shape[1] % chunk_size))
    lr0 = F.pad(lr0, (0, 0, 0, -lr0.shape[1] % chunk_size))
    lr1 = F.pad(lr1, (0, 0, 0, -lr1.shape[1] % chunk_size))
    lr2 = F.pad(lr2, (0, 0, 0, -lr2.shape[1] % chunk_size))
    if momentum is not None:
        momentum = F.pad(momentum, (0, 0, 0, -momentum.shape[1] % chunk_size))
    num_chunks = (q.shape[1] + chunk_size - 1) // chunk_size

    k = rearrange(k, "b (n c) d -> n b c d", n=num_chunks)
    v = rearrange(v, "b (n c) d -> n b c d", n=num_chunks)
    q = rearrange(q, "b (n c) d -> n b c d", n=num_chunks)
    lr0 = rearrange(lr0, "b (n c) d -> n b (c d)", n=num_chunks, d=1)
    lr1 = rearrange(lr1, "b (n c) d -> n b (c d)", n=num_chunks, d=1)
    lr2 = rearrange(lr2, "b (n c) d -> n b (c d)", n=num_chunks, d=1)
    if momentum is not None:
        momentum = rearrange(momentum, "b (n c) 1 -> n b c 1", n=num_chunks)
    output = torch.zeros_like(q)

    e_index = 0
    seq_len = k.shape[1]
    for chunk_idx in range(num_chunks - 1):

        # [b, l, dk]
        ki = k[chunk_idx].contiguous()  # bf16
        # [b, l, dv]
        vi = v[chunk_idx].contiguous()  # bf16
        # [b, dh, l]
        qi = q[chunk_idx].contiguous()
        # [b, l, d/1] fp32
        lr1i = lr1[chunk_idx].contiguous()  # [b, l, d/1] fp32
        lr2i = lr2[chunk_idx].contiguous()  # [b, l, d/1] fp32
        lr0i = lr0[chunk_idx].contiguous()  # [b, l, d/1] fp32

        # apply first, perform swiglu ffn forward pass with the qi.
        with profile_range("ttt_apply"):
            output[chunk_idx] = fused_swiglu_ffn_fwd(w0_w2, w1, qi)

        # then, compute test-time training gradients for w0, w1, w2. under negative dot product loss.
        with profile_range("ttt_grad"):
            dw0_w2, dw1 = fused_lact_swiglu_ffn_fast_weight_grads(
                w0_w2.to(torch.bfloat16), w1.to(torch.bfloat16), ki, vi, lr0i, lr1i, lr2i
            )

        if momentum is not None:
          with profile_range("ttt_momentum"):
            m_i = momentum[chunk_idx].contiguous()
            m_i = m_i.mean(dim=1, keepdim=True)  # [b, 1, 1]

            dw0_w2 = dw0_w2 + dw0_dw2_momentum * m_i
            dw1 = dw1 + dw1_momentum * m_i
            dw0_dw2_momentum = dw0_w2
            dw1_momentum = dw1

        if use_muon:
          with profile_range("ttt_muon"):
            dw1 = zeropower_via_newtonschulz5(dw1)
            dw0_w2 = zeropower_via_newtonschulz5(dw0_w2)

        with profile_range("ttt_update"):
            w0_w2_main = w0_w2_main + dw0_w2
            w1_main = w1_main + dw1

            # cast to bf16
            w0_w2 = (
                w0_w2_main / (w0_w2_main.norm(dim=2, keepdim=True) + 1e-5) * w0_w2_norm
            ).to(torch.bfloat16)
            w1 = (w1_main / (w1_main.norm(dim=2, keepdim=True) + 1e-5) * w1_norm).to(
                torch.bfloat16
            )

    # for the last chunk, don't update the fast weights, directly apply the fast weights to the query.
    qi = q[-1].contiguous()

    with profile_range("ttt_apply_last"):
        output[-1] = fused_swiglu_ffn_fwd(w0_w2, w1, qi)

    output = rearrange(output, "n b c d -> b (n c) d")

    return output[:, :q_original_length]




def _compare(name_a, a, name_b, b, cu_seqlens):
    """Print max_diff overall and per-doc between two packed outputs."""
    num_docs = cu_seqlens.shape[0] - 1
    mx = (a - b).abs().max().item()
    mn = (a - b).abs().mean().item()
    print(f"{name_a} vs {name_b}: max={mx:.6e} mean={mn:.6e}")
    for i in range(num_docs):
        s, e = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
        dd = (a[:, s:e] - b[:, s:e]).abs().max().item()
        print(f"  doc {i} (len={e-s}): max_diff={dd:.6e}")


def _unpack_pad(packed, cu_seqlens, max_len):
    """[nh, packed_len, *] -> [nh, num_docs, max_len, *]  zero-padded."""
    nh = packed.shape[0]
    rest = packed.shape[2:]
    num_docs = cu_seqlens.shape[0] - 1
    out = packed.new_zeros(nh, num_docs, max_len, *rest)
    for i in range(num_docs):
        s, e = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
        out[:, i, :e - s] = packed[:, s:e]
    return out


def _repack(padded, cu_seqlens, packed_len):
    """[nh, num_docs, max_len, d] -> [nh, packed_len, d]"""
    nh, num_docs, _, d = padded.shape
    out = padded.new_zeros(nh, packed_len, d)
    for i in range(num_docs):
        s, e = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
        out[:, s:e] = padded[:, i, :e - s]
    return out


if __name__ == "__main__":
    from ttt_operation_fused_kernel import (
        prenorm_block_causal_lact_swiglu_fused_kernel_triton as reference_fn,
    )

    device = "cuda"
    nh, d, dh, chunk_size = 4, 512, 512, 2048
    doc_lens = [4096, 3072, 2048, 1024]
    cu_seqlens = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(doc_lens), dim=0).tolist()),
        dtype=torch.int32, device=device,
    )
    packed_len = cu_seqlens[-1].item()
    num_docs = len(doc_lens)
    max_doc_len = max(doc_lens)
    print(f"{nh=}, {d=}, {dh=}, {chunk_size=}, {doc_lens=}, {packed_len=}")

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
    kwargs = dict(chunk_size=chunk_size, use_muon=False, momentum=momentum)

    # ---- Reference 1 (loop): per-doc with fresh weights ----
    print("\n--- ref_loop: per-doc loop ---")
    ref_loop_parts = []
    for i in range(num_docs):
        s, e = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
        out = reference_fn(
            w0.clone(), w1.clone(), w2.clone(),
            q[:, s:e].contiguous(), k[:, s:e].contiguous(), v[:, s:e].contiguous(),
            lr0[:, s:e].contiguous(), lr1[:, s:e].contiguous(), lr2[:, s:e].contiguous(),
            **kwargs, momentum=momentum[:, s:e].contiguous(),
        )
        ref_loop_parts.append(out)
    ref_loop = torch.cat(ref_loop_parts, dim=1)

    # ---- Reference 2 (padded batch): expand docs into batch dim ----
    print("--- ref_padded: padded batch ---")
    up = lambda t: _unpack_pad(t, cu_seqlens, max_doc_len)
    B = nh * num_docs
    ref_pad_out = reference_fn(
        w0.repeat_interleave(num_docs, 0), w1.repeat_interleave(num_docs, 0),
        w2.repeat_interleave(num_docs, 0),
        up(q).reshape(B, max_doc_len, d), up(k).reshape(B, max_doc_len, d),
        up(v).reshape(B, max_doc_len, d),
        up(lr0).reshape(B, max_doc_len, 1), up(lr1).reshape(B, max_doc_len, 1),
        up(lr2).reshape(B, max_doc_len, 1),
        **kwargs, momentum=up(momentum).reshape(B, max_doc_len, 1),
    ).reshape(nh, num_docs, max_doc_len, d)
    ref_padded = _repack(ref_pad_out, cu_seqlens, packed_len)

    _compare("ref_loop", ref_loop, "ref_padded", ref_padded, cu_seqlens)

    # ---- Varlen version: single call with packed input + cu_seqlens ----
    print("\n--- varlen (our implementation) ---")
    varlen_output = prenorm_block_causal_lact_swiglu_fused_kernel_triton(
        w0.clone(), w1.clone(), w2.clone(),
        q, k, v, lr0, lr1, lr2, **kwargs, cu_seqlens=cu_seqlens,
    )

    _compare("ref_loop", ref_loop, "varlen", varlen_output, cu_seqlens)
    _compare("ref_padded", ref_padded, "varlen", varlen_output, cu_seqlens)
