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
    from .kernel_dev.utils import compute_varlen_args
    from .profiling_utils import profile_range
except ImportError:

    from kernel_dev.lact_swiglu_ffn import fused_swiglu_ffn_fwd
    from kernel_dev.lact_fw_grad import fused_lact_swiglu_ffn_fast_weight_grads
    from kernel_dev.triton_prenorm_update_with_momentum import (
        fused_prenorm_update_with_momentum_and_l2_norm,
    )

    from kernel_dev.l2norm_triton_kernels import l2_norm_add_fused
    from kernel_dev.utils import compute_varlen_args
    from profiling_utils import profile_range

_PROFILING = bool(os.environ.get("PROFILE_MODE", ""))
_maybe_compile = (lambda fn: fn) if _PROFILING else torch.compile()


@_maybe_compile
def _prenorm_update(w0_w2_main, w1_main, dw0_w2, dw1, w0_w2_norm, w1_norm):
    w0_w2_main = w0_w2_main + dw0_w2
    w1_main = w1_main + dw1
    w0_w2_bf16 = (w0_w2_main / (w0_w2_main.norm(dim=3, keepdim=True) + 1e-5) * w0_w2_norm).to(torch.bfloat16)
    w1_bf16 = (w1_main / (w1_main.norm(dim=3, keepdim=True) + 1e-5) * w1_norm).to(torch.bfloat16)
    return w0_w2_main, w1_main, w0_w2_bf16, w1_bf16


@_maybe_compile
def _momentum_and_mask(dw0_w2, dw1, dw0_mom, dw1_mom, m_mean, grad_mask):
    dw0_w2 = dw0_w2 + dw0_mom * m_mean
    dw1 = dw1 + dw1_mom * m_mean
    dw0_w2 = dw0_w2 * grad_mask
    dw1 = dw1 * grad_mask
    return dw0_w2, dw1


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

@torch._dynamo.disable
# @_maybe_compile
def prenorm_block_causal_lact_swiglu_fused_kernel_triton(
    w0: torch.Tensor,  # [nh, dh, d], fp32 or b16
    w1: torch.Tensor,  # [nh, d, dh], fp32 or b16
    w2: torch.Tensor,  # [nh, dh, d], fp32 or b16
    q: torch.Tensor,  # [nh, packed_len, d], bf16
    k: torch.Tensor,  # [nh, packed_len, d], bf16
    v: torch.Tensor,  # [nh, packed_len, d], bf16
    lr0: torch.Tensor,  # [nh, packed_len, 1], fp32
    lr1: torch.Tensor,  # [nh, packed_len, 1], fp32
    lr2: torch.Tensor,  # [nh, packed_len, 1], fp32
    chunk_size: int = 2048,  # test-time training chunk size
    use_muon: bool = False,
    momentum: torch.Tensor = None,  # [nh, packed_len, 1], fp32 or bf16
    cu_seqlens: torch.Tensor = None,  # [num_docs + 1], int32
    num_chunks: int = None,  # total chunks (seq_len // chunk_size), avoids .item() for torch.compile
):
    """
    Varlen block causal LaCT with SwiGLU fast weight function.
        Apply then Update => Shifted Block Causal LaCT
    w0, w1, w2 are the fast weights. f(x) =  w1 @ (silu(w0 @ x) * (w2 @ x))

    Each document (delimited by cu_seqlens) gets fresh fast weights.
    Documents of different lengths are processed in parallel without padding.

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

    cu_seqlens: 1D int32 tensor of shape [num_docs + 1], cumulative
        sequence lengths following the flash_attn_varlen_func convention.
        E.g. for 3 sequences of lengths 3, 4, 3: cu_seqlens = [0, 3, 7, 10].
        Each sequence gets fresh fast weights.

    Outputs:
        o: [nh, packed_len, d]
    """

    nh = w0.shape[0]
    d = w0.shape[2]
    dh = w0.shape[1]
    num_docs = cu_seqlens.shape[0] - 1
    device = w0.device
    packed_len = q.shape[1]
    G = nh * num_docs
    flat_len = nh * packed_len

    doc_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    doc_nchunks = (doc_lens + chunk_size - 1) // chunk_size
    max_chunks = num_chunks if num_chunks is not None else (packed_len + chunk_size - 1) // chunk_size

    # Zero lr at each doc's last-chunk positions (apply-only, no grad+update)
    tok_idx = torch.arange(packed_len, device=device)
    tok_doc = torch.searchsorted(cu_seqlens[1:].long(), tok_idx.long(), right=True)
    tok_pos = tok_idx - cu_seqlens[tok_doc]
    last_chunk_start = (doc_nchunks - 1) * chunk_size
    lr_mask = (tok_pos < last_chunk_start[tok_doc]).float()[None, :, None]  # [1, packed_len, 1]
    lr0 = lr0 * lr_mask
    lr1 = lr1 * lr_mask
    lr2 = lr2 * lr_mask

    # Memory logging
    # _mem_alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)
    # _mem_reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    # print(f"[varlen_kernel] nh={nh}, d={d}, dh={dh}, num_docs={num_docs}, packed_len={packed_len}, "
    #       f"max_chunks={max_chunks}, chunk_size={chunk_size}, G={G}, flat_len={flat_len}")
    # print(f"[varlen_kernel] doc_lens={doc_lens.tolist()}")
    # print(f"[varlen_kernel] CUDA memory before weight expand: alloc={_mem_alloc:.2f}GiB, reserved={_mem_reserved:.2f}GiB")

    # [nh, num_docs, 1, 1] mask: True for groups that need grad at chunk ci
    nchunks_g = doc_nchunks.unsqueeze(0).expand(nh, -1)

    # Flat views: [nh, packed_len, d] -> [nh * packed_len, d]  (free reshape)
    Q = q.reshape(flat_len, d)
    K = k.reshape(flat_len, d)
    V = v.reshape(flat_len, d)
    LR0 = lr0[:, :, 0].reshape(flat_len)
    LR1 = lr1[:, :, 0].reshape(flat_len)
    LR2 = lr2[:, :, 0].reshape(flat_len)
    M_flat = momentum[:, :, 0].reshape(flat_len) if momentum is not None else None

    # Flat cu_seqlens: [G+1] for G = nh * num_docs groups
    head_off = torch.arange(nh, device=device, dtype=torch.int32) * packed_len
    flat_cu = (cu_seqlens[:-1].unsqueeze(0) + head_off.unsqueeze(1)).reshape(-1)
    flat_cu = torch.cat([flat_cu, (head_off[-1:] + cu_seqlens[-1:])]).int()

    # Per-(head, doc) weights: [nh, num_docs, ...]
    w0_w2 = torch.cat([w0, w2], dim=1).contiguous()

    # _w0_w2_expand_bytes = nh * num_docs * 2 * dh * d * 4  # fp32
    # _w1_expand_bytes = nh * num_docs * d * dh * 4
    # _total_expand_bytes = 2 * _w0_w2_expand_bytes + 2 * _w1_expand_bytes  # main + bf16 copies
    # if momentum is not None:
    #     _total_expand_bytes += _w0_w2_expand_bytes + _w1_expand_bytes  # momentum zeros
    # print(f"[varlen_kernel] weight expand will allocate ~{_total_expand_bytes / (1024**3):.2f}GiB "
    #       f"(w0_w2_main: [nh={nh}, num_docs={num_docs}, 2*dh={2*dh}, d={d}], "
    #       f"w1_main: [nh={nh}, num_docs={num_docs}, d={d}, dh={dh}])")

    w0_w2_norm = w0_w2.norm(dim=2, keepdim=True).unsqueeze(1).expand(-1, num_docs, -1, -1).contiguous()
    w1_norm = w1.norm(dim=2, keepdim=True).unsqueeze(1).expand(-1, num_docs, -1, -1).contiguous()
    w0_w2_main = w0_w2.unsqueeze(1).expand(-1, num_docs, -1, -1).contiguous()
    w1_main = w1.unsqueeze(1).expand(-1, num_docs, -1, -1).contiguous()

    if momentum is not None:
        dw0_dw2_momentum = torch.zeros_like(w0_w2_main)
        dw1_momentum = torch.zeros_like(w1_main)

    w0_w2_bf16 = w0_w2_main.to(torch.bfloat16)
    w1_bf16 = w1_main.to(torch.bfloat16)

    # _mem_alloc2 = torch.cuda.memory_allocated(device) / (1024 ** 3)
    # _mem_reserved2 = torch.cuda.memory_reserved(device) / (1024 ** 3)
    # print(f"[varlen_kernel] CUDA memory after weight expand: alloc={_mem_alloc2:.2f}GiB, reserved={_mem_reserved2:.2f}GiB "
    #       f"(delta={_mem_alloc2 - _mem_alloc:.2f}GiB)")

    output = torch.zeros(flat_len, d, device=device, dtype=q.dtype)

    # Precompute per-chunk args
    chunk_args = []
    for ci in range(max_chunks):
        el, ba, ms = compute_varlen_args(flat_cu, chunk_size, ci)
        chunk_args.append(dict(eff_lens=el, bos_arr=ba, max_sl=ms))

    # Precompute grad masks and momentum means
    grad_masks = [(nchunks_g > ci + 1).unsqueeze(-1).unsqueeze(-1) for ci in range(max_chunks - 1)]
    if momentum is not None:
        rng_cs = torch.arange(chunk_size, device=device)
        m_means = []
        for ci in range(max_chunks - 1):
            bos = chunk_args[ci]['bos_arr']
            m_idx = bos.long().unsqueeze(1) + rng_cs.unsqueeze(0)
            m_idx = m_idx.clamp(max=flat_len - 1)
            m_means.append(M_flat[m_idx].mean(dim=1).reshape(nh, num_docs, 1, 1))

    # Main loop: apply + grad + update (all chunks except last)
    for ci in range(max_chunks - 1):
        ckw = chunk_args[ci]

        with profile_range("ttt_apply"):
            fwd_out = fused_swiglu_ffn_fwd(
                w0_w2_bf16.reshape(G, 2 * dh, d), w1_bf16.reshape(G, d, dh),
                Q, cu_seqlens=flat_cu, **ckw,
                # out=output,  # TODO: use out= when not training (no grad needed)
            )
            output = output + fwd_out

        with profile_range("ttt_grad"):
            dw0_w2, dw1 = fused_lact_swiglu_ffn_fast_weight_grads(
                w0_w2_bf16.reshape(G, 2 * dh, d), w1_bf16.reshape(G, d, dh),
                K, V, LR0, LR1, LR2, cu_seqlens=flat_cu, **ckw,
            )

        dw0_w2 = dw0_w2.reshape(nh, num_docs, 2 * dh, d)
        dw1 = dw1.reshape(nh, num_docs, d, dh)

        grad_mask = grad_masks[ci]
        dw0_w2 = dw0_w2 * grad_mask
        dw1 = dw1 * grad_mask

        if momentum is not None:
          with profile_range("ttt_momentum"):
            m_mean = m_means[ci]
            dw0_w2, dw1 = _momentum_and_mask(dw0_w2, dw1, dw0_dw2_momentum, dw1_momentum, m_mean, grad_mask)
            dw0_dw2_momentum = dw0_w2
            dw1_momentum = dw1

        if use_muon:
          with profile_range("ttt_muon"):
            dw1 = zeropower_via_newtonschulz5(dw1.reshape(G, d, dh)).reshape(nh, num_docs, d, dh)
            dw0_w2 = zeropower_via_newtonschulz5(dw0_w2.reshape(G, 2 * dh, d)).reshape(nh, num_docs, 2 * dh, d)

        with profile_range("ttt_update"):
            w0_w2_main, w1_main, w0_w2_bf16, w1_bf16 = _prenorm_update(
                w0_w2_main, w1_main, dw0_w2, dw1, w0_w2_norm, w1_norm,
            )

    # Last chunk: apply only
    with profile_range("ttt_apply"):
        fwd_out = fused_swiglu_ffn_fwd(
            w0_w2_bf16.reshape(G, 2 * dh, d), w1_bf16.reshape(G, d, dh),
            Q, cu_seqlens=flat_cu, **chunk_args[max_chunks - 1],
            # out=output,  # TODO: use out= when not training (no grad needed)
        )
        output = output + fwd_out

    return output.reshape(nh, packed_len, d)




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
    # doc_lens = [4096, 3072, 2048, 1024]
    doc_lens=[1, 4213, 1906, 295, 537, 1580, 213, 475, 743, 659, 1414, 230, 1520, 116, 745, 327, 181, 193, 187, 96, 110, 303, 340]
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
    _num_chunks = (max_doc_len + chunk_size - 1) // chunk_size
    kwargs = dict(chunk_size=chunk_size, use_muon=False)
    varlen_kw = dict(**kwargs, num_chunks=_num_chunks)

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
        q, k, v, lr0, lr1, lr2, **varlen_kw, momentum=momentum, cu_seqlens=cu_seqlens,
    )

    _compare("ref_loop", ref_loop, "varlen", varlen_output, cu_seqlens)
    _compare("ref_padded", ref_padded, "varlen", varlen_output, cu_seqlens)

    # ---- Benchmark: varlen vs padded batch ----
    from kernel_dev.kernel_test_utils import benchmark

    print("\n" + "=" * 60)
    print("Benchmark: varlen vs padded batch")
    print("=" * 60)

    def varlen_fn():
        return prenorm_block_causal_lact_swiglu_fused_kernel_triton(
            w0.clone(), w1.clone(), w2.clone(),
            q, k, v, lr0, lr1, lr2, **varlen_kw, momentum=momentum, cu_seqlens=cu_seqlens,
        )

    def ref_padded_fn():
        w0_pad = w0.repeat_interleave(num_docs, 0)
        w1_pad = w1.repeat_interleave(num_docs, 0)
        w2_pad = w2.repeat_interleave(num_docs, 0)
        q_pad = up(q).reshape(B, max_doc_len, d)
        k_pad = up(k).reshape(B, max_doc_len, d)
        v_pad = up(v).reshape(B, max_doc_len, d)
        lr0_pad = up(lr0).reshape(B, max_doc_len, 1)
        lr1_pad = up(lr1).reshape(B, max_doc_len, 1)
        lr2_pad = up(lr2).reshape(B, max_doc_len, 1)
        m_pad = up(momentum).reshape(B, max_doc_len, 1)
        out_pad = reference_fn(
            w0_pad, w1_pad, w2_pad,
            q_pad, k_pad, v_pad, lr0_pad, lr1_pad, lr2_pad,
            **kwargs, momentum=m_pad,
        ).reshape(nh, num_docs, max_doc_len, d)
        return _repack(out_pad, cu_seqlens, packed_len)

    benchmark(varlen_fn, ref_padded_fn, (), (), enable_grad=True)

    # ---- Benchmark fwd+bwd ----
    print("\n" + "=" * 60)
    print("Benchmark fwd+bwd: varlen vs padded batch")
    print("=" * 60)

    grad_out_varlen = torch.randn(nh, packed_len, d, device=device, dtype=q.dtype)
    grad_out_ref = torch.randn(B, max_doc_len, d, device=device, dtype=q.dtype)

    def varlen_fwd_bwd():
        _w0 = w0.clone().requires_grad_()
        _w1 = w1.clone().requires_grad_()
        _w2 = w2.clone().requires_grad_()
        out = prenorm_block_causal_lact_swiglu_fused_kernel_triton(
            _w0, _w1, _w2,
            q, k, v, lr0, lr1, lr2, **varlen_kw, momentum=momentum, cu_seqlens=cu_seqlens,
        )
        out.backward(grad_out_varlen)

    def ref_padded_fwd_bwd():
        _w0p = w0.repeat_interleave(num_docs, 0).requires_grad_()
        _w1p = w1.repeat_interleave(num_docs, 0).requires_grad_()
        _w2p = w2.repeat_interleave(num_docs, 0).requires_grad_()
        out = reference_fn(
            _w0p, _w1p, _w2p,
            up(q).reshape(B, max_doc_len, d), up(k).reshape(B, max_doc_len, d),
            up(v).reshape(B, max_doc_len, d),
            up(lr0).reshape(B, max_doc_len, 1), up(lr1).reshape(B, max_doc_len, 1),
            up(lr2).reshape(B, max_doc_len, 1),
            **kwargs, momentum=up(momentum).reshape(B, max_doc_len, 1),
        )
        out.backward(grad_out_ref)

    benchmark(varlen_fwd_bwd, ref_padded_fwd_bwd, (), (), enable_grad=True)

    # ---- Profile each part of the varlen chunk loop ----
    print("\n" + "=" * 60)
    print("Profiling varlen chunk loop parts")
    print("=" * 60)

    import torch.nn.functional as Fp
    from kernel_dev.utils import compute_varlen_args as _cva

    num_runs = 20
    timings = {}

    def timed(name):
        class _T:
            def __enter__(self_):
                self_.start = torch.cuda.Event(enable_timing=True)
                self_.end = torch.cuda.Event(enable_timing=True)
                self_.start.record()
                return self_
            def __exit__(self_, *a):
                self_.end.record()
                torch.cuda.synchronize()
                ms = self_.start.elapsed_time(self_.end)
                timings.setdefault(name, []).append(ms)
        return _T()

    warmup_varlen = 5
    for run_idx in range(warmup_varlen + num_runs):
        if run_idx == warmup_varlen:
            timings.clear()

        _w0, _w1, _w2 = w0.clone(), w1.clone(), w2.clone()
        _nh, _d, _dh = _w0.shape[0], _w0.shape[2], _w0.shape[1]
        _num_docs = cu_seqlens.shape[0] - 1
        _packed_len = cu_seqlens[-1].item()
        _G = _nh * _num_docs
        _flat_len = _nh * _packed_len

        _doc_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        _doc_nchunks = (_doc_lens + chunk_size - 1) // chunk_size
        _max_chunks = _doc_nchunks.max().item()
        _nchunks_g = _doc_nchunks.unsqueeze(0).expand(_nh, -1)

        _Q = q.reshape(_flat_len, _d)
        _K = k.reshape(_flat_len, _d)
        _V = v.reshape(_flat_len, _d)
        _LR0 = lr0[:, :, 0].reshape(_flat_len)
        _LR1 = lr1[:, :, 0].reshape(_flat_len)
        _LR2 = lr2[:, :, 0].reshape(_flat_len)
        _M_flat = momentum[:, :, 0].reshape(_flat_len)

        _head_off = torch.arange(_nh, device=device, dtype=torch.int32) * _packed_len
        _flat_cu = (cu_seqlens[:-1].unsqueeze(0) + _head_off.unsqueeze(1)).reshape(-1)
        _flat_cu = torch.cat([_flat_cu, (_head_off[-1:] + cu_seqlens[-1:])]).int()

        _w0_w2 = torch.cat([_w0, _w2], dim=1).contiguous()
        _w0_w2_norm = _w0_w2.norm(dim=2, keepdim=True).unsqueeze(1).expand(-1, _num_docs, -1, -1).contiguous()
        _w1_norm = _w1.norm(dim=2, keepdim=True).unsqueeze(1).expand(-1, _num_docs, -1, -1).contiguous()
        _w0_w2_main = _w0_w2.unsqueeze(1).expand(-1, _num_docs, -1, -1).contiguous()
        _w1_main = _w1.unsqueeze(1).expand(-1, _num_docs, -1, -1).contiguous()
        _dw0_mom = torch.zeros_like(_w0_w2_main)
        _dw1_mom = torch.zeros_like(_w1_main)
        _w0_w2_bf16 = _w0_w2_main.to(torch.bfloat16)
        _w1_bf16 = _w1_main.to(torch.bfloat16)
        _output = torch.zeros(_flat_len, _d, device=device, dtype=q.dtype)

        # Precompute chunk args, grad masks, momentum means
        _chunk_args = [dict(zip(('eff_lens', 'bos_arr', 'max_sl'), _cva(_flat_cu, chunk_size, ci))) for ci in range(_max_chunks)]
        _grad_masks = [(_nchunks_g > ci + 1).unsqueeze(-1).unsqueeze(-1) for ci in range(_max_chunks - 1)]
        _rng_cs = torch.arange(chunk_size, device=device)
        _m_means = []
        for ci in range(_max_chunks - 1):
            _bos = _chunk_args[ci]['bos_arr']
            _mi = _bos.long().unsqueeze(1) + _rng_cs.unsqueeze(0)
            _mi = _mi.clamp(max=_flat_len - 1)
            _m_means.append(_M_flat[_mi].mean(dim=1).reshape(_nh, _num_docs, 1, 1))

        for ci in range(_max_chunks - 1):
            ckw = _chunk_args[ci]

            with timed("ttt_apply"):
                fused_swiglu_ffn_fwd(
                    _w0_w2_bf16.reshape(_G, 2 * _dh, _d), _w1_bf16.reshape(_G, _d, _dh),
                    _Q, cu_seqlens=_flat_cu, out=_output, **ckw,
                )

            with timed("ttt_grad"):
                dw0_w2, dw1 = fused_lact_swiglu_ffn_fast_weight_grads(
                    _w0_w2_bf16.reshape(_G, 2 * _dh, _d), _w1_bf16.reshape(_G, _d, _dh),
                    _K, _V, _LR0, _LR1, _LR2, cu_seqlens=_flat_cu, **ckw,
                )

            with timed("grad_mask"):
                dw0_w2 = dw0_w2.reshape(_nh, _num_docs, 2 * _dh, _d)
                dw1 = dw1.reshape(_nh, _num_docs, _d, _dh)
                grad_mask = _grad_masks[ci]
                dw0_w2 = dw0_w2 * grad_mask
                dw1 = dw1 * grad_mask

            with timed("momentum"):
                m_mean = _m_means[ci]
                dw0_w2 = dw0_w2 + _dw0_mom * m_mean
                dw1 = dw1 + _dw1_mom * m_mean
                dw0_w2 = dw0_w2 * grad_mask
                dw1 = dw1 * grad_mask
                _dw0_mom = dw0_w2
                _dw1_mom = dw1

            with timed("ttt_update"):
                _w0_w2_main = _w0_w2_main + dw0_w2
                _w1_main = _w1_main + dw1
                _w0_w2_bf16 = (_w0_w2_main / (_w0_w2_main.norm(dim=3, keepdim=True) + 1e-5) * _w0_w2_norm).to(torch.bfloat16)
                _w1_bf16 = (_w1_main / (_w1_main.norm(dim=3, keepdim=True) + 1e-5) * _w1_norm).to(torch.bfloat16)

        # Last chunk: apply only
        with timed("ttt_apply"):
            fused_swiglu_ffn_fwd(
                _w0_w2_bf16.reshape(_G, 2 * _dh, _d), _w1_bf16.reshape(_G, _d, _dh),
                _Q, cu_seqlens=_flat_cu, out=_output, **_chunk_args[_max_chunks - 1],
            )

    print(f"\n{'Section':<20} {'Mean (ms)':>10} {'Total (ms)':>10} {'Calls':>6}")
    print("-" * 50)
    total_all = 0
    for name in ["ttt_apply", "ttt_grad", "grad_mask", "momentum", "ttt_update"]:
        if name in timings:
            vals = timings[name]
            mean_ms = sum(vals) / len(vals)
            total_ms = sum(vals)
            total_all += total_ms
            print(f"{name:<20} {mean_ms:>10.4f} {total_ms:>10.2f} {len(vals):>6}")
    print(f"{'TOTAL':<20} {total_all/num_runs:>10.4f} {total_all:>10.2f}")

    # ---- Profile reference (padded batch) chunk loop ----
    print("\n" + "=" * 60)
    print("Profiling REFERENCE (padded batch) chunk loop parts")
    print("=" * 60)

    from einops import rearrange as rearr
    from ttt_operation_fused_kernel import fused_swiglu_ffn_fwd as ref_fused_swiglu_ffn_fwd
    from ttt_operation_fused_kernel import fused_lact_swiglu_ffn_fast_weight_grads as ref_fused_lact_grads

    ref_timings = {}
    def ref_timed(name):
        class _T:
            def __enter__(self_):
                self_.start = torch.cuda.Event(enable_timing=True)
                self_.end = torch.cuda.Event(enable_timing=True)
                self_.start.record()
                return self_
            def __exit__(self_, *a):
                self_.end.record()
                torch.cuda.synchronize()
                ms = self_.start.elapsed_time(self_.end)
                ref_timings.setdefault(name, []).append(ms)
        return _T()

    # Pre-compute padded inputs for reference profiling
    w0_pad = w0.repeat_interleave(num_docs, 0)
    w1_pad = w1.repeat_interleave(num_docs, 0)
    w2_pad = w2.repeat_interleave(num_docs, 0)
    q_pad = up(q).reshape(B, max_doc_len, d)
    k_pad = up(k).reshape(B, max_doc_len, d)
    v_pad = up(v).reshape(B, max_doc_len, d)
    lr0_pad = up(lr0).reshape(B, max_doc_len, 1)
    lr1_pad = up(lr1).reshape(B, max_doc_len, 1)
    lr2_pad = up(lr2).reshape(B, max_doc_len, 1)
    m_pad = up(momentum).reshape(B, max_doc_len, 1)

    warmup_ref = 5
    for run_idx in range(warmup_ref + num_runs):
        if run_idx == warmup_ref:
            ref_timings.clear()
        _w0r, _w1r, _w2r = w0_pad.clone(), w1_pad.clone(), w2_pad.clone()
        _B = _w0r.shape[0]  # nh * num_docs
        _d = _w0r.shape[2]

        _w0_w2r = torch.cat([_w0r, _w2r], dim=1).contiguous()
        _w0_w2_normr = _w0_w2r.norm(dim=2, keepdim=True)
        _w1_normr = _w1r.norm(dim=2, keepdim=True)
        _w0_w2_mainr = _w0_w2r
        _w1_mainr = _w1r
        _w0_w2r_bf16 = _w0_w2r.to(torch.bfloat16)
        _w1r_bf16 = _w1r.to(torch.bfloat16)

        _dw0_dw2_momr = torch.zeros_like(_w0_w2_mainr)
        _dw1_momr = torch.zeros_like(_w1_mainr)

        _qr = Fp.pad(q_pad, (0, 0, 0, -q_pad.shape[1] % chunk_size))
        _kr = Fp.pad(k_pad, (0, 0, 0, -k_pad.shape[1] % chunk_size))
        _vr = Fp.pad(v_pad, (0, 0, 0, -v_pad.shape[1] % chunk_size))
        _lr0r = Fp.pad(lr0_pad, (0, 0, 0, -lr0_pad.shape[1] % chunk_size))
        _lr1r = Fp.pad(lr1_pad, (0, 0, 0, -lr1_pad.shape[1] % chunk_size))
        _lr2r = Fp.pad(lr2_pad, (0, 0, 0, -lr2_pad.shape[1] % chunk_size))
        _mr = Fp.pad(m_pad, (0, 0, 0, -m_pad.shape[1] % chunk_size))
        _num_chunks = (_qr.shape[1] + chunk_size - 1) // chunk_size

        _kr = rearr(_kr, "b (n c) d -> n b c d", n=_num_chunks)
        _vr = rearr(_vr, "b (n c) d -> n b c d", n=_num_chunks)
        _qr = rearr(_qr, "b (n c) d -> n b c d", n=_num_chunks)
        _lr0r = rearr(_lr0r, "b (n c) d -> n b (c d)", n=_num_chunks, d=1)
        _lr1r = rearr(_lr1r, "b (n c) d -> n b (c d)", n=_num_chunks, d=1)
        _lr2r = rearr(_lr2r, "b (n c) d -> n b (c d)", n=_num_chunks, d=1)
        _mr = rearr(_mr, "b (n c) 1 -> n b c 1", n=_num_chunks)
        _outr = torch.zeros_like(_qr)

        for chunk_idx in range(_num_chunks - 1):
            with ref_timed("slice"):
                ki = _kr[chunk_idx].contiguous()
                vi = _vr[chunk_idx].contiguous()
                qi = _qr[chunk_idx].contiguous()
                lr1i = _lr1r[chunk_idx].contiguous()
                lr2i = _lr2r[chunk_idx].contiguous()
                lr0i = _lr0r[chunk_idx].contiguous()

            with ref_timed("ttt_apply"):
                _outr[chunk_idx] = ref_fused_swiglu_ffn_fwd(_w0_w2r_bf16, _w1r_bf16, qi)

            with ref_timed("ttt_grad"):
                dw0_w2, dw1 = ref_fused_lact_grads(
                    _w0_w2r_bf16, _w1r_bf16, ki, vi, lr0i, lr1i, lr2i
                )

            with ref_timed("momentum"):
                m_i = _mr[chunk_idx].contiguous()
                m_i = m_i.mean(dim=1, keepdim=True)
                dw0_w2 = dw0_w2 + _dw0_dw2_momr * m_i
                dw1 = dw1 + _dw1_momr * m_i
                _dw0_dw2_momr = dw0_w2
                _dw1_momr = dw1

            with ref_timed("ttt_update"):
                _w0_w2_mainr = _w0_w2_mainr + dw0_w2
                _w1_mainr = _w1_mainr + dw1
                _w0_w2r_bf16 = (_w0_w2_mainr / (_w0_w2_mainr.norm(dim=2, keepdim=True) + 1e-5) * _w0_w2_normr).to(torch.bfloat16)
                _w1r_bf16 = (_w1_mainr / (_w1_mainr.norm(dim=2, keepdim=True) + 1e-5) * _w1_normr).to(torch.bfloat16)

        # last chunk
        with ref_timed("ttt_apply"):
            _outr[-1] = ref_fused_swiglu_ffn_fwd(_w0_w2r_bf16, _w1r_bf16, _qr[-1].contiguous())

    print(f"\n{'Section':<20} {'Mean (ms)':>10} {'Total (ms)':>10} {'Calls':>6}")
    print("-" * 50)
    ref_total_all = 0
    for name in ["slice", "ttt_apply", "ttt_grad", "momentum", "ttt_update"]:
        if name in ref_timings:
            vals = ref_timings[name]
            mean_ms = sum(vals) / len(vals)
            total_ms = sum(vals)
            ref_total_all += total_ms
            print(f"{name:<20} {mean_ms:>10.4f} {total_ms:>10.2f} {len(vals):>6}")
    print(f"{'TOTAL':<20} {ref_total_all/num_runs:>10.4f} {ref_total_all:>10.2f}")
