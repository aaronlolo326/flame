import torch

from torch.autograd.function import once_differentiable

try:
    from triton_swiglu_bwd_with_lr import swiglu_backward_three_bmm_with_lr_triton, swiglu_backward_three_bmm_with_lr_varlen_triton

    from triton_fused_matmul_kernels import (
        fused_two_mm_same_out_interface,
        fused_two_mm_same_out_wT_xT_varlen_triton,
        fused_two_mm_same_out_wT_x_varlen_triton,
    )
    from triton_pointwise_kernels import triton_swiglu_bwd_bwd_fused_cat_inp_out, triton_swiglu_bwd_bwd_fused_cat_inp_out_varlen
    from grouped_gemm import grouped_gemm_to_packed, grouped_gemm_reduce
    from utils import compute_varlen_args
except ImportError:
    from .triton_swiglu_bwd_with_lr import swiglu_backward_three_bmm_with_lr_triton, swiglu_backward_three_bmm_with_lr_varlen_triton

    from .triton_fused_matmul_kernels import (
        fused_two_mm_same_out_interface,
        fused_two_mm_same_out_wT_xT_varlen_triton,
        fused_two_mm_same_out_wT_x_varlen_triton,
    )
    from .triton_pointwise_kernels import triton_swiglu_bwd_bwd_fused_cat_inp_out, triton_swiglu_bwd_bwd_fused_cat_inp_out_varlen
    from .grouped_gemm import grouped_gemm_to_packed, grouped_gemm_reduce
    from .utils import compute_varlen_args


class FusedLactSwiGLUFFNBwd(torch.autograd.Function):

    @staticmethod
    def forward(ctx, W0_W2, W1, K, V, lr0, lr1, lr2):
        """
        Args:
            W0_W2:    [B, 2M, K] or [B, 2 * Hidden, D]
            W1:        [B, K, M] or [B, D, Hidden]
            K, V:      [M, N, K] or [B, num_Tokens, D]
            lr0, lr1, lr2:    [B, N]

        Outs:
            Hidden: [B, N, K] or [B, num_tokens, Hidden]
            dW0_W2: [B, 2 * Hidden, D]
            dW1: [B, D, Hidden]
        Total FLOPS: 12 * B * Hidden * D * num_tokens
        """
        #### without this triton kernel, we will materize Y0, Y2, Dhidden;  DY0_with_LR0, DY2_with_LR2, Hidden_with_LR1;
        #### 3 + 3 + 3.   read, write.
        DY0_DY2, Hidden = swiglu_backward_three_bmm_with_lr_triton(
            W0_W2,
            W1,
            K,
            V,
            lr0,
            lr1,
            lr2,
        )

        # groupping below two GEMM togeather can futher reduce launching overhead.
        # [B, 2 * Hidden, num_tokens] @ [B, num_tokens, D] -> [B, 2 * Hidden, D]
        DW0_DW2 = torch.bmm(DY0_DY2, K)
        # [B, D, Hidden] = [B, D, num_tokens] @ [B, num_tokens, Hidden]
        DW1 = torch.bmm(V.transpose(1, 2), Hidden.transpose(1, 2))

        # we don't need to save DY0, DY2, and Hidden, because we will compute them again in the backward pass.
        ctx.save_for_backward(W0_W2, W1, K, V, lr0, lr1, lr2)

        return DW0_DW2, DW1

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_dw0_dw2, grad_dw1):
        """
        Args:
            grad_dw0_dw2: [B, 2 * Hidden, D]
            grad_dw1: [B, D, Hidden]
        Outs:
            grad_W0: [B, Hidden, D]
            grad_W1: [B, D, Hidden]
            grad_W2: [B, Hidden, D]
            grad_K: [B, D, num_tokens]
            grad_V: [B, num_tokens, D]
            grad_lr0: [B, 1]
            grad_lr1: [B, 1]
            grad_lr2: [B, 1]

        Total FLOPS: 24 * B * Hidden * D * num_tokens + 6 * B * Hidden * D * num_tokens
        # 24 for backward matmuls, and 6 for forward recomputation.
        """

        W0_W2, W1, K, V, lr0, lr1, lr2 = ctx.saved_tensors

        # -> [B, 2 * Hidden, num_tokens]
        Y0_Y2 = torch.bmm(W0_W2, K.transpose(1, 2))

        DHidden = torch.bmm(W1.transpose(1, 2), V.transpose(1, 2))
        grad_Hidden_with_lr1 = torch.bmm(grad_dw1.transpose(1, 2), V.transpose(1, 2))

        # [B, Hidden, num_tokens] = [B, Hidden, D] @ [B, num_tokens, D].T
        # grad_DY0_with_lr0, grad_DY2_with_lr2 = torch.ops.lact.two_mm_same_inp(
        #     grad_dw0.contiguous(), grad_dw2.contiguous(), K.contiguous(), False, True
        # )

        # [B, 2 * Hidden, D]
        grad_DY0_with_lr0_and_grad_DY2_with_lr2 = torch.bmm(
            grad_dw0_dw2, K.transpose(1, 2)
        )

        #### Next, we do tones of element-wise ops.
        #### These element-wise ops are compiled with torch.compile. one graph?
        (
            grad_DHidden,  # [B, Hidden, num_tokens]
            grad_Y0_Y2,  # [B, 2 * Hidden, num_tokens]
            grad_lr0,  # [B, L]
            grad_lr1,  # [B, L]
            grad_lr2,  # [B, L]
            DY0_with_lr0_and_DY2_with_lr2,  # [B, 2 * Hidden, L]
            Hidden_with_lr1,  # [B, Hidden, L]
        ) = triton_swiglu_bwd_bwd_fused_cat_inp_out(
            # ) = pytorch_swiglu_bwd_bwd_fused_cat_inp_out(
            DHidden,  # [B, Hidden, num_tokens]
            Y0_Y2,  # [B, 2 * Hidden, num_tokens]
            lr0,  # [B, L]
            lr1,  # [B, L]
            lr2,  # [B, L]
            grad_DY0_with_lr0_and_grad_DY2_with_lr2,  # [B, 2 * Hidden, num_tokens]
            grad_Hidden_with_lr1,  # [B, Hidden, num_tokens]
        )

        grad_K = fused_two_mm_same_out_interface(
            DY0_with_lr0_and_DY2_with_lr2,  # [B, 2 * Hidden, num_tokens]
            grad_dw0_dw2.contiguous(),  # [B, 2 * Hidden, D]
            grad_Y0_Y2,
            W0_W2,
            A_transpose=True,
            B_transpose=False,
        )

        # grad_V = two_mm_same_out_interface_v2(
        grad_V = fused_two_mm_same_out_interface(
            grad_DHidden,
            W1,
            Hidden_with_lr1,
            grad_dw1.contiguous(),
            A_transpose=True,
            B_transpose=True,
        )

        #### For below three matmuls, occupancy is the key, cause their dimension might be small.

        # [B, D, num_tokens] @ [B, num_tokens, Hidden].T -> [B, D, Hidden]
        grad_W1 = torch.bmm(V.transpose(1, 2), grad_DHidden.transpose(1, 2))

        # [B, 2 * Hidden, D] @ [B, D, num_tokens] -> [B, 2 * Hidden, num_tokens]
        grad_W0_W2 = torch.bmm(grad_Y0_Y2, K)

        return (
            grad_W0_W2,
            grad_W1,
            grad_K,
            grad_V,
            grad_lr0,
            grad_lr1,
            grad_lr2,
        )


class FusedLactSwiGLUFFNVarlenBwd(torch.autograd.Function):
    """Varlen autograd Function for fused LaCT SwiGLU FFN backward (ttt_grad).
    W0_W2: [G, 2H, D], W1: [G, D, H] — per-doc weights
    K, V: [T, D] — packed tokens
    lr0, lr1, lr2: [T] — per-token learning rates
    cu_seqlens: [G+1]
    Returns: DW0_DW2 [G, 2H, D], DW1 [G, D, H]
    """

    @staticmethod
    def forward(ctx, W0_W2, W1, K, V, lr0, lr1, lr2, cu_seqlens, eff_lens, bos_arr, max_sl):
        chunk_kw = dict(eff_lens=eff_lens, bos_arr=bos_arr, max_sl=max_sl)
        # DY0_DY2: [T, 2H], Hidden: [T, H]
        DY0_DY2, Hidden = swiglu_backward_three_bmm_with_lr_varlen_triton(
            W0_W2, W1, K, V, lr0, lr1, lr2, cu_seqlens, **chunk_kw,
        )
        # DW0_DW2: [G, 2H, D] = DY0_DY2.T @ K per doc
        DW0_DW2 = grouped_gemm_reduce(DY0_DY2, K, cu_seqlens, **chunk_kw)
        # DW1: [G, D, H] = V.T @ Hidden per doc
        DW1 = grouped_gemm_reduce(V, Hidden, cu_seqlens, **chunk_kw)

        ctx.save_for_backward(W0_W2, W1, K, V, lr0, lr1, lr2, cu_seqlens, eff_lens, bos_arr)
        ctx.max_sl = max_sl
        return DW0_DW2, DW1

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_dw0_dw2, grad_dw1):
        W0_W2, W1, K, V, lr0, lr1, lr2, cu_seqlens, eff_lens, bos_arr = ctx.saved_tensors
        chunk_kw = dict(eff_lens=eff_lens, bos_arr=bos_arr, max_sl=ctx.max_sl)

        need_grad_K = ctx.needs_input_grad[2]
        need_grad_V = ctx.needs_input_grad[3]
        need_grad_lr = ctx.needs_input_grad[4]  # lr0, lr1, lr2 share the same flag pattern

        # Recompute intermediates: all [T, ?] packed
        # Y0_Y2: [T, 2H] = K @ W0_W2.T per doc
        Y0_Y2 = grouped_gemm_to_packed(K, W0_W2, cu_seqlens, trans_W=True, **chunk_kw)
        # DHidden: [T, H] = V @ W1 per doc  (W1.T @ V.T transposed)
        DHidden = grouped_gemm_to_packed(V, W1, cu_seqlens, trans_W=False, **chunk_kw)
        # grad_Hidden_with_lr1: [T, H] = V @ grad_dw1 per doc
        grad_Hidden_with_lr1 = grouped_gemm_to_packed(V, grad_dw1, cu_seqlens, trans_W=False, **chunk_kw)
        # grad_DY0_lr0_and_grad_DY2_lr2: [T, 2H] = K @ grad_dw0_dw2.T per doc
        grad_DY0_lr0_and_grad_DY2_lr2 = grouped_gemm_to_packed(
            K, grad_dw0_dw2.contiguous(), cu_seqlens, trans_W=True, **chunk_kw,
        )

        # Pointwise ops: all [T, ?] packed
        (
            grad_DHidden,       # [T, H]
            grad_Y0_Y2,        # [T, 2H]
            grad_lr0,          # [T]
            grad_lr1,          # [T]
            grad_lr2,          # [T]
            DY0_with_lr0_and_DY2_with_lr2,  # [T, 2H]
            Hidden_with_lr1,   # [T, H]
        ) = triton_swiglu_bwd_bwd_fused_cat_inp_out_varlen(
            DHidden,
            Y0_Y2,
            lr0,
            lr1,
            lr2,
            grad_DY0_lr0_and_grad_DY2_lr2,
            grad_Hidden_with_lr1,
            cu_seqlens, **chunk_kw,
        )

        # grad_K: [T, D] (skip if K doesn't need grad)
        grad_K = None
        if need_grad_K:
            grad_K = fused_two_mm_same_out_wT_x_varlen_triton(
                DY0_with_lr0_and_DY2_with_lr2,
                grad_dw0_dw2.contiguous(),
                grad_Y0_Y2,
                W0_W2,
                cu_seqlens, **chunk_kw,
            )

        # grad_V: [T, D] (skip if V doesn't need grad)
        grad_V = None
        if need_grad_V:
            grad_V = fused_two_mm_same_out_wT_xT_varlen_triton(
                grad_DHidden,
                W1,
                Hidden_with_lr1,
                grad_dw1.contiguous(),
                cu_seqlens, **chunk_kw,
            )

        # grad_W1: [G, D, H] = V.T @ grad_DHidden per doc
        grad_W1 = grouped_gemm_reduce(V, grad_DHidden, cu_seqlens, **chunk_kw)
        # grad_W0_W2: [G, 2H, D] = grad_Y0_Y2.T @ K per doc
        grad_W0_W2 = grouped_gemm_reduce(grad_Y0_Y2, K, cu_seqlens, **chunk_kw)

        return (
            grad_W0_W2,
            grad_W1,
            grad_K,
            grad_V,
            grad_lr0,
            grad_lr1,
            grad_lr2,
            None,  # cu_seqlens
            None,  # eff_lens
            None,  # bos_arr
            None,  # max_sl
        )


def fused_lact_swiglu_ffn_fast_weight_grads(W0_W2, W1, K, V, lr0, lr1, lr2,
                                            cu_seqlens=None, eff_lens=None, bos_arr=None, max_sl=0):
    """
    Args:
        W0_W2:    [B, 2 * Hidden, D]
        W1:       [B, D, Hidden]
        K, V:     [B, num_Tokens, D]
        lr0, lr1, lr2:    [B, N]
        eff_lens, bos_arr, max_sl: precomputed (if None, computed from cu_seqlens)

    Outs:
        dW0_W2: [B, 2 * Hidden, D]
        dW1: [B, D, Hidden]
    """
    if cu_seqlens is not None:
        if eff_lens is None:
            eff_lens, bos_arr, max_sl = compute_varlen_args(cu_seqlens)
        return FusedLactSwiGLUFFNVarlenBwd.apply(W0_W2, W1, K, V, lr0, lr1, lr2,
                                                  cu_seqlens, eff_lens, bos_arr, max_sl)
    return FusedLactSwiGLUFFNBwd.apply(W0_W2, W1, K, V, lr0, lr1, lr2)


@torch.compile()
def pytorch_swiglu_bwd_bwd_fused_cat_inp_out(
    dh: torch.Tensor,  # [b, d, l]
    x0_x2: torch.Tensor,  # [b, 2* d, l]
    lr0: torch.Tensor,  # [b, l]
    lr1: torch.Tensor,  # [b, l]
    lr2: torch.Tensor,  # [b, l]
    grad_dx0_dx2: torch.Tensor,  # [b, 2 * d, l]
    grad_hidden_lr1: torch.Tensor,  # [b, d, l]
):
    """
    In previous fwd pass:
    dx0 = lr0 * dh * x2 * sigma * (1 + x0 * (1 - sigma))
    dx2 = lr2 * dh * silu(x0)
    hidden_lr1 = lr1 * x2 * silu(x0)

    In this backward pass:
    grad_dh = grad_dx0 * lr0 * x2 * sigma * (1 + x0 * (1 - sigma)) + grad_dx2 * lr2 * silu(x0)

    grad_x2 = grad_dx0 * lr0 * dh * sigma * (1 + x0 * (1 - sigma)) + grad_hidden_lr1 * lr1 * sigma * x0
    # for grad_x0, a little bit tricky,
    - grad_sigma = grad_dx0 * lr0 * dh * x2 * (1 + x0 - 2 sigma * x0)
    - grad_x0_naive  = grad_dx2 * lr2 * dh * sigma * (1 + x0 * (1 - sigma)) +  grad_dx0 * lr0 * dh * x2 * sigma * (1 - sigma) + grad_hidden_lr1 * lr1 * x2 * dsilu_x0_multiplier
    grad_x0 = grad_x0_naive + grad_sigma * sigma * (1 - sigma)

    # then sum of the last dimension (the d dimension!)
    grad_lr0 = grad_dx0 * dh * x2 * sigma * (1 + x0 * (1 - sigma)) # need to sum over all the d of the same l
    grad_lr2 = grad_dx2 * dh * silu(x0)
    grad_lr1 = grad_hidden_lr1 * x2 * sigma * x0

    """
    lr0 = lr0.unsqueeze(dim=1)
    lr1 = lr1.unsqueeze(dim=1)
    lr2 = lr2.unsqueeze(dim=1)

    x0, x2 = x0_x2.chunk(2, dim=1)
    grad_dx0, grad_dx2 = grad_dx0_dx2.chunk(2, dim=1)

    sigma = torch.sigmoid(x0)
    silu_x0 = torch.nn.functional.silu(x0)
    silu_bp_multiplier = sigma * (1 + x0 * (1 - sigma))
    grad_dh = grad_dx0 * lr0 * x2 * silu_bp_multiplier + grad_dx2 * lr2 * silu_x0
    grad_x2 = grad_dx0 * lr0 * dh * silu_bp_multiplier + grad_hidden_lr1 * lr1 * silu_x0

    grad_sigma = grad_dx0 * lr0 * dh * x2 * (1 + x0 - 2 * sigma * x0)
    grad_x0_naive = (
        grad_dx2 * lr2 * dh + grad_hidden_lr1 * lr1 * x2
    ) * silu_bp_multiplier + grad_dx0 * lr0 * dh * x2 * sigma * (1 - sigma)
    grad_x0 = grad_x0_naive + grad_sigma * sigma * (1 - sigma)
    grad_lr0 = grad_dx0 * dh * x2 * silu_bp_multiplier
    grad_lr1 = grad_hidden_lr1 * x2 * silu_x0
    grad_lr2 = grad_dx2 * dh * silu_x0

    grad_lr0 = grad_lr0.sum(dim=1, keepdim=False)
    grad_lr1 = grad_lr1.sum(dim=1, keepdim=False)
    grad_lr2 = grad_lr2.sum(dim=1, keepdim=False)

    # also for the first order backward:

    dx2 = silu_x0 * dh
    dx0 = dh * x2 * silu_bp_multiplier

    dx0 = dx0 * lr0
    dx2 = dx2 * lr2

    hidden_lr1 = lr1 * x2 * silu_x0

    grad_x0_x2 = torch.cat([grad_x0, grad_x2], dim=1)
    dx0_x2 = torch.cat([dx0, dx2], dim=1)

    x_dtype = x0.dtype

    # grad_dh_and_hidden_lr1 = torch.cat([grad_dh, hidden_lr1], dim=1)

    return (
        grad_dh.to(x_dtype),
        grad_x0_x2.to(x_dtype),
        grad_lr0,
        grad_lr1,
        grad_lr2,
        dx0_x2.to(x_dtype),
        hidden_lr1.to(x_dtype),
    )


########################################################
# Pytorch Reference implementation
########################################################


@torch.compile
def reference_lact_swiglu_ffn_fast_weight_grads(W0_W2, W1, K, V, lr0, lr1, lr2):
    """
    Args:
        W0, W2: [B, M, K] or [B, Hidden, D]
        W1:     [B, K, M] or [B, D, Hidden]
        X:      [M, N, K] or [B, num_Tokens, D]
    """
    W0, W2 = W0_W2.chunk(2, dim=1)
    Y0 = torch.bmm(W0, K.transpose(1, 2))
    Y2 = torch.bmm(W2, K.transpose(1, 2))

    DHidden = torch.bmm(W1.transpose(1, 2), V.transpose(1, 2))

    # DY0_with_lr0, DY2_with_lr2, Hidden_with_lr1 = ref_pytorch_swiglu_bwd(
    #     DHidden, Y0, Y2, lr0, lr1, lr2
    # )
    ### Element-wise ops
    lr0 = lr0.unsqueeze(dim=1)
    lr1 = lr1.unsqueeze(dim=1)
    lr2 = lr2.unsqueeze(dim=1)

    x0_sigmoid = torch.sigmoid(Y0)

    dx2 = x0_sigmoid * Y0 * DHidden
    dx0 = DHidden * Y2 * x0_sigmoid * (1 + Y0 * (1 - x0_sigmoid))

    DY0_with_lr0 = dx0 * lr0
    DY2_with_lr2 = dx2 * lr2

    Hidden_with_lr1 = lr1 * Y2 * torch.nn.functional.silu(Y0)
    ### Element-wise ops done.

    DY0_with_lr0_and_DY2_with_lr2 = torch.cat([DY0_with_lr0, DY2_with_lr2], dim=1)

    DW0_DW2 = torch.bmm(DY0_with_lr0_and_DY2_with_lr2, K)
    DW1 = torch.bmm(V.transpose(1, 2), Hidden_with_lr1.transpose(1, 2))

    return DW0_DW2, DW1


if __name__ == "__main__":
    from kernel_test_utils import test_correctness, benchmark, get_chunk_info
    from grouped_gemm import _pack_to_padded, _padded_to_pack

    device = "cuda"
    num_docs = 4
    doc_lens = [4096, 3072, 2048, 1024]
    d, dh = 512, 512
    cu_seqlens = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(doc_lens), 0).tolist()),
        dtype=torch.int32, device=device,
    )
    T = cu_seqlens[-1].item()
    max_len = max(doc_lens)

    W0_W2 = torch.randn(num_docs, 2 * dh, d, device=device, dtype=torch.bfloat16)
    W1 = torch.randn(num_docs, d, dh, device=device, dtype=torch.bfloat16)
    K_tok = torch.randn(T, d, device=device, dtype=torch.bfloat16)
    V_tok = torch.randn(T, d, device=device, dtype=torch.bfloat16)
    lr0 = torch.randn(T, device=device, dtype=torch.float32) * 0.01
    lr1 = torch.randn(T, device=device, dtype=torch.float32) * 0.01
    lr2 = torch.randn(T, device=device, dtype=torch.float32) * 0.01

    class FakeCtx:
        def save_for_backward(self, *args): self.saved_tensors = args

    print(f"Config: {num_docs=}, {doc_lens=}, {d=}, {dh=}, {T=}")
    print()

    # ===== Forward =====
    print("=" * 60)
    print("Forward: varlen vs padded FusedLactSwiGLUFFNBwd")
    print("=" * 60)

    # Pre-pad for fair benchmark
    K_pad = _pack_to_padded(K_tok, cu_seqlens, max_len)
    V_pad = _pack_to_padded(V_tok, cu_seqlens, max_len)
    lr0_pad = _pack_to_padded(lr0.unsqueeze(1), cu_seqlens, max_len).squeeze(2)
    lr1_pad = _pack_to_padded(lr1.unsqueeze(1), cu_seqlens, max_len).squeeze(2)
    lr2_pad = _pack_to_padded(lr2.unsqueeze(1), cu_seqlens, max_len).squeeze(2)

    eff_lens, bos_arr, max_sl_val = compute_varlen_args(cu_seqlens)

    def varlen_fwd(W0_W2, W1, K, V, lr0, lr1, lr2):
        ctx = FakeCtx()
        return FusedLactSwiGLUFFNVarlenBwd.forward(ctx, W0_W2, W1, K, V, lr0, lr1, lr2, cu_seqlens, eff_lens, bos_arr, max_sl_val)

    def ref_fwd_for_test(W0_W2, W1, K, V, lr0, lr1, lr2):
        ctx = FakeCtx()
        Kp = _pack_to_padded(K, cu_seqlens, max_len)
        Vp = _pack_to_padded(V, cu_seqlens, max_len)
        lr0p = _pack_to_padded(lr0.unsqueeze(1), cu_seqlens, max_len).squeeze(2)
        lr1p = _pack_to_padded(lr1.unsqueeze(1), cu_seqlens, max_len).squeeze(2)
        lr2p = _pack_to_padded(lr2.unsqueeze(1), cu_seqlens, max_len).squeeze(2)
        return FusedLactSwiGLUFFNBwd.forward(ctx, W0_W2, W1, Kp, Vp, lr0p, lr1p, lr2p)

    def ref_fwd_bench(W0_W2, W1, K_pad, V_pad, lr0_pad, lr1_pad, lr2_pad):
        ctx = FakeCtx()
        return FusedLactSwiGLUFFNBwd.forward(ctx, W0_W2, W1, K_pad, V_pad, lr0_pad, lr1_pad, lr2_pad)

    fwd_args = (W0_W2, W1, K_tok, V_tok, lr0, lr1, lr2)
    fwd_bench_ref_args = (W0_W2, W1, K_pad, V_pad, lr0_pad, lr1_pad, lr2_pad)
    test_correctness(varlen_fwd, ref_fwd_for_test, fwd_args, fwd_args, atol=1e-2, rtol=1e-2)
    benchmark(varlen_fwd, ref_fwd_bench, fwd_args, fwd_bench_ref_args)

    # ===== Backward =====
    print()
    print("=" * 60)
    print("Backward: varlen vs padded FusedLactSwiGLUFFNBwd")
    print("=" * 60)

    grad_dw0_dw2 = torch.randn(num_docs, 2 * dh, d, device=device, dtype=torch.bfloat16) * 0.1
    grad_dw1 = torch.randn(num_docs, d, dh, device=device, dtype=torch.bfloat16) * 0.1

    def varlen_bwd(grad_dw0_dw2, grad_dw1):
        ctx = FakeCtx()
        ctx.saved_tensors = (W0_W2, W1, K_tok, V_tok, lr0, lr1, lr2, cu_seqlens, eff_lens, bos_arr)
        ctx.max_sl = max_sl_val
        return FusedLactSwiGLUFFNVarlenBwd.backward(ctx, grad_dw0_dw2, grad_dw1)[:7]

    def ref_bwd_for_test(grad_dw0_dw2, grad_dw1):
        ctx = FakeCtx()
        ctx.saved_tensors = (W0_W2, W1, K_pad, V_pad, lr0_pad, lr1_pad, lr2_pad)
        grads = FusedLactSwiGLUFFNBwd.backward(ctx, grad_dw0_dw2, grad_dw1)
        return (
            grads[0], grads[1],
            _padded_to_pack(grads[2], cu_seqlens, T),
            _padded_to_pack(grads[3], cu_seqlens, T),
            _padded_to_pack(grads[4].unsqueeze(2), cu_seqlens, T).squeeze(1),
            _padded_to_pack(grads[5].unsqueeze(2), cu_seqlens, T).squeeze(1),
            _padded_to_pack(grads[6].unsqueeze(2), cu_seqlens, T).squeeze(1),
        )

    def ref_bwd_bench(grad_dw0_dw2, grad_dw1):
        ctx = FakeCtx()
        ctx.saved_tensors = (W0_W2, W1, K_pad, V_pad, lr0_pad, lr1_pad, lr2_pad)
        return FusedLactSwiGLUFFNBwd.backward(ctx, grad_dw0_dw2, grad_dw1)

    bwd_args = (grad_dw0_dw2, grad_dw1)
    test_correctness(varlen_bwd, ref_bwd_for_test, bwd_args, bwd_args, atol=1e-2, rtol=1e-2)
    benchmark(varlen_bwd, ref_bwd_bench, bwd_args, bwd_args)

    # ===== Chunk correctness: forward =====
    chunk_size = 2048
    _, _, max_chunks = get_chunk_info(cu_seqlens, chunk_size, 0)

    print()
    print("=" * 60)
    print(f"Chunk correctness: forward (chunk_size={chunk_size})")
    print("=" * 60)

    for chunk_index in range(max_chunks):
        chunk_cu, idx, _ = get_chunk_info(cu_seqlens, chunk_size, chunk_index)
        if len(idx) == 0:
            break
        # Reference: gather chunk tokens, call base varlen forward
        ref_dw0w2, ref_dw1 = fused_lact_swiglu_ffn_fast_weight_grads(
            W0_W2, W1, K_tok[idx], V_tok[idx], lr0[idx], lr1[idx], lr2[idx], cu_seqlens=chunk_cu,
        )
        # Test: call chunk forward on full buffer with precomputed chunk args
        chunk_eff, chunk_bos, chunk_max = compute_varlen_args(cu_seqlens, chunk_size, chunk_index)
        test_dw0w2, test_dw1 = fused_lact_swiglu_ffn_fast_weight_grads(
            W0_W2, W1, K_tok, V_tok, lr0, lr1, lr2,
            cu_seqlens=cu_seqlens, eff_lens=chunk_eff, bos_arr=chunk_bos, max_sl=chunk_max,
        )
        # Outputs are per-doc [G, ...], compare directly
        diff_w0w2 = (test_dw0w2 - ref_dw0w2).abs().max().item()
        diff_w1 = (test_dw1 - ref_dw1).abs().max().item()
        print(f"  chunk_idx={chunk_index}: DW0_W2 max_diff={diff_w0w2:.2e}, DW1 max_diff={diff_w1:.2e}, n_tokens={len(idx)}")
        assert diff_w0w2 == 0.0 and diff_w1 == 0.0, f"chunk_idx={chunk_index} not exact match!"

    print("✓ PASS (exact match)")
