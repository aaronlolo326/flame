import torch


try:
    from triton_swiglu_bwd_kernels import (
        swiglu_backward_three_bmm_triton,
        swiglu_backward_three_bmm_triton_op,
        swiglu_backward_three_bmm_varlen_triton,
    )
    from triton_swiglu_kernels import fused_two_mm_swiglu_triton, fused_two_mm_swiglu_varlen_triton
    from grouped_gemm import grouped_gemm_to_packed, grouped_gemm_reduce, _pack_to_padded, _padded_to_pack
    from utils import compute_varlen_args
except ImportError:
    from .triton_swiglu_bwd_kernels import (
        swiglu_backward_three_bmm_triton,
        swiglu_backward_three_bmm_triton_op,
        swiglu_backward_three_bmm_varlen_triton,
    )
    from .triton_swiglu_kernels import fused_two_mm_swiglu_triton, fused_two_mm_swiglu_varlen_triton
    from .grouped_gemm import grouped_gemm_to_packed, grouped_gemm_reduce, _pack_to_padded, _padded_to_pack
    from .utils import compute_varlen_args
from torch.autograd.function import once_differentiable


class FusedSwiGLUFFNFwd(torch.autograd.Function):

    @staticmethod
    def forward(ctx, W0_W2, W1, X):
        """
        Args:
            W0_W2: [B, 2 * Hidden, D]
            W1:     [B, K, M] or [B, D, Hidden]
            X:      [M, N, K] or [B, num_Tokens, D]
        Outs:
            Hidden: [B, N, K] or [B, num_tokens, Hidden]

        W1 @ [SiLU(W0 @ X.T) * (W2 @ X.T)]
        """

        # [B, Hidden, num_tokens]
        #### Without this triton kernel, we will materize Y2, SiLU(Y0) * Y2.
        #### 2 + 1 read and write.
        # Here we only have one write.
        Hidden = fused_two_mm_swiglu_triton(W0_W2, X)

        # -> [B, num_tokens, D]
        # output = torch.bmm(W1, Hidden).transpose(1, 2)
        output = torch.bmm(Hidden.transpose(1, 2), W1.transpose(1, 2))
        ctx.save_for_backward(W0_W2, W1, X)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        """
        Args:
            grad_out: [B, num_tokens, D]
        Outs:
            grad_W0_W2: [B, 2 * Hidden, D]
            grad_W1: [B, D, Hidden]
            grad_X: [B, D, num_tokens]
        """
        W0_W2, W1, X = ctx.saved_tensors
        # [B, 2 * Hidden, num_tokens]
        DY0_DY2, Hidden = swiglu_backward_three_bmm_triton(
            # DY0_DY2, Hidden = swiglu_backward_three_bmm_triton_op(
            W0_W2,
            W1,
            X,
            grad_out.contiguous(),
        )

        # [B, D, num_tokens] @ [B, num_tokens, Hidden] -> [B, D, Hidden]
        grad_W1 = torch.bmm(grad_out.transpose(1, 2), Hidden.transpose(1, 2))

        # [B, 2 * Hidden, num_tokens] @ [B, num_tokens, D] -> [B, 2 * Hidden, D]
        grad_W0_W2 = torch.bmm(DY0_DY2, X)

        # [B, 2 * Hidden, num_tokens].T @ [B, 2 * Hidden, D] -> [B, 2 * Hidden, D]
        grad_X = torch.bmm(DY0_DY2.transpose(1, 2), W0_W2)

        return (grad_W0_W2, grad_W1, grad_X)


class FusedSwiGLUFFNVarlenFwd(torch.autograd.Function):
    """Varlen autograd Function for fused SwiGLU FFN forward.
    W0_W2: [G, 2*Hidden, D], W1: [G, D, Hidden] — per-doc weights
    X: [T, D] — packed tokens, cu_seqlens: [G+1]
    eff_lens: [G] int32, bos_arr: [G] int64, max_sl: int — precomputed
    """

    @staticmethod
    def forward(ctx, W0_W2, W1, X, cu_seqlens, eff_lens, bos_arr, max_sl, out=None):
        # [T, Hidden] = SiLU(W0 @ X.T) * (W2 @ X.T) per doc, packed
        Hidden = fused_two_mm_swiglu_varlen_triton(
            W0_W2, X, cu_seqlens, eff_lens=eff_lens, bos_arr=bos_arr, max_sl=max_sl,
        )
        # [T, D] = Hidden @ W1.T per doc
        output = grouped_gemm_to_packed(
            Hidden, W1, cu_seqlens, trans_W=True, eff_lens=eff_lens, bos_arr=bos_arr, max_sl=max_sl, out=out,
        )
        ctx.save_for_backward(W0_W2, W1, X, cu_seqlens, eff_lens, bos_arr)
        ctx.max_sl = max_sl
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        """
        grad_out: [T, D]
        Returns: grad_W0_W2 [G, 2H, D], grad_W1 [G, D, H], grad_X [T, D], None * 4
        """
        W0_W2, W1, X, cu_seqlens, eff_lens, bos_arr = ctx.saved_tensors
        max_sl = ctx.max_sl

        chunk_kw = dict(eff_lens=eff_lens, bos_arr=bos_arr, max_sl=max_sl)

        # Recompute DY0_DY2: [T, 2H], Hidden: [T, H]
        DY0_DY2, Hidden = swiglu_backward_three_bmm_varlen_triton(
            W0_W2, W1, X, grad_out.contiguous(), cu_seqlens, **chunk_kw,
        )

        # grad_out.T @ Hidden per doc -> [G, D, H]
        grad_W1 = grouped_gemm_reduce(grad_out, Hidden, cu_seqlens, **chunk_kw)

        # DY0_DY2.T @ X per doc -> [G, 2H, D]
        grad_W0_W2 = grouped_gemm_reduce(DY0_DY2, X, cu_seqlens, **chunk_kw)

        # DY0_DY2 @ W0_W2 per doc -> [T, D] (skip if X doesn't need grad)
        grad_X = None
        if ctx.needs_input_grad[2]:
            grad_X = grouped_gemm_to_packed(
                DY0_DY2, W0_W2, cu_seqlens, trans_W=False, **chunk_kw,
            )

        return (grad_W0_W2, grad_W1, grad_X, None, None, None, None, None)


def fused_swiglu_ffn_fwd(W0_W2, W1, X, cu_seqlens=None, eff_lens=None, bos_arr=None, max_sl=0, out=None):
    """
    Args:
        W0_W2: [B, 2 * Hidden, D]
        W1:     [B, D, Hidden]
        X:      [B, num_Tokens, D]
        cu_seqlens: optional [G+1] int32 for varlen mode
        eff_lens, bos_arr, max_sl: precomputed (if None, computed from cu_seqlens)
        out: optional [T, D] output buffer (kernel writes directly, no alloc)
    Outs:
        output: [B, num_tokens, D]  (batched) or [T, D] (varlen)
    """
    if cu_seqlens is not None:
        if eff_lens is None:
            eff_lens, bos_arr, max_sl = compute_varlen_args(cu_seqlens)
        return FusedSwiGLUFFNVarlenFwd.apply(W0_W2, W1, X, cu_seqlens, eff_lens, bos_arr, max_sl, out)
    return FusedSwiGLUFFNFwd.apply(W0_W2, W1, X)


############################################################
# Pytorch Reference Code Below
############################################################


@torch.compile
def reference_swiglu_ffn_fwd(W0_W2, W1, X):
    """
    Args:
        W0, W2: [B, M, K] or [B, Hidden, D]
        W1:     [B, K, M] or [B, D, Hidden]
        X:      [M, N, K] or [B, num_Tokens, D]
    """
    W0, W2 = W0_W2.chunk(2, dim=1)
    Y0 = torch.bmm(W0, X.transpose(1, 2))
    Y2 = torch.bmm(W2, X.transpose(1, 2))
    Hidden = torch.nn.functional.silu(Y0) * Y2
    return torch.bmm(W1, Hidden).transpose(1, 2)


if __name__ == "__main__":
    from kernel_test_utils import test_correctness, benchmark, get_chunk_info, BENCH_DOC_LENS, make_cu_seqlens
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
    X = torch.randn(T, d, device=device, dtype=torch.bfloat16)

    print(f"Config: {num_docs=}, {doc_lens=}, {d=}, {dh=}, {T=}")

    class FakeCtx:
        def save_for_backward(self, *args): self.saved_tensors = args

    eff_lens, bos_arr, max_sl_val = compute_varlen_args(cu_seqlens)

    # ===== Forward correctness =====
    print()
    print("=" * 60)
    print("Forward: varlen vs padded FusedSwiGLUFFNFwd")
    print("=" * 60)

    X_pad = _pack_to_padded(X, cu_seqlens, max_len)

    def varlen_fwd(W0_W2, W1, X):
        ctx = FakeCtx()
        return FusedSwiGLUFFNVarlenFwd.forward(ctx, W0_W2, W1, X, cu_seqlens, eff_lens, bos_arr, max_sl_val)

    def ref_fwd_for_test(W0_W2, W1, X):
        ctx = FakeCtx()
        X_pad = _pack_to_padded(X, cu_seqlens, max_len)
        return _padded_to_pack(FusedSwiGLUFFNFwd.forward(ctx, W0_W2, W1, X_pad), cu_seqlens, T)

    test_correctness(varlen_fwd, ref_fwd_for_test, (W0_W2, W1, X), (W0_W2, W1, X), atol=1e-2, rtol=1e-2)

    # ===== Backward correctness =====
    print()
    print("=" * 60)
    print("Backward: varlen vs padded FusedSwiGLUFFNFwd")
    print("=" * 60)

    grad_out = torch.randn(T, d, device=device, dtype=torch.bfloat16)

    def varlen_bwd(W0_W2, W1, X):
        ctx = type('Ctx', (), {
            'saved_tensors': (W0_W2, W1, X, cu_seqlens, eff_lens, bos_arr),
            'max_sl': max_sl_val,
            'needs_input_grad': (True, True, True, False, False, False, False, False),
        })()
        return FusedSwiGLUFFNVarlenFwd.backward(ctx, grad_out)[:3]

    def ref_bwd_for_test(W0_W2, W1, X):
        X_pad = _pack_to_padded(X, cu_seqlens, max_len)
        grad_out_pad = _pack_to_padded(grad_out, cu_seqlens, max_len)
        ctx = type('Ctx', (), {'saved_tensors': (W0_W2, W1, X_pad)})()
        gW0W2, gW1, gX_pad = FusedSwiGLUFFNFwd.backward(ctx, grad_out_pad)
        return gW0W2, gW1, _padded_to_pack(gX_pad, cu_seqlens, T)

    test_correctness(varlen_bwd, ref_bwd_for_test, (W0_W2, W1, X), (W0_W2, W1, X), atol=1e-2, rtol=1e-2)

    # ===== Benchmarks: forward + backward across configs =====
    print()
    print("=" * 72)
    print(f"{'Config':<16} {'Fwd ms':>10} {'Fwd Pad':>10} {'Fwd Spd':>8}   {'Bwd ms':>10} {'Bwd Pad':>10} {'Bwd Spd':>8}")
    print("=" * 72)

    for cfg_name, dl in BENCH_DOC_LENS.items():
        cu = make_cu_seqlens(dl, device)
        T_b = cu[-1].item()
        G = len(dl)
        ml = max(dl)
        eff, bos, msl = compute_varlen_args(cu)

        W0_W2_b = torch.randn(G, 2 * dh, d, device=device, dtype=torch.bfloat16)
        W1_b = torch.randn(G, d, dh, device=device, dtype=torch.bfloat16)
        X_b = torch.randn(T_b, d, device=device, dtype=torch.bfloat16)
        X_pad_b = _pack_to_padded(X_b, cu, ml)
        go_b = torch.randn(T_b, d, device=device, dtype=torch.bfloat16)
        go_pad_b = _pack_to_padded(go_b, cu, ml)

        def _varlen_fwd(W, W1, Xv, _cu=cu, _e=eff, _b=bos, _m=msl):
            ctx = FakeCtx()
            return FusedSwiGLUFFNVarlenFwd.forward(ctx, W, W1, Xv, _cu, _e, _b, _m)

        def _ref_fwd(W, W1, Xp):
            ctx = FakeCtx()
            return FusedSwiGLUFFNFwd.forward(ctx, W, W1, Xp)

        rf = benchmark(_varlen_fwd, _ref_fwd,
                       (W0_W2_b, W1_b, X_b), (W0_W2_b, W1_b, X_pad_b), verbose=False)

        def _varlen_bwd(W, W1, Xv, _cu=cu, _e=eff, _b=bos, _m=msl, _go=go_b):
            ctx = type('Ctx', (), {
                'saved_tensors': (W, W1, Xv, _cu, _e, _b),
                'max_sl': _m,
                'needs_input_grad': (True, True, True, False, False, False, False, False),
            })()
            return FusedSwiGLUFFNVarlenFwd.backward(ctx, _go)[:3]

        def _ref_bwd(W, W1, Xp, _go=go_pad_b):
            ctx = type('Ctx', (), {'saved_tensors': (W, W1, Xp)})()
            return FusedSwiGLUFFNFwd.backward(ctx, _go)

        rb = benchmark(_varlen_bwd, _ref_bwd,
                       (W0_W2_b, W1_b, X_b), (W0_W2_b, W1_b, X_pad_b), verbose=False)

        print(f"{cfg_name:<16} {rf['time_triton']:>10.4f} {rf['time_ref']:>10.4f} {rf['speedup']:>7.2f}x"
              f"   {rb['time_triton']:>10.4f} {rb['time_ref']:>10.4f} {rb['speedup']:>7.2f}x")

    print("=" * 72)

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
        ref_out = fused_swiglu_ffn_fwd(W0_W2, W1, X[idx], cu_seqlens=chunk_cu)
        # Test: call chunk forward on full buffer with precomputed chunk args
        chunk_eff, chunk_bos, chunk_max = compute_varlen_args(cu_seqlens, chunk_size, chunk_index)
        test_out = fused_swiglu_ffn_fwd(W0_W2, W1, X, cu_seqlens=cu_seqlens, eff_lens=chunk_eff, bos_arr=chunk_bos, max_sl=chunk_max)
        diff = (test_out[idx] - ref_out).abs().max().item()
        print(f"  chunk_idx={chunk_index}: max_diff={diff:.2e}, n_tokens={len(idx)}")
        assert diff == 0.0, f"chunk_idx={chunk_index} not exact match!"

    print("✓ PASS (exact match)")
