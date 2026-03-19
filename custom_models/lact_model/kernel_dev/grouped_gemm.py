"""Triton grouped GEMM kernels for varlen (cu_seqlens) packed sequences.

Replaces torch.bmm for variable-length document sequences:
  - grouped_gemm_to_packed: per-doc X @ W, output packed as [total_tokens, N]
  - grouped_gemm_reduce: per-doc A.T @ B, output as [num_docs, M, K]
"""

import torch
import triton
import triton.language as tl

_TORCH_TO_TL = {
    torch.bfloat16: tl.bfloat16,
    torch.float16: tl.float16,
    torch.float32: tl.float32,
}


# ======================== grouped_gemm_to_packed ========================
#
# Per doc g:  Y[cu[g]:cu[g+1]] = X[cu[g]:cu[g+1]] @ (W[g].T if trans_W else W[g])
#
# Grid: (cdiv(max_seqlen, BLOCK_M), cdiv(N, BLOCK_N), num_docs)
# Docs shorter than their token-tile simply early-exit.


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
    ],
    key=["N", "K"],
)
@triton.jit
def _grouped_gemm_to_packed_kernel(
    X_ptr, W_ptr, Y_ptr, eff_lens_ptr, bos_arr_ptr,
    N, K,
    stride_x_t, stride_x_k,
    stride_w_g, stride_w_k, stride_w_n,  # "effective" strides: W viewed as [G, K, N]
    stride_y_t, stride_y_n,
    OUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)  # token tile within doc
    pid_n = tl.program_id(1)  # output feature tile
    pid_g = tl.program_id(2)  # doc / group

    doc_len = tl.load(eff_lens_ptr + pid_g).to(tl.int32)

    m_start = pid_m * BLOCK_M
    if m_start >= doc_len:
        return

    doc_start = tl.load(bos_arr_ptr + pid_g).to(tl.int64)

    offs_m = m_start + tl.arange(0, BLOCK_M)
    mask_m = offs_m < doc_len
    glob_m = doc_start + offs_m.to(tl.int64)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    w_base = W_ptr + pid_g.to(tl.int64) * stride_w_g
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        x = tl.load(
            X_ptr + glob_m[:, None] * stride_x_t + offs_k[None, :] * stride_x_k,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        w = tl.load(
            w_base + offs_k[:, None] * stride_w_k + offs_n[None, :] * stride_w_n,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        )
        acc += tl.dot(x, w)

    y_ptrs = Y_ptr + glob_m[:, None] * stride_y_t + offs_n[None, :] * stride_y_n
    tl.store(y_ptrs, acc.to(OUT_DTYPE), mask=mask_m[:, None] & mask_n[None, :])


def grouped_gemm_to_packed(
    X, W, cu_seqlens, trans_W=False,
    chunk_size=0, chunk_idx=0,
    eff_lens=None, bos_arr=None, max_sl=0,
    out=None,
):
    """Per doc g: Y[cu[g]:cu[g+1]] = X[cu[g]:cu[g+1]] @ (W[g].T if trans_W else W[g])

    Args:
        X: [total_tokens, K] -- packed tokens
        W: [num_docs, K, N] (trans_W=False) or [num_docs, N, K] (trans_W=True)
        cu_seqlens: [num_docs + 1] int32
        eff_lens, bos_arr, max_sl: precomputed (if None, computed from cu_seqlens + chunk params)
        out: optional [total_tokens, N] output buffer (writes in-place, no alloc)
    Returns:
        Y: [total_tokens, N]
    """
    try:
        from utils import compute_varlen_args
    except ImportError:
        from .utils import compute_varlen_args

    T, Kx = X.shape
    G = cu_seqlens.shape[0] - 1

    if trans_W:
        N, Kw = W.shape[1], W.shape[2]
        sw_g, sw_k, sw_n = W.stride(0), W.stride(2), W.stride(1)
    else:
        Kw, N = W.shape[1], W.shape[2]
        sw_g, sw_k, sw_n = W.stride(0), W.stride(1), W.stride(2)

    assert Kx == Kw, f"K mismatch: X has {Kx}, W has {Kw}"

    Y = out if out is not None else torch.zeros(T, N, device=X.device, dtype=X.dtype)

    if eff_lens is None:
        eff_lens, bos_arr, max_sl = compute_varlen_args(cu_seqlens, chunk_size, chunk_idx)

    grid = lambda META: (
        triton.cdiv(max_sl, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
        G,
    )
    _grouped_gemm_to_packed_kernel[grid](
        X, W, Y, eff_lens, bos_arr,
        N, Kx,
        X.stride(0), X.stride(1),
        sw_g, sw_k, sw_n,
        Y.stride(0), Y.stride(1),
        OUT_DTYPE=_TORCH_TO_TL[X.dtype],
    )
    return Y


# ======================== grouped_gemm_reduce ========================
#
# Per doc g:  C[g] = A[cu[g]:cu[g+1]].T @ B[cu[g]:cu[g+1]]
#
# Grid: (cdiv(M, BLOCK_M), cdiv(K, BLOCK_K), num_docs)
# Reduction over the token dimension.


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 128, "BLOCK_T": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 64, "BLOCK_T": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 128, "BLOCK_T": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 64, "BLOCK_T": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 64, "BLOCK_T": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 64, "BLOCK_T": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 32, "BLOCK_T": 64}, num_warps=4, num_stages=3),
    ],
    key=["M", "K"],
)
@triton.jit
def _grouped_gemm_reduce_kernel(
    A_ptr, B_ptr, C_ptr, eff_lens_ptr, bos_arr_ptr,
    M, K,
    stride_a_t, stride_a_m,
    stride_b_t, stride_b_k,
    stride_c_g, stride_c_m, stride_c_k,
    OUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_m = tl.program_id(0)  # A-feature tile
    pid_k = tl.program_id(1)  # B-feature tile
    pid_g = tl.program_id(2)  # doc / group

    doc_len = tl.load(eff_lens_ptr + pid_g).to(tl.int32)
    doc_start = tl.load(bos_arr_ptr + pid_g).to(tl.int64)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_k = offs_k < K

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    for t0 in range(0, doc_len, BLOCK_T):
        offs_t = t0 + tl.arange(0, BLOCK_T)
        mask_t = offs_t < doc_len
        glob_t = doc_start + offs_t.to(tl.int64)

        # A transposed: load as [BLOCK_M, BLOCK_T] from A stored as [T, M]
        a_T = tl.load(
            A_ptr + glob_t[None, :] * stride_a_t + offs_m[:, None] * stride_a_m,
            mask=mask_m[:, None] & mask_t[None, :],
            other=0.0,
        )
        # B: [BLOCK_T, BLOCK_K]
        b = tl.load(
            B_ptr + glob_t[:, None] * stride_b_t + offs_k[None, :] * stride_b_k,
            mask=mask_t[:, None] & mask_k[None, :],
            other=0.0,
        )
        acc += tl.dot(a_T, b)

    c_ptrs = (
        C_ptr
        + pid_g.to(tl.int64) * stride_c_g
        + offs_m[:, None] * stride_c_m
        + offs_k[None, :] * stride_c_k
    )
    tl.store(c_ptrs, acc.to(OUT_DTYPE), mask=mask_m[:, None] & mask_k[None, :])


def grouped_gemm_reduce(
    A, B, cu_seqlens,
    chunk_size=0, chunk_idx=0,
    eff_lens=None, bos_arr=None, max_sl=0,
):
    """Per doc g: C[g] = A[cu[g]:cu[g+1]].T @ B[cu[g]:cu[g+1]]

    Args:
        A: [total_tokens, M] -- packed
        B: [total_tokens, K] -- packed
        cu_seqlens: [num_docs + 1] int32
        eff_lens, bos_arr, max_sl: precomputed (if None, computed from cu_seqlens + chunk params)
    Returns:
        C: [num_docs, M, K]
    """
    try:
        from utils import compute_varlen_args
    except ImportError:
        from .utils import compute_varlen_args

    _, M = A.shape
    _, K = B.shape
    G = cu_seqlens.shape[0] - 1

    C = torch.empty(G, M, K, device=A.device, dtype=A.dtype)

    if eff_lens is None:
        eff_lens, bos_arr, max_sl = compute_varlen_args(cu_seqlens, chunk_size, chunk_idx)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(K, META["BLOCK_K"]),
        G,
    )
    _grouped_gemm_reduce_kernel[grid](
        A, B, C, eff_lens, bos_arr,
        M, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1), C.stride(2),
        OUT_DTYPE=_TORCH_TO_TL[A.dtype],
    )
    return C


# ======================== Padding helpers ========================


def _pack_to_padded(X, cu_seqlens, max_len):
    """[total_tokens, D] -> [G, max_len, D] zero-padded."""
    G = cu_seqlens.shape[0] - 1
    D = X.shape[1]
    out = X.new_zeros(G, max_len, D)
    for i in range(G):
        s, e = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
        out[i, : e - s] = X[s:e]
    return out


def _padded_to_pack(Y_padded, cu_seqlens, total_tokens):
    """[G, max_len, D] -> [total_tokens, D]."""
    D = Y_padded.shape[2]
    out = Y_padded.new_zeros(total_tokens, D)
    G = cu_seqlens.shape[0] - 1
    for i in range(G):
        s, e = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
        out[s:e] = Y_padded[i, : e - s]
    return out


# ======================== Reference implementations (padded bmm) ========================


def ref_grouped_gemm_to_packed(X, W, cu_seqlens, trans_W=False):
    """Pad -> torch.bmm -> unpack.  Fair baseline for benchmarking."""
    G = cu_seqlens.shape[0] - 1
    T = X.shape[0]
    doc_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    max_len = doc_lens.max().item()

    X_pad = _pack_to_padded(X, cu_seqlens, max_len)  # [G, max_len, K]
    if trans_W:
        # X @ W.T:  [G, max_len, K] @ [G, K, N]  (W is [G, N, K])
        Y_pad = torch.bmm(X_pad, W.transpose(1, 2))
    else:
        # X @ W:  [G, max_len, K] @ [G, K, N]
        Y_pad = torch.bmm(X_pad, W)
    return _padded_to_pack(Y_pad, cu_seqlens, T)


def ref_grouped_gemm_reduce(A, B, cu_seqlens):
    """Pad -> torch.bmm -> done.  Fair baseline for benchmarking."""
    G = cu_seqlens.shape[0] - 1
    doc_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    max_len = doc_lens.max().item()

    A_pad = _pack_to_padded(A, cu_seqlens, max_len)  # [G, max_len, M]
    B_pad = _pack_to_padded(B, cu_seqlens, max_len)  # [G, max_len, K]
    # A.T @ B:  [G, M, max_len] @ [G, max_len, K] -> [G, M, K]
    return torch.bmm(A_pad.transpose(1, 2), B_pad)


# ======================== Main ========================

if __name__ == "__main__":
    import argparse
    from kernel_test_utils import test_correctness, benchmark, get_chunk_info

    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    device = "cuda"

    num_docs = 4
    doc_lens = [4096, 3072, 2048, 1024]
    d, dh = 512, 512
    cu_seqlens = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(doc_lens), 0).tolist()),
        dtype=torch.int32, device=device,
    )
    packed_len = cu_seqlens[-1].item()
    tol = dict(atol=1e-2, rtol=1e-2) if dtype == torch.bfloat16 else dict(atol=1e-4, rtol=1e-4)

    print(f"Config: {num_docs=}, {doc_lens=}, {d=}, {dh=}, {dtype=}, {packed_len=}")
    print()

    # ===== Test 1: grouped_gemm_to_packed (X @ W) =====
    print("=" * 60)
    print("Test 1: grouped_gemm_to_packed  X @ W")
    print("=" * 60)

    X = torch.randn(packed_len, d, device=device, dtype=dtype)
    W = torch.randn(num_docs, d, dh, device=device, dtype=dtype)

    max_len = max(doc_lens)
    X_pad = _pack_to_padded(X, cu_seqlens, max_len)

    try:
        from utils import compute_varlen_args
    except ImportError:
        from .utils import compute_varlen_args
    eff_lens, bos_arr, max_sl_val = compute_varlen_args(cu_seqlens)

    test_correctness(
        grouped_gemm_to_packed, ref_grouped_gemm_to_packed,
        (X, W, cu_seqlens), (X, W, cu_seqlens),
        debug=args.debug, **tol,
    )
    benchmark(
        lambda *a: grouped_gemm_to_packed(*a, eff_lens=eff_lens, bos_arr=bos_arr, max_sl=max_sl_val),
        lambda X_pad, W: torch.bmm(X_pad, W),
        (X, W, cu_seqlens), (X_pad, W),
    )

    # ===== Test 2: grouped_gemm_to_packed (X @ W.T) =====
    print()
    print("=" * 60)
    print("Test 2: grouped_gemm_to_packed  X @ W.T  (trans_W=True)")
    print("=" * 60)

    Wt = torch.randn(num_docs, dh, d, device=device, dtype=dtype)

    test_correctness(
        lambda *a: grouped_gemm_to_packed(*a, trans_W=True),
        lambda *a: ref_grouped_gemm_to_packed(*a, trans_W=True),
        (X, Wt, cu_seqlens), (X, Wt, cu_seqlens),
        debug=args.debug, **tol,
    )
    benchmark(
        lambda *a: grouped_gemm_to_packed(*a, trans_W=True, eff_lens=eff_lens, bos_arr=bos_arr, max_sl=max_sl_val),
        lambda X_pad, Wt: torch.bmm(X_pad, Wt.transpose(1, 2)),
        (X, Wt, cu_seqlens), (X_pad, Wt),
    )

    # ===== Test 3: grouped_gemm_reduce (A.T @ B) =====
    print()
    print("=" * 60)
    print("Test 3: grouped_gemm_reduce  A.T @ B")
    print("=" * 60)

    A = torch.randn(packed_len, 2 * dh, device=device, dtype=dtype)
    B = torch.randn(packed_len, d, device=device, dtype=dtype)

    A_pad = _pack_to_padded(A, cu_seqlens, max_len)
    B_pad = _pack_to_padded(B, cu_seqlens, max_len)

    test_correctness(
        grouped_gemm_reduce, ref_grouped_gemm_reduce,
        (A, B, cu_seqlens), (A, B, cu_seqlens),
        debug=args.debug, **tol,
    )
    benchmark(
        lambda *a: grouped_gemm_reduce(*a, eff_lens=eff_lens, bos_arr=bos_arr, max_sl=max_sl_val),
        lambda A_pad, B_pad: torch.bmm(A_pad.transpose(1, 2), B_pad),
        (A, B, cu_seqlens), (A_pad, B_pad),
    )

    # ===== Test 4: chunk correctness for grouped_gemm_to_packed =====
    chunk_size = 2048
    _, _, max_chunks = get_chunk_info(cu_seqlens, chunk_size, 0)

    print()
    print("=" * 60)
    print(f"Test 4: grouped_gemm_to_packed chunk correctness (chunk_size={chunk_size})")
    print("=" * 60)

    for chunk_index in range(max_chunks):
        chunk_cu, idx, _ = get_chunk_info(cu_seqlens, chunk_size, chunk_index)
        if len(idx) == 0:
            break
        # Reference: gather chunk tokens, call base varlen kernel
        ref_out = grouped_gemm_to_packed(X[idx], W, chunk_cu)
        # Test: call chunk kernel on full buffer
        test_out = grouped_gemm_to_packed(X, W, cu_seqlens, chunk_size=chunk_size, chunk_idx=chunk_index)
        diff = (test_out[idx] - ref_out).abs().max().item()
        print(f"  chunk_idx={chunk_index}: max_diff={diff:.2e}, n_tokens={len(idx)}")
        assert diff == 0.0, f"chunk_idx={chunk_index} not exact match!"

    # Also test trans_W
    for chunk_index in range(max_chunks):
        chunk_cu, idx, _ = get_chunk_info(cu_seqlens, chunk_size, chunk_index)
        if len(idx) == 0:
            break
        ref_out = grouped_gemm_to_packed(X[idx], Wt, chunk_cu, trans_W=True)
        test_out = grouped_gemm_to_packed(X, Wt, cu_seqlens, chunk_size=chunk_size, chunk_idx=chunk_index, trans_W=True)
        diff = (test_out[idx] - ref_out).abs().max().item()
        print(f"  chunk_idx={chunk_index} (trans_W): max_diff={diff:.2e}, n_tokens={len(idx)}")
        assert diff == 0.0, f"chunk_idx={chunk_index} trans_W not exact match!"

    print("✓ PASS (exact match)")

    # ===== Test 5: chunk correctness for grouped_gemm_reduce =====
    print()
    print("=" * 60)
    print(f"Test 5: grouped_gemm_reduce chunk correctness (chunk_size={chunk_size})")
    print("=" * 60)

    for chunk_index in range(max_chunks):
        chunk_cu, idx, _ = get_chunk_info(cu_seqlens, chunk_size, chunk_index)
        if len(idx) == 0:
            break
        ref_out = grouped_gemm_reduce(A[idx], B[idx], chunk_cu)
        test_out = grouped_gemm_reduce(A, B, cu_seqlens, chunk_size=chunk_size, chunk_idx=chunk_index)
        diff = (test_out - ref_out).abs().max().item()
        print(f"  chunk_idx={chunk_index}: max_diff={diff:.2e}, n_tokens={len(idx)}")
        assert diff == 0.0, f"chunk_idx={chunk_index} not exact match!"

    print("✓ PASS (exact match)")
