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
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=4),
    ],
    key=["N", "K"],
)
@triton.heuristics({
    'EVEN_K': lambda args: args['K'] % args['BLOCK_K'] == 0,
    'EVEN_N': lambda args: args['N'] % args['BLOCK_N'] == 0,
})
@triton.jit
def _grouped_gemm_to_packed_kernel(
    X_ptr, W_ptr, Y_ptr, eff_lens_ptr, bos_arr_ptr,
    max_sl, N, K,
    stride_x_t, stride_x_k,
    stride_w_g, stride_w_k, stride_w_n,  # "effective" strides: W viewed as [G, K, N]
    stride_y_t, stride_y_n,
    OUT_DTYPE: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # L2 cache swizzle: group M-tiles so adjacent PIDs share X rows across N-tiles
    pid = tl.program_id(0)
    pid_g = tl.program_id(1)
    num_pid_m = tl.cdiv(max_sl, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    doc_len = tl.load(eff_lens_ptr + pid_g).to(tl.int32)
    m_start = pid_m * BLOCK_M
    if m_start >= doc_len:
        return

    doc_start = tl.load(bos_arr_ptr + pid_g).to(tl.int64)

    # Stride hints for compiler address optimization
    tl.assume(stride_x_t > 0)
    tl.assume(stride_x_k > 0)
    tl.assume(stride_w_k > 0)
    tl.assume(stride_w_n > 0)
    tl.assume(stride_y_t > 0)
    tl.assume(stride_y_n > 0)

    # Modulo wrapping: edge tiles wrap to valid positions, masked only at store
    offs_m = (m_start + tl.arange(0, BLOCK_M)) % doc_len
    glob_m = doc_start + offs_m.to(tl.int64)
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    # Initialize pointer blocks — advance through K loop
    x_ptrs = X_ptr + glob_m[:, None] * stride_x_t + offs_k[None, :] * stride_x_k
    w_base = W_ptr + pid_g.to(tl.int64) * stride_w_g
    w_ptrs = w_base + offs_k[:, None] * stride_w_k + offs_n[None, :] * stride_w_n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if EVEN_K:
        # Fast path: K perfectly tiles — zero masking on all loads
        for _ in range(0, K, BLOCK_K):
            x = tl.load(x_ptrs)
            w = tl.load(w_ptrs)
            acc = tl.dot(x, w, acc)
            x_ptrs += BLOCK_K * stride_x_k
            w_ptrs += BLOCK_K * stride_w_k
    else:
        # General path: K masking needed
        for k0 in range(0, K, BLOCK_K):
            mask_k = (k0 + offs_k) < K
            x = tl.load(x_ptrs, mask=mask_k[None, :], other=0.0)
            w = tl.load(w_ptrs, mask=mask_k[:, None], other=0.0)
            acc = tl.dot(x, w, acc)
            x_ptrs += BLOCK_K * stride_x_k
            w_ptrs += BLOCK_K * stride_w_k

    # Store: mask only on M (wrapping requires real offsets for store)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_mask = (offs_cm[:, None] < doc_len) & (offs_cn[None, :] < N)
    y_ptrs = Y_ptr + (doc_start + offs_cm.to(tl.int64))[:, None] * stride_y_t + offs_cn[None, :] * stride_y_n
    tl.store(y_ptrs, acc.to(OUT_DTYPE), mask=c_mask)


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
        triton.cdiv(max_sl, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        G,
    )
    _grouped_gemm_to_packed_kernel[grid](
        X, W, Y, eff_lens, bos_arr,
        max_sl, N, Kx,
        X.stride(0), X.stride(1),
        sw_g, sw_k, sw_n,
        Y.stride(0), Y.stride(1),
        OUT_DTYPE=_TORCH_TO_TL[X.dtype],
        GROUP_SIZE_M=8,
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
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 128, "BLOCK_T": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 128, "BLOCK_T": 32}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 128, "BLOCK_T": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 64, "BLOCK_T": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 64, "BLOCK_T": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 64, "BLOCK_T": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 128, "BLOCK_T": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 128, "BLOCK_T": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 128, "BLOCK_T": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 64, "BLOCK_T": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 64, "BLOCK_T": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 64, "BLOCK_T": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 64, "BLOCK_T": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 64, "BLOCK_T": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 64, "BLOCK_T": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 64, "BLOCK_T": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 32, "BLOCK_T": 64}, num_warps=4, num_stages=4),
    ],
    key=["M", "K"],
    reset_to_zero=["C_ptr"],
)
@triton.heuristics({
    'EVEN_M': lambda args: args['M'] % args['BLOCK_M'] == 0,
    'EVEN_K': lambda args: args['K'] % args['BLOCK_K'] == 0,
})
@triton.jit
def _grouped_gemm_reduce_kernel(
    A_ptr, B_ptr, C_ptr, eff_lens_ptr, bos_arr_ptr,
    M, K,
    stride_a_t, stride_a_m,
    stride_b_t, stride_b_k,
    stride_c_g, stride_c_m, stride_c_k,
    num_groups,
    SPLIT_K: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_m = tl.program_id(0)  # A-feature tile
    pid_k = tl.program_id(1)  # B-feature tile
    pid_z = tl.program_id(2)  # doc * split_k

    # Split-K: divide token loop across split_k blocks per group
    pid_g = pid_z % num_groups
    split_idx = pid_z // num_groups

    full_doc_len = tl.load(eff_lens_ptr + pid_g).to(tl.int32)
    full_doc_start = tl.load(bos_arr_ptr + pid_g).to(tl.int64)

    tokens_per_split = tl.cdiv(full_doc_len, SPLIT_K)
    t_start = split_idx * tokens_per_split
    t_end = min(t_start + tokens_per_split, full_doc_len)
    doc_len = t_end - t_start
    doc_start = full_doc_start + t_start.to(tl.int64)

    # Stride hints
    tl.assume(stride_a_t > 0)
    tl.assume(stride_a_m > 0)
    tl.assume(stride_b_t > 0)
    tl.assume(stride_b_k > 0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_t = tl.arange(0, BLOCK_T)

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    if EVEN_M and EVEN_K:
        # Fast path: no M/K masking, pointer advancement, split token loop
        glob_t_init = doc_start + offs_t.to(tl.int64)
        a_ptrs = A_ptr + glob_t_init[None, :] * stride_a_t + offs_m[:, None] * stride_a_m
        b_ptrs = B_ptr + glob_t_init[:, None] * stride_b_t + offs_k[None, :] * stride_b_k

        n_full_t = (doc_len // BLOCK_T) * BLOCK_T
        for _ in range(0, n_full_t, BLOCK_T):
            a_T = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc = tl.dot(a_T, b, acc)
            a_ptrs += BLOCK_T * stride_a_t
            b_ptrs += BLOCK_T * stride_b_t
        # Remainder token tile (at most one, with token masking)
        if n_full_t < doc_len:
            rem_mask = offs_t < (doc_len - n_full_t)
            a_T = tl.load(a_ptrs, mask=rem_mask[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=rem_mask[:, None], other=0.0)
            acc = tl.dot(a_T, b, acc)
        c_ptrs = (C_ptr + pid_g.to(tl.int64) * stride_c_g
                  + offs_m[:, None] * stride_c_m + offs_k[None, :] * stride_c_k)
        if SPLIT_K > 1:
            tl.atomic_add(c_ptrs, acc.to(tl.float32))
        else:
            tl.store(c_ptrs, acc.to(OUT_DTYPE))
    else:
        # General path: full masking
        mask_m = offs_m < M
        mask_k = offs_k < K
        for t0 in range(0, doc_len, BLOCK_T):
            cur_offs_t = t0 + offs_t
            mask_t = cur_offs_t < doc_len
            glob_t = doc_start + cur_offs_t.to(tl.int64)
            a_T = tl.load(A_ptr + glob_t[None, :] * stride_a_t + offs_m[:, None] * stride_a_m,
                          mask=mask_m[:, None] & mask_t[None, :], other=0.0)
            b = tl.load(B_ptr + glob_t[:, None] * stride_b_t + offs_k[None, :] * stride_b_k,
                        mask=mask_t[:, None] & mask_k[None, :], other=0.0)
            acc = tl.dot(a_T, b, acc)
        c_ptrs = (C_ptr + pid_g.to(tl.int64) * stride_c_g
                  + offs_m[:, None] * stride_c_m + offs_k[None, :] * stride_c_k)
        if SPLIT_K > 1:
            tl.atomic_add(c_ptrs, acc.to(tl.float32), mask=mask_m[:, None] & mask_k[None, :])
        else:
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

    if eff_lens is None:
        eff_lens, bos_arr, max_sl = compute_varlen_args(cu_seqlens, chunk_size, chunk_idx)

    # Split-K: increase parallelism only when severely underutilizing SMs
    grid_tiles = triton.cdiv(M, 64) * triton.cdiv(K, 64) * G
    split_k = 1
    if grid_tiles < 200:
        split_k = min(8, max(1, 512 // grid_tiles))
        while split_k > 1 and max_sl // split_k < 128:
            split_k //= 2

    # split_k > 1: float32 output + atomic_add (reset_to_zero handles autotune)
    if split_k > 1:
        C = torch.zeros(G, M, K, device=A.device, dtype=torch.float32)
    else:
        C = torch.empty(G, M, K, device=A.device, dtype=A.dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(K, META["BLOCK_K"]),
        G * split_k,
    )
    _grouped_gemm_reduce_kernel[grid](
        A, B, C, eff_lens, bos_arr,
        M, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1), C.stride(2),
        G, SPLIT_K=split_k,
        OUT_DTYPE=_TORCH_TO_TL[A.dtype],
    )
    if split_k > 1:
        return C.to(A.dtype)
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
    from kernel_test_utils import (
        test_correctness, benchmark, get_chunk_info,
        BENCH_DOC_LENS, make_cu_seqlens,
    )
    from utils import compute_varlen_args

    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    device = "cuda"
    d, dh = 512, 512
    tol = dict(atol=1e-2, rtol=1e-2) if dtype == torch.bfloat16 else dict(atol=1e-4, rtol=1e-4)

    # ================================================================
    #  CORRECTNESS  (use varied-size config that exercises edge cases)
    # ================================================================
    doc_lens = [4096, 3072, 2048, 1024]
    cu = make_cu_seqlens(doc_lens, device)
    T = cu[-1].item()
    G = len(doc_lens)
    chunk_size = 2048

    X = torch.randn(T, d, device=device, dtype=dtype)
    W = torch.randn(G, d, dh, device=device, dtype=dtype)
    Wt = torch.randn(G, dh, d, device=device, dtype=dtype)
    A = torch.randn(T, 2 * dh, device=device, dtype=dtype)
    B = torch.randn(T, d, device=device, dtype=dtype)

    print(f"Correctness: doc_lens={doc_lens}, {d=}, {dh=}, {dtype=}")

    # varlen correctness
    for name, tri_fn, ref_fn, tri_a, ref_a in [
        ("to_packed  X@W",     grouped_gemm_to_packed, ref_grouped_gemm_to_packed, (X, W, cu), (X, W, cu)),
        ("to_packed  X@W.T",
         lambda *a: grouped_gemm_to_packed(*a, trans_W=True),
         lambda *a: ref_grouped_gemm_to_packed(*a, trans_W=True),
         (X, Wt, cu), (X, Wt, cu)),
        ("reduce     A.T@B",   grouped_gemm_reduce, ref_grouped_gemm_reduce, (A, B, cu), (A, B, cu)),
    ]:
        print(f"\n  {name}: ", end="")
        ok = test_correctness(tri_fn, ref_fn, tri_a, ref_a, debug=args.debug, **tol)
        assert ok, f"{name} failed!"

    # chunk correctness
    _, _, max_chunks = get_chunk_info(cu, chunk_size, 0)
    print(f"\n  chunk correctness (chunk_size={chunk_size}):")
    for ci in range(max_chunks):
        chunk_cu, idx, _ = get_chunk_info(cu, chunk_size, ci)
        if len(idx) == 0:
            break
        for label, fn, kw, gather_input in [
            ("to_packed",      grouped_gemm_to_packed, {},                [X, W]),
            ("to_packed transW", grouped_gemm_to_packed, {"trans_W": True}, [X, Wt]),
            ("reduce",         grouped_gemm_reduce,     {},                [A, B]),
        ]:
            is_reduce = "reduce" in label
            ref_inputs = [t[idx] for t in gather_input] if is_reduce else [gather_input[0][idx], gather_input[1]]
            ref_out = fn(*ref_inputs, chunk_cu, **kw)
            test_out = fn(*gather_input, cu, chunk_size=chunk_size, chunk_idx=ci, **kw)
            compare = test_out - ref_out if is_reduce else test_out[idx] - ref_out
            diff = compare.abs().max().item()
            assert diff == 0.0, f"{label} ci={ci} diff={diff}"
        print(f"    ci={ci}: exact match (n={len(idx)})")
    print("✓ All correctness tests passed\n")

    # ================================================================
    #  BENCHMARKS  (4 configurations)
    # ================================================================
    print("=" * 72)
    print(f"{'Config':<16} {'Kernel':<20} {'Triton ms':>10} {'cuBLAS ms':>10} {'Speedup':>8}")
    print("=" * 72)

    for cfg_name, dl in BENCH_DOC_LENS.items():
        cu = make_cu_seqlens(dl, device)
        T = cu[-1].item()
        G = len(dl)
        max_len = max(dl)
        eff, bos, msl = compute_varlen_args(cu)

        X = torch.randn(T, d, device=device, dtype=dtype)
        W = torch.randn(G, d, dh, device=device, dtype=dtype)
        Wt = torch.randn(G, dh, d, device=device, dtype=dtype)
        A = torch.randn(T, 2 * dh, device=device, dtype=dtype)
        B = torch.randn(T, d, device=device, dtype=dtype)
        X_pad = _pack_to_padded(X, cu, max_len)
        A_pad = _pack_to_padded(A, cu, max_len)
        B_pad = _pack_to_padded(B, cu, max_len)

        for kernel_name, tri_fn, ref_fn, tri_a, ref_a in [
            ("to_packed X@W",
             lambda *a, _e=eff, _b=bos, _m=msl: grouped_gemm_to_packed(*a, eff_lens=_e, bos_arr=_b, max_sl=_m),
             lambda Xp, W: torch.bmm(Xp, W),
             (X, W, cu), (X_pad, W)),
            ("to_packed X@W.T",
             lambda *a, _e=eff, _b=bos, _m=msl: grouped_gemm_to_packed(*a, trans_W=True, eff_lens=_e, bos_arr=_b, max_sl=_m),
             lambda Xp, Wt: torch.bmm(Xp, Wt.transpose(1, 2)),
             (X, Wt, cu), (X_pad, Wt)),
            ("reduce A.T@B",
             lambda *a, _e=eff, _b=bos, _m=msl: grouped_gemm_reduce(*a, eff_lens=_e, bos_arr=_b, max_sl=_m),
             lambda Ap, Bp: torch.bmm(Ap.transpose(1, 2), Bp),
             (A, B, cu), (A_pad, B_pad)),
        ]:
            r = benchmark(tri_fn, ref_fn, tri_a, ref_a, verbose=False)
            print(f"{cfg_name:<16} {kernel_name:<20} {r['time_triton']:>10.4f} {r['time_ref']:>10.4f} {r['speedup']:>7.2f}x")

        print("-" * 72)
