import torch
import triton
import triton.language as tl
import itertools


def two_mm(W0, X0, W1, X1, A_transpose=True, B_transpose=True):
    if A_transpose and B_transpose:
        return W0.transpose(1, 2) @ X0.transpose(1, 2) + W1.transpose(
            1, 2
        ) @ X1.transpose(1, 2)
    if A_transpose and not B_transpose:
        return W0.transpose(1, 2) @ X0 + W1.transpose(1, 2) @ X1
    raise NotImplementedError(
        "Only the two variants with A_transpose=True are implemented here."
    )


def get_autotune_configs(
    block_M_list=(64, 128),
    block_N_list=(64, 128, 256),
    block_K_list=(32, 64),
    num_stages_list=(2, 3),
    threads_list=(128, 256),
):
    configs = []
    for BM, BN, BK, stages, threads in itertools.product(
        block_M_list, block_N_list, block_K_list, num_stages_list, threads_list
    ):
        configs.append(
            triton.Config(
                {"BLOCK_M": BM, "BLOCK_N": BN, "BLOCK_K": BK},
                num_warps=threads // 32,
                num_stages=stages,
            )
        )
    return configs


########################################################
# O = W1.transpose(1, 2) @ X1T.transpose(1, 2) + W0.transpose(1, 2) @ X0T.transpose(1, 2)
# W0 and W1 of shape [B, K, M]   (M = token axis for varlen)
# X0T and X1T of shape [B, N, K] (per-doc for varlen)
# O of shape [B, M, N]           ([T, N] packed for varlen)
#
# IS_VARLEN: W has packed tokens in M axis, X is per-doc, O is packed.
########################################################
@triton.autotune(configs=get_autotune_configs(), key=["M", "N", "K"])
@triton.jit
def fused_two_mm_wT_xT_kernel(
    W0,
    W1,
    X0T,
    X1T,
    O,
    eff_lens,
    bos_arr,
    B,
    M,
    N,
    K,
    # W0 strides: [B, K, M]
    stride_w0_b,
    stride_w0_k,
    stride_w0_m,
    # W1 strides: [B, K, M]
    stride_w1_b,
    stride_w1_k,
    stride_w1_m,
    # X0T strides: [B, N, K]
    stride_x0t_b,
    stride_x0t_n,
    stride_x0t_k,
    # X1T strides: [B, N, K]
    stride_x1t_b,
    stride_x1t_n,
    stride_x1t_k,
    # O strides: [B, M, N]
    stride_o_b,
    stride_o_m,
    stride_o_n,
    OUT_IS_BF16: tl.constexpr,
    OUT_IS_FP16: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    M_eff = M
    if IS_VARLEN:
        M_eff = tl.load(eff_lens + pid_b).to(tl.int32)
        if pid_m * BLOCK_M >= M_eff:
            return
        bos = tl.load(bos_arr + pid_b).to(tl.int64)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    tl.multiple_of(offs_k, BLOCK_K)
    tl.max_contiguous(offs_m, BLOCK_M)
    tl.max_contiguous(offs_n, BLOCK_N)

    # W has tokens in M axis: bos offset for varlen, batch offset otherwise
    if IS_VARLEN:
        w0_off = bos * stride_w0_m
        w1_off = bos * stride_w1_m
        o_off = bos * stride_o_m
    else:
        w0_off = pid_b * stride_w0_b
        w1_off = pid_b * stride_w1_b
        o_off = pid_b * stride_o_b

    # X is per-doc: always use pid_b * batch_stride
    x0_off = pid_b * stride_x0t_b
    x1_off = pid_b * stride_x1t_b

    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k

        a0_ptrs = W0 + w0_off + offs_m[:, None] * stride_w0_m + k[None, :] * stride_w0_k
        b0_ptrs = X0T + x0_off + k[:, None] * stride_x0t_k + offs_n[None, :] * stride_x0t_n

        mask_a = (offs_m[:, None] < M_eff) & (k[None, :] < K)
        mask_b = (k[:, None] < K) & (offs_n[None, :] < N)

        a0 = tl.load(a0_ptrs, mask=mask_a, other=0.0)
        b0 = tl.load(b0_ptrs, mask=mask_b, other=0.0)
        acc += tl.dot(a0, b0)

    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k

        a0_ptrs = W1 + w1_off + offs_m[:, None] * stride_w1_m + k[None, :] * stride_w1_k
        b0_ptrs = X1T + x1_off + k[:, None] * stride_x1t_k + offs_n[None, :] * stride_x1t_n

        mask_a = (offs_m[:, None] < M_eff) & (k[None, :] < K)
        mask_b = (k[:, None] < K) & (offs_n[None, :] < N)

        a0 = tl.load(a0_ptrs, mask=mask_a, other=0.0)
        b0 = tl.load(b0_ptrs, mask=mask_b, other=0.0)
        acc += tl.dot(a0, b0)

    o_ptrs = O + o_off + offs_m[:, None] * stride_o_m + offs_n[None, :] * stride_o_n
    mask_o = (offs_m[:, None] < M_eff) & (offs_n[None, :] < N)

    if OUT_IS_BF16:
        tl.store(o_ptrs, acc.to(tl.bfloat16), mask=mask_o)
    elif OUT_IS_FP16:
        tl.store(o_ptrs, acc.to(tl.float16), mask=mask_o)
    else:
        tl.store(o_ptrs, acc, mask=mask_o)


########################################################
# O = W1.transpose(1, 2) @ X1 + W0.transpose(1, 2) @ X0
# W0 and W1 of shape [B, K, M]  (M = token axis for varlen)
# X0 and X1 of shape [B, K, N]  (per-doc for varlen)
# O of shape [B, M, N]          ([T, N] packed for varlen)
########################################################
@triton.autotune(configs=get_autotune_configs(), key=["M", "N", "K"])
@triton.jit
def fused_two_mm_wT_x_kernel(
    W0,
    W1,
    X0,
    X1,
    O,
    eff_lens,
    bos_arr,
    B,
    M,
    N,
    K,
    # W0 strides: [B, K, M]
    stride_w0_b,
    stride_w0_k,
    stride_w0_m,
    # W1 strides: [B, K, M]
    stride_w1_b,
    stride_w1_k,
    stride_w1_m,
    # X0 strides: [B, K, N]
    stride_x0_b,
    stride_x0_k,
    stride_x0_n,
    # X1 strides: [B, K, N]
    stride_x1_b,
    stride_x1_k,
    stride_x1_n,
    # O strides: [B, M, N]
    stride_o_b,
    stride_o_m,
    stride_o_n,
    OUT_IS_BF16: tl.constexpr,
    OUT_IS_FP16: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    M_eff = M
    if IS_VARLEN:
        M_eff = tl.load(eff_lens + pid_b).to(tl.int32)
        if pid_m * BLOCK_M >= M_eff:
            return
        bos = tl.load(bos_arr + pid_b).to(tl.int64)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    tl.multiple_of(offs_k, BLOCK_K)
    tl.max_contiguous(offs_m, BLOCK_M)
    tl.max_contiguous(offs_n, BLOCK_N)

    # W has tokens in M axis: bos offset for varlen, batch offset otherwise
    if IS_VARLEN:
        w0_off = bos * stride_w0_m
        w1_off = bos * stride_w1_m
        o_off = bos * stride_o_m
    else:
        w0_off = pid_b * stride_w0_b
        w1_off = pid_b * stride_w1_b
        o_off = pid_b * stride_o_b

    # X is per-doc: always use pid_b * batch_stride
    x0_off = pid_b * stride_x0_b
    x1_off = pid_b * stride_x1_b

    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k

        # A tiles: W^T -> treat W as (M,K) via strides (W is stored [K, M])
        a0_ptrs = (
            W0
            + w0_off
            + offs_m[:, None] * stride_w0_m
            + k[None, :] * stride_w0_k
        )

        # B tiles: X stored as (K,N)
        b0_ptrs = (
            X0
            + x0_off
            + k[:, None] * stride_x0_k
            + offs_n[None, :] * stride_x0_n
        )

        mask_a = (offs_m[:, None] < M_eff) & (k[None, :] < K)
        mask_b = (k[:, None] < K) & (offs_n[None, :] < N)

        a0 = tl.load(a0_ptrs, mask=mask_a, other=0.0)
        b0 = tl.load(b0_ptrs, mask=mask_b, other=0.0)
        acc += tl.dot(a0, b0)

    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k

        # A tiles: W^T -> treat W as (M,K) via strides (W is stored [K, M])

        a0_ptrs = (
            W1
            + w1_off
            + offs_m[:, None] * stride_w1_m
            + k[None, :] * stride_w1_k
        )

        b0_ptrs = (
            X1
            + x1_off
            + k[:, None] * stride_x1_k
            + offs_n[None, :] * stride_x1_n
        )

        mask_a = (offs_m[:, None] < M_eff) & (k[None, :] < K)
        mask_b = (k[:, None] < K) & (offs_n[None, :] < N)

        a0 = tl.load(a0_ptrs, mask=mask_a, other=0.0)
        b0 = tl.load(b0_ptrs, mask=mask_b, other=0.0)
        acc += tl.dot(a0, b0)

    o_ptrs = (
        O
        + o_off
        + offs_m[:, None] * stride_o_m
        + offs_n[None, :] * stride_o_n
    )
    mask_o = (offs_m[:, None] < M_eff) & (offs_n[None, :] < N)

    if OUT_IS_BF16:
        tl.store(o_ptrs, acc.to(tl.bfloat16), mask=mask_o)
    elif OUT_IS_FP16:
        tl.store(o_ptrs, acc.to(tl.float16), mask=mask_o)
    else:
        tl.store(o_ptrs, acc, mask=mask_o)


@triton.autotune(configs=get_autotune_configs(), key=["M", "N", "K"])
@triton.jit
def fused_four_mm_wT_x_kernel(
    W0,
    W1,
    W2,
    W3,
    X0,
    X1,
    X2,
    X3,
    O,
    B,
    M,
    N,
    K,
    # W0 strides: [B, K, M]
    stride_w0_b,
    stride_w0_k,
    stride_w0_m,
    # W1 strides: [B, K, M]
    stride_w1_b,
    stride_w1_k,
    stride_w1_m,
    # W2 strides: [B, K, M]
    stride_w2_b,
    stride_w2_k,
    stride_w2_m,
    # W3 strides: [B, K, M]
    stride_w3_b,
    stride_w3_k,
    stride_w3_m,
    # X0 strides: [B, K, N]
    stride_x0_b,
    stride_x0_k,
    stride_x0_n,
    # X1 strides: [B, K, N]
    stride_x1_b,
    stride_x1_k,
    stride_x1_n,
    # X2 strides: [B, K, N]
    stride_x2_b,
    stride_x2_k,
    stride_x2_n,
    # X3 strides: [B, K, N]
    stride_x3_b,
    stride_x3_k,
    stride_x3_n,
    # O strides: [B, M, N]
    stride_o_b,
    stride_o_m,
    stride_o_n,
    OUT_IS_BF16: tl.constexpr,
    OUT_IS_FP16: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    tl.multiple_of(offs_k, BLOCK_K)
    tl.max_contiguous(offs_m, BLOCK_M)
    tl.max_contiguous(offs_n, BLOCK_N)

    # O = W0.T @ X0
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k

        # A tiles: W^T -> treat W as (M,K) via strides (W is stored [K, M])
        a0_ptrs = (
            W0
            + pid_b * stride_w0_b
            + offs_m[:, None] * stride_w0_m
            + k[None, :] * stride_w0_k
        )

        # B tiles: X stored as (K,N)
        b0_ptrs = (
            X0
            + pid_b * stride_x0_b
            + k[:, None] * stride_x0_k
            + offs_n[None, :] * stride_x0_n
        )

        mask_a = (offs_m[:, None] < M) & (k[None, :] < K)
        mask_b = (k[:, None] < K) & (offs_n[None, :] < N)

        a0 = tl.load(a0_ptrs, mask=mask_a, other=0.0)
        b0 = tl.load(b0_ptrs, mask=mask_b, other=0.0)
        acc += tl.dot(a0, b0)

    # O += W1.T @ X1
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k

        # A tiles: W^T -> treat W as (M,K) via strides (W is stored [K, M])

        a0_ptrs = (
            W1
            + pid_b * stride_w1_b
            + offs_m[:, None] * stride_w1_m
            + k[None, :] * stride_w1_k
        )

        b0_ptrs = (
            X1
            + pid_b * stride_x1_b
            + k[:, None] * stride_x1_k
            + offs_n[None, :] * stride_x1_n
        )

        mask_a = (offs_m[:, None] < M) & (k[None, :] < K)
        mask_b = (k[:, None] < K) & (offs_n[None, :] < N)

        a0 = tl.load(a0_ptrs, mask=mask_a, other=0.0)
        b0 = tl.load(b0_ptrs, mask=mask_b, other=0.0)
        acc += tl.dot(a0, b0)

    # O += W2.T @ X2
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k

        # A tiles: W^T -> treat W as (M,K) via strides (W is stored [K, M])
        a0_ptrs = (
            W2
            + pid_b * stride_w2_b
            + offs_m[:, None] * stride_w2_m
            + k[None, :] * stride_w2_k
        )

        # B tiles: X stored as (K,N)
        b0_ptrs = (
            X2
            + pid_b * stride_x2_b
            + k[:, None] * stride_x2_k
            + offs_n[None, :] * stride_x2_n
        )

        mask_a = (offs_m[:, None] < M) & (k[None, :] < K)
        mask_b = (k[:, None] < K) & (offs_n[None, :] < N)

        a0 = tl.load(a0_ptrs, mask=mask_a, other=0.0)
        b0 = tl.load(b0_ptrs, mask=mask_b, other=0.0)
        acc += tl.dot(a0, b0)

    # O += W3.T @ X3
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k

        # A tiles: W^T -> treat W as (M,K) via strides (W is stored [K, M])
        a0_ptrs = (
            W3
            + pid_b * stride_w3_b
            + offs_m[:, None] * stride_w3_m
            + k[None, :] * stride_w3_k
        )

        b0_ptrs = (
            X3
            + pid_b * stride_x3_b
            + k[:, None] * stride_x3_k
            + offs_n[None, :] * stride_x3_n
        )

        mask_a = (offs_m[:, None] < M) & (k[None, :] < K)
        mask_b = (k[:, None] < K) & (offs_n[None, :] < N)

        a0 = tl.load(a0_ptrs, mask=mask_a, other=0.0)
        b0 = tl.load(b0_ptrs, mask=mask_b, other=0.0)
        acc += tl.dot(a0, b0)

    o_ptrs = (
        O
        + pid_b * stride_o_b
        + offs_m[:, None] * stride_o_m
        + offs_n[None, :] * stride_o_n
    )
    mask_o = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    if OUT_IS_BF16:
        tl.store(o_ptrs, acc.to(tl.bfloat16), mask=mask_o)
    elif OUT_IS_FP16:
        tl.store(o_ptrs, acc.to(tl.float16), mask=mask_o)
    else:
        tl.store(o_ptrs, acc, mask=mask_o)


# -----------------------------
# Python launchers
# -----------------------------
def _out_dtype_flags(t: torch.Tensor):
    return t.dtype == torch.bfloat16, t.dtype == torch.float16


def fused_two_mm_same_out_wT_xT_triton(W0, X0T, W1, X1T):
    """
    O = W0^T @ X0T.T + W1^T @ X1T.T

    Batched:
      W0, W1: [B, K, M], X0T, X1T: [B, N, K] -> O: [B, M, N]
    """
    assert W0.ndim == W1.ndim == X0T.ndim == X1T.ndim == 3
    B, K, M = W0.shape
    Bx, N, Kx = X0T.shape
    assert (Bx == B) and (Kx == K), f"Bx: {Bx}, B: {B}, Kx: {Kx}, K: {K}"
    assert W1.shape == (B, K, M), f"W1.shape: {W1.shape}, B: {B}, K: {K}, M: {M}"
    assert X1T.shape == (B, N, K), f"X1T.shape: {X1T.shape}, B: {B}, N: {N}, K: {K}"

    _dummy = torch.empty(0, dtype=torch.int32, device=W0.device)
    out = torch.empty((B, M, N), device=W0.device, dtype=X0T.dtype)

    W0 = W0.contiguous()
    W1 = W1.contiguous()
    X0T = X0T.contiguous()
    X1T = X1T.contiguous()

    out_is_bf16, out_is_fp16 = _out_dtype_flags(out)
    s_w_b, s_w_k, s_w_m = K * M, M, 1
    s_x_b, s_x_n, s_x_k = X0T.stride()
    s_o_b, s_o_m, s_o_n = M * N, N, 1

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]), B)

    fused_two_mm_wT_xT_kernel[grid](
        W0,
        W1,
        X0T,
        X1T,
        out,
        _dummy,
        _dummy,
        B,
        M,
        N,
        K,
        s_w_b,
        s_w_k,
        s_w_m,
        s_w_b,
        s_w_k,
        s_w_m,
        s_x_b,
        s_x_n,
        s_x_k,
        s_x_b,
        s_x_n,
        s_x_k,
        s_o_b,
        s_o_m,
        s_o_n,
        out_is_bf16,
        out_is_fp16,
        IS_VARLEN=False,
    )
    return out


def fused_two_mm_same_out_wT_xT_varlen_triton(
    W0, X0T, W1, X1T, cu_seqlens,
    chunk_size=0, chunk_idx=0,
    eff_lens=None, bos_arr=None, max_sl=0,
):
    """
    O = W0^T @ X0T.T + W1^T @ X1T.T  (varlen)

      W0, W1: [T, K] packed (tokens in M axis)
      X0T, X1T: [G, N, K] per-doc
      cu_seqlens: [G+1] int32 -> O: [T, N] packed
      eff_lens, bos_arr, max_sl: precomputed (if None, computed from cu_seqlens + chunk params)
    """
    try:
        from utils import compute_varlen_args
    except ImportError:
        from .utils import compute_varlen_args

    T, K = W0.shape
    G = cu_seqlens.shape[0] - 1
    N = X0T.shape[1]

    out = torch.zeros((T, N), device=W0.device, dtype=X0T.dtype)

    if eff_lens is None:
        eff_lens, bos_arr, max_sl = compute_varlen_args(cu_seqlens, chunk_size, chunk_idx)
    M = max_sl

    W0 = W0.contiguous()
    W1 = W1.contiguous()
    X0T = X0T.contiguous()
    X1T = X1T.contiguous()

    out_is_bf16, out_is_fp16 = _out_dtype_flags(out)
    # W is [T, K]: stride_b=0, stride_k=1, stride_m=K
    s_w_b, s_w_k, s_w_m = 0, W0.stride(1), W0.stride(0)
    s_x_b, s_x_n, s_x_k = X0T.stride()
    # O is [T, N]: stride_b=0, stride_m=N, stride_n=1
    s_o_b, s_o_m, s_o_n = 0, out.stride(0), out.stride(1)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]), G)

    fused_two_mm_wT_xT_kernel[grid](
        W0,
        W1,
        X0T,
        X1T,
        out,
        eff_lens,
        bos_arr,
        G,
        M,
        N,
        K,
        s_w_b,
        s_w_k,
        s_w_m,
        s_w_b,
        s_w_k,
        s_w_m,
        s_x_b,
        s_x_n,
        s_x_k,
        s_x_b,
        s_x_n,
        s_x_k,
        s_o_b,
        s_o_m,
        s_o_n,
        out_is_bf16,
        out_is_fp16,
        IS_VARLEN=True,
    )
    return out


def fused_two_mm_same_out_wT_x_triton(W0, X0, W1, X1, out=None):
    """
    W0, W1: [B, K, M]
    X0, X1: [B, K, N]
    Returns O: [B, M, N] with O = W0^T @ X0 + W1^T @ X1
    """
    assert W0.ndim == W1.ndim == X0.ndim == X1.ndim == 3
    B, K, M = W0.shape
    Bx, Kx, N = X0.shape
    assert (Bx == B) and (Kx == K)
    assert W1.shape == (B, K, M)
    assert X1.shape == (B, K, N)

    _dummy = torch.empty(0, dtype=torch.int32, device=W0.device)
    if out is None:
        out = torch.empty((B, M, N), device=W0.device, dtype=X0.dtype)

    W0 = W0.contiguous()
    W1 = W1.contiguous()
    X0 = X0.contiguous()
    X1 = X1.contiguous()
    out = out.contiguous()

    out_is_bf16, out_is_fp16 = _out_dtype_flags(out)
    s_w_b, s_w_k, s_w_m = K * M, M, 1
    s_x_b, s_x_k, s_x_n = X0.stride()
    s_o_b, s_o_m, s_o_n = M * N, N, 1

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]), B)

    fused_two_mm_wT_x_kernel[grid](
        W0,
        W1,
        X0,
        X1,
        out,
        _dummy,
        _dummy,
        B,
        M,
        N,
        K,
        s_w_b,
        s_w_k,
        s_w_m,
        s_w_b,
        s_w_k,
        s_w_m,
        s_x_b,
        s_x_k,
        s_x_n,
        s_x_b,
        s_x_k,
        s_x_n,
        s_o_b,
        s_o_m,
        s_o_n,
        out_is_bf16,
        out_is_fp16,
        IS_VARLEN=False,
    )
    return out


def fused_two_mm_same_out_wT_x_varlen_triton(
    W0, X0, W1, X1, cu_seqlens,
    chunk_size=0, chunk_idx=0,
    eff_lens=None, bos_arr=None, max_sl=0,
):
    """
    O = W0^T @ X0 + W1^T @ X1  (varlen)

      W0, W1: [T, K] packed (tokens in M axis)
      X0, X1: [G, K, N] per-doc
      cu_seqlens: [G+1] int32 -> O: [T, N] packed
      eff_lens, bos_arr, max_sl: precomputed (if None, computed from cu_seqlens + chunk params)
    """
    try:
        from utils import compute_varlen_args
    except ImportError:
        from .utils import compute_varlen_args

    T, K = W0.shape
    G = cu_seqlens.shape[0] - 1
    N = X0.shape[2]

    out = torch.zeros((T, N), device=W0.device, dtype=X0.dtype)

    if eff_lens is None:
        eff_lens, bos_arr, max_sl = compute_varlen_args(cu_seqlens, chunk_size, chunk_idx)
    M = max_sl

    W0 = W0.contiguous()
    W1 = W1.contiguous()
    X0 = X0.contiguous()
    X1 = X1.contiguous()

    out_is_bf16, out_is_fp16 = _out_dtype_flags(out)
    s_w_b, s_w_k, s_w_m = 0, W0.stride(1), W0.stride(0)
    s_x_b, s_x_k, s_x_n = X0.stride()
    s_o_b, s_o_m, s_o_n = 0, out.stride(0), out.stride(1)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]), G)

    fused_two_mm_wT_x_kernel[grid](
        W0,
        W1,
        X0,
        X1,
        out,
        eff_lens,
        bos_arr,
        G,
        M,
        N,
        K,
        s_w_b,
        s_w_k,
        s_w_m,
        s_w_b,
        s_w_k,
        s_w_m,
        s_x_b,
        s_x_k,
        s_x_n,
        s_x_b,
        s_x_k,
        s_x_n,
        s_o_b,
        s_o_m,
        s_o_n,
        out_is_bf16,
        out_is_fp16,
        IS_VARLEN=True,
    )
    return out


def fused_two_mm_same_out_interface(W0, X0, W1, X1, A_transpose=True, B_transpose=True):
    assert A_transpose, "Only A_transpose=True is implemented"
    if B_transpose:
        return fused_two_mm_same_out_wT_xT_triton(W0, X0, W1, X1)
    else:
        return fused_two_mm_same_out_wT_x_triton(W0, X0, W1, X1)


def fused_four_mm_same_out_interface(
    W0, X0, W1, X1, W2, X2, W3, X3, A_transpose=True, B_transpose=False
):
    assert (
        A_transpose and not B_transpose
    ), "Only A_transpose=True and B_transpose=False is implemented"
    B, K, M = W0.shape
    Bx, Kx, N = X0.shape
    assert (Bx == B) and (Kx == K)
    assert W1.shape == (B, K, M)
    assert X1.shape == (B, K, N)
    assert W2.shape == (B, K, M)
    assert X2.shape == (B, K, N)
    assert W3.shape == (B, K, M)
    assert X3.shape == (B, K, N)

    # assume contiguous inputs
    s_w_b, s_w_k, s_w_m = K * M, M, 1
    s_x_b, s_x_k, s_x_n = K * N, N, 1
    s_o_b, s_o_m, s_o_n = M * N, N, 1

    device = W0.device
    out = torch.empty((B, M, N), device=device, dtype=X0.dtype)
    out_is_bf16, out_is_fp16 = _out_dtype_flags(X0)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]), B)

    fused_four_mm_wT_x_kernel[grid](
        W0,
        W1,
        W2,
        W3,
        X0,
        X1,
        X2,
        X3,
        out,
        B,
        M,
        N,
        K,
        s_w_b,
        s_w_k,
        s_w_m,
        s_w_b,
        s_w_k,
        s_w_m,
        s_w_b,
        s_w_k,
        s_w_m,
        s_w_b,
        s_w_k,
        s_w_m,
        s_x_b,
        s_x_k,
        s_x_n,
        s_x_b,
        s_x_k,
        s_x_n,
        s_x_b,
        s_x_k,
        s_x_n,
        s_x_b,
        s_x_k,
        s_x_n,
        s_o_b,
        s_o_m,
        s_o_n,
        out_is_bf16,
        out_is_fp16,
    )
    return out


# -----------------------------
# Correctness checks and benchmark code below
# -----------------------------


def correctness_check_wT_x(
    device="cuda",
):
    from .benchmark import report_error

    device = torch.device("cuda", torch.cuda.current_device())

    def make_inputs(B, M, N, K, dtype=torch.bfloat16):
        return (
            torch.randn(B, K, M, device=device, dtype=dtype),
            torch.randn(B, K, N, device=device, dtype=dtype),
            torch.randn(B, K, M, device=device, dtype=dtype),
            torch.randn(B, K, N, device=device, dtype=dtype),
        )

    def ref_func(W0, X0, W1, X1):
        return two_mm(W0, X0, W1, X1, A_transpose=True, B_transpose=False)

    shape_list = [[1, 512, 512, 512], [2, 2048, 1024, 4096], [1, 678, 724, 996]]
    for B, M, N, K in shape_list:
        _inputs = make_inputs(B, M, N, K)
        O_triton = fused_two_mm_same_out_wT_x_triton(*_inputs)
        fp32_inputs = [_inp.float() for _inp in _inputs]
        O_ref = ref_func(*fp32_inputs)
        report_error(O_ref, O_triton, "O")


def correctness_check_wT_xT():
    """
    Validates fused_two_mm_same_out_wT_xT_triton against a float32 PyTorch reference.
    """
    from .benchmark import report_error

    device = torch.device("cuda", torch.cuda.current_device())

    def make_inputs(B, M, N, K, dtype=torch.bfloat16):
        return (
            torch.randn(B, K, M, device=device, dtype=dtype),
            torch.randn(B, N, K, device=device, dtype=dtype),
            torch.randn(B, K, M, device=device, dtype=dtype),
            torch.randn(B, N, K, device=device, dtype=dtype),
        )

    def ref_func(W0, X0T, W1, X1T):
        return two_mm(W0, X0T, W1, X1T, A_transpose=True, B_transpose=True)

    shape_list = [[1, 512, 512, 512], [2, 2048, 1024, 4096], [1, 678, 724, 996]]
    for B, M, N, K in shape_list:
        _inputs = make_inputs(B, M, N, K)
        O_triton = fused_two_mm_same_out_wT_xT_triton(*_inputs)
        fp32_inputs = [_inp.float() for _inp in _inputs]
        O_ref = ref_func(*fp32_inputs)
        report_error(O_ref, O_triton, "O")


def check_correctness_four_mm():
    """
    Validates fused_four_mm_same_out_interface against a float32 PyTorch reference.
    """
    from .benchmark import report_error

    device = torch.device("cuda", torch.cuda.current_device())

    def make_inputs(B, M, N, K, dtype=torch.bfloat16):
        return (
            torch.randn(B, K, M, device=device, dtype=dtype),
            torch.randn(B, K, N, device=device, dtype=dtype),
            torch.randn(B, K, M, device=device, dtype=dtype),
            torch.randn(B, K, N, device=device, dtype=dtype),
            torch.randn(B, K, M, device=device, dtype=dtype),
            torch.randn(B, K, N, device=device, dtype=dtype),
            torch.randn(B, K, M, device=device, dtype=dtype),
            torch.randn(B, K, N, device=device, dtype=dtype),
        )

    def ref_func(W0, X0, W1, X1, W2, X2, W3, X3):
        O = (
            W0.transpose(1, 2) @ X0
            + W1.transpose(1, 2) @ X1
            + W2.transpose(1, 2) @ X2
            + W3.transpose(1, 2) @ X3
        )
        return O

    shape_list = [[1, 512, 512, 512], [2, 2048, 1024, 4096], [1, 678, 724, 996]]
    for B, M, N, K in shape_list:
        _inputs = make_inputs(B, M, N, K)
        O_triton = fused_four_mm_same_out_interface(*_inputs)
        fp32_inputs = [_inp.float() for _inp in _inputs]
        O_ref = ref_func(*fp32_inputs)
        print(f"B={B}, M={M}, N={N}, K={K} results")
        report_error(O_ref, O_triton, "O")


def _reference_wT_xT_varlen_padded(W0, X0T, W1, X1T, cu_seqlens):
    """Pad W -> batched kernel -> unpack."""
    from grouped_gemm import _pack_to_padded, _padded_to_pack

    G = cu_seqlens.shape[0] - 1
    T = W0.shape[0]
    max_len = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

    # W0, W1 are [T, K] packed -> pad to [G, max_len, K] -> transpose to [G, K, max_len]
    W0_pad = _pack_to_padded(W0, cu_seqlens, max_len).transpose(1, 2).contiguous()
    W1_pad = _pack_to_padded(W1, cu_seqlens, max_len).transpose(1, 2).contiguous()
    # X0T, X1T are [G, N, K] per-doc — already batched
    out_batch = fused_two_mm_same_out_wT_xT_triton(W0_pad, X0T, W1_pad, X1T)
    # out_batch: [G, max_len, N] -> unpack to [T, N]
    return _padded_to_pack(out_batch, cu_seqlens, T)


def _reference_wT_x_varlen_padded(W0, X0, W1, X1, cu_seqlens):
    """Pad W -> batched kernel -> unpack."""
    from grouped_gemm import _pack_to_padded, _padded_to_pack

    G = cu_seqlens.shape[0] - 1
    T = W0.shape[0]
    max_len = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

    # W0, W1 are [T, K] packed -> pad to [G, max_len, K] -> transpose to [G, K, max_len]
    W0_pad = _pack_to_padded(W0, cu_seqlens, max_len).transpose(1, 2).contiguous()
    W1_pad = _pack_to_padded(W1, cu_seqlens, max_len).transpose(1, 2).contiguous()
    # X0, X1 are [G, K, N] per-doc — already batched
    out_batch = fused_two_mm_same_out_wT_x_triton(W0_pad, X0, W1_pad, X1)
    # out_batch: [G, max_len, N] -> unpack to [T, N]
    return _padded_to_pack(out_batch, cu_seqlens, T)


if __name__ == "__main__":
    import argparse
    from kernel_test_utils import (
        test_correctness, benchmark, get_chunk_info,
        BENCH_DOC_LENS, make_cu_seqlens,
    )
    from grouped_gemm import _pack_to_padded
    from utils import compute_varlen_args

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    device = "cuda"
    K, N = 512, 256
    tol = dict(atol=1e-2, rtol=1e-2)

    # ================================================================
    #  CORRECTNESS
    # ================================================================
    doc_lens = [4096, 3072, 2048, 1024]
    cu = make_cu_seqlens(doc_lens, device)
    T = cu[-1].item()
    G = len(doc_lens)
    chunk_size = 2048

    W0 = torch.randn(T, K, device=device, dtype=torch.bfloat16)
    W1 = torch.randn(T, K, device=device, dtype=torch.bfloat16)
    X0T = torch.randn(G, N, K, device=device, dtype=torch.bfloat16)
    X1T = torch.randn(G, N, K, device=device, dtype=torch.bfloat16)
    X0 = torch.randn(G, K, N, device=device, dtype=torch.bfloat16)
    X1 = torch.randn(G, K, N, device=device, dtype=torch.bfloat16)

    print(f"Correctness: doc_lens={doc_lens}, {K=}, {N=}")

    # varlen vs padded
    for name, tri_fn, ref_fn, tri_a, ref_a in [
        ("wT_xT varlen",
         fused_two_mm_same_out_wT_xT_varlen_triton,
         _reference_wT_xT_varlen_padded,
         (W0, X0T, W1, X1T, cu), (W0, X0T, W1, X1T, cu)),
        ("wT_x  varlen",
         fused_two_mm_same_out_wT_x_varlen_triton,
         _reference_wT_x_varlen_padded,
         (W0, X0, W1, X1, cu), (W0, X0, W1, X1, cu)),
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
        for label, fn, extra in [
            ("wT_xT", fused_two_mm_same_out_wT_xT_varlen_triton, (X0T, W1, X1T)),
            ("wT_x",  fused_two_mm_same_out_wT_x_varlen_triton,  (X0, W1, X1)),
        ]:
            ref_out = fn(W0[idx], extra[0], extra[1][idx], extra[2], chunk_cu)
            test_out = fn(W0, extra[0], W1, extra[2], cu, chunk_size=chunk_size, chunk_idx=ci)
            diff = (test_out[idx] - ref_out).abs().max().item()
            assert diff == 0.0, f"{label} ci={ci} diff={diff}"
        print(f"    ci={ci}: exact match (n={len(idx)})")
    print("✓ All correctness tests passed\n")

    # ================================================================
    #  BENCHMARKS
    # ================================================================
    print("=" * 72)
    print(f"{'Config':<16} {'Kernel':<12} {'Triton ms':>10} {'Padded ms':>10} {'Speedup':>8}")
    print("=" * 72)

    for cfg_name, dl in BENCH_DOC_LENS.items():
        cu = make_cu_seqlens(dl, device)
        T = cu[-1].item()
        G = len(dl)
        max_len = max(dl)
        eff, bos, msl = compute_varlen_args(cu)

        W0 = torch.randn(T, K, device=device, dtype=torch.bfloat16)
        W1 = torch.randn(T, K, device=device, dtype=torch.bfloat16)
        X0T = torch.randn(G, N, K, device=device, dtype=torch.bfloat16)
        X1T = torch.randn(G, N, K, device=device, dtype=torch.bfloat16)
        X0 = torch.randn(G, K, N, device=device, dtype=torch.bfloat16)
        X1 = torch.randn(G, K, N, device=device, dtype=torch.bfloat16)
        W0_pad = _pack_to_padded(W0, cu, max_len).transpose(1, 2).contiguous()
        W1_pad = _pack_to_padded(W1, cu, max_len).transpose(1, 2).contiguous()

        for kernel_name, tri_fn, ref_fn, tri_a, ref_a in [
            ("wT_xT",
             lambda *a, _e=eff, _b=bos, _m=msl: fused_two_mm_same_out_wT_xT_varlen_triton(
                 *a, eff_lens=_e, bos_arr=_b, max_sl=_m),
             lambda W0p, X0T, W1p, X1T: fused_two_mm_same_out_wT_xT_triton(W0p, X0T, W1p, X1T),
             (W0, X0T, W1, X1T, cu), (W0_pad, X0T, W1_pad, X1T)),
            ("wT_x",
             lambda *a, _e=eff, _b=bos, _m=msl: fused_two_mm_same_out_wT_x_varlen_triton(
                 *a, eff_lens=_e, bos_arr=_b, max_sl=_m),
             lambda W0p, X0, W1p, X1: fused_two_mm_same_out_wT_x_triton(W0p, X0, W1p, X1),
             (W0, X0, W1, X1, cu), (W0_pad, X0, W1_pad, X1)),
        ]:
            r = benchmark(tri_fn, ref_fn, tri_a, ref_a, verbose=False)
            print(f"{cfg_name:<16} {kernel_name:<12} {r['time_triton']:>10.4f} {r['time_ref']:>10.4f} {r['speedup']:>7.2f}x")

        print("-" * 72)
