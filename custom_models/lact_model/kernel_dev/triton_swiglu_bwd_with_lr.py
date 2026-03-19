import math
import time
import torch
import triton
import triton.language as tl
import itertools


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
# Three fused GEMMs (W0@X.T, W2@X.T, W1.T@V.T) with
# SiLU backward epilogue + per-token lr scaling.
#
# IS_VARLEN=False: batched mode
#   W0_W2: [B, 2M, K], W1: [B, K, M], X/V: [B, N, K], lr: [B, N]
#   → DY0_DY2: [B, 2M, N], Hidden: [B, M, N]
#
# IS_VARLEN=True: packed varlen mode
#   W0_W2: [G, 2M, K], W1: [G, K, M], X/V: [T, K], lr: [T]
#   → DY0_DY2: [T, 2M], Hidden: [T, M]
########################################################


@triton.autotune(configs=get_autotune_configs(), key=["B", "M", "N", "K"])
@triton.jit
def _swiglu_three_bmm_with_lr_kernel(
    w0_w2_ptr,
    w1_ptr,
    x_ptr,
    v_ptr,
    lr0_ptr,  # scales DY0
    lr1_ptr,  # scales Hidden
    lr2_ptr,  # scales DY2
    # outputs
    dy0_dy2_ptr,
    hidden_ptr,
    eff_lens,
    bos_arr,
    B,
    M: tl.constexpr,
    N,
    K: tl.constexpr,
    # strides for W0_W2 [B, 2M, K]
    s_w0w2_b,
    s_w0w2_m,
    s_w0w2_k,
    # strides for W1 [B, K, M]  (note axes)
    s_w1_b,
    s_w1_k,
    s_w1_m,
    # strides for X and V [B, N, K] or [T, K]
    s_x_b,
    s_x_n,
    s_x_k,
    # strides for lr* [B, N] or unused for varlen (lr is [T], stride=1)
    s_lr_b,
    s_lr_n,
    # strides for dy0_dy2 [B, 2M, N] or [T, 2M]
    s_dy0_dy2_b,
    s_dy0_dy2_m,
    s_dy0_dy2_n,
    # strides for Hidden [B, M, N] or [T, M]
    s_h_b,
    s_h_m,
    s_h_n,
    out_dtype: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    # meta
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    N_eff = N
    if IS_VARLEN:
        N_eff = tl.load(eff_lens + pid_b).to(tl.int32)
        if pid_n * BLOCK_N >= N_eff:
            return
        bos = tl.load(bos_arr + pid_b).to(tl.int64)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Masks for ragged tiles
    mask_m = offs_m < M
    mask_n = offs_n < N_eff

    # FP32 accumulators
    acc_y0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_y2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_dh = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Bases for this batch/doc (weights always [B/G, ...])
    w0_batch = w0_w2_ptr + pid_b * s_w0w2_b
    w2_batch = w0_w2_ptr + pid_b * s_w0w2_b + M * s_w0w2_m
    w1_batch = w1_ptr + pid_b * s_w1_b

    if IS_VARLEN:
        x_base = x_ptr + bos * s_x_n
        v_base = v_ptr + bos * s_x_n
    else:
        x_base = x_ptr + pid_b * s_x_b
        v_base = v_ptr + pid_b * s_x_b

    # --- K loop ---
    num_k_tiles = tl.cdiv(K, BLOCK_K)
    for ko in range(0, num_k_tiles):
        k0 = ko * BLOCK_K
        k_ids = k0 + offs_k
        mask_k = k_ids < K

        # A0 = W0[offs_m, k_ids] -> [M, K]
        a0_ptrs = w0_batch + (offs_m[:, None] * s_w0w2_m + k_ids[None, :] * s_w0w2_k)
        a0 = tl.load(a0_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # A2 = W2[offs_m, k_ids] -> [M, K]
        a2_ptrs = w2_batch + (offs_m[:, None] * s_w0w2_m + k_ids[None, :] * s_w0w2_k)
        a2 = tl.load(a2_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # A1 = (W1^T)[offs_m, k_ids] = W1[k_ids, offs_m]  -> [M, K]
        a1_ptrs = w1_batch + (k_ids[None, :] * s_w1_k + offs_m[:, None] * s_w1_m)
        a1 = tl.load(a1_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Bx = X^T[k_ids, offs_n] = X[offs_n, k_ids] -> [K, N]
        bx_ptrs = x_base + (offs_n[None, :] * s_x_n + k_ids[:, None] * s_x_k)
        bx = tl.load(bx_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # Bv = V^T[k_ids, offs_n] -> [K, N]
        bv_ptrs = v_base + (offs_n[None, :] * s_x_n + k_ids[:, None] * s_x_k)
        bv = tl.load(bv_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # Three GEMMs (FP32 accumulate)
        acc_y0 += tl.dot(a0, bx, out_dtype=tl.float32)
        acc_y2 += tl.dot(a2, bx, out_dtype=tl.float32)
        acc_dh += tl.dot(a1, bv, out_dtype=tl.float32)

    # --- Epilogue on fragments (FP32 math) ---
    y0 = acc_y0
    y2 = acc_y2
    dh = acc_dh

    sigma = tl.sigmoid(y0)
    swish = sigma * y0

    hidden_tile = swish * y2
    dy0_tile = sigma * y2 * dh * (1.0 + y0 * (1.0 - sigma))
    dy2_tile = swish * dh

    # --- Load per-token scaling vectors, cast to fp32 ---
    if IS_VARLEN:
        # lr is packed [T], load via global token indices
        glob_n = bos + offs_n.to(tl.int64)
        lr0_vec = tl.load(lr0_ptr + glob_n, mask=mask_n, other=0.0).to(tl.float32)
        lr1_vec = tl.load(lr1_ptr + glob_n, mask=mask_n, other=0.0).to(tl.float32)
        lr2_vec = tl.load(lr2_ptr + glob_n, mask=mask_n, other=0.0).to(tl.float32)
    else:
        lr0_batch = lr0_ptr + pid_b * s_lr_b
        lr1_batch = lr1_ptr + pid_b * s_lr_b
        lr2_batch = lr2_ptr + pid_b * s_lr_b
        lr0_vec = tl.load(lr0_batch + offs_n * s_lr_n, mask=mask_n, other=0.0).to(tl.float32)
        lr1_vec = tl.load(lr1_batch + offs_n * s_lr_n, mask=mask_n, other=0.0).to(tl.float32)
        lr2_vec = tl.load(lr2_batch + offs_n * s_lr_n, mask=mask_n, other=0.0).to(tl.float32)

    # Broadcast to [BLOCK_M, BLOCK_N] and scale.
    dy0_tile *= lr0_vec[None, :]
    dy2_tile *= lr2_vec[None, :]
    hidden_tile *= lr1_vec[None, :]

    # Store with Casting
    out_dtype_tl = (
        tl.float16
        if out_dtype == "fp16"
        else tl.bfloat16 if out_dtype == "bf16" else tl.float32
    )

    if IS_VARLEN:
        # Store transposed to packed [T, 2M] and [T, M]
        mask_out_T = mask_n[:, None] & mask_m[None, :]

        dy0_ptrs = dy0_dy2_ptr + glob_n[:, None] * s_dy0_dy2_n + offs_m[None, :] * s_dy0_dy2_m
        tl.store(dy0_ptrs, tl.trans(dy0_tile).to(out_dtype_tl), mask=mask_out_T)

        dy2_ptrs = dy0_dy2_ptr + glob_n[:, None] * s_dy0_dy2_n + (offs_m[None, :] + M) * s_dy0_dy2_m
        tl.store(dy2_ptrs, tl.trans(dy2_tile).to(out_dtype_tl), mask=mask_out_T)

        hid_ptrs = hidden_ptr + glob_n[:, None] * s_h_n + offs_m[None, :] * s_h_m
        tl.store(hid_ptrs, tl.trans(hidden_tile).to(out_dtype_tl), mask=mask_out_T)
    else:
        # [B, 2M, N]
        dy0_ptrs = (
            dy0_dy2_ptr
            + pid_b * s_dy0_dy2_b
            + (offs_m[:, None] * s_dy0_dy2_m + offs_n[None, :] * s_dy0_dy2_n)
        )
        dy2_ptrs = (
            dy0_dy2_ptr
            + pid_b * s_dy0_dy2_b
            + M * s_dy0_dy2_m
            + (offs_m[:, None] * s_dy0_dy2_m + offs_n[None, :] * s_dy0_dy2_n)
        )
        hid_ptrs = (
            hidden_ptr + pid_b * s_h_b + (offs_m[:, None] * s_h_m + offs_n[None, :] * s_h_n)
        )

        mask_out = mask_m[:, None] & mask_n[None, :]
        tl.store(dy0_ptrs, dy0_tile.to(out_dtype_tl), mask=mask_out)
        tl.store(dy2_ptrs, dy2_tile.to(out_dtype_tl), mask=mask_out)
        tl.store(hid_ptrs, hidden_tile.to(out_dtype_tl), mask=mask_out)


def swiglu_backward_three_bmm_with_lr_triton(W0_W2, W1, X, V, lr0, lr1, lr2):
    """
    Args:
        W0_W2: [B, 2M, K], W1: [B, K, M], X/V: [B, N, K], lr0/lr1/lr2: [B, N]
    Returns:
        DY0_DY2: [B, 2M, N], Hidden: [B, M, N]
    """
    B, M_times_2, K = W0_W2.shape
    M = M_times_2 // 2
    Bx, N, Kx = X.shape
    assert W1.shape == (B, K, M)
    assert V.shape == (B, N, K)
    assert (
        W0_W2.dtype == torch.bfloat16 and V.dtype == torch.bfloat16
    ), "W0_W2 and V must be bf16"
    assert (
        W0_W2.is_contiguous()
        and W1.is_contiguous()
        and X.is_contiguous()
        and V.is_contiguous()
    )
    assert lr0.shape == (B, N) and lr1.shape == (B, N) and lr2.shape == (B, N)

    # compute strides assuming contigous inputs
    s_w0w2_b, s_w0w2_m, s_w0w2_k = K * M_times_2, K, 1
    s_w1_b, s_w1_k, s_w1_m = K * M, M, 1
    s_x_b, s_x_n, s_x_k = K * N, K, 1
    s_lr_b, s_lr_n = lr0.stride(0), lr0.stride(1)

    s_dy0_dy2_b, s_dy0_dy2_m, s_dy0_dy2_n = M_times_2 * N, N, 1
    s_h_b, s_h_m, s_h_n = M * N, N, 1

    # Allocate outputs (compute dtype)
    Hidden = torch.empty((B, M, N), device=X.device, dtype=X.dtype)
    DY0_DY2 = torch.empty((B, M_times_2, N), device=X.device, dtype=X.dtype)
    _dummy = torch.empty(0, dtype=torch.int32, device=X.device)

    # Make the store dtype match the destination tensors (robust if W0.dtype != X.dtype)
    out_dtype_str = (
        "float32"
        if Hidden.dtype == torch.float32
        else "bf16" if Hidden.dtype == torch.bfloat16 else "fp16"
    )

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]), B)

    _swiglu_three_bmm_with_lr_kernel[grid](
        W0_W2, W1, X, V, lr0, lr1, lr2,
        DY0_DY2, Hidden, _dummy, _dummy,
        B, M, N, K,
        s_w0w2_b, s_w0w2_m, s_w0w2_k,
        s_w1_b, s_w1_k, s_w1_m,
        s_x_b, s_x_n, s_x_k,
        s_lr_b, s_lr_n,
        s_dy0_dy2_b, s_dy0_dy2_m, s_dy0_dy2_n,
        s_h_b, s_h_m, s_h_n,
        out_dtype=out_dtype_str,
        IS_VARLEN=False,
    )

    return DY0_DY2, Hidden


def swiglu_backward_three_bmm_with_lr_varlen_triton(
    W0_W2, W1, X, V, lr0, lr1, lr2, cu_seqlens,
    chunk_size=0, chunk_idx=0,
    eff_lens=None, bos_arr=None, max_sl=0,
):
    """
    Varlen variant. Shapes:
      W0_W2 : [G, 2M, K]  (bf16) — per-doc weights
      W1    : [G, K, M]   (bf16) — per-doc weights
      X, V  : [T, K]      (bf16) — packed tokens
      lr0, lr1, lr2: [T]  (fp32) — per-token learning rates
      cu_seqlens: [G+1]   (int32)
      eff_lens, bos_arr, max_sl: precomputed (if None, computed from cu_seqlens + chunk params)
    Returns:
      DY0_DY2: [T, 2M] (bf16) — packed, lr-scaled
      Hidden:  [T, M]  (bf16) — packed, lr-scaled
    """
    try:
        from utils import compute_varlen_args
    except ImportError:
        from .utils import compute_varlen_args

    assert W0_W2.dtype == torch.bfloat16 and V.dtype == torch.bfloat16
    assert W0_W2.is_contiguous() and W1.is_contiguous()
    assert X.is_contiguous() and V.is_contiguous()

    G, M2, K = W0_W2.shape
    M = M2 // 2
    T = X.shape[0]

    DY0_DY2 = torch.empty((T, M2), device=X.device, dtype=X.dtype)
    Hidden = torch.empty((T, M), device=X.device, dtype=X.dtype)

    out_dtype_str = (
        "float32" if Hidden.dtype == torch.float32
        else "bf16" if Hidden.dtype == torch.bfloat16 else "fp16"
    )

    if eff_lens is None:
        eff_lens, bos_arr, max_sl = compute_varlen_args(cu_seqlens, chunk_size, chunk_idx)

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(max_sl, meta["BLOCK_N"]),
            G,
        )

    _swiglu_three_bmm_with_lr_kernel[grid](
        W0_W2, W1, X, V, lr0, lr1, lr2,
        DY0_DY2, Hidden, eff_lens, bos_arr,
        G, M, max_sl, K,
        W0_W2.stride(0), W0_W2.stride(1), W0_W2.stride(2),
        W1.stride(0), W1.stride(1), W1.stride(2),
        0, X.stride(0), X.stride(1),                           # s_x_b=0, s_x_n, s_x_k
        0, 1,                                                   # s_lr_b=0, s_lr_n=1 (unused)
        0, DY0_DY2.stride(1), DY0_DY2.stride(0),              # s_dy_b=0, s_dy_m, s_dy_n
        0, Hidden.stride(1), Hidden.stride(0),                  # s_h_b=0, s_h_m, s_h_n
        out_dtype=out_dtype_str,
        IS_VARLEN=True,
    )
    return DY0_DY2, Hidden


@torch.no_grad()
def ref_func(W0_W2, W1, X, dOut, lr0, lr1, lr2):
    """
    Shapes:
      W0: [B, M, K]
      W1: [B, K, M]
      W2: [B, M, K]
      X : [B, N, K]
      dOut (a.k.a. V): [B, N, K]
      lr0, lr1, lr2: [B, N]
    Returns (for convenience): DY0, DY2, Hidden
    """
    W0, W2 = W0_W2.chunk(2, dim=1)
    Y0 = torch.bmm(W0, X.transpose(1, 2))  # [B, M, N]
    Y2 = torch.bmm(W2, X.transpose(1, 2))  # [B, M, N]
    Hidden = torch.nn.functional.silu(Y0) * Y2
    DHidden = torch.bmm(W1.transpose(1, 2), dOut.transpose(1, 2))  # [B, M, N]
    DY0 = DHidden * Y2 * torch.sigmoid(Y0) * (1 + Y0 * (1 - torch.sigmoid(Y0)))
    DY2 = DHidden * torch.nn.functional.silu(Y0)

    # Column-wise scalings: match kernel
    DY0 = DY0 * lr0.unsqueeze(1)  # [B, 1, N]
    DY2 = DY2 * lr2.unsqueeze(1)  # [B, 1, N]
    Hidden = Hidden * lr1.unsqueeze(1)
    return torch.cat([DY0, DY2], dim=1), Hidden


def make_inputs(B, H, D, L, lr_dtype=torch.bfloat16):
    """
    W0, W1: [B, K, M]
    X0, X1: [B, K, N]

    W2:     [B, M, N]
    X2:     [B, K, M]
    """
    device = torch.device("cuda", torch.cuda.current_device())
    W0_W2 = torch.randn(
        B, 2 * H, D, device=device, dtype=torch.bfloat16, requires_grad=True
    )

    W1 = torch.randn(B, D, H, device=device, dtype=torch.bfloat16, requires_grad=True)

    X = torch.randn(B, L, D, device=device, dtype=torch.bfloat16, requires_grad=True)

    dOut = torch.randn(B, L, D, device=device, dtype=torch.bfloat16, requires_grad=True)

    lr0 = torch.randn(B, L, device=device, dtype=lr_dtype, requires_grad=True)
    lr1 = torch.randn(B, L, device=device, dtype=lr_dtype, requires_grad=True)
    lr2 = torch.randn(B, L, device=device, dtype=lr_dtype, requires_grad=True)

    return W0_W2, W1, X, dOut, lr0, lr1, lr2


def check_correctness():

    device = torch.device("cuda", torch.cuda.current_device())

    from .benchmark import report_error

    inps = make_inputs(4, 2048, 1024, 8192, lr_dtype=torch.float32)
    DY0_DY2, Hidden = swiglu_backward_three_bmm_with_lr_triton(*inps)

    fp32_inps = [_.to(torch.float32) for _ in inps]
    DY0_DY2_ref, Hidden_ref = ref_func(*fp32_inps)

    # print(torch.allclose(DY0, DY0_ref))
    # print(torch.allclose(DY2, DY2_ref))
    # print(torch.allclose(Hidden, Hidden_ref))

    report_error(DY0_DY2_ref, DY0_DY2, "DY0_DY2")
    report_error(Hidden_ref, Hidden, "Hidden")


def _reference_varlen_padded(W0_W2, W1, X, V, lr0, lr1, lr2, cu_seqlens):
    """Pad -> batched Triton kernel -> unpack. Same precision as varlen kernel."""
    from grouped_gemm import _pack_to_padded, _padded_to_pack

    G = cu_seqlens.shape[0] - 1
    T = X.shape[0]
    max_len = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

    X_pad = _pack_to_padded(X, cu_seqlens, max_len)
    V_pad = _pack_to_padded(V, cu_seqlens, max_len)
    lr0_pad = _pack_to_padded(lr0.unsqueeze(1), cu_seqlens, max_len).squeeze(2)
    lr1_pad = _pack_to_padded(lr1.unsqueeze(1), cu_seqlens, max_len).squeeze(2)
    lr2_pad = _pack_to_padded(lr2.unsqueeze(1), cu_seqlens, max_len).squeeze(2)

    DY0_DY2_b, Hidden_b = swiglu_backward_three_bmm_with_lr_triton(
        W0_W2, W1, X_pad, V_pad, lr0_pad, lr1_pad, lr2_pad,
    )
    # [G, 2M, max_len] -> [G, max_len, 2M] -> packed [T, 2M]
    return (
        _padded_to_pack(DY0_DY2_b.transpose(1, 2), cu_seqlens, T),
        _padded_to_pack(Hidden_b.transpose(1, 2), cu_seqlens, T),
    )


if __name__ == "__main__":
    import argparse
    from kernel_test_utils import test_correctness, benchmark, get_chunk_info

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

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
    W1 = torch.randn(num_docs, d, dh, device=device, dtype=torch.bfloat16)
    X = torch.randn(packed_len, d, device=device, dtype=torch.bfloat16)
    V = torch.randn(packed_len, d, device=device, dtype=torch.bfloat16)
    lr0 = torch.randn(packed_len, device=device, dtype=torch.float32) * 0.01
    lr1 = torch.randn(packed_len, device=device, dtype=torch.float32) * 0.01
    lr2 = torch.randn(packed_len, device=device, dtype=torch.float32) * 0.01

    print(f"Config: {num_docs=}, {doc_lens=}, {d=}, {dh=}, {packed_len=}")
    print()

    # ===== Correctness vs padded batched kernel =====
    print("=" * 60)
    print("Varlen correctness vs padded batched Triton kernel")
    print("=" * 60)

    triton_args = (W0_W2, W1, X, V, lr0, lr1, lr2, cu_seqlens)

    test_correctness(
        lambda *a: swiglu_backward_three_bmm_with_lr_varlen_triton(*a),
        lambda *a: _reference_varlen_padded(*a),
        triton_args, triton_args,
        debug=args.debug, atol=1e-2, rtol=1e-2,
    )

    # ===== Benchmark =====
    print()
    print("=" * 60)
    print("Varlen benchmark vs padded batched kernel")
    print("=" * 60)

    from grouped_gemm import _pack_to_padded
    max_len = max(doc_lens)
    X_pad = _pack_to_padded(X, cu_seqlens, max_len)
    V_pad = _pack_to_padded(V, cu_seqlens, max_len)
    lr0_pad = _pack_to_padded(lr0.unsqueeze(1), cu_seqlens, max_len).squeeze(2)
    lr1_pad = _pack_to_padded(lr1.unsqueeze(1), cu_seqlens, max_len).squeeze(2)
    lr2_pad = _pack_to_padded(lr2.unsqueeze(1), cu_seqlens, max_len).squeeze(2)

    ref_bench_args = (W0_W2, W1, X_pad, V_pad, lr0_pad, lr1_pad, lr2_pad)

    try:
        from utils import compute_varlen_args
    except ImportError:
        from .utils import compute_varlen_args
    eff_lens, bos_arr, max_sl_val = compute_varlen_args(cu_seqlens)

    benchmark(
        lambda *a: swiglu_backward_three_bmm_with_lr_varlen_triton(
            *a, eff_lens=eff_lens, bos_arr=bos_arr, max_sl=max_sl_val,
        ),
        lambda *a: swiglu_backward_three_bmm_with_lr_triton(*a),
        triton_args, ref_bench_args,
    )

    # ===== Chunk correctness =====
    chunk_size = 2048
    _, _, max_chunks = get_chunk_info(cu_seqlens, chunk_size, 0)

    print()
    print("=" * 60)
    print(f"Chunk correctness (chunk_size={chunk_size})")
    print("=" * 60)

    for chunk_index in range(max_chunks):
        chunk_cu, idx, _ = get_chunk_info(cu_seqlens, chunk_size, chunk_index)
        if len(idx) == 0:
            break
        # Reference: gather chunk tokens, call base varlen kernel
        ref_dy, ref_h = swiglu_backward_three_bmm_with_lr_varlen_triton(
            W0_W2, W1, X[idx], V[idx], lr0[idx], lr1[idx], lr2[idx], chunk_cu,
        )
        # Test: call chunk kernel on full buffer
        test_dy, test_h = swiglu_backward_three_bmm_with_lr_varlen_triton(
            W0_W2, W1, X, V, lr0, lr1, lr2, cu_seqlens,
            chunk_size=chunk_size, chunk_idx=chunk_index,
        )
        diff_dy = (test_dy[idx] - ref_dy).abs().max().item()
        diff_h = (test_h[idx] - ref_h).abs().max().item()
        print(f"  chunk_idx={chunk_index}: DY0_DY2 max_diff={diff_dy:.2e}, Hidden max_diff={diff_h:.2e}, n_tokens={len(idx)}")
        assert diff_dy == 0.0 and diff_h == 0.0, f"chunk_idx={chunk_index} not exact match!"

    print("✓ PASS (exact match)")
