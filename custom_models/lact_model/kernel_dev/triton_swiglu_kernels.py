# fused_two_mm_swiglu_triton.py
import math
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
# W0_W2: [B, 2M, K]
# X: [B, N, K]  (batched)  or  [T, K]  (varlen)
# O: [B, M, N]  (batched)  or  [T, M]  (varlen)
#
# IS_VARLEN=False: standard batched mode
# IS_VARLEN=True:  packed varlen mode — eff_lens/bos_arr precomputed
########################################################


@triton.autotune(
    configs=get_autotune_configs(),
    key=["M", "N", "K"],
)
@triton.jit
def _fused_two_mm_swiglu_kernel(
    W0_W2,
    X,
    O,
    eff_lens,   # [G] per-doc effective token count (varlen), or dummy (batched)
    bos_arr,    # [G] per-doc start offset in X (varlen), or dummy (batched)
    B,
    M: tl.constexpr,
    N,
    K: tl.constexpr,  # mark the reduce axis as constexpr
    stride_w_b,  # = 2M * K
    stride_w_m,  # = K
    stride_w_k,
    stride_x_b,
    stride_x_n,
    stride_x_k,
    stride_o_b,
    stride_o_m,
    stride_o_n,
    IS_VARLEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 3D launch grid: (m-tiles, n-tiles, batch/docs)
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    pid_b = tl.program_id(axis=2)

    N_eff = N
    if IS_VARLEN:
        N_eff = tl.load(eff_lens + pid_b).to(tl.int32)
        if pid_n * BLOCK_N >= N_eff:
            return
        bos = tl.load(bos_arr + pid_b).to(tl.int64)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    offs_k = tl.arange(0, BLOCK_K)  # [BLOCK_K]

    mask_m = offs_m < M
    mask_n = offs_n < N_eff

    # Pointers base for this batch/doc
    w0_batch = W0_W2 + pid_b * stride_w_b
    w2_batch = W0_W2 + pid_b * stride_w_b + M * stride_w_m

    if IS_VARLEN:
        x_base = X + bos * stride_x_n
    else:
        x_base = X + pid_b * stride_x_b

    # Accumulators in fp32
    if IS_VARLEN:
        # Accumulate as [BLOCK_N, BLOCK_M] to match [T, M] output — no transpose on store
        acc0 = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    else:
        acc0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K loop
    for k0 in range(0, K, BLOCK_K):
        k_ids = k0 + offs_k  # [BLOCK_K]
        k_mask = k_ids < K

        # Load tiles: W0/W2 shape -> (BLOCK_M, BLOCK_K), X shape -> (BLOCK_N, BLOCK_K)
        w0_ptrs = (
            w0_batch + (offs_m[:, None] * stride_w_m) + (k_ids[None, :] * stride_w_k)
        )
        w2_ptrs = (
            w2_batch + (offs_m[:, None] * stride_w_m) + (k_ids[None, :] * stride_w_k)
        )
        x_ptrs = (
            x_base + (offs_n[:, None] * stride_x_n) + (k_ids[None, :] * stride_x_k)
        )

        w_mask = mask_m[:, None] & k_mask[None, :]
        x_mask = mask_n[:, None] & k_mask[None, :]

        w0 = tl.load(w0_ptrs, mask=w_mask, other=0).to(tl.bfloat16)
        w2 = tl.load(w2_ptrs, mask=w_mask, other=0).to(tl.bfloat16)
        x = tl.load(x_ptrs, mask=x_mask, other=0).to(tl.bfloat16)  # (BLOCK_N, BLOCK_K)

        if IS_VARLEN:
            # (N,K) x (K,M): x @ w^T -> [BLOCK_N, BLOCK_M]
            acc0 += tl.dot(x, tl.trans(w0), out_dtype=tl.float32)
            acc2 += tl.dot(x, tl.trans(w2), out_dtype=tl.float32)
        else:
            # (M,K) x (K,N): w @ x^T -> [BLOCK_M, BLOCK_N]
            acc0 += tl.dot(w0, tl.trans(x), out_dtype=tl.float32)
            acc2 += tl.dot(w2, tl.trans(x), out_dtype=tl.float32)

    # Apply SiLU in fp32 and fuse multiply
    # SiLU(x) = x * sigmoid(x)
    out = acc2 * (acc0 * tl.sigmoid(acc0))

    # Store to bf16
    if IS_VARLEN:
        # out is [BLOCK_N, BLOCK_M], output is [T, M] — direct store
        glob_n = bos + offs_n.to(tl.int64)
        o_ptrs = O + glob_n[:, None] * stride_o_n + offs_m[None, :] * stride_o_m
        tl.store(o_ptrs, out.to(tl.bfloat16), mask=mask_n[:, None] & mask_m[None, :])
    else:
        o_batch = O + pid_b * stride_o_b
        o_ptrs = o_batch + (offs_m[:, None] * stride_o_m) + (offs_n[None, :] * stride_o_n)
        tl.store(o_ptrs, out.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])


def fused_two_mm_swiglu_triton(
    W0_W2: torch.Tensor,
    X: torch.Tensor,
):
    """
    Wraps the Triton kernel. Shapes:
      W0, W2: [B, M, K]  (bf16)
      X     : [B, N, K]  (bf16)
      returns O: [B, M, N] (bf16) where O = SiLU(W0 @ X^T) * (W2 @ X^T)
    """
    assert W0_W2.dtype == torch.bfloat16 and X.dtype == torch.bfloat16
    assert W0_W2.is_contiguous() and X.is_contiguous(), "W0_W2 and X must be contiguous"

    B, M_times_2, K = W0_W2.shape
    Bx, N, Kx = X.shape
    assert Bx == B and Kx == K, "X must be [B, N, K] with matching B,K."

    M = M_times_2 // 2

    O = torch.empty((B, M, N), device=X.device, dtype=torch.bfloat16)
    _dummy = torch.empty(0, dtype=torch.int32, device=X.device)

    # Strides (PyTorch: element strides)
    stride_w_b, stride_w_m, stride_w_k = W0_W2.stride()
    stride_x_b, stride_x_n, stride_x_k = X.stride()
    stride_o_b, stride_o_m, stride_o_n = O.stride()

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]), B)

    _fused_two_mm_swiglu_kernel[grid](
        W0_W2, X, O, _dummy, _dummy,
        B, M, N, K,
        stride_w_b, stride_w_m, stride_w_k,
        stride_x_b, stride_x_n, stride_x_k,
        stride_o_b, stride_o_m, stride_o_n,
        IS_VARLEN=False,
    )
    return O


def fused_two_mm_swiglu_varlen_triton(
    W0_W2: torch.Tensor,
    X: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_size: int = 0,
    chunk_idx: int = 0,
    eff_lens: torch.Tensor = None,
    bos_arr: torch.Tensor = None,
    max_sl: int = 0,
    out: torch.Tensor = None,
):
    """
    Varlen variant. Shapes:
      W0_W2 : [G, 2M, K]  (bf16) — per-doc weights
      X     : [T, K]       (bf16) — packed tokens
      cu_seqlens: [G+1]    (int32)
      eff_lens, bos_arr, max_sl: precomputed (if None, computed from cu_seqlens + chunk params)
      out   : optional [T, M] output buffer (writes in-place, no alloc)
      returns O: [T, M]    (bf16) — packed output
    """
    assert W0_W2.dtype == torch.bfloat16 and X.dtype == torch.bfloat16
    assert W0_W2.is_contiguous() and X.is_contiguous()

    G, M2, K = W0_W2.shape
    T, Kx = X.shape
    assert Kx == K
    M = M2 // 2

    O = out if out is not None else torch.empty((T, M), device=X.device, dtype=torch.bfloat16)

    if eff_lens is None:
        eff_lens, bos_arr, max_sl = compute_varlen_args(cu_seqlens, chunk_size, chunk_idx)

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(max_sl, meta["BLOCK_N"]),
            G,
        )

    _fused_two_mm_swiglu_kernel[grid](
        W0_W2, X, O, eff_lens, bos_arr,
        G, M, max_sl, K,
        W0_W2.stride(0), W0_W2.stride(1), W0_W2.stride(2),
        0, X.stride(0), X.stride(1),          # stride_x_b=0, stride_x_n, stride_x_k
        0, O.stride(1), O.stride(0),          # stride_o_b=0, stride_o_m, stride_o_n
        IS_VARLEN=True,
    )
    return O


try:
    from utils import compute_varlen_args
except ImportError:
    from .utils import compute_varlen_args


# -------------------------
# Test harness vs PyTorch
# -------------------------
@torch.compile
def _reference_pytorch(W0_W2, X):
    # Compute in fp32 and cast to bf16 to match kernel's final cast
    W0, W2 = W0_W2.chunk(2, dim=1)
    Y0 = torch.matmul(W0, X.transpose(-1, -2))  # [B, M, N]
    Y2 = torch.matmul(W2, X.transpose(-1, -2))  # [B, M, N]
    O = torch.nn.functional.silu(Y0) * Y2
    return O


def make_inputs_ffn(B, M, K, N, require_grad=True):
    W0_W2 = torch.randn(
        B, 2 * M, K, device="cuda", dtype=torch.bfloat16, requires_grad=require_grad
    )
    K_input = torch.randn(
        B, N, K, device="cuda", dtype=torch.bfloat16, requires_grad=require_grad
    )

    return W0_W2, K_input


def check_correctness():
    from .benchmark import report_error

    torch.manual_seed(0)
    device = "cuda"

    # Example sizes; feel free to change
    B, H, D, L = 4, 1536, 3072, 16384

    inputs = make_inputs_ffn(B, H, D, L)

    # Triton
    O_triton = fused_two_mm_swiglu_triton(*inputs)

    # Reference in fp32
    fp32_inputs = [_.to(torch.float32) for _ in inputs]
    O_ref = _reference_pytorch(*fp32_inputs)

    report_error(O_ref, O_triton, "triton_swiglu_kernel")


def _reference_varlen_padded(W0_W2, X, cu_seqlens):
    """Pad -> batched Triton kernel -> unpack. Same precision as varlen kernel."""
    from grouped_gemm import _pack_to_padded, _padded_to_pack

    G = cu_seqlens.shape[0] - 1
    T = X.shape[0]
    doc_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    max_len = doc_lens.max().item()

    X_pad = _pack_to_padded(X, cu_seqlens, max_len)  # [G, max_len, K]
    O_batched = fused_two_mm_swiglu_triton(W0_W2, X_pad)  # [G, M, max_len]
    O_padded = O_batched.transpose(1, 2)  # [G, max_len, M]
    return _padded_to_pack(O_padded, cu_seqlens, T)


if __name__ == "__main__":
    import argparse
    from kernel_test_utils import (
        test_correctness, benchmark, get_chunk_info,
        BENCH_DOC_LENS, make_cu_seqlens,
    )
    from grouped_gemm import _pack_to_padded

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    device = "cuda"
    d, dh = 512, 512
    tol = dict(atol=1e-2, rtol=1e-2)

    # ================================================================
    #  CORRECTNESS
    # ================================================================
    doc_lens = [4096, 3072, 2048, 1024]
    cu = make_cu_seqlens(doc_lens, device)
    T = cu[-1].item()
    G = len(doc_lens)
    chunk_size = 2048

    W0_W2 = torch.randn(G, 2 * dh, d, device=device, dtype=torch.bfloat16)
    X = torch.randn(T, d, device=device, dtype=torch.bfloat16)

    print(f"Correctness: doc_lens={doc_lens}, {d=}, {dh=}")

    # varlen vs padded batched
    print("\n  varlen vs padded: ", end="")
    ok = test_correctness(
        fused_two_mm_swiglu_varlen_triton, _reference_varlen_padded,
        (W0_W2, X, cu), (W0_W2, X, cu),
        debug=args.debug, **tol,
    )
    assert ok

    # chunk correctness
    _, _, max_chunks = get_chunk_info(cu, chunk_size, 0)
    print(f"\n  chunk correctness (chunk_size={chunk_size}):")
    for ci in range(max_chunks):
        chunk_cu, idx, _ = get_chunk_info(cu, chunk_size, ci)
        if len(idx) == 0:
            break
        ref_out = fused_two_mm_swiglu_varlen_triton(W0_W2, X[idx], chunk_cu)
        test_out = fused_two_mm_swiglu_varlen_triton(W0_W2, X, cu, chunk_size=chunk_size, chunk_idx=ci)
        diff = (test_out[idx] - ref_out).abs().max().item()
        assert diff == 0.0, f"ci={ci} diff={diff}"
        print(f"    ci={ci}: exact match (n={len(idx)})")
    print("✓ All correctness tests passed\n")

    # ================================================================
    #  BENCHMARKS
    # ================================================================
    print("=" * 72)
    print(f"{'Config':<16} {'Triton ms':>10} {'Padded ms':>10} {'Speedup':>8}")
    print("=" * 72)

    for cfg_name, dl in BENCH_DOC_LENS.items():
        cu = make_cu_seqlens(dl, device)
        T = cu[-1].item()
        G = len(dl)
        max_len = max(dl)
        eff, bos, msl = compute_varlen_args(cu)

        W0_W2 = torch.randn(G, 2 * dh, d, device=device, dtype=torch.bfloat16)
        X = torch.randn(T, d, device=device, dtype=torch.bfloat16)
        X_pad = _pack_to_padded(X, cu, max_len)

        r = benchmark(
            lambda W, Xv, c, _e=eff, _b=bos, _m=msl: fused_two_mm_swiglu_varlen_triton(
                W, Xv, c, eff_lens=_e, bos_arr=_b, max_sl=_m),
            lambda W, Xp: fused_two_mm_swiglu_triton(W, Xp),
            (W0_W2, X, cu), (W0_W2, X_pad),
            verbose=False,
        )
        print(f"{cfg_name:<16} {r['time_triton']:>10.4f} {r['time_ref']:>10.4f} {r['speedup']:>7.2f}x")

    print("=" * 72)
