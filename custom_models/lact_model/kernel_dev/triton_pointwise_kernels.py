import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 32, "BLOCK_L": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_D": 32, "BLOCK_L": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_D": 32, "BLOCK_L": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_D": 64, "BLOCK_L": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_D": 64, "BLOCK_L": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_D": 64, "BLOCK_L": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_D": 64, "BLOCK_L": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_D": 128, "BLOCK_L": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_D": 128, "BLOCK_L": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_D": 128, "BLOCK_L": 256}, num_warps=8, num_stages=2),
        # More aggressive configurations for larger problems
        triton.Config({"BLOCK_D": 64, "BLOCK_L": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_D": 128, "BLOCK_L": 128}, num_warps=8, num_stages=3),
    ],
    key=["D", "L"], reset_to_zero=["OUT_GRAD_LR0", "OUT_GRAD_LR1", "OUT_GRAD_LR2"] # Autotune based on these dimensions
)
@triton.jit
def _swiglu_bwd_bwd_fused_kernel(
    # ---- inputs ----
    DH,  # *[B, D, L] or [T, D]
    X0X2,  # *[B, 2D, L] or [T, 2D]
    LR0,
    LR1,
    LR2,  # *[B, L] or [T]
    GDX0DX2,  # *[B, 2D, L] or [T, 2D]
    GH_LR1,  # *[B, D, L] or [T, D]
    # ---- outputs ----
    OUT_GRAD_DH,  # *[B, D, L] or [T, D]
    OUT_GRAD_X0X2,  # *[B, 2D, L] or [T, 2D]
    OUT_GRAD_LR0,  # *[B, L] or [T]
    OUT_GRAD_LR1,
    OUT_GRAD_LR2,
    OUT_DX0X2,  # *[B, 2D, L] or [T, 2D]
    OUT_HIDDEN_LR1,  # *[B, D, L] or [T, D]
    eff_lens,
    bos_arr,
    # ---- sizes ----
    B,
    D: tl.constexpr,
    L,
    # ---- strides: [B, D, L] or [T, D] ----
    s_dh_b,
    s_dh_d,
    s_dh_l,
    # for X0X2: [B, 2D, L] or [T, 2D]
    s_x0x2_b,
    s_x0x2_d,
    s_x0x2_l,
    # for LR*
    s_lr_b,
    s_lr_l,
    # for GDX0DX2: [B, 2D, L] or [T, 2D]
    s_gdx0dx2_b,
    s_gdx0dx2_d,
    s_gdx0dx2_l,
    # for GH_LR1: [B, D, L] or [T, D]
    s_gh_lr1_b,
    s_gh_lr1_d,
    s_gh_lr1_l,
    s_out_grad_lr_b,
    s_out_grad_lr_l,
    IS_VARLEN: tl.constexpr,
    # ---- meta ----
    BLOCK_D: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    # program ids
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_d = tl.program_id(2)

    # --- varlen: load precomputed eff_len and bos ---
    L_eff = L
    if IS_VARLEN:
        L_eff = tl.load(eff_lens + pid_b).to(tl.int32)
        if pid_l * BLOCK_L >= L_eff:
            return
        bos = tl.load(bos_arr + pid_b).to(tl.int64)

    # --- base pointers: bos*token_stride (varlen) or pid_b*batch_stride (batched) ---
    if IS_VARLEN:
        dh_off = bos * s_dh_l
        x0x2_off = bos * s_x0x2_l
        gdx0dx2_off = bos * s_gdx0dx2_l
        gh_lr1_off = bos * s_gh_lr1_l
        lr_off = bos
        out_lr_off = bos
    else:
        dh_off = pid_b * s_dh_b
        x0x2_off = pid_b * s_x0x2_b
        gdx0dx2_off = pid_b * s_gdx0dx2_b
        gh_lr1_off = pid_b * s_gh_lr1_b
        lr_off = pid_b * s_lr_b
        out_lr_off = pid_b * s_out_grad_lr_b

    # tile coordinates
    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    mask_l = offs_l < L_eff
    mask_d = offs_d < D
    mask = mask_d[:, None] & mask_l[None, :]

    # ----- load input tiles (cast to fp32 for math) -----
    # dh [B, D, L] or [T, D]
    dh_ptrs = DH + dh_off + offs_d[:, None] * s_dh_d + offs_l[None, :] * s_dh_l
    dh = tl.load(dh_ptrs, mask=mask, other=0.0).to(tl.float32)

    # x0, x2 from [B, 2D, L] or [T, 2D]
    x0_ptrs = (
        X0X2
        + x0x2_off
        + offs_d[:, None] * s_x0x2_d
        + offs_l[None, :] * s_x0x2_l
    )
    x2_ptrs = (
        X0X2
        + x0x2_off
        + (offs_d[:, None] + D) * s_x0x2_d
        + offs_l[None, :] * s_x0x2_l
    )
    x0 = tl.load(x0_ptrs, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x2_ptrs, mask=mask, other=0.0).to(tl.float32)

    # grad_dx0, grad_dx2 from [B, 2D, L] or [T, 2D]
    gdx0_ptrs = (
        GDX0DX2
        + gdx0dx2_off
        + offs_d[:, None] * s_gdx0dx2_d
        + offs_l[None, :] * s_gdx0dx2_l
    )
    gdx2_ptrs = (
        GDX0DX2
        + gdx0dx2_off
        + (offs_d[:, None] + D) * s_gdx0dx2_d
        + offs_l[None, :] * s_gdx0dx2_l
    )
    grad_dx0 = tl.load(gdx0_ptrs, mask=mask, other=0.0).to(tl.float32)
    grad_dx2 = tl.load(gdx2_ptrs, mask=mask, other=0.0).to(tl.float32)

    # grad_hidden_lr1 [B, D, L] or [T, D]
    gh_ptrs = (
        GH_LR1
        + gh_lr1_off
        + offs_d[:, None] * s_gh_lr1_d
        + offs_l[None, :] * s_gh_lr1_l
    )
    grad_hidden_lr1 = tl.load(gh_ptrs, mask=mask, other=0.0).to(tl.float32)

    # lr0/lr1/lr2 vectors [B, L] or [T] -> broadcast on D
    lr0_ptrs = LR0 + lr_off + offs_l * s_lr_l
    lr1_ptrs = LR1 + lr_off + offs_l * s_lr_l
    lr2_ptrs = LR2 + lr_off + offs_l * s_lr_l
    lr0 = tl.load(lr0_ptrs, mask=mask_l, other=0.0).to(tl.float32)[
        None, :
    ]  # [1, BLOCK_L]
    lr1 = tl.load(lr1_ptrs, mask=mask_l, other=0.0).to(tl.float32)[None, :]
    lr2 = tl.load(lr2_ptrs, mask=mask_l, other=0.0).to(tl.float32)[None, :]

    # ----- common terms -----
    sigma = tl.sigmoid(x0)  # [D,L]
    one_m_sigma = 1.0 - sigma
    silu_x0 = x0 * sigma
    silu_bp_multiplier = sigma * (1.0 + x0 * one_m_sigma)  # sigma * (1 + x0*(1-sigma))

    # ----- outputs that are fully pointwise -----
    # grad_dh
    grad_dh = grad_dx0 * lr0 * x2 * silu_bp_multiplier + grad_dx2 * lr2 * silu_x0

    # grad_x2
    grad_x2 = grad_dx0 * lr0 * dh * silu_bp_multiplier + grad_hidden_lr1 * lr1 * silu_x0

    # grad_x0 (naive + correction)
    # grad_sigma = grad_dx0 * lr0 * dh * x2 * (1 + x0 - 2 * sigma * x0)
    grad_sigma = grad_dx0 * lr0 * dh * x2 * (1.0 + x0 - 2.0 * sigma * x0)
    grad_x0_naive = (
        grad_dx2 * lr2 * dh + grad_hidden_lr1 * lr1 * x2
    ) * silu_bp_multiplier + grad_dx0 * lr0 * dh * x2 * sigma * one_m_sigma
    grad_x0 = grad_x0_naive + grad_sigma * sigma * one_m_sigma

    # ---- reductions for grad_lr* (sum over D of same L) ----
    # Partial tile contributions
    part_lr0 = grad_dx0 * dh * x2 * silu_bp_multiplier
    part_lr1 = grad_hidden_lr1 * x2 * silu_x0
    part_lr2 = grad_dx2 * dh * silu_x0

    # Reduce along D-axis of the tile -> [BLOCK_L]
    red_lr0 = tl.sum(part_lr0, axis=0)  # [Ltile]
    red_lr1 = tl.sum(part_lr1, axis=0)
    red_lr2 = tl.sum(part_lr2, axis=0)

    # Atomic add into [B, L] or [T]
    out_lr0_ptrs = OUT_GRAD_LR0 + out_lr_off + offs_l * s_out_grad_lr_l
    out_lr1_ptrs = OUT_GRAD_LR1 + out_lr_off + offs_l * s_out_grad_lr_l
    out_lr2_ptrs = OUT_GRAD_LR2 + out_lr_off + offs_l * s_out_grad_lr_l
    tl.atomic_add(out_lr0_ptrs, red_lr0, mask=mask_l)
    tl.atomic_add(out_lr1_ptrs, red_lr1, mask=mask_l)
    tl.atomic_add(out_lr2_ptrs, red_lr2, mask=mask_l)

    # ----- first-order backward extras -----
    dx2 = silu_x0 * dh
    dx0 = dh * x2 * silu_bp_multiplier
    dx0 = dx0 * lr0
    dx2 = dx2 * lr2
    hidden_lr1 = lr1 * x2 * silu_x0

    # ----- stores -----
    # grad_dh [B, D, L] or [T, D]
    out_gdh_ptrs = (
        OUT_GRAD_DH
        + dh_off
        + offs_d[:, None] * s_dh_d
        + offs_l[None, :] * s_dh_l
    )
    tl.store(out_gdh_ptrs, tl.cast(grad_dh, tl.bfloat16), mask=mask)

    # grad_x0_x2 [B, 2D, L] or [T, 2D]
    out_gx0_ptrs = (
        OUT_GRAD_X0X2
        + x0x2_off
        + offs_d[:, None] * s_x0x2_d
        + offs_l[None, :] * s_x0x2_l
    )
    out_gx2_ptrs = (
        OUT_GRAD_X0X2
        + x0x2_off
        + (offs_d[:, None] + D) * s_x0x2_d
        + offs_l[None, :] * s_x0x2_l
    )
    tl.store(out_gx0_ptrs, tl.cast(grad_x0, tl.bfloat16), mask=mask)
    tl.store(out_gx2_ptrs, tl.cast(grad_x2, tl.bfloat16), mask=mask)

    # dx0_x2 [B, 2D, L] or [T, 2D]
    out_dx0_ptrs = (
        OUT_DX0X2
        + x0x2_off
        + offs_d[:, None] * s_x0x2_d
        + offs_l[None, :] * s_x0x2_l
    )
    out_dx2_ptrs = (
        OUT_DX0X2
        + x0x2_off
        + (offs_d[:, None] + D) * s_x0x2_d
        + offs_l[None, :] * s_x0x2_l
    )
    tl.store(out_dx0_ptrs, tl.cast(dx0, tl.bfloat16), mask=mask)
    tl.store(out_dx2_ptrs, tl.cast(dx2, tl.bfloat16), mask=mask)

    # hidden_lr1 [B, D, L] or [T, D]
    out_h_ptrs = (
        OUT_HIDDEN_LR1
        + gh_lr1_off
        + offs_d[:, None] * s_gh_lr1_d
        + offs_l[None, :] * s_gh_lr1_l
    )
    tl.store(out_h_ptrs, tl.cast(hidden_lr1, tl.bfloat16), mask=mask)


def triton_swiglu_bwd_bwd_fused_cat_inp_out(
    dh: torch.Tensor,  # [B, D, L]      bf16/fp16/fp32
    x0_x2: torch.Tensor,  # [B, 2D, L]     bf16/fp16/fp32
    lr0: torch.Tensor,  # [B, L]         fp32 preferred
    lr1: torch.Tensor,  # [B, L]
    lr2: torch.Tensor,  # [B, L]
    grad_dx0_dx2: torch.Tensor,  # [B, 2D, L]     bf16/fp16/fp32
    grad_hidden_lr1: torch.Tensor,  # [B, D, L]      bf16/fp16/fp32
    # BLOCK_D: int = 128,
    # BLOCK_L: int = 128,
    # num_warps: int = 4,
    # num_stages: int = 2,
):
    """
    Fused Triton kernel for the 'ref_pytorch_swiglu_bwd_bwd_fused_cat_inp_out' computation.
    - Computes all pointwise outputs in one pass.
    - Reduces grad_lr{0,1,2} over D using atomic adds (fast and simple).
    """

    B, D, L = dh.shape
    assert x0_x2.shape == (B, 2 * D, L)
    assert grad_dx0_dx2.shape == (B, 2 * D, L)
    assert grad_hidden_lr1.shape == (B, D, L)
    assert lr0.shape == (B, L) and lr1.shape == (B, L) and lr2.shape == (B, L)

    # assume input is contigous to compute the strides
    s_dh_b, s_dh_d, s_dh_l = D * L, L, 1
    s_x0x2_b, s_x0x2_d, s_x0x2_l = 2 * D * L, L, 1
    s_gdx0dx2_b, s_gdx0dx2_d, s_gdx0dx2_l = 2 * D * L, L, 1
    s_gh_lr1_b, s_gh_lr1_d, s_gh_lr1_l = D * L, L, 1
    s_lr_b, s_lr_l = L, 1
    s_out_grad_lr_b, s_out_grad_lr_l = L, 1

    device = dh.device
    x_dtype = x0_x2.dtype
    lr_dtype = lr0.dtype  # keep grad_lr* in this dtype (usually fp32)

    # Allocate outputs as fp32 for the kernel; cast back to x_dtype where required
    grad_dh = torch.empty_like(dh, dtype=x_dtype, device=device)
    grad_x0_x2 = torch.empty_like(x0_x2, dtype=x_dtype, device=device)
    dx0_x2 = torch.empty_like(x0_x2, dtype=x_dtype, device=device)
    hidden_lr1 = torch.empty_like(dh, dtype=x_dtype, device=device)

    grad_lr0 = torch.zeros((B, L), dtype=torch.float32, device=device)
    grad_lr1 = torch.zeros((B, L), dtype=torch.float32, device=device)
    grad_lr2 = torch.zeros((B, L), dtype=torch.float32, device=device)

    # grid = (B, triton.cdiv(L, BLOCK_L), triton.cdiv(D, BLOCK_D))

    def grid(meta):
        return (
            B,
            triton.cdiv(L, meta["BLOCK_L"]),
            triton.cdiv(D, meta["BLOCK_D"]),
        )

    _dummy = torch.empty(0, dtype=torch.int32, device=device)

    _swiglu_bwd_bwd_fused_kernel[grid](
        # inputs
        dh,
        x0_x2,
        lr0,
        lr1,
        lr2,
        grad_dx0_dx2,
        grad_hidden_lr1,
        # outputs
        grad_dh,
        grad_x0_x2,
        grad_lr0,
        grad_lr1,
        grad_lr2,
        dx0_x2,
        hidden_lr1,
        _dummy,
        _dummy,
        # sizes
        B,
        D,
        L,
        # strides
        s_dh_b,
        s_dh_d,
        s_dh_l,
        s_x0x2_b,
        s_x0x2_d,
        s_x0x2_l,
        s_lr_b,
        s_lr_l,
        s_gdx0dx2_b,
        s_gdx0dx2_d,
        s_gdx0dx2_l,
        s_gh_lr1_b,
        s_gh_lr1_d,
        s_gh_lr1_l,
        s_out_grad_lr_b,
        s_out_grad_lr_l,
        IS_VARLEN=False,
    )

    # Cast the grad_lr* to fp32
    return (
        grad_dh,
        grad_x0_x2,
        grad_lr0.to(lr_dtype),
        grad_lr1.to(lr_dtype),
        grad_lr2.to(lr_dtype),
        dx0_x2,
        hidden_lr1,
    )


def triton_swiglu_bwd_bwd_fused_cat_inp_out_varlen(
    dh: torch.Tensor,            # [T, D]
    x0_x2: torch.Tensor,         # [T, 2D]
    lr0: torch.Tensor,           # [T]
    lr1: torch.Tensor,           # [T]
    lr2: torch.Tensor,           # [T]
    grad_dx0_dx2: torch.Tensor,  # [T, 2D]
    grad_hidden_lr1: torch.Tensor,  # [T, D]
    cu_seqlens: torch.Tensor,    # [G+1]
    chunk_size: int = 0,
    chunk_idx: int = 0,
    eff_lens: torch.Tensor = None,
    bos_arr: torch.Tensor = None,
    max_sl: int = 0,
):
    """Varlen variant. All 2D tensors are [T, feat] packed, lr/grad_lr are [T]."""
    try:
        from utils import compute_varlen_args
    except ImportError:
        from .utils import compute_varlen_args

    T, D = dh.shape
    G = cu_seqlens.shape[0] - 1

    if eff_lens is None:
        eff_lens, bos_arr, max_sl = compute_varlen_args(cu_seqlens, chunk_size, chunk_idx)

    device = dh.device
    x_dtype = x0_x2.dtype
    lr_dtype = lr0.dtype

    grad_dh = torch.empty_like(dh, dtype=x_dtype)
    grad_x0_x2 = torch.empty_like(x0_x2, dtype=x_dtype)
    dx0_x2 = torch.empty_like(x0_x2, dtype=x_dtype)
    hidden_lr1 = torch.empty_like(dh, dtype=x_dtype)

    grad_lr0 = torch.zeros(T, dtype=torch.float32, device=device)
    grad_lr1 = torch.zeros(T, dtype=torch.float32, device=device)
    grad_lr2 = torch.zeros(T, dtype=torch.float32, device=device)

    # For [T, D]: s_b=0, s_d=stride(1), s_l=stride(0)
    s_dh_l, s_dh_d = dh.stride()
    s_x0x2_l, s_x0x2_d = x0_x2.stride()
    s_gdx0dx2_l, s_gdx0dx2_d = grad_dx0_dx2.stride()
    s_gh_lr1_l, s_gh_lr1_d = grad_hidden_lr1.stride()

    def grid(meta):
        return (G, triton.cdiv(max_sl, meta["BLOCK_L"]), triton.cdiv(D, meta["BLOCK_D"]))

    _swiglu_bwd_bwd_fused_kernel[grid](
        dh, x0_x2, lr0, lr1, lr2, grad_dx0_dx2, grad_hidden_lr1,
        grad_dh, grad_x0_x2, grad_lr0, grad_lr1, grad_lr2, dx0_x2, hidden_lr1,
        eff_lens, bos_arr,
        G, D, max_sl,
        0, s_dh_d, s_dh_l,                # s_dh_b=0, s_dh_d, s_dh_l
        0, s_x0x2_d, s_x0x2_l,            # s_x0x2_b=0
        0, 1,                              # s_lr_b=0, s_lr_l=1
        0, s_gdx0dx2_d, s_gdx0dx2_l,      # s_gdx0dx2_b=0
        0, s_gh_lr1_d, s_gh_lr1_l,         # s_gh_lr1_b=0
        0, 1,                              # s_out_grad_lr_b=0, s_out_grad_lr_l=1
        IS_VARLEN=True,
    )

    return (
        grad_dh,
        grad_x0_x2,
        grad_lr0.to(lr_dtype),
        grad_lr1.to(lr_dtype),
        grad_lr2.to(lr_dtype),
        dx0_x2,
        hidden_lr1,
    )


@torch.compile()
def ref_pytorch_swiglu_bwd_bwd_fused_cat_inp_out(
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


def make_inputs(B, D, L, lr_dtype=torch.float32):
    dh = torch.randn(B, D, L, device="cuda", dtype=torch.bfloat16)
    x0_x2 = torch.randn(B, 2 * D, L, device="cuda", dtype=torch.bfloat16)
    lr0 = torch.randn(B, L, device="cuda", dtype=lr_dtype)
    lr1 = torch.randn(B, L, device="cuda", dtype=lr_dtype)
    lr2 = torch.randn(B, L, device="cuda", dtype=lr_dtype)
    grad_dx0_dx2 = torch.randn(B, 2 * D, L, device="cuda", dtype=torch.bfloat16)
    grad_hidden_lr1 = torch.randn(B, D, L, device="cuda", dtype=torch.bfloat16)
    return dh, x0_x2, lr0, lr1, lr2, grad_dx0_dx2, grad_hidden_lr1


def check_correctness():
    from .benchmark import report_error

    B = 4
    D = 2048
    L = 1024
    lr_dtype = torch.float32

    # lr_dtype = torch.bfloat16
    inps = make_inputs(B, D, L, lr_dtype)
    grad_dh, grad_x0_x2, grad_lr0, grad_lr1, grad_lr2, dx0_x2, hidden_lr1_out = (
        triton_swiglu_bwd_bwd_fused_cat_inp_out(*inps)
    )

    fp32_inps = [_.to(torch.float32) for _ in inps]
    (
        grad_dh_ref,
        grad_x0_x2_ref,
        grad_lr0_ref,
        grad_lr1_ref,
        grad_lr2_ref,
        dx0_x2_ref,
        hidden_lr1_out_ref,
    ) = ref_pytorch_swiglu_bwd_bwd_fused_cat_inp_out(*fp32_inps)
    report_error(grad_dh, grad_dh_ref, "grad_dh")
    report_error(grad_x0_x2, grad_x0_x2_ref, "grad_x0_x2")
    report_error(grad_lr0, grad_lr0_ref, "grad_lr0")
    report_error(grad_lr1, grad_lr1_ref, "grad_lr1")
    report_error(grad_lr2, grad_lr2_ref, "grad_lr2")
    report_error(dx0_x2, dx0_x2_ref, "dx0_x2")
    report_error(hidden_lr1_out, hidden_lr1_out_ref, "hidden_lr1_out")
    print("=> Done testing correctness")


def _reference_varlen_padded(dh, x0_x2, lr0, lr1, lr2, grad_dx0_dx2, grad_hidden_lr1, cu_seqlens):
    """Pad -> batched Triton kernel -> unpack."""
    from grouped_gemm import _pack_to_padded, _padded_to_pack

    G = cu_seqlens.shape[0] - 1
    T = dh.shape[0]
    D = dh.shape[1]
    max_len = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

    # Pad [T, feat] -> [G, max_len, feat], then transpose to [G, feat, max_len] for batched kernel
    dh_p = _pack_to_padded(dh, cu_seqlens, max_len).transpose(1, 2)
    x0_x2_p = _pack_to_padded(x0_x2, cu_seqlens, max_len).transpose(1, 2)
    gdx_p = _pack_to_padded(grad_dx0_dx2, cu_seqlens, max_len).transpose(1, 2)
    gh_p = _pack_to_padded(grad_hidden_lr1, cu_seqlens, max_len).transpose(1, 2)
    lr0_p = _pack_to_padded(lr0.unsqueeze(1), cu_seqlens, max_len).squeeze(2)
    lr1_p = _pack_to_padded(lr1.unsqueeze(1), cu_seqlens, max_len).squeeze(2)
    lr2_p = _pack_to_padded(lr2.unsqueeze(1), cu_seqlens, max_len).squeeze(2)

    out = triton_swiglu_bwd_bwd_fused_cat_inp_out(
        dh_p.contiguous(), x0_x2_p.contiguous(),
        lr0_p.contiguous(), lr1_p.contiguous(), lr2_p.contiguous(),
        gdx_p.contiguous(), gh_p.contiguous(),
    )
    # Unpack: [G, feat, max_len] -> transpose -> [G, max_len, feat] -> packed [T, feat]
    # out[0]=grad_dh [G,D,L], out[1]=grad_x0x2 [G,2D,L], out[5]=dx0x2 [G,2D,L], out[6]=hidden_lr1 [G,D,L]
    # out[2,3,4]=grad_lr [G,L]
    return (
        _padded_to_pack(out[0].transpose(1, 2), cu_seqlens, T),
        _padded_to_pack(out[1].transpose(1, 2), cu_seqlens, T),
        _padded_to_pack(out[2].unsqueeze(2), cu_seqlens, T).squeeze(1),
        _padded_to_pack(out[3].unsqueeze(2), cu_seqlens, T).squeeze(1),
        _padded_to_pack(out[4].unsqueeze(2), cu_seqlens, T).squeeze(1),
        _padded_to_pack(out[5].transpose(1, 2), cu_seqlens, T),
        _padded_to_pack(out[6].transpose(1, 2), cu_seqlens, T),
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
    D = 512
    cu_seqlens = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(doc_lens), 0).tolist()),
        dtype=torch.int32, device=device,
    )
    T = cu_seqlens[-1].item()

    dh = torch.randn(T, D, device=device, dtype=torch.bfloat16)
    x0_x2 = torch.randn(T, 2 * D, device=device, dtype=torch.bfloat16)
    lr0 = torch.randn(T, device=device, dtype=torch.float32) * 0.01
    lr1 = torch.randn(T, device=device, dtype=torch.float32) * 0.01
    lr2 = torch.randn(T, device=device, dtype=torch.float32) * 0.01
    gdx = torch.randn(T, 2 * D, device=device, dtype=torch.bfloat16)
    gh = torch.randn(T, D, device=device, dtype=torch.bfloat16)

    print(f"Config: {num_docs=}, {doc_lens=}, {D=}, {T=}")
    print()

    print("=" * 60)
    print("Varlen correctness vs padded batched Triton kernel")
    print("=" * 60)

    triton_args = (dh, x0_x2, lr0, lr1, lr2, gdx, gh, cu_seqlens)

    test_correctness(
        lambda *a: triton_swiglu_bwd_bwd_fused_cat_inp_out_varlen(*a),
        lambda *a: _reference_varlen_padded(*a),
        triton_args, triton_args,
        debug=args.debug, atol=1e-2, rtol=1e-2,
    )

    print()
    print("=" * 60)
    print("Varlen benchmark vs padded batched kernel")
    print("=" * 60)

    from grouped_gemm import _pack_to_padded
    max_len = max(doc_lens)
    # Pre-pad for fair benchmark: [T, feat] -> [G, feat, max_len] (batched kernel layout)
    dh_p = _pack_to_padded(dh, cu_seqlens, max_len).transpose(1, 2).contiguous()
    x0_x2_p = _pack_to_padded(x0_x2, cu_seqlens, max_len).transpose(1, 2).contiguous()
    gdx_p = _pack_to_padded(gdx, cu_seqlens, max_len).transpose(1, 2).contiguous()
    gh_p = _pack_to_padded(gh, cu_seqlens, max_len).transpose(1, 2).contiguous()
    lr0_p = _pack_to_padded(lr0.unsqueeze(1), cu_seqlens, max_len).squeeze(2).contiguous()
    lr1_p = _pack_to_padded(lr1.unsqueeze(1), cu_seqlens, max_len).squeeze(2).contiguous()
    lr2_p = _pack_to_padded(lr2.unsqueeze(1), cu_seqlens, max_len).squeeze(2).contiguous()

    ref_bench_args = (dh_p, x0_x2_p, lr0_p, lr1_p, lr2_p, gdx_p, gh_p)

    try:
        from utils import compute_varlen_args
    except ImportError:
        from .utils import compute_varlen_args
    eff_lens, bos_arr, max_sl_val = compute_varlen_args(cu_seqlens)

    benchmark(
        lambda *a: triton_swiglu_bwd_bwd_fused_cat_inp_out_varlen(
            *a, eff_lens=eff_lens, bos_arr=bos_arr, max_sl=max_sl_val,
        ),
        lambda *a: triton_swiglu_bwd_bwd_fused_cat_inp_out(*a),
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
        ref_out = triton_swiglu_bwd_bwd_fused_cat_inp_out_varlen(
            dh[idx], x0_x2[idx], lr0[idx], lr1[idx], lr2[idx], gdx[idx], gh[idx], chunk_cu,
        )
        # Test: call chunk kernel on full buffer
        test_out = triton_swiglu_bwd_bwd_fused_cat_inp_out_varlen(
            dh, x0_x2, lr0, lr1, lr2, gdx, gh, cu_seqlens,
            chunk_size=chunk_size, chunk_idx=chunk_index,
        )
        # 7 outputs: grad_dh[T,D], grad_x0_x2[T,2D], grad_lr0[T], grad_lr1[T], grad_lr2[T], dx0_x2[T,2D], hidden_lr1[T,D]
        names = ["grad_dh", "grad_x0_x2", "grad_lr0", "grad_lr1", "grad_lr2", "dx0_x2", "hidden_lr1"]
        diffs = []
        for name, t, r in zip(names, test_out, ref_out):
            d = (t[idx] - r).abs().max().item()
            diffs.append(d)
        max_d = max(diffs)
        print(f"  chunk_idx={chunk_index}: max_diff={max_d:.2e}, n_tokens={len(idx)}")
        # grad_lr outputs use atomic_add — ordering differs between chunk and full, so allow tolerance
        assert max_d < 1e-4, f"chunk_idx={chunk_index} mismatch! diffs={dict(zip(names, diffs))}"

    print("✓ PASS")
