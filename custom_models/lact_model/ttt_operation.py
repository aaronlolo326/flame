import torch
import torch.nn.functional as F
import os


_LOG_VARLEN_SEG_STATS = os.getenv("LACT_LOG_VARLEN_SEG_STATS", "0") == "1"
_LOG_VARLEN_SEG_STATS_EVERY = max(int(os.getenv("LACT_LOG_VARLEN_SEG_STATS_EVERY", "1")), 1)
_varlen_seg_stats_calls = 0


@torch.compile()
def silu_backprop(dy: torch.Tensor, x: torch.Tensor):
    """
    Args:
        dy: [b, d, l], gradient of the outer loss wrt the y
        x: [b, d, l], input of the silu activation
    outs:
        dx: [b, d, l], gradient of the outer loss wrt the x
        dx = dy * sigma * (1 + x * (1 - sigma))
    """
    sigma = torch.sigmoid(x)
    dx = dy * sigma * (1 + x * (1 - sigma))
    return dx


@torch.compile()
def l2_norm(x: torch.Tensor):
    """
    x: [b, l, d]
    """
    x_type = x.dtype
    ret = x / (x.norm(dim=-1, keepdim=True) + 1e-5)  # norm will upcast to float32
    return ret.type(x_type)


@torch.compile()
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
        A = X @ X.transpose(1, 2)
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    return X


@torch.compile()
@torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16)
def block_causal_lact_swiglu(
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    chunk_size: int = 2048,  # test-time training chunk size
    use_muon: bool = False,
    momentum: torch.Tensor = None,  # [b, s, 1]
):
    """
    Block causal LaCT with SwiGLU fast weight function.
        Apply then Update => Shifted Block Causal LaCT
    w0, w1, w2 are the fast weights. f(x) =  w1 @ (silu(w0 @ x) * (w2 @ x))

    About precision:
        w0, w1, w2 are mostly likely fp32.
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

    # adding detach here sometimes improves stability.
    w0_norm = w0.norm(dim=2, keepdim=True)
    w1_norm = w1.norm(dim=2, keepdim=True)
    w2_norm = w2.norm(dim=2, keepdim=True)

    if momentum is not None:
        dw1_momentum = torch.zeros_like(w1)
        dw0_momentum = torch.zeros_like(w0)
        dw2_momentum = torch.zeros_like(w2)

    q = q.transpose(1, 2)  # [b, dk, l]
    v = v.transpose(1, 2)

    output = torch.zeros_like(v)

    e_index = 0
    seq_len = k.shape[1]
    for i in range(0, seq_len - chunk_size, chunk_size):
        s_index = i
        e_index = s_index + chunk_size

        # [b, l, dk]
        ki = k[:, s_index:e_index, :]  # bf16
        # [b, dv, l]
        vi = v[:, :, s_index:e_index]  # bf16
        # [b, dh, l]
        qi = q[:, :, s_index:e_index]
        # [b, l, d/1] fp32
        lr1i = lr1[:, s_index:e_index, :]  # [b, l, d/1] fp32
        lr2i = lr2[:, s_index:e_index, :]  # [b, l, d/1] fp32
        lr0i = lr0[:, s_index:e_index, :]  # [b, l, d/1] fp32

        # use previous w0 and w1 to get the final output
        # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
        h = torch.bmm(w2, qi)
        gate = F.silu(torch.bmm(w0, qi), inplace=True)
        # [b, dv, dh] @ [b, dh, l] -> [b, dv, l] -> [b, l, dv]
        output[:, :, s_index:e_index] = torch.bmm(w1, gate * h)

        # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
        gate_before_act = torch.bmm(w0, ki.transpose(1, 2))
        hidden_before_mul = torch.bmm(w2, ki.transpose(1, 2))

        hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul

        # [b, dh, dv] @ [b, dv, l] -> [b, dh, l]
        dhidden = torch.bmm(w1.transpose(1, 2), vi)

        dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)

        dgate = dhidden * hidden_before_mul
        dgate_before_act = silu_backprop(dgate, gate_before_act)

        # [b, d_2, l] @ [b, l, d_1] -> [b, d_2, d_1]
        # in bmm two mat is fp32, but the result is bf16.
        # it's better to cast the mat to bf16 before bmm.
        # [b, dv, l] @ [b, l, dh] -> [b, dv, dh]
        # it's better to cast the mat to bf16 before bmm.
        dw1 = torch.bmm(vi, (hidden.transpose(1, 2) * lr1i).type_as(vi))  # [b, d, d]
        # [b, dh, l] @ [b, l, dk] -> [b, dh, dk]
        dw0 = torch.bmm(dgate_before_act, (ki * lr0i).type_as(dgate_before_act))
        dw2 = torch.bmm(dhidden_before_mul, (ki * lr2i).type_as(dhidden_before_mul))

        if momentum is not None:
            m_i = momentum[:, s_index:e_index, :]
            m_i = m_i.mean(dim=1, keepdim=True)

            dw0 = dw0 + dw0_momentum * m_i
            dw1 = dw1 + dw1_momentum * m_i
            dw2 = dw2 + dw2_momentum * m_i
            dw0_momentum = dw0
            dw1_momentum = dw1
            dw2_momentum = dw2

        if use_muon:
            dw1 = zeropower_via_newtonschulz5(dw1)
            dw0 = zeropower_via_newtonschulz5(dw0)
            dw2 = zeropower_via_newtonschulz5(dw2)
            # legacy code for different global lr for muon. Conclusion: 1.0 is good
            # if muon_w0_lr is not None:
            #     # lr is fp32 (after softplus)
            #     # in future version, we can cast it before input. TODO
            #     dw1 = (dw1 * muon_w1_lr).type_as(w1)
            #     dw0 = (dw0 * muon_w0_lr).type_as(w0)
            #     dw2 = (dw2 * muon_w2_lr).type_as(w2)

        w1 = w1 + dw1
        w0 = w0 + dw0
        w2 = w2 + dw2

        # Do channel-wise l2 norm.  conceptually like post-norm.
        w0 = w0 / (w0.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
        w1 = w1 / (w1.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
        w2 = w2 / (w2.norm(dim=2, keepdim=True) + 1e-5) * w2_norm

    # for the last chunk, don't update the fast weights, directly apply the fast weights to the query.
    s_index = e_index
    e_index = seq_len

    qi = q[:, :, s_index:e_index]
    # use the last w0 and w1 to get the final output
    # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
    h = torch.bmm(w2, qi)
    gate = F.silu(torch.bmm(w0, qi), inplace=True)
    # [b, dv, dh] @ [b, dh, l] -> [b, dv, l] -> [b, l, dv]
    output[:, :, s_index:e_index] = torch.bmm(w1, gate * h)

    return output.transpose(1, 2)


@torch.compile()
@torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16)
def prenorm_block_causal_lact_swiglu(
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    chunk_size: int = 2048,  # test-time training chunk size
    use_muon: bool = False,
    momentum: torch.Tensor = None,  # [b, s, 1]
):
    """
    Block causal LaCT with SwiGLU fast weight function.
        Apply then Update => Shifted Block Causal LaCT
    w0, w1, w2 are the fast weights. f(x) =  w1 @ (silu(w0 @ x) * (w2 @ x))

    About precision:
        w0, w1, w2 are mostly likely fp32.
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

    # adding detach here sometimes improves stability.
    w0_norm = w0.norm(dim=2, keepdim=True)
    w1_norm = w1.norm(dim=2, keepdim=True)
    w2_norm = w2.norm(dim=2, keepdim=True)

    w0_main, w1_main, w2_main = w0, w1, w2

    if momentum is not None:
        dw1_momentum = torch.zeros_like(w1)
        dw0_momentum = torch.zeros_like(w0)
        dw2_momentum = torch.zeros_like(w2)

    q = q.transpose(1, 2)  # [b, dk, l]
    v = v.transpose(1, 2)

    output = torch.zeros_like(v)

    e_index = 0
    seq_len = k.shape[1]
    for i in range(0, seq_len - chunk_size, chunk_size):
        s_index = i
        e_index = s_index + chunk_size

        # [b, l, dk]
        ki = k[:, s_index:e_index, :]  # bf16
        # [b, dv, l]
        vi = v[:, :, s_index:e_index]  # bf16
        # [b, dh, l]
        qi = q[:, :, s_index:e_index]
        # [b, l, d/1] fp32
        lr1i = lr1[:, s_index:e_index, :]  # [b, l, d/1] fp32
        lr2i = lr2[:, s_index:e_index, :]  # [b, l, d/1] fp32
        lr0i = lr0[:, s_index:e_index, :]  # [b, l, d/1] fp32

        # use previous w0 and w1 to get the final output
        # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
        h = torch.bmm(w2, qi)
        gate = F.silu(torch.bmm(w0, qi), inplace=True)
        # [b, dv, dh] @ [b, dh, l] -> [b, dv, l] -> [b, l, dv]
        output[:, :, s_index:e_index] = torch.bmm(w1, gate * h)

        # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
        gate_before_act = torch.bmm(w0, ki.transpose(1, 2))
        hidden_before_mul = torch.bmm(w2, ki.transpose(1, 2))

        hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul

        # [b, dh, dv] @ [b, dv, l] -> [b, dh, l]
        dhidden = torch.bmm(w1.transpose(1, 2), vi)

        dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)

        dgate = dhidden * hidden_before_mul
        dgate_before_act = silu_backprop(dgate, gate_before_act)

        # [b, d_2, l] @ [b, l, d_1] -> [b, d_2, d_1]
        # in bmm two mat is fp32, but the result is bf16.
        # it's better to cast the mat to bf16 before bmm.
        # [b, dv, l] @ [b, l, dh] -> [b, dv, dh]
        # it's better to cast the mat to bf16 before bmm.
        dw1 = torch.bmm(vi, (hidden.transpose(1, 2) * lr1i).type_as(vi))  # [b, d, d]
        # [b, dh, l] @ [b, l, dk] -> [b, dh, dk]
        dw0 = torch.bmm(dgate_before_act, (ki * lr0i).type_as(dgate_before_act))
        dw2 = torch.bmm(dhidden_before_mul, (ki * lr2i).type_as(dhidden_before_mul))

        if momentum is not None:
            m_i = momentum[:, s_index:e_index, :]
            m_i = m_i.mean(dim=1, keepdim=True)

            dw0 = dw0 + dw0_momentum * m_i
            dw1 = dw1 + dw1_momentum * m_i
            dw2 = dw2 + dw2_momentum * m_i
            dw0_momentum = dw0
            dw1_momentum = dw1
            dw2_momentum = dw2

        if use_muon:
            dw1 = zeropower_via_newtonschulz5(dw1)
            dw0 = zeropower_via_newtonschulz5(dw0)
            dw2 = zeropower_via_newtonschulz5(dw2)
            # legacy code for different global lr for muon. Conclusion: 1.0 is good
            # if muon_w0_lr is not None:
            #     # lr is fp32 (after softplus)
            #     # in future version, we can cast it before input. TODO
            #     dw1 = (dw1 * muon_w1_lr).type_as(w1)
            #     dw0 = (dw0 * muon_w0_lr).type_as(w0)
            #     dw2 = (dw2 * muon_w2_lr).type_as(w2)

        w1_main = w1_main + dw1
        w0_main = w0_main + dw0
        w2_main = w2_main + dw2

        # Do channel-wise l2 norm.  conceptually like post-norm.
        w0 = w0_main / (w0_main.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
        w1 = w1_main / (w1_main.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
        w2 = w2_main / (w2_main.norm(dim=2, keepdim=True) + 1e-5) * w2_norm

    # for the last chunk, don't update the fast weights, directly apply the fast weights to the query.
    s_index = e_index
    e_index = seq_len

    qi = q[:, :, s_index:e_index]
    # use the last w0 and w1 to get the final output
    # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
    h = torch.bmm(w2, qi)
    gate = F.silu(torch.bmm(w0, qi), inplace=True)
    # [b, dv, dh] @ [b, dh, l] -> [b, dv, l] -> [b, l, dv]
    output[:, :, s_index:e_index] = torch.bmm(w1, gate * h)

    return output.transpose(1, 2)

######################### NAIVE IMPLEMENTATION #########################

# @torch._dynamo.disable()
# def prenorm_block_causal_lact_swiglu_varlen(
#     w0: torch.Tensor,
#     w1: torch.Tensor,
#     w2: torch.Tensor,
#     q: torch.Tensor,          # [bh, total_len, d]
#     k: torch.Tensor,          # [bh, total_len, d]
#     v: torch.Tensor,          # [bh, total_len, d]
#     lr0: torch.Tensor,        # [bh, total_len, lr_dim]
#     lr1: torch.Tensor,        # [bh, total_len, lr_dim]
#     lr2: torch.Tensor,        # [bh, total_len, lr_dim]
#     cu_seqlens: torch.Tensor, # [n_seq + 1]
#     chunk_size: int = 2048,
#     use_muon: bool = False,
#     momentum: torch.Tensor = None,  # [bh, total_len, 1] or None
# ):
#     """
#     Segment-aware wrapper for packed varlen training.

#     Semantics:
#       - Each segment [cu_seqlens[i], cu_seqlens[i+1]) is isolated.
#       - Fast weights / momentum are reset per segment.
#       - If seg_len > chunk_size: do real TTT update path.
#       - If seg_len <= chunk_size: do apply-only path, and attach zero-anchor
#         to lr/momentum so DDP does not see unused parameters.
#     """
#     if cu_seqlens.dim() != 1:
#         raise ValueError(f"cu_seqlens must be 1D, got {tuple(cu_seqlens.shape)}")

#     total_len = q.shape[1]
#     if int(cu_seqlens[-1].item()) != total_len:
#         raise ValueError(
#             f"cu_seqlens[-1] ({int(cu_seqlens[-1].item())}) != total_len ({total_len})"
#         )

#     out = torch.zeros_like(v)
#     zero_anchor = q.new_zeros((), dtype=torch.float32)
#     num_update_seg = 0
#     num_apply_only_seg = 0
#     num_empty_seg = 0

#     n_seg = cu_seqlens.numel() - 1
#     for seg_idx in range(n_seg):
#         s = int(cu_seqlens[seg_idx].item())
#         e = int(cu_seqlens[seg_idx + 1].item())
#         if e <= s:
#             num_empty_seg += 1
#             continue

#         seg_len = e - s

#         # Reset fast weights per segment
#         seg_w0 = w0.clone()
#         seg_w1 = w1.clone()
#         seg_w2 = w2.clone()

#         if seg_len > chunk_size:
#             num_update_seg += 1
#             # Real update path: lr/momentum are genuinely used
#             seg_out = prenorm_block_causal_lact_swiglu(
#                 w0=seg_w0,
#                 w1=seg_w1,
#                 w2=seg_w2,
#                 q=q[:, s:e, :],
#                 k=k[:, s:e, :],
#                 v=v[:, s:e, :],
#                 lr0=lr0[:, s:e, :],
#                 lr1=lr1[:, s:e, :],
#                 lr2=lr2[:, s:e, :],
#                 chunk_size=chunk_size,
#                 use_muon=use_muon,
#                 momentum=None if momentum is None else momentum[:, s:e, :],
#             )
#         else:
#             num_apply_only_seg += 1
#             # Apply-only path: no update should happen for this segment
#             seg_out = prenorm_block_causal_lact_swiglu_apply_only(
#                 w0=seg_w0,
#                 w1=seg_w1,
#                 w2=seg_w2,
#                 q=q[:, s:e, :],
#             )

#             # DDP-safe zero anchor: keep lr/momentum connected to graph
#             zero_anchor = zero_anchor + lr0[:, s:e, :].float().sum() * 0.0
#             zero_anchor = zero_anchor + lr1[:, s:e, :].float().sum() * 0.0
#             zero_anchor = zero_anchor + lr2[:, s:e, :].float().sum() * 0.0
#             if momentum is not None:
#                 zero_anchor = zero_anchor + momentum[:, s:e, :].float().sum() * 0.0

#         out[:, s:e, :] = seg_out

#     global _varlen_seg_stats_calls
#     _varlen_seg_stats_calls += 1
#     if _LOG_VARLEN_SEG_STATS and _varlen_seg_stats_calls % _LOG_VARLEN_SEG_STATS_EVERY == 0:
#         should_log = True
#         if torch.distributed.is_available() and torch.distributed.is_initialized():
#             should_log = torch.distributed.get_rank() == 0
#         if should_log:
#             cs = cu_seqlens.detach().reshape(-1).to(torch.int64).cpu()
#             cs_list = cs.tolist()
#             if len(cs_list) <= 16:
#                 cs_preview = cs_list
#             else:
#                 cs_preview = cs_list[:8] + ["..."] + cs_list[-8:]
#             print(
#                 "[LaCT varlen seg stats] "
#                 f"call={_varlen_seg_stats_calls} "
#                 f"chunk_size={chunk_size} total_seg={n_seg} "
#                 f"update_seg={num_update_seg} apply_only_seg={num_apply_only_seg} empty_seg={num_empty_seg} "
#                 f"cu_seqlens_n={len(cs_list)} cu_seqlens={cs_preview}"
#             )

#     return out + zero_anchor.to(dtype=out.dtype)

    

# @torch.compile()
# @torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16)
# def prenorm_block_causal_lact_swiglu_apply_only(
#     w0: torch.Tensor,
#     w1: torch.Tensor,
#     w2: torch.Tensor,
#     q: torch.Tensor,   # [bh, s, d]
# ):
#     q_t = q.transpose(1, 2)  # [bh, d, s]

#     h = torch.bmm(w2, q_t)
#     gate = F.silu(torch.bmm(w0, q_t), inplace=True)
#     out = torch.bmm(w1, gate * h)  # [bh, d, s]

#     return out.transpose(1, 2)     # [bh, s, d]

######################### BUCKET WAY IMPLEMENTATION #########################


from collections import defaultdict
from typing import List, Tuple, Optional


# def _round_up_to_multiple(x: int, multiple: int) -> int:
#     if multiple <= 1:
#         return x
#     return ((x + multiple - 1) // multiple) * multiple


def _repeat_fast_weights_per_segment(w: torch.Tensor, n_seg: int) -> torch.Tensor:
    """
    w: [bh, ..., ...]
    return: [n_seg * bh, ..., ...]
    """
    if n_seg == 1:
        return w.clone()
    return w.repeat(n_seg, 1, 1)


def _pack_varlen_segments_dense(
    x: torch.Tensor,  # [bh, total_len, d]
    seg_meta: List[Tuple[int, int, int]],
    pad_len: int,
) -> torch.Tensor:
    """
    Pack a list of segments into a dense tensor.

    Args:
        x: [bh, total_len, d]
        seg_meta: list of (s, e, seg_len)
        pad_len: padded sequence length for this bucket

    Returns:
        packed: [n_seg * bh, pad_len, d]
    """
    bh, _, d = x.shape
    n_seg = len(seg_meta)

    packed = x.new_zeros((n_seg, bh, pad_len, d))
    for i, (s, e, seg_len) in enumerate(seg_meta):
        packed[i, :, :seg_len, :] = x[:, s:e, :]

    return packed.reshape(n_seg * bh, pad_len, d)


def _scatter_varlen_segments_dense(
    out: torch.Tensor,         # [bh, total_len, d]
    packed_out: torch.Tensor,  # [n_seg * bh, pad_len, d]
    seg_meta: List[Tuple[int, int, int]],
) -> None:
    bh, _, d = out.shape
    n_seg = len(seg_meta)
    pad_len = packed_out.shape[1]

    packed_out = packed_out.view(n_seg, bh, pad_len, d)
    for i, (s, e, seg_len) in enumerate(seg_meta):
        out[:, s:e, :] = packed_out[i, :, :seg_len, :]


def _iter_bucket_minibatches(
    seg_meta: List[Tuple[int, int, int]],
    max_segments_per_bucket: Optional[int],
):
    if max_segments_per_bucket is None or max_segments_per_bucket <= 0:
        yield seg_meta
        return
    for i in range(0, len(seg_meta), max_segments_per_bucket):
        yield seg_meta[i : i + max_segments_per_bucket]



def _build_varlen_buckets_pad_full_chunk(
    cu_seqlens: torch.Tensor,
    chunk_size: int,
):
    """
    Bucket segments by:
      - short/apply-only: seg_len <= chunk_size, bucket by pad_len=chunk_size or seg_len
      - long/update: key = n_updates only, and always pad total length to (n_updates + 1) * chunk_size

    Definitions:
        n_updates = ceil(seg_len / chunk_size) - 1
                  = (seg_len - 1) // chunk_size
    """
    cs = cu_seqlens.to(torch.int64).cpu().tolist()

    apply_only_buckets = defaultdict(list)   # key: pad_len
    update_buckets = defaultdict(list)       # key: n_updates

    num_empty_seg = 0
    num_apply_only_seg = 0
    num_update_seg = 0

    for seg_idx in range(len(cs) - 1):
        s, e = cs[seg_idx], cs[seg_idx + 1]
        if e <= s:
            num_empty_seg += 1
            continue

        seg_len = e - s

        if seg_len <= chunk_size:
            num_apply_only_seg += 1
            # 这里你有两个选择：
            # 1. 继续保留真实 seg_len，减少短段浪费
            # 2. 也统一 pad 到 chunk_size，shape 最简单
            #
            # 如果你想“全都最简单”，就直接用下面这一行：
            pad_len = chunk_size

            apply_only_buckets[pad_len].append((s, e, seg_len))
        else:
            num_update_seg += 1
            n_updates = (seg_len - 1) // chunk_size
            update_buckets[n_updates].append((s, e, seg_len))

    stats = {
        "total_seg": len(cs) - 1,
        "update_seg": num_update_seg,
        "apply_only_seg": num_apply_only_seg,
        "empty_seg": num_empty_seg,
        "n_apply_buckets": len(apply_only_buckets),
        "n_update_buckets": len(update_buckets),
        "cu_seqlens_n": len(cs),
        "cu_seqlens_preview": cs if len(cs) <= 16 else cs[:8] + ["..."] + cs[-8:],
    }
    return apply_only_buckets, update_buckets, stats

@torch._dynamo.disable()
def prenorm_block_causal_lact_swiglu_varlen_bucketed(
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    q: torch.Tensor,          # [bh, total_len, d]
    k: torch.Tensor,          # [bh, total_len, d]
    v: torch.Tensor,          # [bh, total_len, d]
    lr0: torch.Tensor,        # [bh, total_len, lr_dim]
    lr1: torch.Tensor,        # [bh, total_len, lr_dim]
    lr2: torch.Tensor,        # [bh, total_len, lr_dim]
    cu_seqlens: torch.Tensor, # [n_seq + 1]
    chunk_size: int = 2048,
    use_muon: bool = False,
    momentum: torch.Tensor = None,  # [bh, total_len, 1] or None
    max_segments_per_bucket: Optional[int] = None,
):
    """
    Bucketed dense-batching version of varlen LaCT.

    Simpler policy:
      - short segments: optionally all pad to chunk_size
      - long segments: final chunk always pad to full chunk_size
      - update bucket key = n_updates only
    """
    if cu_seqlens.dim() != 1:
        raise ValueError(f"cu_seqlens must be 1D, got {tuple(cu_seqlens.shape)}")

    total_len = q.shape[1]
    if int(cu_seqlens[-1].item()) != total_len:
        raise ValueError(
            f"cu_seqlens[-1] ({int(cu_seqlens[-1].item())}) != total_len ({total_len})"
        )

    out = torch.zeros_like(v)
    zero_anchor = q.new_zeros((), dtype=torch.float32)

    apply_only_buckets, update_buckets, stats = _build_varlen_buckets_pad_full_chunk(
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )

    # ----------------------------------------------------------
    # 1) Apply-only buckets
    # ----------------------------------------------------------
    for pad_len, seg_list in apply_only_buckets.items():
        for seg_meta in _iter_bucket_minibatches(seg_list, max_segments_per_bucket):
            packed_q = _pack_varlen_segments_dense(q, seg_meta, pad_len=pad_len)

            packed_w0 = _repeat_fast_weights_per_segment(w0, len(seg_meta))
            packed_w1 = _repeat_fast_weights_per_segment(w1, len(seg_meta))
            packed_w2 = _repeat_fast_weights_per_segment(w2, len(seg_meta))

            packed_out = prenorm_block_causal_lact_swiglu_apply_only(
                w0=packed_w0,
                w1=packed_w1,
                w2=packed_w2,
                q=packed_q,
            )

            _scatter_varlen_segments_dense(out, packed_out, seg_meta)

            for s, e, _ in seg_meta:
                zero_anchor = zero_anchor + lr0[:, s:e, :].float().sum() * 0.0
                zero_anchor = zero_anchor + lr1[:, s:e, :].float().sum() * 0.0
                zero_anchor = zero_anchor + lr2[:, s:e, :].float().sum() * 0.0
                if momentum is not None:
                    zero_anchor = zero_anchor + momentum[:, s:e, :].float().sum() * 0.0

    # ----------------------------------------------------------
    # 2) Update buckets
    # ----------------------------------------------------------
    for n_updates, seg_list in update_buckets.items():
        pad_len = (n_updates + 1) * chunk_size

        for seg_meta in _iter_bucket_minibatches(seg_list, max_segments_per_bucket):
            packed_q = _pack_varlen_segments_dense(q, seg_meta, pad_len=pad_len)
            packed_k = _pack_varlen_segments_dense(k, seg_meta, pad_len=pad_len)
            packed_v = _pack_varlen_segments_dense(v, seg_meta, pad_len=pad_len)
            packed_lr0 = _pack_varlen_segments_dense(lr0, seg_meta, pad_len=pad_len)
            packed_lr1 = _pack_varlen_segments_dense(lr1, seg_meta, pad_len=pad_len)
            packed_lr2 = _pack_varlen_segments_dense(lr2, seg_meta, pad_len=pad_len)
            packed_momentum = (
                None if momentum is None
                else _pack_varlen_segments_dense(momentum, seg_meta, pad_len=pad_len)
            )

            packed_w0 = _repeat_fast_weights_per_segment(w0, len(seg_meta))
            packed_w1 = _repeat_fast_weights_per_segment(w1, len(seg_meta))
            packed_w2 = _repeat_fast_weights_per_segment(w2, len(seg_meta))

            packed_out = prenorm_block_causal_lact_swiglu(
                w0=packed_w0,
                w1=packed_w1,
                w2=packed_w2,
                q=packed_q,
                k=packed_k,
                v=packed_v,
                lr0=packed_lr0,
                lr1=packed_lr1,
                lr2=packed_lr2,
                chunk_size=chunk_size,
                use_muon=use_muon,
                momentum=packed_momentum,
            )

            _scatter_varlen_segments_dense(out, packed_out, seg_meta)

    global _varlen_seg_stats_calls
    _varlen_seg_stats_calls += 1
    if _LOG_VARLEN_SEG_STATS and _varlen_seg_stats_calls % _LOG_VARLEN_SEG_STATS_EVERY == 0:
        should_log = True
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            should_log = torch.distributed.get_rank() == 0
        if should_log:
            print(
                "[LaCT varlen bucketed fullpad stats] "
                f"call={_varlen_seg_stats_calls} "
                f"chunk_size={chunk_size} "
                f"total_seg={stats['total_seg']} "
                f"update_seg={stats['update_seg']} "
                f"apply_only_seg={stats['apply_only_seg']} "
                f"empty_seg={stats['empty_seg']} "
                f"apply_buckets={stats['n_apply_buckets']} "
                f"update_buckets={stats['n_update_buckets']} "
                f"cu_seqlens_n={stats['cu_seqlens_n']} "
                f"cu_seqlens={stats['cu_seqlens_preview']}"
            )

    return out + zero_anchor.to(dtype=out.dtype)

    

@torch.compile()
@torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16)
def prenorm_block_causal_lact_swiglu_apply_only(
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    q: torch.Tensor,   # [bh, s, d]
):
    q_t = q.transpose(1, 2)  # [bh, d, s]

    h = torch.bmm(w2, q_t)
    gate = F.silu(torch.bmm(w0, q_t), inplace=True)
    out = torch.bmm(w1, gate * h)  # [bh, d, s]

    return out.transpose(1, 2)     # [bh, s, d]