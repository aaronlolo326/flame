import torch.nn.functional as F
import torch
from typing import Any, Dict, Optional


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


def _clone_optional_tensor(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Clone an optional tensor while preserving ``None`` for missing state."""
    return None if x is None else x.clone()


def _empty_pending_like(x: torch.Tensor) -> torch.Tensor:
    """Create an empty ``[batch, 0, dim]`` buffer that matches an existing tensor."""
    return x[:, :0, :].clone()


def _clear_pending_buffers(state: Dict[str, Any]) -> Dict[str, Any]:
    """Reset the buffered partial chunk after it has been consumed by an update."""
    state["pending_k"] = None
    state["pending_v"] = None
    state["pending_lr0"] = None
    state["pending_lr1"] = None
    state["pending_lr2"] = None
    state["pending_momentum"] = None
    state["pending_len"] = 0
    return state


def init_lact_decode_state(
    fw_w0: torch.Tensor,
    fw_w1: torch.Tensor,
    fw_w2: torch.Tensor,
    *,
    ttt_prenorm: bool,
    use_momentum: bool,
) -> Dict[str, Any]:
    """Initialize the persistent decode state from the slow-weight snapshot."""
    state: Dict[str, Any] = {
        "fw_w0": fw_w0.clone(),
        "fw_w1": fw_w1.clone(),
        "fw_w2": fw_w2.clone(),
        "pending_k": None,
        "pending_v": None,
        "pending_lr0": None,
        "pending_lr1": None,
        "pending_lr2": None,
        "pending_momentum": None,
        "pending_len": 0,
    }
    if ttt_prenorm:
        state["fw_w0_main"] = fw_w0.clone()
        state["fw_w1_main"] = fw_w1.clone()
        state["fw_w2_main"] = fw_w2.clone()
    if use_momentum:
        state["dw0_momentum"] = torch.zeros_like(fw_w0)
        state["dw1_momentum"] = torch.zeros_like(fw_w1)
        state["dw2_momentum"] = torch.zeros_like(fw_w2)
    else:
        state["dw0_momentum"] = None
        state["dw1_momentum"] = None
        state["dw2_momentum"] = None
    return state


def pending_chunk_ready(state: Dict[str, Any], chunk_size: int) -> bool:
    """Return whether the buffered tokens are enough to close one update chunk."""
    return int(state["pending_len"]) >= int(chunk_size)


def append_token_to_pending(
    state: Dict[str, Any],
    k: torch.Tensor,
    v: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    momentum: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Append newly decoded tokens into the pending chunk buffer."""
    if state["pending_k"] is None:
        state["pending_k"] = k.clone()
        state["pending_v"] = v.clone()
        state["pending_lr0"] = lr0.clone()
        state["pending_lr1"] = lr1.clone()
        state["pending_lr2"] = lr2.clone()
        state["pending_momentum"] = _clone_optional_tensor(momentum)
    else:
        state["pending_k"] = torch.cat([state["pending_k"], k], dim=1)
        state["pending_v"] = torch.cat([state["pending_v"], v], dim=1)
        state["pending_lr0"] = torch.cat([state["pending_lr0"], lr0], dim=1)
        state["pending_lr1"] = torch.cat([state["pending_lr1"], lr1], dim=1)
        state["pending_lr2"] = torch.cat([state["pending_lr2"], lr2], dim=1)
        if momentum is not None:
            if state["pending_momentum"] is None:
                state["pending_momentum"] = momentum.clone()
            else:
                state["pending_momentum"] = torch.cat(
                    [state["pending_momentum"], momentum], dim=1
                )
    state["pending_len"] = int(state["pending_k"].shape[1])
    return state


def _apply_fast_weights(
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    q: torch.Tensor,
) -> torch.Tensor:
    """Run the fast SwiGLU network on queries without mutating the cached state."""
    w0_compute = w0.to(q.dtype)
    w1_compute = w1.to(q.dtype)
    w2_compute = w2.to(q.dtype)
    q = q.transpose(1, 2)
    h = torch.bmm(w2_compute, q)
    gate = F.silu(torch.bmm(w0_compute, q), inplace=True)
    return torch.bmm(w1_compute, gate * h).transpose(1, 2)


def lact_apply_only_postnorm(
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    q: torch.Tensor,
) -> torch.Tensor:
    """Apply-only step for post-norm TTT; identical math, different caller semantics."""
    return _apply_fast_weights(w0, w1, w2, q)


def lact_apply_only_prenorm(
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    q: torch.Tensor,
) -> torch.Tensor:
    """Apply-only step for pre-norm TTT; identical math, different caller semantics."""
    return _apply_fast_weights(w0, w1, w2, q)


def lact_update_chunk_postnorm(
    state: Dict[str, Any],
    k_chunk: torch.Tensor,
    v_chunk: torch.Tensor,
    lr0_chunk: torch.Tensor,
    lr1_chunk: torch.Tensor,
    lr2_chunk: torch.Tensor,
    momentum_chunk: Optional[torch.Tensor] = None,
    *,
    use_muon: bool = False,
) -> Dict[str, Any]:
    """Update one closed chunk of post-norm fast weights and write the next state in place."""
    w0 = state["fw_w0"]
    w1 = state["fw_w1"]
    w2 = state["fw_w2"]
    compute_dtype = k_chunk.dtype
    state_dtype = w0.dtype
    w0_compute = w0.to(compute_dtype)
    w1_compute = w1.to(compute_dtype)
    w2_compute = w2.to(compute_dtype)

    w0_norm = w0.norm(dim=2, keepdim=True)
    w1_norm = w1.norm(dim=2, keepdim=True)
    w2_norm = w2.norm(dim=2, keepdim=True)

    vi = v_chunk.transpose(1, 2)
    ki_t = k_chunk.transpose(1, 2)

    gate_before_act = torch.bmm(w0_compute, ki_t)
    hidden_before_mul = torch.bmm(w2_compute, ki_t)
    hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul
    dhidden = torch.bmm(w1_compute.transpose(1, 2), vi)
    dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
    dgate = dhidden * hidden_before_mul
    dgate_before_act = silu_backprop(dgate, gate_before_act)

    dw1 = torch.bmm(vi, (hidden.transpose(1, 2) * lr1_chunk).type_as(vi)).to(state_dtype)
    dw0 = torch.bmm(
        dgate_before_act,
        (k_chunk * lr0_chunk).type_as(dgate_before_act),
    ).to(state_dtype)
    dw2 = torch.bmm(
        dhidden_before_mul,
        (k_chunk * lr2_chunk).type_as(dhidden_before_mul),
    ).to(state_dtype)

    if momentum_chunk is not None:
        m_i = momentum_chunk.mean(dim=1, keepdim=True).to(state_dtype)
        dw0 = dw0 + state["dw0_momentum"] * m_i
        dw1 = dw1 + state["dw1_momentum"] * m_i
        dw2 = dw2 + state["dw2_momentum"] * m_i
        state["dw0_momentum"] = dw0
        state["dw1_momentum"] = dw1
        state["dw2_momentum"] = dw2

    if use_muon:
        dw1 = zeropower_via_newtonschulz5(dw1)
        dw0 = zeropower_via_newtonschulz5(dw0)
        dw2 = zeropower_via_newtonschulz5(dw2)

    w0 = w0 + dw0
    w1 = w1 + dw1
    w2 = w2 + dw2

    state["fw_w0"] = w0 / (w0.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
    state["fw_w1"] = w1 / (w1.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
    state["fw_w2"] = w2 / (w2.norm(dim=2, keepdim=True) + 1e-5) * w2_norm
    return state


def lact_update_chunk_prenorm(
    state: Dict[str, Any],
    k_chunk: torch.Tensor,
    v_chunk: torch.Tensor,
    lr0_chunk: torch.Tensor,
    lr1_chunk: torch.Tensor,
    lr2_chunk: torch.Tensor,
    momentum_chunk: Optional[torch.Tensor] = None,
    *,
    use_muon: bool = False,
) -> Dict[str, Any]:
    """Update one closed chunk of pre-norm fast weights, keeping both main and normalized states."""
    w0 = state["fw_w0"]
    w1 = state["fw_w1"]
    w2 = state["fw_w2"]
    w0_main = state["fw_w0_main"]
    w1_main = state["fw_w1_main"]
    w2_main = state["fw_w2_main"]
    compute_dtype = k_chunk.dtype
    state_dtype = w0_main.dtype
    w0_compute = w0.to(compute_dtype)
    w1_compute = w1.to(compute_dtype)
    w2_compute = w2.to(compute_dtype)

    w0_norm = w0.norm(dim=2, keepdim=True)
    w1_norm = w1.norm(dim=2, keepdim=True)
    w2_norm = w2.norm(dim=2, keepdim=True)

    vi = v_chunk.transpose(1, 2)
    ki_t = k_chunk.transpose(1, 2)

    gate_before_act = torch.bmm(w0_compute, ki_t)
    hidden_before_mul = torch.bmm(w2_compute, ki_t)
    hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul
    dhidden = torch.bmm(w1_compute.transpose(1, 2), vi)
    dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
    dgate = dhidden * hidden_before_mul
    dgate_before_act = silu_backprop(dgate, gate_before_act)

    dw1 = torch.bmm(vi, (hidden.transpose(1, 2) * lr1_chunk).type_as(vi)).to(state_dtype)
    dw0 = torch.bmm(
        dgate_before_act,
        (k_chunk * lr0_chunk).type_as(dgate_before_act),
    ).to(state_dtype)
    dw2 = torch.bmm(
        dhidden_before_mul,
        (k_chunk * lr2_chunk).type_as(dhidden_before_mul),
    ).to(state_dtype)

    if momentum_chunk is not None:
        m_i = momentum_chunk.mean(dim=1, keepdim=True).to(state_dtype)
        dw0 = dw0 + state["dw0_momentum"] * m_i
        dw1 = dw1 + state["dw1_momentum"] * m_i
        dw2 = dw2 + state["dw2_momentum"] * m_i
        state["dw0_momentum"] = dw0
        state["dw1_momentum"] = dw1
        state["dw2_momentum"] = dw2

    if use_muon:
        dw1 = zeropower_via_newtonschulz5(dw1)
        dw0 = zeropower_via_newtonschulz5(dw0)
        dw2 = zeropower_via_newtonschulz5(dw2)

    w0_main = w0_main + dw0
    w1_main = w1_main + dw1
    w2_main = w2_main + dw2

    state["fw_w0_main"] = w0_main
    state["fw_w1_main"] = w1_main
    state["fw_w2_main"] = w2_main
    state["fw_w0"] = w0_main / (w0_main.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
    state["fw_w1"] = w1_main / (w1_main.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
    state["fw_w2"] = w2_main / (w2_main.norm(dim=2, keepdim=True) + 1e-5) * w2_norm
    return state


def _take_pending_chunk(
    state: Dict[str, Any],
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Pop one full chunk from the pending buffer and keep any leftover tail buffered."""
    if not pending_chunk_ready(state, chunk_size):
        raise ValueError("Pending chunk is not ready yet")
    k_chunk = state["pending_k"][:, :chunk_size, :]
    v_chunk = state["pending_v"][:, :chunk_size, :]
    lr0_chunk = state["pending_lr0"][:, :chunk_size, :]
    lr1_chunk = state["pending_lr1"][:, :chunk_size, :]
    lr2_chunk = state["pending_lr2"][:, :chunk_size, :]
    momentum_chunk = None
    if state["pending_momentum"] is not None:
        momentum_chunk = state["pending_momentum"][:, :chunk_size, :]

    remaining = state["pending_len"] - chunk_size
    if remaining > 0:
        state["pending_k"] = state["pending_k"][:, chunk_size:, :]
        state["pending_v"] = state["pending_v"][:, chunk_size:, :]
        state["pending_lr0"] = state["pending_lr0"][:, chunk_size:, :]
        state["pending_lr1"] = state["pending_lr1"][:, chunk_size:, :]
        state["pending_lr2"] = state["pending_lr2"][:, chunk_size:, :]
        if state["pending_momentum"] is not None:
            state["pending_momentum"] = state["pending_momentum"][:, chunk_size:, :]
        state["pending_len"] = remaining
    else:
        _clear_pending_buffers(state)

    return k_chunk, v_chunk, lr0_chunk, lr1_chunk, lr2_chunk, momentum_chunk


def run_lact_sequence_with_state(
    state: Dict[str, Any],
    q_seq: torch.Tensor,
    k_seq: torch.Tensor,
    v_seq: torch.Tensor,
    lr0_seq: torch.Tensor,
    lr1_seq: torch.Tensor,
    lr2_seq: torch.Tensor,
    momentum_seq: Optional[torch.Tensor] = None,
    *,
    chunk_size: int,
    ttt_prenorm: bool,
    use_muon: bool = False,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    """Advance the decode state over a sequence using apply-then-update chunk semantics."""
    if q_seq.shape[1] == 0:
        return q_seq, state

    apply_fn = lact_apply_only_prenorm if ttt_prenorm else lact_apply_only_postnorm
    update_fn = lact_update_chunk_prenorm if ttt_prenorm else lact_update_chunk_postnorm

    outputs = []
    seq_len = q_seq.shape[1]
    pos = 0

    if state["pending_len"] > 0:
        take = min(chunk_size - state["pending_len"], seq_len)
        outputs.append(apply_fn(state["fw_w0"], state["fw_w1"], state["fw_w2"], q_seq[:, :take, :]))
        append_token_to_pending(
            state,
            k_seq[:, :take, :],
            v_seq[:, :take, :],
            lr0_seq[:, :take, :],
            lr1_seq[:, :take, :],
            lr2_seq[:, :take, :],
            None if momentum_seq is None else momentum_seq[:, :take, :],
        )
        if pending_chunk_ready(state, chunk_size):
            chunk = _take_pending_chunk(state, chunk_size)
            print("--------------------decode update weights--------------------")
            update_fn(state, *chunk, use_muon=use_muon)
        pos = take

    while pos + chunk_size <= seq_len:
        q_chunk = q_seq[:, pos:pos + chunk_size, :]
        outputs.append(apply_fn(state["fw_w0"], state["fw_w1"], state["fw_w2"], q_chunk))
        print("--------------------prefill update weights--------------------")
        update_fn(
            state,
            k_seq[:, pos:pos + chunk_size, :],
            v_seq[:, pos:pos + chunk_size, :],
            lr0_seq[:, pos:pos + chunk_size, :],
            lr1_seq[:, pos:pos + chunk_size, :],
            lr2_seq[:, pos:pos + chunk_size, :],
            None if momentum_seq is None else momentum_seq[:, pos:pos + chunk_size, :],
            use_muon=use_muon,
        )
        pos += chunk_size

    if pos < seq_len:
        outputs.append(apply_fn(state["fw_w0"], state["fw_w1"], state["fw_w2"], q_seq[:, pos:, :]))
        append_token_to_pending(
            state,
            k_seq[:, pos:, :],
            v_seq[:, pos:, :],
            lr0_seq[:, pos:, :],
            lr1_seq[:, pos:, :],
            lr2_seq[:, pos:, :],
            None if momentum_seq is None else momentum_seq[:, pos:, :],
        )

    return torch.cat(outputs, dim=1), state
