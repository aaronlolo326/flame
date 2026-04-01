import torch
import torch.nn.functional as F
from typing import Optional

from .ttt_operation import silu_backprop, zeropower_via_newtonschulz5


def topk_sparse_softmax(logits: torch.Tensor, topk: int) -> torch.Tensor:
    """Softmax over the largest ``topk`` logits and mask the rest to zero."""
    if logits.shape[-1] <= topk:
        return logits.softmax(dim=-1)

    values, indices = torch.topk(logits, k=topk, dim=-1)
    masked_logits = torch.full_like(logits, torch.finfo(logits.dtype).min)
    masked_logits.scatter_(-1, indices, values)
    return masked_logits.softmax(dim=-1)


def _apply_slot_fast_weights(
    w0s: torch.Tensor,
    w1s: torch.Tensor,
    w2s: torch.Tensor,
    q: torch.Tensor,
) -> torch.Tensor:
    compute_dtype = q.dtype
    q_t = q.transpose(1, 2).unsqueeze(1)
    h = torch.matmul(w2s.to(compute_dtype), q_t)
    gate = F.silu(torch.matmul(w0s.to(compute_dtype), q_t), inplace=True)
    return torch.matmul(w1s.to(compute_dtype), gate * h).sum(dim=1)


def _apply_muon_over_slots(dw: torch.Tensor) -> torch.Tensor:
    batch_size, num_slots, rows, cols = dw.shape
    dw = dw.reshape(batch_size * num_slots, rows, cols)
    dw = zeropower_via_newtonschulz5(dw)
    return dw.reshape(batch_size, num_slots, rows, cols)


@torch.compile()
@torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16)
def block_causal_srlact_swiglu(
    w0s: torch.Tensor,
    w1s: torch.Tensor,
    w2s: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    router_gates: torch.Tensor,
    chunk_size: int = 2048,
    use_muon: bool = False,
    momentum: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    batch_size, num_slots, _, _ = w0s.shape

    w0_norms = w0s.norm(dim=3, keepdim=True)
    w1_norms = w1s.norm(dim=3, keepdim=True)
    w2_norms = w2s.norm(dim=3, keepdim=True)

    if momentum is not None:
        dw0_momentum = torch.zeros_like(w0s)
        dw1_momentum = torch.zeros_like(w1s)
        dw2_momentum = torch.zeros_like(w2s)

    q_t = q.transpose(1, 2)
    v_t = v.transpose(1, 2)
    output = torch.zeros_like(v_t)

    seq_len = k.shape[1]
    e_index = 0
    num_update_chunks = int(router_gates.shape[1])

    for chunk_idx in range(num_update_chunks):
        s_index = chunk_idx * chunk_size
        e_index = s_index + chunk_size

        ki = k[:, s_index:e_index, :]
        vi = v_t[:, :, s_index:e_index]
        qi = q_t[:, :, s_index:e_index]

        lr0i = lr0[:, s_index:e_index, :]
        lr1i = lr1[:, s_index:e_index, :]
        lr2i = lr2[:, s_index:e_index, :]
        gate_i = router_gates[:, chunk_idx, :].to(w0s.dtype)

        output[:, :, s_index:e_index] = _apply_slot_fast_weights(
            w0s,
            w1s,
            w2s,
            qi.transpose(1, 2),
        )

        ki_t = ki.transpose(1, 2).unsqueeze(1)
        vi = vi.unsqueeze(1)
        ki = ki.unsqueeze(1)
        lr0i = lr0i.unsqueeze(1)
        lr1i = lr1i.unsqueeze(1)
        lr2i = lr2i.unsqueeze(1)

        w0_compute = w0s.to(q.dtype)
        w1_compute = w1s.to(q.dtype)
        w2_compute = w2s.to(q.dtype)

        gate_before_act = torch.matmul(w0_compute, ki_t)
        hidden_before_mul = torch.matmul(w2_compute, ki_t)
        hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul

        dhidden = torch.matmul(w1_compute.transpose(-1, -2), vi)
        dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
        dgate = dhidden * hidden_before_mul
        dgate_before_act = silu_backprop(dgate, gate_before_act)

        dw1 = torch.matmul(
            vi,
            (hidden.transpose(-1, -2) * lr1i).type_as(vi),
        ).to(w1s.dtype)
        dw0 = torch.matmul(
            dgate_before_act,
            (ki * lr0i).type_as(dgate_before_act),
        ).to(w0s.dtype)
        dw2 = torch.matmul(
            dhidden_before_mul,
            (ki * lr2i).type_as(dhidden_before_mul),
        ).to(w2s.dtype)

        if momentum is not None:
            m_i = momentum[:, s_index:e_index, :].mean(dim=1, keepdim=True).to(w0s.dtype)
            m_i = m_i.unsqueeze(1)
            dw0 = dw0 + dw0_momentum * m_i
            dw1 = dw1 + dw1_momentum * m_i
            dw2 = dw2 + dw2_momentum * m_i
            dw0_momentum = dw0
            dw1_momentum = dw1
            dw2_momentum = dw2

        if use_muon:
            dw1 = _apply_muon_over_slots(dw1)
            dw0 = _apply_muon_over_slots(dw0)
            dw2 = _apply_muon_over_slots(dw2)

        gate_i = gate_i.view(batch_size, num_slots, 1, 1)
        w0s = w0s + gate_i * dw0
        w1s = w1s + gate_i * dw1
        w2s = w2s + gate_i * dw2

        w0s = w0s / (w0s.norm(dim=3, keepdim=True) + 1e-5) * w0_norms
        w1s = w1s / (w1s.norm(dim=3, keepdim=True) + 1e-5) * w1_norms
        w2s = w2s / (w2s.norm(dim=3, keepdim=True) + 1e-5) * w2_norms

    s_index = e_index
    e_index = seq_len
    qi = q_t[:, :, s_index:e_index]
    if e_index > s_index:
        output[:, :, s_index:e_index] = _apply_slot_fast_weights(
            w0s,
            w1s,
            w2s,
            qi.transpose(1, 2),
        )

    return output.transpose(1, 2)


@torch.compile()
@torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16)
def prenorm_block_causal_srlact_swiglu(
    w0s: torch.Tensor,
    w1s: torch.Tensor,
    w2s: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    router_gates: torch.Tensor,
    chunk_size: int = 2048,
    use_muon: bool = False,
    momentum: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    batch_size, num_slots, _, _ = w0s.shape
    w0s_main = w0s
    w1s_main = w1s
    w2s_main = w2s

    w0_norms = w0s.norm(dim=3, keepdim=True)
    w1_norms = w1s.norm(dim=3, keepdim=True)
    w2_norms = w2s.norm(dim=3, keepdim=True)

    if momentum is not None:
        dw0_momentum = torch.zeros_like(w0s)
        dw1_momentum = torch.zeros_like(w1s)
        dw2_momentum = torch.zeros_like(w2s)

    q_t = q.transpose(1, 2)
    v_t = v.transpose(1, 2)
    output = torch.zeros_like(v_t)

    seq_len = k.shape[1]
    e_index = 0
    num_update_chunks = int(router_gates.shape[1])

    for chunk_idx in range(num_update_chunks):
        s_index = chunk_idx * chunk_size
        e_index = s_index + chunk_size

        ki = k[:, s_index:e_index, :]
        vi = v_t[:, :, s_index:e_index]
        qi = q_t[:, :, s_index:e_index]

        lr0i = lr0[:, s_index:e_index, :]
        lr1i = lr1[:, s_index:e_index, :]
        lr2i = lr2[:, s_index:e_index, :]
        gate_i = router_gates[:, chunk_idx, :].to(w0s.dtype)

        output[:, :, s_index:e_index] = _apply_slot_fast_weights(
            w0s,
            w1s,
            w2s,
            qi.transpose(1, 2),
        )

        ki_t = ki.transpose(1, 2).unsqueeze(1)
        vi = vi.unsqueeze(1)
        ki = ki.unsqueeze(1)
        lr0i = lr0i.unsqueeze(1)
        lr1i = lr1i.unsqueeze(1)
        lr2i = lr2i.unsqueeze(1)

        w0_compute = w0s.to(q.dtype)
        w1_compute = w1s.to(q.dtype)
        w2_compute = w2s.to(q.dtype)

        gate_before_act = torch.matmul(w0_compute, ki_t)
        hidden_before_mul = torch.matmul(w2_compute, ki_t)
        hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul

        dhidden = torch.matmul(w1_compute.transpose(-1, -2), vi)
        dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
        dgate = dhidden * hidden_before_mul
        dgate_before_act = silu_backprop(dgate, gate_before_act)

        dw1 = torch.matmul(
            vi,
            (hidden.transpose(-1, -2) * lr1i).type_as(vi),
        ).to(w1s_main.dtype)
        dw0 = torch.matmul(
            dgate_before_act,
            (ki * lr0i).type_as(dgate_before_act),
        ).to(w0s_main.dtype)
        dw2 = torch.matmul(
            dhidden_before_mul,
            (ki * lr2i).type_as(dhidden_before_mul),
        ).to(w2s_main.dtype)

        if momentum is not None:
            m_i = momentum[:, s_index:e_index, :].mean(dim=1, keepdim=True).to(w0s.dtype)
            m_i = m_i.unsqueeze(1)
            dw0 = dw0 + dw0_momentum * m_i
            dw1 = dw1 + dw1_momentum * m_i
            dw2 = dw2 + dw2_momentum * m_i
            dw0_momentum = dw0
            dw1_momentum = dw1
            dw2_momentum = dw2

        if use_muon:
            dw1 = _apply_muon_over_slots(dw1)
            dw0 = _apply_muon_over_slots(dw0)
            dw2 = _apply_muon_over_slots(dw2)

        gate_i = gate_i.view(batch_size, num_slots, 1, 1)
        w0s_main = w0s_main + gate_i * dw0
        w1s_main = w1s_main + gate_i * dw1
        w2s_main = w2s_main + gate_i * dw2

        w0s = w0s_main / (w0s_main.norm(dim=3, keepdim=True) + 1e-5) * w0_norms
        w1s = w1s_main / (w1s_main.norm(dim=3, keepdim=True) + 1e-5) * w1_norms
        w2s = w2s_main / (w2s_main.norm(dim=3, keepdim=True) + 1e-5) * w2_norms

    s_index = e_index
    e_index = seq_len
    qi = q_t[:, :, s_index:e_index]
    if e_index > s_index:
        output[:, :, s_index:e_index] = _apply_slot_fast_weights(
            w0s,
            w1s,
            w2s,
            qi.transpose(1, 2),
        )

    return output.transpose(1, 2)
