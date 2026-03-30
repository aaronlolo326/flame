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
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    q: torch.Tensor,
) -> torch.Tensor:
    compute_dtype = q.dtype
    q = q.transpose(1, 2)
    h = torch.bmm(w2.to(compute_dtype), q)
    gate = F.silu(torch.bmm(w0.to(compute_dtype), q), inplace=True)
    return torch.bmm(w1.to(compute_dtype), gate * h)


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
    d_out = w1s.shape[2]
    compute_dtype = q.dtype

    w0s = w0s.clone()
    w1s = w1s.clone()
    w2s = w2s.clone()

    w0s_norm = w0s.norm(dim=3, keepdim=True)
    w1s_norm = w1s.norm(dim=3, keepdim=True)
    w2s_norm = w2s.norm(dim=3, keepdim=True)

    if momentum is not None:
        dw0_momentum = [torch.zeros_like(w0s[:, slot]) for slot in range(num_slots)]
        dw1_momentum = [torch.zeros_like(w1s[:, slot]) for slot in range(num_slots)]
        dw2_momentum = [torch.zeros_like(w2s[:, slot]) for slot in range(num_slots)]

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

        chunk_output = torch.zeros(
            batch_size,
            d_out,
            e_index - s_index,
            dtype=compute_dtype,
            device=q.device,
        )
        for slot in range(num_slots):
            chunk_output = chunk_output + _apply_slot_fast_weights(
                w0s[:, slot], w1s[:, slot], w2s[:, slot], qi.transpose(1, 2)
            )
        output[:, :, s_index:e_index] = chunk_output

        ki_t = ki.transpose(1, 2)
        if momentum is not None:
            m_i = momentum[:, s_index:e_index, :].mean(dim=1, keepdim=True).to(w0s.dtype)

        for slot in range(num_slots):
            w0_slot = w0s[:, slot]
            w1_slot = w1s[:, slot]
            w2_slot = w2s[:, slot]

            w0_compute = w0_slot.to(compute_dtype)
            w1_compute = w1_slot.to(compute_dtype)
            w2_compute = w2_slot.to(compute_dtype)

            gate_before_act = torch.bmm(w0_compute, ki_t)
            hidden_before_mul = torch.bmm(w2_compute, ki_t)
            hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul

            dhidden = torch.bmm(w1_compute.transpose(1, 2), vi)
            dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
            dgate = dhidden * hidden_before_mul
            dgate_before_act = silu_backprop(dgate, gate_before_act)

            dw1 = torch.bmm(vi, (hidden.transpose(1, 2) * lr1i).type_as(vi)).to(w1_slot.dtype)
            dw0 = torch.bmm(
                dgate_before_act,
                (ki * lr0i).type_as(dgate_before_act),
            ).to(w0_slot.dtype)
            dw2 = torch.bmm(
                dhidden_before_mul,
                (ki * lr2i).type_as(dhidden_before_mul),
            ).to(w2_slot.dtype)

            if momentum is not None:
                dw0 = dw0 + dw0_momentum[slot] * m_i
                dw1 = dw1 + dw1_momentum[slot] * m_i
                dw2 = dw2 + dw2_momentum[slot] * m_i
                dw0_momentum[slot] = dw0
                dw1_momentum[slot] = dw1
                dw2_momentum[slot] = dw2

            if use_muon:
                dw1 = zeropower_via_newtonschulz5(dw1)
                dw0 = zeropower_via_newtonschulz5(dw0)
                dw2 = zeropower_via_newtonschulz5(dw2)

            gate_slot = gate_i[:, slot].view(batch_size, 1, 1)
            w0_slot = w0_slot + gate_slot * dw0
            w1_slot = w1_slot + gate_slot * dw1
            w2_slot = w2_slot + gate_slot * dw2

            w0s[:, slot] = w0_slot / (w0_slot.norm(dim=2, keepdim=True) + 1e-5) * w0s_norm[:, slot]
            w1s[:, slot] = w1_slot / (w1_slot.norm(dim=2, keepdim=True) + 1e-5) * w1s_norm[:, slot]
            w2s[:, slot] = w2_slot / (w2_slot.norm(dim=2, keepdim=True) + 1e-5) * w2s_norm[:, slot]

    s_index = e_index
    e_index = seq_len
    qi = q_t[:, :, s_index:e_index]
    if e_index > s_index:
        chunk_output = torch.zeros(
            batch_size,
            d_out,
            e_index - s_index,
            dtype=compute_dtype,
            device=q.device,
        )
        for slot in range(num_slots):
            chunk_output = chunk_output + _apply_slot_fast_weights(
                w0s[:, slot], w1s[:, slot], w2s[:, slot], qi.transpose(1, 2)
            )
        output[:, :, s_index:e_index] = chunk_output

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
    d_out = w1s.shape[2]
    compute_dtype = q.dtype

    w0s = w0s.clone()
    w1s = w1s.clone()
    w2s = w2s.clone()
    w0s_main = w0s.clone()
    w1s_main = w1s.clone()
    w2s_main = w2s.clone()

    w0s_norm = w0s.norm(dim=3, keepdim=True)
    w1s_norm = w1s.norm(dim=3, keepdim=True)
    w2s_norm = w2s.norm(dim=3, keepdim=True)

    if momentum is not None:
        dw0_momentum = [torch.zeros_like(w0s[:, slot]) for slot in range(num_slots)]
        dw1_momentum = [torch.zeros_like(w1s[:, slot]) for slot in range(num_slots)]
        dw2_momentum = [torch.zeros_like(w2s[:, slot]) for slot in range(num_slots)]

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

        chunk_output = torch.zeros(
            batch_size,
            d_out,
            e_index - s_index,
            dtype=compute_dtype,
            device=q.device,
        )
        for slot in range(num_slots):
            chunk_output = chunk_output + _apply_slot_fast_weights(
                w0s[:, slot], w1s[:, slot], w2s[:, slot], qi.transpose(1, 2)
            )
        output[:, :, s_index:e_index] = chunk_output

        ki_t = ki.transpose(1, 2)
        if momentum is not None:
            m_i = momentum[:, s_index:e_index, :].mean(dim=1, keepdim=True).to(w0s.dtype)

        for slot in range(num_slots):
            w0_compute = w0s[:, slot].to(compute_dtype)
            w1_compute = w1s[:, slot].to(compute_dtype)
            w2_compute = w2s[:, slot].to(compute_dtype)

            gate_before_act = torch.bmm(w0_compute, ki_t)
            hidden_before_mul = torch.bmm(w2_compute, ki_t)
            hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul

            dhidden = torch.bmm(w1_compute.transpose(1, 2), vi)
            dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
            dgate = dhidden * hidden_before_mul
            dgate_before_act = silu_backprop(dgate, gate_before_act)

            dw1 = torch.bmm(vi, (hidden.transpose(1, 2) * lr1i).type_as(vi)).to(w1s_main[:, slot].dtype)
            dw0 = torch.bmm(
                dgate_before_act,
                (ki * lr0i).type_as(dgate_before_act),
            ).to(w0s_main[:, slot].dtype)
            dw2 = torch.bmm(
                dhidden_before_mul,
                (ki * lr2i).type_as(dhidden_before_mul),
            ).to(w2s_main[:, slot].dtype)

            if momentum is not None:
                dw0 = dw0 + dw0_momentum[slot] * m_i
                dw1 = dw1 + dw1_momentum[slot] * m_i
                dw2 = dw2 + dw2_momentum[slot] * m_i
                dw0_momentum[slot] = dw0
                dw1_momentum[slot] = dw1
                dw2_momentum[slot] = dw2

            if use_muon:
                dw1 = zeropower_via_newtonschulz5(dw1)
                dw0 = zeropower_via_newtonschulz5(dw0)
                dw2 = zeropower_via_newtonschulz5(dw2)

            gate_slot = gate_i[:, slot].view(batch_size, 1, 1)
            w0s_main[:, slot] = w0s_main[:, slot] + gate_slot * dw0
            w1s_main[:, slot] = w1s_main[:, slot] + gate_slot * dw1
            w2s_main[:, slot] = w2s_main[:, slot] + gate_slot * dw2

            w0s[:, slot] = (
                w0s_main[:, slot] / (w0s_main[:, slot].norm(dim=2, keepdim=True) + 1e-5)
            ) * w0s_norm[:, slot]
            w1s[:, slot] = (
                w1s_main[:, slot] / (w1s_main[:, slot].norm(dim=2, keepdim=True) + 1e-5)
            ) * w1s_norm[:, slot]
            w2s[:, slot] = (
                w2s_main[:, slot] / (w2s_main[:, slot].norm(dim=2, keepdim=True) + 1e-5)
            ) * w2s_norm[:, slot]

    s_index = e_index
    e_index = seq_len
    qi = q_t[:, :, s_index:e_index]
    if e_index > s_index:
        chunk_output = torch.zeros(
            batch_size,
            d_out,
            e_index - s_index,
            dtype=compute_dtype,
            device=q.device,
        )
        for slot in range(num_slots):
            chunk_output = chunk_output + _apply_slot_fast_weights(
                w0s[:, slot], w1s[:, slot], w2s[:, slot], qi.transpose(1, 2)
            )
        output[:, :, s_index:e_index] = chunk_output

    return output.transpose(1, 2)
