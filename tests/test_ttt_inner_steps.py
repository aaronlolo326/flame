import math
import unittest
from unittest import mock

import torch

from custom_models.hybrid_qwen3_lact_model.configuration_hybrid_qwen3_lact import (
    HybridQwen3LaCTConfig,
)
from custom_models.hybrid_qwen3_lact_model.modeling_hybrid_qwen3_lact import (
    HybridQwen3LaCTBranch,
)
from custom_models.lact_model.configuration_lact_swiglu import LaCTSWIGLUConfig
from custom_models.lact_model.modeling_lact import LaCTBlock
from custom_models.lact_model.ttt_operation import (
    _iter_update_segments,
    block_causal_lact_swiglu,
    prenorm_block_causal_lact_swiglu,
    silu_backprop,
)


def _manual_two_step_block(
    w0,
    w1,
    w2,
    q,
    k,
    v,
    lr0,
    lr1,
    lr2,
    *,
    chunk_size,
    ttt_inner_steps,
):
    w0 = w0.clone()
    w1 = w1.clone()
    w2 = w2.clone()
    w0_norm = w0.norm(dim=2, keepdim=True)
    w1_norm = w1.norm(dim=2, keepdim=True)
    w2_norm = w2.norm(dim=2, keepdim=True)

    q_t = q.transpose(1, 2)
    v_t = v.transpose(1, 2)
    output = torch.zeros_like(v_t)

    for s_index, e_index, should_update in _iter_update_segments(
        k.shape[1], chunk_size, chunk_size - 1
    ):
        ki = k[:, s_index:e_index, :]
        vi = v_t[:, :, s_index:e_index]
        qi = q_t[:, :, s_index:e_index]
        lr0i = lr0[:, s_index:e_index, :]
        lr1i = lr1[:, s_index:e_index, :]
        lr2i = lr2[:, s_index:e_index, :]

        h = torch.bmm(w2, qi)
        gate = torch.nn.functional.silu(torch.bmm(w0, qi), inplace=False)
        output[:, :, s_index:e_index] = torch.bmm(w1, gate * h)

        if not should_update:
            continue

        for _ in range(ttt_inner_steps):
            gate_before_act = torch.bmm(w0, ki.transpose(1, 2))
            hidden_before_mul = torch.bmm(w2, ki.transpose(1, 2))
            hidden = torch.nn.functional.silu(gate_before_act, inplace=False) * hidden_before_mul
            dhidden = torch.bmm(w1.transpose(1, 2), vi)
            dhidden_before_mul = dhidden * torch.nn.functional.silu(
                gate_before_act, inplace=False
            )
            dgate = dhidden * hidden_before_mul
            dgate_before_act = silu_backprop(dgate, gate_before_act)

            dw1 = torch.bmm(vi, (hidden.transpose(1, 2) * lr1i).type_as(vi))
            dw0 = torch.bmm(
                dgate_before_act, (ki * lr0i).type_as(dgate_before_act)
            )
            dw2 = torch.bmm(
                dhidden_before_mul, (ki * lr2i).type_as(dhidden_before_mul)
            )

            w1 = w1 + dw1
            w0 = w0 + dw0
            w2 = w2 + dw2

            w0 = w0 / (w0.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
            w1 = w1 / (w1.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
            w2 = w2 / (w2.norm(dim=2, keepdim=True) + 1e-5) * w2_norm

    return output.transpose(1, 2)


class TTTInnerStepsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_lact_block_receives_ttt_inner_steps_from_config(self):
        config = LaCTSWIGLUConfig(
            hidden_size=16,
            num_hidden_layers=1,
            num_attn_heads=4,
            num_lact_heads=4,
            intermediate_size=32,
            lact_chunk_size=2,
            ttt_inner_steps=3,
            use_fused_kernel=False,
            use_momentum=False,
            learnable_ttt_scale=False,
            max_position_embeddings=8,
            vocab_size=32,
        )
        block = LaCTBlock(config, layer_idx=0)
        self.assertEqual(block.attn.ttt_inner_steps, 3)

    def test_block_causal_multiple_inner_steps_are_effective_and_match_manual_reference(
        self,
    ):
        batch_heads = 2
        seq_len = 4
        d_in = 3
        d_h = 5
        chunk_size = 2

        w0 = torch.randn(batch_heads, d_h, d_in, dtype=torch.float32) / math.sqrt(d_in)
        w1 = torch.randn(batch_heads, d_in, d_h, dtype=torch.float32) / math.sqrt(d_h)
        w2 = torch.randn(batch_heads, d_h, d_in, dtype=torch.float32) / math.sqrt(d_in)
        q = torch.randn(batch_heads, seq_len, d_in, dtype=torch.float32)
        k = torch.randn(batch_heads, seq_len, d_in, dtype=torch.float32)
        v = torch.randn(batch_heads, seq_len, d_in, dtype=torch.float32)
        lr0 = torch.full((batch_heads, seq_len, 1), 0.05, dtype=torch.float32)
        lr1 = torch.full((batch_heads, seq_len, 1), 0.04, dtype=torch.float32)
        lr2 = torch.full((batch_heads, seq_len, 1), 0.03, dtype=torch.float32)

        one_step = block_causal_lact_swiglu(
            w0.clone(),
            w1.clone(),
            w2.clone(),
            q,
            k,
            v,
            lr0,
            lr1,
            lr2,
            chunk_size=chunk_size,
            ttt_inner_steps=1,
            use_muon=False,
            momentum=None,
        )
        two_steps = block_causal_lact_swiglu(
            w0.clone(),
            w1.clone(),
            w2.clone(),
            q,
            k,
            v,
            lr0,
            lr1,
            lr2,
            chunk_size=chunk_size,
            ttt_inner_steps=2,
            use_muon=False,
            momentum=None,
        )
        manual_two_steps = _manual_two_step_block(
            w0,
            w1,
            w2,
            q,
            k,
            v,
            lr0,
            lr1,
            lr2,
            chunk_size=chunk_size,
            ttt_inner_steps=2,
        )

        self.assertFalse(torch.allclose(one_step, two_steps))
        self.assertTrue(torch.allclose(two_steps, manual_two_steps, atol=1e-5, rtol=1e-5))

    def test_hybrid_branch_forwards_ttt_inner_steps_to_ttt_kernel(self):
        config = HybridQwen3LaCTConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=4,
            hybrid_layer_types=["lact"],
            num_lact_heads=4,
            lact_chunk_size=2,
            ttt_inner_steps=4,
            use_fused_kernel=False,
            use_momentum=False,
            learnable_ttt_scale=False,
            max_position_embeddings=8,
        )
        branch = HybridQwen3LaCTBranch(config, layer_idx=0)

        hidden_states = torch.randn(1, 4, 16)
        position_ids = torch.arange(4).unsqueeze(0)
        fast_q = torch.randn(1, 4, 4, 4)
        fast_k = torch.randn(1, 4, 4, 4)
        fast_v = torch.randn(1, 4, 4, 4)
        captured = {}

        def fake_prenorm(
            w0,
            w1,
            w2,
            q,
            k,
            v,
            lr0,
            lr1,
            lr2,
            *,
            chunk_size,
            ttt_inner_steps,
            update_phase,
            use_muon,
            momentum,
        ):
            del w0, w1, w2, k, v, lr0, lr1, lr2, chunk_size, update_phase, use_muon, momentum
            captured["ttt_inner_steps"] = ttt_inner_steps
            return torch.zeros_like(q)

        with mock.patch(
            "custom_models.hybrid_qwen3_lact_model.modeling_hybrid_qwen3_lact.prenorm_block_causal_lact_swiglu",
            side_effect=fake_prenorm,
        ):
            output, _ = branch(
                hidden_states,
                position_ids,
                fast_q=fast_q,
                fast_k=fast_k,
                fast_v=fast_v,
            )

        self.assertEqual(captured["ttt_inner_steps"], 4)
        self.assertEqual(tuple(output.shape), (1, 4, 16))

    def test_prenorm_multiple_inner_steps_are_effective(self):
        batch_heads = 1
        seq_len = 4
        d_in = 2
        d_h = 3

        w0 = torch.randn(batch_heads, d_h, d_in, dtype=torch.float32)
        w1 = torch.randn(batch_heads, d_in, d_h, dtype=torch.float32)
        w2 = torch.randn(batch_heads, d_h, d_in, dtype=torch.float32)
        q = torch.randn(batch_heads, seq_len, d_in, dtype=torch.float32)
        k = torch.randn(batch_heads, seq_len, d_in, dtype=torch.float32)
        v = torch.randn(batch_heads, seq_len, d_in, dtype=torch.float32)
        lr = torch.full((batch_heads, seq_len, 1), 0.02, dtype=torch.float32)

        one_step = prenorm_block_causal_lact_swiglu(
            w0.clone(),
            w1.clone(),
            w2.clone(),
            q,
            k,
            v,
            lr,
            lr,
            lr,
            chunk_size=2,
            ttt_inner_steps=1,
            use_muon=False,
            momentum=None,
        )
        three_steps = prenorm_block_causal_lact_swiglu(
            w0.clone(),
            w1.clone(),
            w2.clone(),
            q,
            k,
            v,
            lr,
            lr,
            lr,
            chunk_size=2,
            ttt_inner_steps=3,
            use_muon=False,
            momentum=None,
        )

        self.assertFalse(torch.allclose(one_step, three_steps))


if __name__ == "__main__":
    unittest.main()
