import pytest
import torch

try:
    from custom_models.hybrid_qwen3_lact_model.configuration_hybrid_qwen3_lact import (
        HybridQwen3LaCTConfig,
    )
    from custom_models.hybrid_qwen3_lact_model.modeling_hybrid_qwen3_lact import (
        HybridQwen3LaCTForCausalLM,
        HybridQwen3LaCTModel,
    )
except Exception as exc:
    pytest.skip(f"HybridQwen3LaCT test dependencies unavailable: {exc}", allow_module_level=True)


def _make_config() -> HybridQwen3LaCTConfig:
    return HybridQwen3LaCTConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        use_sliding_window=False,
        hybrid_layer_types=["fa", "fa"],
        fuse_cross_entropy=False,
        use_fused_kernel=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


def test_prepare_inputs_for_generation_disables_cache_and_keeps_full_context() -> None:
    model = HybridQwen3LaCTForCausalLM(_make_config())
    input_ids = torch.tensor([[0, 0, 5, 6, 7]])
    attention_mask = torch.tensor([[0, 0, 1, 1, 1]])

    prepared = model.prepare_inputs_for_generation(
        input_ids=input_ids,
        attention_mask=attention_mask,
        cache_position=torch.arange(input_ids.shape[1]),
        use_cache=True,
    )

    assert prepared["use_cache"] is False
    assert prepared["past_key_values"] is None
    assert prepared["input_ids"] is not None
    assert prepared["input_ids"].shape == input_ids.shape
    assert torch.equal(prepared["input_ids"], input_ids)


def test_forward_uses_attention_mask_position_ids_for_left_padding() -> None:
    model = HybridQwen3LaCTModel(_make_config())
    captured = {}
    original_forward = model.layers[0].forward

    def capture_forward(hidden_states, attention_mask=None, position_ids=None):
        captured["position_ids"] = position_ids.detach().clone()
        return original_forward(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    model.layers[0].forward = capture_forward

    input_ids = torch.tensor([[0, 0, 5, 6, 7]])
    attention_mask = torch.tensor([[0, 0, 1, 1, 1]])
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    assert outputs.last_hidden_state.shape[:2] == input_ids.shape
    assert torch.equal(captured["position_ids"], torch.tensor([[0, 0, 0, 1, 2]]))
