import sys
from pathlib import Path

import torch


FLAME_ROOT = Path(__file__).resolve().parents[2]
CUSTOM_MODELS = FLAME_ROOT / "custom_models"
sys.path.insert(0, str(CUSTOM_MODELS))

from qwen3_tttip.configuration_qwen3 import Qwen3Config
from qwen3_tttip.modeling_qwen3 import Qwen3ForCausalLM


def _cuda_device_or_skip():
    if torch.cuda.is_available():
        return torch.device("cuda")
    message = "qwen3_tttip one-pass test uses the Liger causal-LM loss, which requires CUDA"
    if __name__ == "__main__":
        raise SystemExit(message)
    import pytest

    pytest.skip(message)


def print_tensor(name, x):
    if x is None:
        print(f"{name}: None")
        return
    if isinstance(x, (tuple, list)):
        print(f"{name}: tuple/list len={len(x)}")
        x = x[0]
    if not torch.is_tensor(x):
        print(f"{name}: {type(x)}")
        return

    y = x.detach()
    print(
        f"{name}: shape={tuple(y.shape)} "
        f"dtype={y.dtype} device={y.device} "
        f"mean={y.float().mean().item():.6f} "
        f"std={y.float().std(unbiased=False).item():.6f} "
        f"min={y.float().min().item():.6f} "
        f"max={y.float().max().item():.6f}"
    )


def add_hooks(model):
    handles = []

    def hook(name):
        def _hook(module, inputs, output):
            print_tensor(f"{name}.input[0]", inputs[0] if inputs else None)
            print_tensor(f"{name}.output", output)
        return _hook

    handles.append(model.model.layers[0].mlp.register_forward_hook(hook("layer0.normal_mlp")))
    handles.append(model.model.layers[1].mlp.register_forward_hook(hook("layer1.ttt_mlp")))
    handles.append(model.model.layers[1].mlp.ttt_conv.register_forward_hook(hook("layer1.ttt_conv")))

    return handles


def test_two_layer_qwen3_tttip_one_training_pass():
    device = _cuda_device_or_skip()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    config = Qwen3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        max_position_embeddings=64,
        ttt_mode=True,
        ttt_layers=[1],
        ttt_chunk=8,
        ttt_proj=True,
        ttt_lr=0.3,
        ttt_target="hidden_states",
        use_cache=False,
        _attn_implementation="eager",
    )
    model = Qwen3ForCausalLM(config).to(device)
    model.train()

    normal_mlp = model.model.layers[0].mlp
    ttt_mlp = model.model.layers[1].mlp
    assert not hasattr(normal_mlp, "ttt_conv")
    assert hasattr(ttt_mlp, "ttt_conv")

    batch_size = 2
    seq_len = 17
    input_ids = torch.randint(
        0,
        config.vocab_size,
        (batch_size, seq_len),
        dtype=torch.long,
        device=device,
    )
    labels = input_ids.clone()
    position_ids = (
        torch.arange(seq_len, dtype=torch.int32, device=device)
        .unsqueeze(0)
        .repeat(batch_size, 1)
    )

    output = model(
        input_ids=input_ids,
        labels=labels,
        position_ids=position_ids,
        cu_seqlens=None,
        use_cache=False,
    )

    assert output.loss is not None
    assert torch.isfinite(output.loss)
    if output.logits is not None:
        assert output.logits.shape == (batch_size, seq_len, config.vocab_size)

    output.loss.backward()

    assert normal_mlp.down_proj.weight.grad is not None
    assert ttt_mlp.down_proj.weight.grad is not None
    assert ttt_mlp.ttt_conv.weight.grad is not None
    assert ttt_mlp.ttt_proj.weight.grad is not None


if __name__ == "__main__":
    test_two_layer_qwen3_tttip_one_training_pass()
    print("two-layer qwen3_tttip one-pass training test passed")

