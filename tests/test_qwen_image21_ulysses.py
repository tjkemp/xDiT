"""Qwen-Image-2.1 Ulysses parity against stock diffusers on a tiny random transformer.

    torchrun --nproc_per_node=2 -m pytest tests/test_qwen_image21_ulysses.py
    torchrun --nproc_per_node=4 -m pytest tests/test_qwen_image21_ulysses.py
    torchrun --nproc_per_node=8 -m pytest tests/test_qwen_image21_ulysses.py
"""

import os
from types import SimpleNamespace

import pytest
import torch

if int(os.environ.get("WORLD_SIZE", "1")) < 2 or not torch.cuda.is_available():
    pytest.skip("needs a multi-GPU torchrun launch", allow_module_level=True)

qwenimage21 = pytest.importorskip("diffusers.models.transformers.transformer_qwenimage21")

from xfuser.core.distributed import init_distributed_environment, initialize_model_parallel
from xfuser.core.distributed.attention_backend import AttentionBackendType
from xfuser.core.distributed.parallel_state import model_parallel_is_initialized
from xfuser.model_executor.layers import usp
from xfuser.model_executor.models.transformers import transformer_qwenimage21 as xfuser_qwenimage21


NUM_LAYERS = 2
# Must be divisible by every tested Ulysses degree.
NUM_HEADS = 8


@pytest.fixture(scope="module")
def device():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not model_parallel_is_initialized():
        world_size = int(os.environ["WORLD_SIZE"])
        init_distributed_environment(rank=int(os.environ["RANK"]), world_size=world_size)
        initialize_model_parallel(ulysses_degree=world_size, ring_degree=1)
    return torch.device(f"cuda:{local_rank}")


@pytest.fixture(autouse=True)
def sdpa_backend(monkeypatch):
    monkeypatch.setattr(
        usp, "get_runtime_state", lambda: SimpleNamespace(attention_backend=AttentionBackendType.SDPA)
    )


def _tiny_model(device):
    torch.manual_seed(0)
    return qwenimage21.QwenImage21Transformer2DModel(
        num_layers=NUM_LAYERS,
        attention_head_dim=16,
        num_attention_heads=NUM_HEADS,
        context_in_dim=8,
        in_channels=4,
        out_channels=4,
        axes_dims_rope=(4, 6, 6),
    ).to(device).eval()


def _inputs(device):
    # 3 text, a 2x2 condition image (one VLM slot), 3 text, then a 2x6 target (three slots): a 22-token
    # prefill and a 12-token decode, so both are padded at some of the tested degrees.
    generator = torch.Generator().manual_seed(1)
    return {
        "hidden_states": torch.randn(1, 4 + 12, 4, generator=generator).to(device),
        "encoder_hidden_states": torch.randn(1, 7, 8, generator=generator).to(device),
        "img_shapes": [[(1, 2, 2), (1, 2, 6)]],
        "img_mask": torch.tensor([[False] * 3 + [True] + [False] * 3 + [True] * 3], device=device),
    }


def _prefill_then_decode(model, device, encoder_hidden_states_mask=None):
    inputs = _inputs(device)
    kv_cache = qwenimage21.QwenImage21KVCache(NUM_LAYERS)
    outputs = []
    for timestep, mode in ((0.9, "extract"), (0.4, "cached")):
        with torch.no_grad():
            outputs.append(
                model(
                    **inputs,
                    timestep=torch.tensor([timestep], device=device),
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    kv_cache=kv_cache,
                    kv_cache_mode=mode,
                ).sample
            )
    return outputs


@pytest.mark.parametrize(
    "processor_name",
    ["xFuserQwenImage21AttnProcessor", "xFuserQwenImage21FlexAttnProcessor"],
)
@pytest.mark.parametrize("padded_prompt", [False, True])
def test_ulysses_matches_single_gpu_diffusers(device, processor_name, padded_prompt):
    if processor_name == "xFuserQwenImage21FlexAttnProcessor" and not qwenimage21._FLEX_AVAILABLE:
        pytest.skip("flex_attention unavailable")
    mask = None
    if padded_prompt:
        mask = torch.ones(1, 7, dtype=torch.bool, device=device)
        mask[0, -1] = False

    stock = _tiny_model(device)
    reference = _prefill_then_decode(stock, device, mask)

    wrapper = xfuser_qwenimage21.xFuserQwenImage21TransformerWrapper.from_config(stock.config).to(device).eval()
    wrapper.load_state_dict(stock.state_dict())
    for block in wrapper.transformer_blocks:
        block.attn.set_processor(getattr(xfuser_qwenimage21, processor_name)())
    actual = _prefill_then_decode(wrapper, device, mask)

    for ref, out in zip(reference, actual):
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
