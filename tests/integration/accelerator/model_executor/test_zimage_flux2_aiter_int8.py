import pytest
import torch
from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel
from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel

from xfuser.model_executor.models.runner_models.loading.contracts import (
    QuantizationBackend,
    QuantizationFormat,
)
from xfuser.model_executor.models.runner_models.loading.format_backends import (
    AiterInt8BackendAdapter,
)


def _aiter_int8_available() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is None:
        return False
    try:
        from aiter.ops.flydsl.kernels.gemm_a8w8_int8_4wave import (
            compile_int8_gemm_4w,  # noqa: F401
        )
    except ImportError:
        return False
    return "gfx950" in torch.cuda.get_device_properties(0).gcnArchName


pytestmark = pytest.mark.skipif(
    not _aiter_int8_available(),
    reason="AITER INT8 integration requires the local gfx950 kernel",
)


def _reference_and_quantized(model_cls, config):
    torch.manual_seed(3)
    reference = model_cls(**config).eval()
    for name, parameter in reference.named_parameters():
        if name.endswith("pad_token"):
            torch.nn.init.normal_(parameter)
    reference = reference.to(device="cuda", dtype=torch.bfloat16)
    quantized = model_cls(**config).eval().to(device="cuda", dtype=torch.bfloat16)
    quantized.load_state_dict(reference.state_dict())
    return reference, quantized


def _convert(model, target_names):
    adapter = AiterInt8BackendAdapter(
        backend=QuantizationBackend.AITER,
        format_=QuantizationFormat.INT8,
    )
    for name in target_names:
        target = getattr(model, name)
        assert adapter.convert_module(target, device="cuda") > 0
        assert not any(isinstance(m, torch.nn.Linear) for m in target.modules())


def _assert_close(actual, expected):
    assert actual.shape == expected.shape
    assert torch.isfinite(actual.float()).all()
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().flatten(),
        expected.float().flatten(),
        dim=0,
    )
    assert cosine > 0.999


@torch.no_grad()
def test_tiny_z_image_transformer_runs_with_aiter_int8_blocks():
    # dim / 3 * 8 keeps the FFN hidden size a multiple of 128, as in the 3840-wide checkpoint.
    config = {
        "in_channels": 4,
        "dim": 384,
        "n_layers": 1,
        "n_refiner_layers": 1,
        "n_heads": 3,
        "n_kv_heads": 3,
        "cap_feat_dim": 64,
        "axes_dims": [32, 48, 48],
        "axes_lens": [64, 32, 32],
    }
    reference, quantized = _reference_and_quantized(ZImageTransformer2DModel, config)
    _convert(quantized, ["layers", "noise_refiner", "context_refiner"])

    x = [torch.randn((4, 1, 8, 8), device="cuda", dtype=torch.bfloat16)]
    cap_feats = [torch.randn((7, 64), device="cuda", dtype=torch.bfloat16)]
    t = torch.tensor([0.5], device="cuda")
    expected = reference(x, t, cap_feats, return_dict=False)[0][0]
    actual = quantized(x, t, cap_feats, return_dict=False)[0][0]

    _assert_close(actual, expected)


@torch.no_grad()
def test_tiny_flux2_klein_transformer_runs_with_aiter_int8_blocks():
    config = {
        "in_channels": 8,
        "num_layers": 1,
        "num_single_layers": 1,
        "attention_head_dim": 128,
        "num_attention_heads": 2,
        "joint_attention_dim": 64,
        "axes_dims_rope": (32, 32, 32, 32),
        "guidance_embeds": False,
    }
    reference, quantized = _reference_and_quantized(Flux2Transformer2DModel, config)
    _convert(quantized, ["transformer_blocks", "single_transformer_blocks"])

    image_tokens, text_tokens = 16, 7
    img_ids = torch.zeros((image_tokens, 4), device="cuda")
    img_ids[:, 1] = torch.arange(image_tokens // 4).repeat_interleave(4)
    img_ids[:, 2] = torch.arange(4).repeat(image_tokens // 4)
    txt_ids = torch.zeros((text_tokens, 4), device="cuda")
    txt_ids[:, 3] = torch.arange(text_tokens)
    inputs = {
        "hidden_states": torch.randn(
            (1, image_tokens, 8), device="cuda", dtype=torch.bfloat16
        ),
        "encoder_hidden_states": torch.randn(
            (1, text_tokens, 64), device="cuda", dtype=torch.bfloat16
        ),
        "timestep": torch.tensor([0.4], device="cuda"),
        "img_ids": img_ids,
        "txt_ids": txt_ids,
        "return_dict": False,
    }
    expected = reference(**inputs)[0]
    actual = quantized(**inputs)[0]

    _assert_close(actual, expected)
