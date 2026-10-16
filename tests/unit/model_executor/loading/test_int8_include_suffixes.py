"""int8_gemm_include_suffixes keeps base Z-Image's feed_forward.w2 and adaLN in bf16 on each INT8 path."""

from types import SimpleNamespace

import pytest

from xfuser.model_executor.models.runner_models.loading import placement, shard
from xfuser.model_executor.models.runner_models.loading.quantization_ledger import QuantizationLedger
from xfuser.model_executor.models.runner_models.z_image import (
    xFuserZImageModel,
    xFuserZImageTurboModel,
)


class _Plan:
    def __init__(self, targets):
        self.targets = targets

    def module_list(self, format_name="fp8"):
        return list(self.targets)

    def targets_for(self, component_name, format_name="fp8"):
        prefix = f"{component_name}."
        return [t[len(prefix):] for t in self.targets if t.startswith(prefix)]


class _FormatAdapter:
    def __init__(self):
        self.block_calls = []
        self.module_calls = []

    def convert_block(self, block, **kwargs):
        self.block_calls.append(kwargs)

    def convert_module(self, module, **kwargs):
        self.module_calls.append(kwargs)


def _loader(model_cls):
    settings = model_cls.settings
    targets = list(settings.int8_gemm_module_list)
    adapter = _FormatAdapter()
    model = SimpleNamespace(
        config=SimpleNamespace(
            use_fp4_gemms=False,
            use_fp8_gemms=False,
            use_int8_gemms=True,
            use_hybrid_gemm_schedule=False,
        ),
        settings=settings,
        pipe=SimpleNamespace(
            transformer=SimpleNamespace(**{t.partition(".")[2]: object() for t in targets})
        ),
    )
    loader = SimpleNamespace(
        model=model,
        quantization_plan=_Plan(targets),
        quantization_ledger=QuantizationLedger(),
        backends=SimpleNamespace(
            format=adapter,
            format_targets_for=lambda name: _Plan(targets).targets_for(name),
        ),
    )
    return loader, adapter


def _converted(filter_fn, prefix=""):
    if filter_fn is None:
        return {"attention.to_q", "feed_forward.w1", "feed_forward.w2", "adaLN_modulation.0"}
    return {
        name
        for name in ("attention.to_q", "feed_forward.w1", "feed_forward.w2", "adaLN_modulation.0")
        if filter_fn(object(), f"{prefix}{name}")
    }


QUANTIZED_EXCEPT_W2_AND_ADALN = {"attention.to_q", "feed_forward.w1"}
EVERY_LINEAR = {"attention.to_q", "feed_forward.w1", "feed_forward.w2", "adaLN_modulation.0"}


@pytest.mark.parametrize(
    ("model_cls", "expected"),
    [(xFuserZImageModel, QUANTIZED_EXCEPT_W2_AND_ADALN), (xFuserZImageTurboModel, EVERY_LINEAR)],
)
def test_blockwise_int8_converts_the_model_declared_linears(model_cls, expected):
    loader, adapter = _loader(model_cls)

    quantize = shard.build_block_quantize_fn(loader, "transformer", ["layers"], local_rank=0)
    quantize(object(), 0)

    assert _converted(adapter.block_calls[0]["filter_fn"]) == expected


@pytest.mark.parametrize(
    ("model_cls", "expected"),
    [(xFuserZImageModel, QUANTIZED_EXCEPT_W2_AND_ADALN), (xFuserZImageTurboModel, EVERY_LINEAR)],
)
def test_eager_int8_converts_the_model_declared_linears(monkeypatch, model_cls, expected):
    loader, adapter = _loader(model_cls)
    monkeypatch.setattr(placement, "log", lambda *_args: None)
    monkeypatch.setattr(
        placement,
        "prepare_native_transformer_format_load",
        lambda *_args, **_kwargs: SimpleNamespace(descriptor=SimpleNamespace(log_message=lambda: "")),
    )

    placement._convert_int8_on_device(loader, local_rank=0)

    assert len(adapter.module_calls) == len(loader.model.settings.int8_gemm_module_list)
    for call in adapter.module_calls:
        assert _converted(call.get("filter_fn"), prefix="0.") == expected
