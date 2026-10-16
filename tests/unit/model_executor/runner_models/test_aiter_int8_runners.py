"""AITER INT8 load contracts for the Z-Image and Flux 2 Klein runners.

Builds runner models from a parsed config only; nothing loads weights or touches the network.
"""

import pytest

from xfuser.model_executor.models.runner_models.loading.contracts import (
    LoadDeclaration,
    MaterializationMode,
    QuantizationBackend,
    QuantizationFormat,
    select_load_contract,
)


@pytest.fixture(autouse=True)
def _single_process(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")


def _build(model_name, **config):
    from xfuser import xFuserArgs
    from xfuser.model_executor.models.runner_models.base_model import MODEL_REGISTRY

    return MODEL_REGISTRY[model_name](xFuserArgs(model=model_name, **config))


@pytest.mark.parametrize(
    "model_name",
    ["Z-Image", "Z-Image-Turbo", "FLUX.2-klein-9B", "FLUX.2-klein-4B"],
)
def test_runner_accepts_aiter_int8_transformer_gemms(model_name):
    model = _build(model_name, gemm_quantization="int8")
    declaration = LoadDeclaration.for_runner(
        model.capabilities,
        load_support=model.load_support,
        fsdp_strategy=model.settings.fsdp_strategy,
    )

    contract = select_load_contract(
        requested_format=QuantizationFormat.INT8,
        selected_backend=QuantizationBackend.AITER,
        materialization_mode=MaterializationMode.EAGER,
        declaration=declaration,
        fsdp_strategy=model.settings.fsdp_strategy,
        runner_name=type(model).__name__,
    )

    assert contract.selected_backend is QuantizationBackend.AITER
