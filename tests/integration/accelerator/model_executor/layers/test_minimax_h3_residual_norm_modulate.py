import pytest
import torch

pytest.importorskip("triton")
transformer_minimax_h3 = pytest.importorskip("diffusers.models.transformers.transformer_minimax_h3")

from xfuser.model_executor.layers import fused_residual_norm_modulate_minimax_h3_triton as fused  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an accelerator")

_HIDDEN = 5376
# One bf16 rounding of an fp32 result: half a ULP, relative to the value.
_BF16_ROUNDING = {"rtol": 2**-8, "atol": 1e-5}


def _inputs(batch, seq_len, timesteps):
    generator = torch.Generator(device="cuda").manual_seed(0)

    def randn(*shape):
        return torch.randn(*shape, device="cuda", generator=generator)

    residual = randn(batch, seq_len, _HIDDEN).to(torch.bfloat16)
    branch = randn(batch, seq_len, _HIDDEN).to(torch.bfloat16)
    adaln = transformer_minimax_h3.MiniMaxH3AdaLayerNormModulation(64, _HIDDEN).to("cuda", torch.bfloat16)
    norm = torch.nn.RMSNorm(_HIDDEN, eps=1e-5).to("cuda", torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(torch.rand(_HIDDEN, device="cuda", generator=generator) + 0.5)
        shift, scale, gate = adaln(randn(timesteps, 64))[:3]
    rows = timesteps * transformer_minimax_h3.MINIMAX_H3_MODALITY_NUM
    gate_index = torch.randint(0, rows, (seq_len,), device="cuda", generator=generator)
    mod_index = torch.randint(0, rows, (seq_len,), device="cuda", generator=generator)
    return residual, branch, gate, gate_index, shift, scale, mod_index, norm


def _fp32_reference(residual, branch, gate, gate_index, shift, scale, mod_index, norm):
    norm32 = torch.nn.RMSNorm(_HIDDEN, eps=norm.eps).to("cuda")
    norm32.weight.data.copy_(norm.weight.float())

    def up(tensor):
        return None if tensor is None else tensor.float()

    return fused._reference(up(residual), up(branch), up(gate), gate_index, up(shift), up(scale), mod_index, norm32)


def _max_error(actual, expected):
    return (actual.float() - expected).abs().max().item()


@pytest.mark.parametrize("batch, seq_len, timesteps", [(1, 77, 3), (2, 64, 2)])
def test_residual_norm_modulate_rounds_the_fp32_result_once(batch, seq_len, timesteps):
    inputs = _inputs(batch, seq_len, timesteps)
    residual, branch, gate, gate_index, shift, scale, mod_index, norm = inputs

    with torch.no_grad():
        hidden, normed = torch.ops.xfuser.minimax_h3_residual_norm_modulate(
            residual, branch, gate, gate_index, shift, scale, mod_index, norm.weight, norm.eps
        )
        exact_hidden, exact_normed = _fp32_reference(*inputs)
        eager_hidden, eager_normed = fused._reference(*inputs)

    torch.testing.assert_close(hidden.float(), exact_hidden, **_BF16_ROUNDING)
    torch.testing.assert_close(normed.float(), exact_normed, **_BF16_ROUNDING)
    assert _max_error(normed, exact_normed) <= _max_error(eager_normed, exact_normed)


def test_norm_modulate_without_a_branch_rounds_the_fp32_result_once():
    residual, _, _, _, shift, scale, mod_index, norm = _inputs(1, 77, 3)

    with torch.no_grad():
        normed = torch.ops.xfuser.minimax_h3_norm_modulate(residual, shift, scale, mod_index, norm.weight, norm.eps)
        _, exact_normed = _fp32_reference(residual, None, None, None, shift, scale, mod_index, norm)

    torch.testing.assert_close(normed.float(), exact_normed, **_BF16_ROUNDING)


def test_in_envelope_calls_select_the_fused_kernels():
    inputs = _inputs(1, 96, 3)
    residual, branch, gate, gate_index, shift, scale, mod_index, norm = inputs

    with torch.no_grad():
        selected = fused.minimax_h3_residual_norm_modulate(*inputs)
        kernel = torch.ops.xfuser.minimax_h3_residual_norm_modulate(
            residual, branch, gate, gate_index, shift, scale, mod_index, norm.weight, norm.eps
        )
        selected_first = fused.minimax_h3_residual_norm_modulate(
            residual, None, None, None, shift, scale, mod_index, norm
        )
        kernel_first = torch.ops.xfuser.minimax_h3_norm_modulate(
            residual, shift, scale, mod_index, norm.weight, norm.eps
        )

    for actual, expected in zip(selected, kernel):
        assert torch.equal(actual, expected)
    assert selected_first[0] is residual
    assert torch.equal(selected_first[1], kernel_first)


def test_grad_enabled_call_runs_the_reference():
    inputs = _inputs(1, 32, 3)

    with torch.enable_grad():
        selected = fused.minimax_h3_residual_norm_modulate(*inputs)
    with torch.no_grad():
        reference = fused._reference(*inputs)

    for actual, expected in zip(selected, reference):
        assert torch.equal(actual, expected)


def test_compiled_graph_keeps_the_fused_op():
    inputs = _inputs(1, 128, 3)
    norm = inputs[-1]
    graph_targets = []

    def recording_inductor(graph_module, example_inputs):
        graph_targets.extend(str(node.target) for node in graph_module.graph.nodes)
        return torch._inductor.compile(graph_module, example_inputs)

    def step(residual, branch, gate, gate_index, shift, scale, mod_index):
        return fused.minimax_h3_residual_norm_modulate(
            residual, branch, gate, gate_index, shift, scale, mod_index, norm
        )

    torch._dynamo.reset()
    compiled = torch.compile(step, backend=recording_inductor, fullgraph=True)
    with torch.no_grad():
        expected = _fp32_reference(*inputs)
        actual = compiled(*inputs[:-1])

    assert any("minimax_h3_residual_norm_modulate" in target for target in graph_targets)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a.float(), e, **_BF16_ROUNDING)
