import pytest
import torch

pytest.importorskip("triton")
transformer_minimax_h3 = pytest.importorskip("diffusers.models.transformers.transformer_minimax_h3")

from xfuser.model_executor.layers import fused_qkv_norm_rope_minimax_h3_triton as fused  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an accelerator")
# One bf16 rounding of an fp32 result: half a ULP, relative to the value.
_BF16_ROUNDING = {"rtol": 2**-8, "atol": 1e-5}


def _inputs(batch, seq_len, heads=4, head_dim=128, rope_freq_dim=16):
    generator = torch.Generator(device="cuda").manual_seed(0)
    qkv = torch.randn(batch, seq_len, 3 * heads * head_dim, device="cuda", generator=generator).to(torch.bfloat16)
    norm_q = torch.nn.RMSNorm(head_dim, eps=1e-5).to("cuda", torch.bfloat16)
    norm_k = torch.nn.RMSNorm(head_dim, eps=1e-5).to("cuda", torch.bfloat16)
    with torch.no_grad():
        norm_q.weight.copy_(torch.rand(head_dim, device="cuda", generator=generator) + 0.5)
        norm_k.weight.copy_(torch.rand(head_dim, device="cuda", generator=generator) + 0.5)
    positions = torch.arange(seq_len, device="cuda", dtype=torch.float32)
    position_ids = torch.stack((positions % 5, positions % 7, positions), dim=-1)
    rope = transformer_minimax_h3.MiniMaxH3RotaryPosEmbed(rope_freq_dim=rope_freq_dim).to("cuda")
    cos, sin = rope(position_ids)
    return qkv, norm_q, norm_k, cos, sin, heads


def _fp32_reference(qkv, norm_q, norm_k, cos, sin, heads):
    def fp32_norm(norm):
        norm32 = torch.nn.RMSNorm(norm.normalized_shape, eps=norm.eps).to("cuda")
        norm32.weight.data.copy_(norm.weight.float())
        return norm32

    return fused._reference(qkv.float(), fp32_norm(norm_q), fp32_norm(norm_k), cos.float(), sin.float(), heads)


def _max_error(actual, expected):
    return (actual.float() - expected).abs().max().item()


@pytest.mark.parametrize("batch, seq_len", [(1, 77), (2, 64)])
def test_fused_qkv_norm_rope_rounds_the_fp32_result_once(batch, seq_len):
    inputs = _inputs(batch, seq_len)
    qkv, norm_q, norm_k, cos, sin, heads = inputs

    with torch.no_grad():
        query, key, value = torch.ops.xfuser.minimax_h3_qkv_norm_rope(
            qkv, norm_q.weight, norm_k.weight, cos, sin, heads, norm_q.eps, norm_k.eps
        )
        exact_query, exact_key, _ = _fp32_reference(*inputs)
        eager_query, eager_key, eager_value = fused._reference(*inputs)

    for tensor in (query, key, value):
        assert tensor.shape == eager_query.shape
        assert tensor.is_contiguous()
    for actual, exact, eager in ((query, exact_query, eager_query), (key, exact_key, eager_key)):
        torch.testing.assert_close(actual.float(), exact, **_BF16_ROUNDING)
        assert _max_error(actual, exact) <= _max_error(eager, exact)
    assert torch.equal(value, eager_value)


def test_in_envelope_call_selects_the_fused_kernel():
    inputs = _inputs(1, 96)

    with torch.no_grad():
        selected = fused.minimax_h3_qkv_norm_rope(*inputs)
        qkv, norm_q, norm_k, cos, sin, heads = inputs
        kernel = torch.ops.xfuser.minimax_h3_qkv_norm_rope(
            qkv, norm_q.weight, norm_k.weight, cos, sin, heads, norm_q.eps, norm_k.eps
        )

    for actual, expected in zip(selected, kernel):
        assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "batch, heads, world_size, chunk_sizes",
    [(1, 56, 8, [4, 3]), (2, 28, 4, [3, 2, 2]), (1, 8, 2, [2, 2])],
)
def test_ulysses_variant_writes_the_packed_bshd_result(batch, heads, world_size, chunk_sizes):
    from xfuser.model_executor.layers.usp import ulysses_send_buffers

    inputs = _inputs(batch, 40, heads=heads)

    with torch.no_grad():
        buffers = fused.minimax_h3_qkv_norm_rope_ulysses(*inputs, world_size, chunk_sizes)
        query, key, value = fused.minimax_h3_qkv_norm_rope(*inputs)
    expected = ulysses_send_buffers(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), world_size, chunk_sizes
    )

    for chunk, expected_chunk in zip(buffers, expected, strict=True):
        for actual, packed in zip(chunk, expected_chunk, strict=True):
            assert actual.is_contiguous()
            assert torch.equal(actual, packed)


def test_grad_enabled_call_runs_the_reference():
    inputs = _inputs(1, 32)

    with torch.enable_grad():
        selected = fused.minimax_h3_qkv_norm_rope(*inputs)
    with torch.no_grad():
        reference = fused._reference(*inputs)

    for actual, expected in zip(selected, reference):
        assert torch.equal(actual, expected)


def test_compiled_graph_keeps_the_fused_op():
    inputs = _inputs(1, 128)
    qkv, norm_q, norm_k, cos, sin, heads = inputs
    graph_targets = []

    def recording_inductor(graph_module, example_inputs):
        graph_targets.extend(str(node.target) for node in graph_module.graph.nodes)
        return torch._inductor.compile(graph_module, example_inputs)

    def step(qkv, cos, sin):
        return fused.minimax_h3_qkv_norm_rope(qkv, norm_q, norm_k, cos, sin, heads)

    torch._dynamo.reset()
    compiled = torch.compile(step, backend=recording_inductor, fullgraph=True)
    with torch.no_grad():
        exact_query, exact_key, _ = _fp32_reference(*inputs)
        query, key, value = compiled(qkv, cos, sin)
        _, _, eager_value = fused._reference(*inputs)

    assert any("minimax_h3_qkv_norm_rope" in target for target in graph_targets)
    torch.testing.assert_close(query.float(), exact_query, **_BF16_ROUNDING)
    torch.testing.assert_close(key.float(), exact_key, **_BF16_ROUNDING)
    assert torch.equal(value, eager_value)
