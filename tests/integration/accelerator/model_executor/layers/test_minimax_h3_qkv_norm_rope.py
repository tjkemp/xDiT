import pytest
import torch

pytest.importorskip("triton")
transformer_minimax_h3 = pytest.importorskip("diffusers.models.transformers.transformer_minimax_h3")

from xfuser.model_executor.layers import fused_qkv_norm_rope_minimax_h3_triton as fused  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an accelerator")


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


@pytest.mark.parametrize("batch, seq_len", [(1, 77), (2, 64)])
def test_fused_qkv_norm_rope_matches_the_unfused_reference(batch, seq_len):
    qkv, norm_q, norm_k, cos, sin, heads = _inputs(batch, seq_len)

    with torch.no_grad():
        query, key, value = torch.ops.xfuser.minimax_h3_qkv_norm_rope(
            qkv, norm_q.weight, norm_k.weight, cos, sin, heads, norm_q.eps, norm_k.eps
        )
        ref_query, ref_key, ref_value = fused._reference(qkv, norm_q, norm_k, cos, sin, heads)

    for tensor in (query, key, value):
        assert tensor.shape == ref_query.shape
        assert tensor.is_contiguous()
    torch.testing.assert_close(query, ref_query, atol=1e-2, rtol=1.6e-2)
    torch.testing.assert_close(key, ref_key, atol=1e-2, rtol=1.6e-2)
    assert torch.equal(value, ref_value)


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


def test_grad_enabled_call_runs_the_reference():
    inputs = _inputs(1, 32)

    with torch.enable_grad():
        selected = fused.minimax_h3_qkv_norm_rope(*inputs)
    with torch.no_grad():
        reference = fused._reference(*inputs)

    for actual, expected in zip(selected, reference):
        assert torch.equal(actual, expected)


def test_fused_qkv_norm_rope_compiles_fullgraph():
    inputs = _inputs(1, 128)

    def step(qkv, cos, sin):
        _, norm_q, norm_k, _, _, heads = inputs
        return fused.minimax_h3_qkv_norm_rope(qkv, norm_q, norm_k, cos, sin, heads)

    torch._dynamo.reset()
    compiled = torch.compile(step, fullgraph=True)
    qkv, _, _, cos, sin, _ = inputs
    with torch.no_grad():
        expected = step(qkv, cos, sin)
        actual = compiled(qkv, cos, sin)

    for a, e in zip(actual, expected):
        assert torch.equal(a, e)
