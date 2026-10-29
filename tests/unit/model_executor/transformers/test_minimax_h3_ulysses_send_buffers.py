from types import SimpleNamespace

import pytest
import torch

minimax_h3 = pytest.importorskip(
    "diffusers.models.transformers.transformer_minimax_h3",
    reason="installed diffusers does not include MiniMax-H3",
)

from xfuser.model_executor.layers import fused_qkv_norm_rope_minimax_h3_triton as fused  # noqa: E402
from xfuser.model_executor.layers import usp  # noqa: E402
from xfuser.model_executor.models.transformers import transformer_minimax_h3 as xfuser_minimax_h3  # noqa: E402

HEADS, HEAD_DIM, SEQ_LEN = 4, 32, 6


def _attention():
    torch.manual_seed(0)
    attn = minimax_h3.MiniMaxH3Attention(hidden_size=32, heads=HEADS, dim_head=HEAD_DIM).eval()
    attn.fuse_projections()
    return attn


def _inputs():
    generator = torch.Generator().manual_seed(1)
    hidden_states = torch.randn(1, SEQ_LEN, 32, generator=generator)
    position_ids = torch.randint(0, 5, (SEQ_LEN, 3), generator=generator).float()
    return hidden_states, minimax_h3.MiniMaxH3RotaryPosEmbed(rope_freq_dim=4)(position_ids)


def _packed_reference(attn, hidden_states, rotary_emb, world_size, chunk_sizes):
    query, key, value = fused._reference(attn.to_qkv(hidden_states), attn.norm_q, attn.norm_k, *rotary_emb, HEADS)
    return usp.ulysses_send_buffers(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), world_size, chunk_sizes
    )


@pytest.mark.parametrize(
    "chunk_sizes, layout",
    [([4, 3], (3, 1)), ([3, 2, 2], (2, 1)), ([2, 2], (2, 0)), ([3], (3, 0)), ([2, 3], None), ([3, 1], None)],
)
def test_chunk_layout_is_the_divmod_split(chunk_sizes, layout):
    assert fused._chunk_layout(chunk_sizes) == layout


def test_ulysses_variant_falls_back_to_the_packed_reference():
    attn = _attention()
    hidden_states, rotary_emb = _inputs()

    with torch.no_grad():
        actual = fused.minimax_h3_qkv_norm_rope_ulysses(
            attn.to_qkv(hidden_states), attn.norm_q, attn.norm_k, *rotary_emb, HEADS, 2, [1, 1]
        )
        expected = _packed_reference(attn, hidden_states, rotary_emb, 2, [1, 1])

    for chunk, expected_chunk in zip(actual, expected, strict=True):
        for buffer, expected_buffer in zip(chunk, expected_chunk, strict=True):
            assert torch.equal(buffer, expected_buffer)


def test_processor_hands_the_chunked_path_its_send_buffers(monkeypatch):
    runtime_state = SimpleNamespace(
        fp8_comms=None, attention_backend=None, runtime_config=SimpleNamespace(use_spargeattn_head_balance=False)
    )
    monkeypatch.setattr(usp, "get_runtime_state", lambda: runtime_state)
    monkeypatch.setattr(usp, "get_ulysses_parallel_world_size", lambda: 2)
    monkeypatch.setattr(usp, "get_ring_parallel_world_size", lambda: 1)
    monkeypatch.setattr(usp, "_ULYSSES_HEAD_CHUNKS", 2)
    monkeypatch.setattr(xfuser_minimax_h3, "get_ulysses_parallel_world_size", lambda: 2)
    attended = torch.randn(1, HEADS, SEQ_LEN, HEAD_DIM)
    calls = []

    def from_send_buffers(send_buffers, **kwargs):
        calls.append(send_buffers)
        return attended

    monkeypatch.setattr(xfuser_minimax_h3, "ulysses_attention_from_send_buffers", from_send_buffers)
    attn = _attention()
    hidden_states, rotary_emb = _inputs()
    processor = xfuser_minimax_h3.xFuserMiniMaxH3AttnProcessor(use_ulysses_parallel_attention=True)

    with torch.no_grad():
        output = processor(attn, hidden_states, rotary_emb)
        expected_buffers = _packed_reference(attn, hidden_states, rotary_emb, 2, [1, 1])
        expected = attn.to_out[0](attended.transpose(1, 2).flatten(2, 3))

    (send_buffers,) = calls
    for chunk, expected_chunk in zip(send_buffers, expected_buffers, strict=True):
        for buffer, expected_buffer in zip(chunk, expected_chunk, strict=True):
            assert torch.equal(buffer, expected_buffer)
    assert torch.equal(output, expected)
