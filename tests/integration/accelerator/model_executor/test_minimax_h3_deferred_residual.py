import copy

import pytest
import torch

pytest.importorskip("triton")
minimax_h3 = pytest.importorskip("diffusers.models.transformers.transformer_minimax_h3")

from xfuser.model_executor.models.transformers import transformer_minimax_h3 as xfuser_minimax_h3  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an accelerator")

TIMESTEPS = 2
SEQ_LEN = 77
HIDDEN = 512


def _inputs():
    generator = torch.Generator(device="cuda").manual_seed(1)
    hidden_states = torch.randn(1, SEQ_LEN, HIDDEN, device="cuda", generator=generator).to(torch.bfloat16)
    temb = torch.randn(TIMESTEPS, 64, device="cuda", generator=generator)
    rows = TIMESTEPS * minimax_h3.MINIMAX_H3_MODALITY_NUM
    adaln_indices = torch.randint(0, rows, (SEQ_LEN,), device="cuda", generator=generator)
    position_ids = torch.randint(0, 5, (SEQ_LEN, 3), device="cuda", generator=generator).float()
    rotary_emb = minimax_h3.MiniMaxH3RotaryPosEmbed(rope_freq_dim=16).to("cuda")(position_ids)
    return hidden_states, temb, adaln_indices, rotary_emb


def _max_error(actual, expected):
    return (actual.float() - expected).abs().max().item()


def test_fused_deferred_residual_blocks_are_as_accurate_as_the_upstream_blocks():
    torch.manual_seed(0)
    upstream = torch.nn.ModuleList(
        minimax_h3.MiniMaxH3TransformerBlock(
            hidden_size=HIDDEN,
            num_attention_heads=4,
            attention_head_dim=128,
            ffn_dim=1024,
            time_embed_dim=64,
            norm_eps=1e-5,
            qk_norm_eps=1e-5,
        )
        for _ in range(3)
    ).to("cuda", torch.bfloat16)
    exact = copy.deepcopy(upstream).float()
    deferred = copy.deepcopy(upstream)
    for block in deferred:
        block.__class__ = xfuser_minimax_h3.xFuserMiniMaxH3TransformerBlock
    hidden_states, temb, adaln_indices, rotary_emb = _inputs()

    with torch.no_grad():
        expected = hidden_states.float()
        eager = hidden_states
        for exact_block, block in zip(exact, upstream):
            expected = exact_block(expected, temb, adaln_indices, rotary_emb)
            eager = block(eager, temb, adaln_indices, rotary_emb)
        actual = xfuser_minimax_h3.run_blocks_deferring_residual(
            deferred, hidden_states, temb, adaln_indices, rotary_emb
        )

    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual.float(), expected, rtol=3e-2, atol=3e-2)
    assert _max_error(actual, expected) <= _max_error(eager, expected)
