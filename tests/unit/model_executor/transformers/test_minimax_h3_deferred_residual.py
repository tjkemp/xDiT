import copy

import pytest
import torch

minimax_h3 = pytest.importorskip(
    "diffusers.models.transformers.transformer_minimax_h3",
    reason="installed diffusers does not include MiniMax-H3",
)

from xfuser.model_executor.models.transformers import transformer_minimax_h3 as xfuser_minimax_h3  # noqa: E402

TIMESTEPS = 2
SEQ_LEN = 11


def _blocks(count):
    torch.manual_seed(0)
    return torch.nn.ModuleList(
        minimax_h3.MiniMaxH3TransformerBlock(
            hidden_size=64,
            num_attention_heads=2,
            attention_head_dim=32,
            ffn_dim=128,
            time_embed_dim=16,
            norm_eps=1e-5,
            qk_norm_eps=1e-5,
        )
        for _ in range(count)
    ).eval()


def _inputs():
    generator = torch.Generator().manual_seed(1)
    hidden_states = torch.randn(1, SEQ_LEN, 64, generator=generator)
    temb = torch.randn(TIMESTEPS, 16, generator=generator)
    adaln_indices = torch.randint(0, TIMESTEPS * minimax_h3.MINIMAX_H3_MODALITY_NUM, (SEQ_LEN,), generator=generator)
    position_ids = torch.randint(0, 5, (SEQ_LEN, 3), generator=generator).float()
    rotary_emb = minimax_h3.MiniMaxH3RotaryPosEmbed(rope_freq_dim=4)(position_ids)
    return hidden_states, temb, adaln_indices, rotary_emb


@pytest.mark.parametrize("count", [1, 3])
def test_deferred_residual_blocks_match_the_upstream_blocks(count):
    upstream = _blocks(count)
    deferred = copy.deepcopy(upstream)
    calls = []
    for block in deferred:
        block.__class__ = xfuser_minimax_h3.xFuserMiniMaxH3TransformerBlock
        block.register_forward_pre_hook(lambda module, args: calls.append(module))
    hidden_states, temb, adaln_indices, rotary_emb = _inputs()

    with torch.no_grad():
        expected = hidden_states
        for block in upstream:
            expected = block(expected, temb, adaln_indices, rotary_emb)
        actual = xfuser_minimax_h3.run_blocks_deferring_residual(
            deferred, hidden_states, temb, adaln_indices, rotary_emb
        )

    assert torch.equal(actual, expected)
    assert calls == list(deferred)


def test_block_without_deferral_is_the_upstream_block():
    upstream = _blocks(1)[0]
    block = copy.deepcopy(upstream)
    block.__class__ = xfuser_minimax_h3.xFuserMiniMaxH3TransformerBlock
    inputs = _inputs()

    with torch.no_grad():
        assert torch.equal(block(*inputs), upstream(*inputs))
