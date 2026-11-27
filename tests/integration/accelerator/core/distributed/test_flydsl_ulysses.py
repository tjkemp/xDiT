"""bf16 Ulysses Q/K/V exchange on AITER's FlyDSL attention all-to-all, against the RCCL path."""

import pytest

pytestmark = [pytest.mark.multi_gpu, pytest.mark.rocm]

HEADS = 56  # MiniMax-H3
HEAD_DIM = 128


def _setup(rank, world_size, init_method, inbox):
    import os

    import torch

    os.environ["XDIT_ULYSSES_A2A_FLYDSL"] = "1"
    os.environ["XDIT_FLYDSL_A2A_INBOX"] = inbox
    from xfuser.config.args import xFuserArgs
    from xfuser.core.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
        initialize_runtime_state,
    )

    torch.cuda.set_device(rank)
    init_distributed_environment(rank=rank, world_size=world_size, local_rank=rank, distributed_init_method=init_method)
    initialize_model_parallel(ulysses_degree=world_size)
    engine_config, _ = xFuserArgs(ulysses_degree=world_size).create_config()
    initialize_runtime_state(engine_config=engine_config)
    return torch.device("cuda", rank)


def _bhsd(seed, device, seq):
    """A [1, H, s, d] view of a contiguous [1, s, H, d] tensor, as MiniMax-H3 hands Q/K/V to USP."""
    import torch

    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(1, seq, HEADS, HEAD_DIM, generator=generator, device=device, dtype=torch.bfloat16)
    return x.transpose(1, 2)


def _trio(seed, device, seq):
    import torch.distributed as dist

    base = 1000 * seed + 10 * dist.get_rank()
    return [_bhsd(base + role, device, seq) for role in range(3)]


def _worker(rank, world_size, init_method, inbox):
    import torch

    from xfuser.core.distributed import flydsl_ulysses, get_runtime_state
    from yunchang.globals import PROCESS_GROUP
    from xfuser.model_executor.layers.usp import _combined_qkv_all_to_all

    device = _setup(rank, world_size, init_method, inbox)
    group = PROCESS_GROUP.ULYSSES_PG
    try:
        for seq in (256, 200):  # a local length that is not a multiple of 32 too
            q, k, v = _trio(1, device, seq)
            assert flydsl_ulysses.eligible(q, k, v, world_size)
            want = _combined_qkv_all_to_all(q, k, v)
            got = flydsl_ulysses.exchange(q, k, v, group)
            for role, (g, w) in enumerate(zip(got, want)):
                assert g.shape == w.shape, (role, g.shape, w.shape)
                assert torch.equal(g, w), f"seq={seq} role={'QKV'[role]} differs from RCCL"

        # Outputs are owned: later trios (which reuse both op buffer parities) leave them intact.
        kept = []
        for seed in range(2, 8):
            q, k, v = _trio(seed, device, 256)
            got = flydsl_ulysses.exchange(q, k, v, group)
            kept.append((got, [x.clone() for x in _combined_qkv_all_to_all(q, k, v)]))
        torch.cuda.synchronize()
        for seed, (got, want) in enumerate(kept, start=2):
            for g, w in zip(got, want):
                assert torch.equal(g, w), f"trio {seed} output was overwritten"

        # Compiled, as MiniMax-H3 runs it (fullgraph), with and without CUDA graphs.
        def block(q, k, v):
            q, k, v = flydsl_ulysses.exchange(q, k, v, group)
            return q * 2 + k - v

        def block_rccl(q, k, v):
            q, k, v = _combined_qkv_all_to_all(q, k, v)
            return q * 2 + k - v

        # Both sides compiled: Inductor fuses the pointwise tail in fp32, unlike eager.
        for mode in ("default", "reduce-overhead"):
            torch._dynamo.reset()
            compiled = torch.compile(block, fullgraph=True, mode=mode)
            compiled_rccl = torch.compile(block_rccl, fullgraph=True, mode="default")
            for seed in range(10, 14):
                q, k, v = _trio(seed, device, 256)
                got = compiled(q, k, v).clone()
                want = compiled_rccl(q, k, v)
                assert torch.equal(got, want), f"compiled mode={mode} seed={seed} differs from RCCL"
    finally:
        torch.cuda.synchronize()
        get_runtime_state().destroy_distributed_env()


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("inbox", ["uncached", "finegrained"])
def test_flydsl_ulysses_matches_rccl(accelerator_ranks, world_size, inbox):
    pytest.importorskip("aiter.ops.flydsl.attention_a2a_intranode")
    if inbox == "uncached":
        pytest.importorskip("mori.shmem")
    accelerator_ranks(_worker, world_size=world_size, args=(inbox,))
