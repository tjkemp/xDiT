"""Lossless bf16 Ulysses Q/K/V all-to-all over AITER's FlyDSL ``AttentionA2AIntraNodeOp``.

Opt-in with ``XDIT_ULYSSES_A2A_FLYDSL=1``. Only the pre-attention Q/K/V exchange moves here;
the post-attention output exchange stays on RCCL. The op pushes each rank's head slices
straight into its peers' receive buffers ("inboxes") with peer-to-peer stores, so it needs no
layout copy before the exchange. ``XDIT_FLYDSL_A2A_INBOX`` selects the inbox memory:

- ``finegrained`` (default; also what ``auto`` means here): coherent memory that the L2 may
  cache. Plain HIP IPC buffers, no MoRI.
- ``uncached``: bypasses the L2. Lives in the MoRI symmetric heap (needs ``mori.shmem``).

On MI350P (PCIe), MiniMax-H3 shape: finegrained is as fast as uncached at 2 ranks and 16x faster
at 4. Uncached at 2 ranks also made full MiniMax-H3 runs nondeterministic (outputs differed run to
run, PSNR ~45 dB) although the exchange alone matched RCCL bit for bit over hundreds of trios;
finegrained runs match RCCL exactly. AITER's own ``auto`` (uncached at 2 ranks) is not used.

The op's outputs are double-buffered and overwritten by peers two trios later, so each
output is copied into a tensor the caller owns. That copy writes the contiguous BSHD layout
the attention backends would otherwise build themselves.
"""

import functools
import os

import torch
import torch.distributed as dist

import xfuser.envs as envs
from xfuser.logger import init_logger

logger = init_logger(__name__)

SUPPORTED_WORLD_SIZES = (2, 4)
_HEAD_DIM = 128

_OPS = {}
_SIDE_STREAMS = {}
_CPU_GROUPS = {}


def enabled() -> bool:
    return str(envs.XDIT_ULYSSES_A2A_FLYDSL).strip().lower() in ("1", "true", "yes", "on")


def _inbox_setting() -> str:
    inbox = str(envs.XDIT_FLYDSL_A2A_INBOX).strip().lower()
    if inbox not in ("auto", "uncached", "finegrained"):
        raise ValueError(f"XDIT_FLYDSL_A2A_INBOX must be auto, uncached or finegrained; got {inbox!r}")
    return inbox


def eligible(query, key, value, world_size: int) -> bool:
    """Whether a ``[b, H, S_local, d]`` Q/K/V trio can take the FlyDSL exchange."""
    return (
        enabled()
        and world_size in SUPPORTED_WORLD_SIZES
        and query.shape == key.shape == value.shape
        and query.dim() == 4
        and query.shape[0] == 1
        and query.shape[3] == _HEAD_DIM
        and query.shape[1] % world_size == 0
        and query.dtype == key.dtype == value.dtype == torch.bfloat16
        and query.is_cuda
    )


@functools.lru_cache(maxsize=None)
def _resolved_inbox(setting: str, world_size: int) -> str:
    from aiter.ops.flydsl.comm_collectives_host_shared import resolve_inbox_flags

    return resolve_inbox_flags("finegrained" if setting == "auto" else setting, world_size)[1]


def _ensure_mori_heap(nbytes: int) -> None:
    if "MORI_SHMEM_HEAP_SIZE" in os.environ:
        return
    # Room for this op's buffers plus the MoRI bookkeeping, rounded up to whole GiB.
    gib = -(-int(nbytes * 1.25 + (512 << 20)) // (1 << 30))
    os.environ["MORI_SHMEM_HEAP_SIZE"] = f"{max(gib, 4)}G"


def _cpu_group(group, ranks, inbox):
    if inbox == "uncached":
        from xfuser.core.distributed import attention_a2a

        attention_a2a._init_mori(group, ranks)
        return attention_a2a._MORI_CPU_GROUP
    cpu_group = _CPU_GROUPS.get(ranks)
    if cpu_group is None:
        cpu_group = dist.new_group(ranks=list(ranks), backend="gloo", use_local_synchronization=True)
        _CPU_GROUPS[ranks] = cpu_group
    return cpu_group


def _get_op(group, shape, device):
    from aiter.ops.flydsl.attention_a2a_intranode import AttentionA2AIntraNodeOp
    ranks = tuple(dist.get_process_group_ranks(group))
    world_size = len(ranks)
    inbox = _resolved_inbox(_inbox_setting(), world_size)
    key = (ranks, device.index, tuple(shape), inbox)
    op = _OPS.get(key)
    if op is not None:
        return op
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "FlyDSL Ulysses A2A: the op must be built before CUDA graph capture (its construction is a collective)"
        )
    if inbox == "uncached":
        _ensure_mori_heap(6 * shape[1] * shape[2] * shape[3] * 2)
    op = AttentionA2AIntraNodeOp(
        rank=dist.get_rank(group),
        world_size=world_size,
        shape=tuple(shape),
        group=_cpu_group(group, ranks, inbox),
        quant=None,
        return_packed=False,
        inbox_memory=inbox,
    )
    _OPS[key] = op
    logger.info(f"FlyDSL Ulysses A2A: bf16 op for local shape {tuple(shape)}, world {world_size}, inbox {inbox}")
    return op


def _side_stream(device):
    # The op requires every trio on one stream; a dedicated stream keeps that true whatever
    # stream the caller (eager, Inductor, CUDA-graph partitions) happens to run on.
    stream = _SIDE_STREAMS.get(device.index)
    if stream is None:
        stream = torch.cuda.Stream(device=device)
        _SIDE_STREAMS[device.index] = stream
    return stream


def _all_to_all_rccl(q, k, v, group):
    """Same result as the FlyDSL exchange over RCCL, for use under CUDA graph capture."""
    world_size = dist.get_world_size(group)
    b, s, h, d = q.shape
    qkv = torch.stack([q, k, v], dim=0).view(3, b, s, world_size, h // world_size, d)
    send = qkv.permute(3, 0, 1, 2, 4, 5).contiguous()
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    # [P, 3, b, s, h/P, d] -> [3, b, P*s, h/P, d]
    out = recv.permute(1, 2, 0, 3, 4, 5).reshape(3, b, world_size * s, h // world_size, d)
    return [t.contiguous() for t in out.unbind(0)]


@torch.library.custom_op(
    "xfuser::flydsl_ulysses_qkv",
    mutates_args=(),
    tags=(torch.Tag.cudagraph_unsafe,) if hasattr(torch.Tag, "cudagraph_unsafe") else (),
)
def _flydsl_ulysses_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, group_name: str) -> list[torch.Tensor]:
    """``[1, S_local, H, d]`` Q/K/V (BSHD) -> three ``[1, S, H/P, d]`` contiguous BSHD tensors."""
    group = dist.distributed_c10d._resolve_process_group(group_name)
    if torch.cuda.is_current_stream_capturing():
        return _all_to_all_rccl(q, k, v, group)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    world_size = dist.get_world_size(group)
    b, s, h, d = q.shape
    op = _get_op(group, q.shape, q.device)
    outs = [q.new_empty(b, s * world_size, h // world_size, d) for _ in range(3)]
    main = torch.cuda.current_stream(q.device)
    side = _side_stream(q.device)
    side.wait_stream(main)
    with torch.cuda.stream(side):
        op.submit_role(0, q)
        op.submit_role(1, k)
        received = op.submit_role(2, v)
        for out, buf in zip(outs, received):
            out.copy_(buf.view(b, h // world_size, s * world_size, d).transpose(1, 2))
    # Inputs and outputs were allocated on main; main waiting here orders their reuse after the op.
    main.wait_stream(side)
    return outs


@_flydsl_ulysses_qkv.register_fake
def _(q, k, v, group_name):
    world_size = dist.get_world_size(dist.distributed_c10d._resolve_process_group(group_name))
    b, s, h, d = q.shape
    return [q.new_empty(b, s * world_size, h // world_size, d) for _ in range(3)]


def exchange(query, key, value, group):
    """Ulysses input exchange: ``[1, H, S_local, d]`` Q/K/V -> ``[1, H/P, S, d]``, like
    ``usp._combined_qkv_all_to_all`` (rank r gets heads ``[r*H/P, (r+1)*H/P)``)."""
    outs = torch.ops.xfuser.flydsl_ulysses_qkv(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), group.group_name
    )
    return tuple(t.transpose(1, 2) for t in outs)


def shutdown() -> None:
    """Collectively close every op. Call on all ranks, before MoRI is finalized."""
    if not (_OPS or _CPU_GROUPS):
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        for op in _OPS.values():
            op.close()
    finally:
        _OPS.clear()
        _SIDE_STREAMS.clear()
        for cpu_group in _CPU_GROUPS.values():
            if dist.is_initialized():
                dist.destroy_process_group(cpu_group)
        _CPU_GROUPS.clear()
