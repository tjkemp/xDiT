"""All-to-alls on the SDMA copy engines, with MoRI's `All2allSdma` (ROCm, one node).

RCCL moves data with kernels on the compute units. MoRI's SDMA all-to-all (`mori.ccl`) moves it
with the copy engines instead, so compute that does not need the data runs alongside: ulysses.py
starts a head group's all-to-all as soon as the group is packed and waits for it only right
before the group's attention.

`Group` is one process group's all-to-alls, created collectively (`group_for`): MoRI shmem is
initialised once per process, on that group, so a process takes part in one SDMA group. Each
all-to-all in flight has a handle of its own (`Group.handle`, keyed by the caller), because a
handle runs one all-to-all at a time; a handle holds MoRI's output buffer, which `wait` returns
a view of until the handle is next used.

MoRI's wait blocks the host until the stream has received the data (`All2allSdma.wait_async`).
"""

import atexit
import os
from typing import Dict, Hashable, Optional

import torch
import torch.distributed as dist

from xfuser.logger import init_logger

logger = init_logger(__name__)

MORI_MISSING = (
    "--ulysses_a2a sdma needs MoRI (https://github.com/ROCm/mori), and it was not found. Install it, "
    "e.g. `pip install git+https://github.com/ROCm/mori`. On TheRock ROCm, MoRI must link ROCm's libdrm "
    "(rocm_sysdeps): build it with LDFLAGS pointing at a directory where libdrm.so and libdrm_amdgpu.so "
    "are those libraries."
)

_group: Optional["Group"] = None


def available() -> bool:
    """Whether this process can use SDMA all-to-alls: a ROCm GPU and MoRI."""
    if torch.version.hip is None or not torch.cuda.is_available():
        return False
    try:
        import mori.ccl  # noqa: F401
    except ImportError:
        return False
    return True


def check_available() -> None:
    """Raise a RuntimeError saying what to install if MoRI cannot be imported."""
    try:
        import mori.ccl  # noqa: F401
    except ImportError as e:
        raise RuntimeError(MORI_MISSING) from e


class Group:
    """MoRI SDMA all-to-alls over one process group (collective to create)."""

    def __init__(self, process_group: dist.ProcessGroup):
        check_available()
        import mori.shmem as shmem

        # MoRI reads this when shmem is initialised and when a handle is made.
        os.environ.setdefault("MORI_ENABLE_SDMA", "1")
        self.process_group = process_group
        self.rank, self.world = dist.get_rank(process_group), process_group.size()
        # As shmem_torch_process_group_init, but broadcasting the unique id from the group's first
        # rank: MoRI's broadcasts from global rank 0, which a sub-group (CFG parallel) may not hold.
        shmem.api._ensure_shmem_module()
        uid = [shmem.shmem_get_unique_id() if self.rank == 0 else None]
        dist.broadcast_object_list(uid, group_src=0, group=process_group)
        rc = shmem.shmem_init_attr(shmem.MORI_SHMEM_INIT_WITH_UNIQUEID, self.rank, self.world, uid[0])
        if rc != 0:
            raise RuntimeError(f"MoRI shmem init failed with {rc}")
        self._handles: Dict[Hashable, tuple] = {}
        logger.info(f"MoRI SDMA all-to-all for group ranks {dist.get_process_group_ranks(process_group)}")

    def handle(self, key: Hashable, nbytes: int):
        """The handle `key` for all-to-alls of `nbytes` per rank (collective when it must grow)."""
        from mori.ccl import All2allSdma

        held = self._handles.get(key)
        if held is None or held[1] < nbytes:
            if held is not None:
                torch.cuda.synchronize()
            # copy_output_to_user=False: `wait` returns a view of MoRI's output buffer.
            h = All2allSdma(self.rank, self.world, nbytes, nbytes, copy_output_to_user=False)
            held = self._handles[key] = (h, nbytes)
        return held[0]

    def start(self, key: Hashable, x: torch.Tensor) -> None:
        """Start the all-to-all of `x` (contiguous, one equal chunk per rank) on handle `key`, once the
        current stream gets here. `x` is read until `wait`."""
        nbytes = x.numel() * x.element_size()
        if nbytes % (4 * self.world):
            raise ValueError(f"an SDMA all-to-all moves whole 4-byte words per rank; got {nbytes} bytes")
        h = self.handle(key, nbytes)
        h.start_async(x.view(torch.uint8), x.view(torch.uint8), nbytes // 4 // self.world, torch.cuda.current_stream())

    def wait(self, key: Hashable, like: torch.Tensor) -> torch.Tensor:
        """Wait for handle `key`'s all-to-all; returns what it received, shaped and typed as `like`."""
        h = self.handle(key, 0)
        h.wait_async(torch.cuda.current_stream())
        # As uint32: MoRI (b80cbe4) sizes a view of a dtype it does not map, such as uint8, in bytes
        # but types it uint32, four times the buffer.
        buf = h.get_output_transit_buffer(dtype=torch.uint32, device=like.device).view(torch.uint8)
        return buf[: like.numel() * like.element_size()].view(like.dtype).view(like.shape)

    def close(self) -> None:
        # Handles free shmem memory: they must go before shmem is finalised.
        torch.cuda.synchronize()
        self._handles.clear()
        import mori.shmem as shmem

        shmem.shmem_finalize()


def group_for(process_group: dist.ProcessGroup) -> Group:
    """The SDMA group of `process_group`, created (collectively) on first use."""
    global _group
    if _group is None:
        _group = Group(process_group)
        atexit.register(_close)
    elif _group.process_group is not process_group and dist.get_process_group_ranks(
        _group.process_group
    ) != dist.get_process_group_ranks(process_group):
        raise RuntimeError("MoRI SDMA all-to-alls serve one process group per process")
    return _group


def _close() -> None:
    global _group
    if _group is not None:
        _group.close()
        _group = None
