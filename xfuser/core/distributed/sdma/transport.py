"""Collective data movement on the SDMA copy engines (ROCm), over MoRI's anvil queues.

RCCL moves data with kernels on the compute units, and a kernel that needs the data waits for
all of it. This transport moves it with the copy engines instead: each rank pushes into its
peers' buffers, one SDMA queue per peer (on the engine KFD recommends for the link), all peers
at once, and uses no compute units, so compute that does not need the data runs alongside.

`PeerGroup` is one process group's transport, created collectively (`group_for`):
- Buffers that peers write into (`SharedBuffer`, `PeerGroup.buffer`) are allocated by the
  library with hipMalloc and mapped into every peer with HIP IPC handles, exchanged over the
  group's device process group.
- A route (`PeerGroup.route`) is the copies of one send to every peer, registered once as
  device tables and sent per call with that call's source address. A copy is (source offset,
  destination buffer, destination offset, bytes), or 2D: (source offset, destination buffer,
  destination offset, width, height, source pitch, destination pitch), `height` rows of
  `width` bytes. Routes are cached by key and rebuilt when a buffer they name was re-created,
  so callers never see addresses.
- A route may raise a signal on a channel ([0, channels)) behind its copies at every peer.
  A signal is a flag word: channel c holds one word per sender at each rank, in uncached
  memory, so a kernel polling them sees the engines' writes.
- A call (`PeerGroup.begin`) sends routes and waits for channels. `Call.send` is one kernel
  on the sending stream that writes each peer's copy packets, and a FENCE packet setting the
  flag, into the peer's queue (transport.hip). The engine runs a queue in order, so a channel's
  signal lands at a peer only after every copy this rank sent to that peer since, flagged or
  not (tests/integration/accelerator/core/distributed/test_sdma_ulysses.py checks every
  element each call reads). `Call.wait` makes a stream wait for every peer's flag word on a
  channel with a one-thread kernel, which resets them for their next use. This rank's own data
  is copied on the stream by the caller, so it is not waited for.

A route's copies read their source once the sending stream reaches the send; the caller keeps
the source alive until the peers have it. Nothing is held per call: the queues are rings that
the engines drain, and the caller's protocol says when a source or buffer can be reused.

Waits are bounded. After XDIT_SDMA_WAIT_S seconds (default 30) the wait kernel records which
flags never landed. By default (`abort`) the kernel keeps the stream held, so nothing runs on
data that never came, and a watchdog thread logs what was missing and aborts the process, as
NCCL's async error handling does. Tests can call `set_on_timeout("raise")`: the stream then
moves on (that call's output is wrong) and the next call on this process raises an error
naming the missing flags.

The library is loaded with ctypes, from the first of: XDIT_SDMA_LIB; a library built into this
package's directory from the current source (`python xfuser/core/distributed/sdma/build.py
--out <this directory>`, for image builds: no GPU and no xfuser import needed); a library built
with hipcc for this GPU on first use and cached. Either way it is built against MoRI
(build.py `mori_paths`). The transport runs on the GPU architectures in VALIDATED_ARCHS; others
need XDIT_SDMA_ALLOW_UNTESTED=1. ROCm only, within one node.
"""

import ctypes
import fcntl
import hashlib
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Hashable, List, Optional, Tuple

import torch
import torch.distributed as dist

import xfuser.envs as envs
from xfuser.core.distributed.sdma.build import LIB_NAME, MORI_MISSING, SOURCE, build, mori_paths, source_hash
from xfuser.logger import init_logger

logger = init_logger(__name__)

# GPU architectures the transport is validated on; others need XDIT_SDMA_ALLOW_UNTESTED=1.
VALIDATED_ARCHS = ("gfx942", "gfx950")
_CHANNELS = 64
MAX_FLAGS_PER_WAIT = 64

_lib = None
_lib_path = None
_groups: Dict[Tuple[int, ...], "PeerGroup"] = {}
_by_process_group: Dict[dist.ProcessGroup, "PeerGroup"] = {}
_call = 0
# Test hook (debug_drop_flag): call id -> (send index, peer index) whose flag write is dropped.
_drops: Dict[int, Tuple[int, int]] = {}
_watchdog = None
_init_lock = threading.Lock()


def _check(rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{what} failed with {rc}")


def _env(name: str) -> str:
    return envs.environment_variables[name]().strip()


def wait_seconds() -> float:
    return float(_env("XDIT_SDMA_WAIT_S"))


_on_timeout = "abort"


def on_timeout() -> str:
    """What a flag wait that gives up does: abort the process, or let the next call raise."""
    return _on_timeout


def set_on_timeout(mode: str) -> str:
    """Set `on_timeout` to "abort" (the default) or "raise" (tests); returns the previous mode.
    Set it before the transport's first use in the process: abort mode starts a watchdog."""
    global _on_timeout
    if mode not in ("abort", "raise"):
        raise ValueError(f"on_timeout must be abort or raise; got {mode!r}")
    previous, _on_timeout = _on_timeout, mode
    return previous


def _arch() -> str:
    return torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(":")[0]


def _packaged() -> Optional[Path]:
    """The library built into this package's directory, if it matches the source."""
    packaged = SOURCE.with_name(LIB_NAME)
    stamp = SOURCE.with_name(LIB_NAME + ".source-sha1")
    if packaged.is_file() and stamp.is_file() and stamp.read_text().strip() == source_hash():
        return packaged
    return None


def available() -> bool:
    """Whether this process can use the transport: a ROCm build with a GPU, and the library
    prebuilt (XDIT_SDMA_LIB or packaged) or MoRI to build it against."""
    if torch.version.hip is None or not torch.cuda.is_available():
        return False
    return bool(_env("XDIT_SDMA_LIB")) or _packaged() is not None or mori_paths() is not None


def check_available() -> None:
    """Raise a RuntimeError saying what to install if the library can be neither found nor built."""
    if not _env("XDIT_SDMA_LIB") and _packaged() is None and mori_paths() is None:
        raise RuntimeError(MORI_MISSING)


def _library_path() -> Path:
    """XDIT_SDMA_LIB if set; else the packaged library; else one built for this GPU, cached by
    source, arch, HIP version and MoRI location."""
    prebuilt = _env("XDIT_SDMA_LIB")
    if prebuilt:
        if not Path(prebuilt).is_file():
            raise RuntimeError(f"XDIT_SDMA_LIB={prebuilt} is not a file")
        return Path(prebuilt)
    packaged = _packaged()
    if packaged is not None:
        return packaged
    check_available()
    arch = _arch()
    stamp = f" {arch} {torch.version.hip} {mori_paths()[1]}"
    key = hashlib.sha1(SOURCE.read_bytes() + stamp.encode()).hexdigest()[:12]
    cache = Path(os.environ.get("XDIT_CACHE_DIR", Path.home() / ".cache" / "xfuser")) / f"sdma_{key}"
    so = cache / LIB_NAME
    if not so.exists():
        cache.mkdir(parents=True, exist_ok=True)
        with open(cache / "build.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not so.exists():
                build(cache, [arch])
    return so


def load():
    """Load the transport library (prebuilt or built on first use)."""
    global _lib, _lib_path
    if _lib is not None:
        return _lib
    _lib_path = _library_path()
    lib = ctypes.CDLL(str(_lib_path))
    i, u64, vp, dbl = ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_double
    ip, u64p, u32p = ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint32)
    for name, argtypes in {
        "sx_kfd_node": [i],
        "sx_init": [i],
        "sx_queue": [i, i, u64p],
        "sx_malloc": [u64, i, u64p],
        "sx_free": [u64],
        "sx_ipc_handle_size": [],
        "sx_ipc_get": [u64, vp],
        "sx_ipc_open": [vp, u64p],
        "sx_ipc_close": [u64],
        "sx_route_create": [i, u64p, ip, u64p, u64p, u64p, u64p, u32p, u32p, u32p, u32p],
        "sx_route_free": [i],
        "sx_send": [vp, i, u64, i],
        "sx_wait_flags": [vp, u64, i, i, dbl, i, i, i],
        "sx_take_error": [ip, ip, u64p],
        "sx_peek_error": [ip, ip, u64p],
    }.items():
        getattr(lib, name).argtypes = argtypes
        getattr(lib, name).restype = i
    _lib = lib
    return lib


class _CudaArray:
    """Exposes raw device memory to torch.as_tensor through __cuda_array_interface__."""

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 2,
            "strides": None,
        }


def device_bytes(ptr: int, nbytes: int, device: torch.device) -> torch.Tensor:
    """A uint8 tensor viewing `nbytes` of device memory at `ptr` (not owned by torch)."""
    return torch.as_tensor(_CudaArray(ptr, nbytes), device=device)


class SharedBuffer:
    """Device memory of the same size on every rank of a group, mapped into every peer.

    `local` is this rank's buffer as a uint8 tensor; `peer_bases` holds the peers' base
    addresses in this process, in `PeerGroup.peers` order. `uncached` memory is for flag
    words. Collective: every rank of the group creates (and closes) it together.
    """

    def __init__(self, group: "PeerGroup", nbytes: int, uncached: bool = False):
        lib = load()
        self.nbytes = nbytes
        # False once released: routes naming it are rebuilt.
        self.live = True
        ptr = ctypes.c_uint64()
        _check(lib.sx_malloc(nbytes, int(uncached), ctypes.byref(ptr)), f"sx_malloc({nbytes}, uncached={uncached})")
        self.ptr = ptr.value
        size = lib.sx_ipc_handle_size()
        handle = (ctypes.c_uint8 * size)()
        _check(lib.sx_ipc_get(self.ptr, handle), "sx_ipc_get")
        mine = torch.tensor(list(handle), dtype=torch.uint8, device=group.device)
        every = torch.empty(group.world * size, dtype=torch.uint8, device=group.device)
        dist.all_gather_into_tensor(every, mine, group=group.process_group)
        every = every.view(group.world, size).cpu()
        self.peer_bases = []
        for p in group.peers:
            theirs = (ctypes.c_uint8 * size)(*every[p].tolist())
            out = ctypes.c_uint64()
            _check(lib.sx_ipc_open(theirs, ctypes.byref(out)), f"sx_ipc_open (peer {p})")
            self.peer_bases.append(out.value)
        self.local = device_bytes(self.ptr, nbytes, group.device)

    def unmap(self) -> None:
        """Unmap the peers' buffers; every rank must be past its last use of them."""
        self.live = False
        for p in self.peer_bases:
            load().sx_ipc_close(p)
        self.peer_bases = []

    def free(self) -> None:
        """Free this rank's buffer; every peer must have unmapped it: a buffer freed while a peer
        still maps it leaves later allocations that HIP cannot export."""
        self.local = None
        load().sx_free(self.ptr)


class PeerGroup:
    """Per process group: the peers in push order with one SDMA queue each, named shared buffers
    that grow on demand, and the routes registered on them."""

    channels = _CHANNELS

    def __init__(self, process_group: dist.ProcessGroup):
        check_available()
        lib = load()
        self.process_group = process_group
        self.rank, self.world = process_group.rank(), process_group.size()
        self.device = torch.device("cuda", torch.cuda.current_device())
        arch = _arch()
        if arch not in VALIDATED_ARCHS and _env("XDIT_SDMA_ALLOW_UNTESTED") != "1":
            raise RuntimeError(
                f"the SDMA transport is validated on {', '.join(VALIDATED_ARCHS)}, not {arch}; set "
                "XDIT_SDMA_ALLOW_UNTESTED=1 to run it anyway"
            )
        rc = lib.sx_init(self.device.index)
        if rc == -11:
            raise RuntimeError(f"{_lib_path} has no GPU code for {arch}; rebuild it with --offload-arch {arch}")
        _check(rc, "sx_init")
        node = torch.tensor([lib.sx_kfd_node(self.device.index)], dtype=torch.int32, device=self.device)
        nodes = torch.empty(self.world, dtype=torch.int32, device=self.device)
        dist.all_gather_into_tensor(nodes, node, group=process_group)
        nodes = nodes.tolist()
        if len(set(nodes)) != self.world:
            raise RuntimeError(f"the SDMA transport needs one GPU per rank within a node; KFD nodes {nodes}")
        # Peers in step order from this rank, so at each step every rank sends to a different peer.
        self.peers = [(self.rank + step) % self.world for step in range(1, self.world)]
        queues = []
        for p in self.peers:
            out = ctypes.c_uint64()
            _check(lib.sx_queue(nodes[self.rank], nodes[p], ctypes.byref(out)), f"sx_queue (peer {p})")
            queues.append(out.value)
        self.queues = (ctypes.c_uint64 * len(queues))(*queues)
        self._buffers: Dict[str, SharedBuffer] = {}
        self._routes: Dict[tuple, Route] = {}
        # Channel c's words at each rank: [c * world, (c + 1) * world), word p is rank p's.
        self._flags = self._buffer("signals", self.channels * self.world * 4, uncached=True)
        _start_watchdog(lib)
        logger.info(f"SDMA transport for group ranks {dist.get_process_group_ranks(process_group)}")

    def barrier(self) -> None:
        """Every rank's device work, and so every copy it waited for, is done past this point."""
        torch.cuda.synchronize()
        dist.barrier(group=self.process_group)
        torch.cuda.synchronize()

    def buffer(self, name: str, nbytes: int) -> SharedBuffer:
        """The shared buffer `name`, at least `nbytes`; collective when it must be (re)created."""
        return self._buffer(name, nbytes)

    def _buffer(self, name: str, nbytes: int, uncached: bool = False) -> SharedBuffer:
        buf = self._buffers.get(name)
        if buf is not None and buf.nbytes >= nbytes:
            return buf
        if buf is not None:
            self._release([buf])
        buf = SharedBuffer(self, nbytes, uncached)
        self._buffers[name] = buf
        self.barrier()
        return buf

    def route(self, key: Hashable, build, signal: Optional[int]) -> "Route":
        """The route `key`: `build()` gives each peer's copies in `peers` order, called only when
        the route is not cached or names a re-created buffer. With `signal`, a channel id, the
        route raises this rank's signal on that channel at each peer behind its copies."""
        cached = self._routes.get((key, signal))
        if cached is not None and all(buf.live for buf in cached.buffers):
            return cached
        if cached is not None:
            cached.free()
        copies = build()
        if len(copies) != len(self.peers):
            raise ValueError(f"a route needs the copies of each of the {len(self.peers)} peers")
        flags = None
        if signal is not None:
            if not 0 <= signal < self.channels:
                raise ValueError(f"channel {signal} is not in [0, {self.channels})")
            word = (signal * self.world + self.rank) * 4
            flags = [base + word for base in self._flags.peer_bases]
        route = Route(self, copies, flags)
        self._routes[(key, signal)] = route
        return route

    def begin(self) -> "Call":
        """Open a call."""
        return Call(self)

    def _release(self, buffers) -> None:
        # Peers may write into a buffer until every rank is past its last use, and its owner may
        # free it only once every peer has unmapped it.
        self.barrier()
        for buf in buffers:
            buf.unmap()
        self.barrier()
        for buf in buffers:
            buf.free()

    def close(self) -> None:
        for route in self._routes.values():
            route.free()
        self._routes = {}
        self._release(list(self._buffers.values()))
        self._buffers = {}


def _u32(value: int, what: str) -> int:
    if not 0 < value < 1 << 32:
        raise ValueError(f"a 2D copy's {what} must be in [1, 2**32); got {value}")
    return value


class Route:
    """A route's copies as device tables, sent per call with that call's source; `flags[i]` is
    the address of the flag word set behind peer i's copies, or `flags` is None."""

    def __init__(self, g: PeerGroup, copies, flags: Optional[List[int]]):
        # The buffers the copies write into.
        self.buffers = list({id(c[1]): c[1] for peer in copies for c in peer}.values())
        # Per copy: source offset, destination address, bytes, width, height, pitches.
        rows = []
        for i, peer in enumerate(copies):
            for c in peer:
                dst = c[1].peer_bases[i] + c[2]
                if len(c) == 4:
                    rows.append((c[0], dst, c[3], 0, 0, 0, 0))
                else:
                    width, height, src_pitch, dst_pitch = c[3:]
                    sizes = (_u32(width, "width"), _u32(height, "height"))
                    rows.append((c[0], dst, 0, *sizes, _u32(src_pitch, "pitch"), _u32(dst_pitch, "pitch")))
        n, m = len(copies), max(1, len(rows))
        columns = list(zip(*rows)) or [()] * 7
        u64, u32 = ctypes.c_uint64 * m, ctypes.c_uint32 * m
        self.id = load().sx_route_create(
            n,
            g.queues,
            (ctypes.c_int * n)(*[len(peer) for peer in copies]),
            None if flags is None else (ctypes.c_uint64 * n)(*flags),
            *[u64(*col) for col in columns[:3]],
            *[u32(*col) for col in columns[3:]],
        )
        if self.id < 0:
            raise RuntimeError(f"sx_route_create failed with {self.id}")

    def free(self) -> None:
        if self.id >= 0:
            load().sx_route_free(self.id)
            self.id = -1


def group_for(process_group: dist.ProcessGroup) -> PeerGroup:
    """The PeerGroup of `process_group`, created (collectively) on first use."""
    g = _by_process_group.get(process_group)
    if g is not None:
        return g
    key = tuple(dist.get_process_group_ranks(process_group))
    with _init_lock:
        if key not in _groups:
            _groups[key] = PeerGroup(process_group)
        _by_process_group[process_group] = _groups[key]
        return _groups[key]


def _timeout_message(count: int, tag: int, call: int, missing: int) -> str:
    words = [i for i in range(MAX_FLAGS_PER_WAIT) if missing >> i & 1]
    rank = dist.get_rank() if dist.is_initialized() else "?"
    return (
        f"rank {rank}: {count} SDMA flag wait(s) gave up after {wait_seconds():g} s. First: call {call}, wait "
        f"{tag}, flag words {words} never set (word r is group rank r's). Groups: {list(_groups)}"
    )


def _start_watchdog(lib) -> None:
    """In abort mode, a thread that aborts the process once a flag wait has given up."""
    global _watchdog
    if _watchdog is not None or on_timeout() != "abort":
        return

    def watch():
        tag, call, missing = ctypes.c_int(), ctypes.c_int(), ctypes.c_uint64()
        while True:
            count = lib.sx_peek_error(ctypes.byref(tag), ctypes.byref(call), ctypes.byref(missing))
            if count:
                message = _timeout_message(count, tag.value, call.value, missing.value) + "; aborting"
                logger.error(message)
                print(message, file=sys.stderr, flush=True)
                os.abort()
            time.sleep(0.1)

    _watchdog = threading.Thread(target=watch, name="xfuser-sdma-watchdog", daemon=True)
    _watchdog.start()


def raise_if_waits_gave_up() -> None:
    """Raise if a bounded flag wait of an earlier call on this process ran out."""
    tag, call, missing = ctypes.c_int(), ctypes.c_int(), ctypes.c_uint64()
    count = load().sx_take_error(ctypes.byref(tag), ctypes.byref(call), ctypes.byref(missing))
    if count:
        raise RuntimeError(_timeout_message(count, tag.value, call.value, missing.value))


class Call:
    """One call: a send is one post kernel, and each wait is tagged with its order in the call,
    for the timeout report. Nothing is held per call."""

    def __init__(self, g: PeerGroup):
        global _call
        if on_timeout() == "raise":
            raise_if_waits_gave_up()
        self._g = g
        _call += 1
        self.id = _call
        self._sends = 0
        self._waits = 0

    def send(self, route: Route, src: torch.Tensor, stream: torch.cuda.Stream) -> None:
        """Start `route`'s copies from `src` once `stream` reaches this point."""
        drop = -1
        planned = _drops.get(self.id)
        if planned is not None and planned[0] == self._sends:
            drop = _drops.pop(self.id)[1]
        _check(load().sx_send(ctypes.c_void_p(stream.cuda_stream), route.id, src.data_ptr(), drop), "sx_send")
        self._sends += 1

    def wait(self, channel: int, stream: torch.cuda.Stream) -> None:
        """Make `stream` wait (bounded) until every peer's latest signal on `channel` has landed."""
        g = self._g
        if g.world > MAX_FLAGS_PER_WAIT:
            raise ValueError(f"at most {MAX_FLAGS_PER_WAIT} ranks per group; got {g.world}")
        rc = load().sx_wait_flags(
            ctypes.c_void_p(stream.cuda_stream),
            g._flags.ptr + channel * g.world * 4,
            g.world,
            g.rank,
            wait_seconds(),
            self._waits,
            self.id,
            int(on_timeout() == "abort"),
        )
        _check(rc, "sx_wait_flags")
        self._waits += 1


def debug_drop_flag(ahead: int, issue: int, peer_index: int) -> None:
    """Test hook: in the `ahead`-th call started from now, the `issue`-th send (from 0) loses its
    flag write to peers[peer_index]."""
    _drops[_call + ahead] = (issue, peer_index)
