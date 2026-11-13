"""Ulysses attention with its all-to-alls on the SDMA engines, pipelined over head groups.

Ulysses sends each rank's Q/K/V for head block p to rank p, which then attends over the whole
sequence for its heads and sends each rank's rows back. This does both exchanges with
all-to-alls on the copy engines (transport.py, MoRI's `All2allSdma`) and splits each rank's
heads into head groups: group g's all-to-all starts as soon as the group is packed, its
attention once its data has landed, while later groups are still moving, and its output goes
back as soon as it is done, while later groups attend. With `rccl` (and while a CUDA graph is
captured) the same calls run their all-to-alls with RCCL instead.

What one call exchanges is an `Exchange`: tensors with their own row counts and head counts (Q
first). That covers joint-text attention (Q carries the joint rows), K/V with fewer heads
(grouped-query attention), and extra per-head tensors that ride with Q. Head groups split the
heads in units that keep every tensor's heads whole.

A call is `begin`, then `send` once per head group with that group's packed tensor
(`pack_group`), or in parts (the next tensors in order, e.g. Q/K before V is projected), then
`finish`, which attends group by group and returns the output. The steps are separate so the
packing, or even the projection that produces a group, can run in between (and be fused there
by torch.compile).

Layouts, for P ranks, batch b, head dim d; tensor t has s_t rows on this rank and h_tg heads in
group g:

- packed part (this rank's): [P][t][b][s_t][h_tg][d] for the part's tensors, so the rows for
  each peer are one contiguous chunk: one all-to-all per part.
- received part: the same, from every rank; each tensor is gathered into bshd over the whole
  sequence, [b][P * s_t][h_tg][d], with this rank's reserved rows (such as joint-text K/V that
  every rank holds) in front of or behind it.
- attention output of group g: bshd [b][P * s_q][h_qg][d], sent back as [P][b][s_q][h_qg][d]
  and gathered into the [b, s_q, H, d] output, which the output projection reads as
  [b, s_q, H * d].

A received part is a view of the all-to-all's buffer until that buffer's next use, the same part
of the next call; a peer writes it only after it has received this rank's output of this call,
which this rank sends after the group's attention has read the part. A packed part is read by the
copy engines until `finish` has waited for it, so the call holds it until then (under
torch.compile its producer's buffer may be freed as soon as `send` returns).
"""

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

from xfuser.core.distributed.sdma import transport

# What `attend` gets: one head group's tensors, each a [b, h, rows, d] view of a bshd
# tensor (Q, K, V, then the extras); it returns the group's attention output as
# [b, h_q, rows_q, d].
Attend = Callable[..., torch.Tensor]

# Head groups at most.
MAX_HEAD_GROUPS = 16


def split_heads(heads: int, groups: int) -> List[int]:
    """Sizes of `groups` head groups covering `heads` heads, as even as possible, larger first."""
    groups = max(1, min(groups, heads, MAX_HEAD_GROUPS))
    return [heads // groups + (1 if i < heads % groups else 0) for i in range(groups)]


def head_split(heads: Sequence[int], groups: int) -> Tuple[Tuple[int, ...], ...]:
    """Per head group, each tensor's heads per rank in it: the groups split the heads in units
    that keep every tensor whole (one unit is one K/V head and its query heads)."""
    units = math.gcd(*heads)
    return tuple(tuple(u * h // units for h in heads) for u in split_heads(units, groups))


def encode(world, batch, head_dim, rows, heads, reserved, head_groups) -> List[int]:
    """An exchange as integers, for a custom op's arguments (its dtype travels with a tensor)."""
    out = [world, batch, head_dim, head_groups, len(rows)]
    for r, h, (before, after) in zip(rows, heads, reserved):
        out += [r, h, before, after]
    return out


@dataclass(frozen=True)
class Exchange:
    """The tensors one Ulysses call exchanges, and how they are split into head groups.

    `rows[t]` is tensor t's rows on this rank and `heads[t]` its heads per rank (Q is
    tensor 0). `reserved[t]` is (rows before, rows after) of this rank's own
    rows put around tensor t's gathered sequence, such as the joint-text K/V that every rank holds.
    """

    world: int
    batch: int
    head_dim: int
    dtype: torch.dtype
    rows: Tuple[int, ...]
    heads: Tuple[int, ...]
    reserved: Tuple[Tuple[int, int], ...]
    head_groups: int
    # Derived in __post_init__.
    sizes: Tuple[Tuple[int, ...], ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "sizes", head_split(self.heads, self.head_groups))

    @staticmethod
    def make(world, batch, head_dim, dtype, rows, heads, reserved, head_groups) -> "Exchange":
        return _exchange(world, batch, head_dim, dtype, tuple(rows), tuple(heads), tuple(reserved), head_groups)

    def encode(self) -> List[int]:
        return encode(self.world, self.batch, self.head_dim, self.rows, self.heads, self.reserved, self.head_groups)

    @staticmethod
    def decode(spec: Sequence[int], dtype: torch.dtype) -> "Exchange":
        world, batch, head_dim, head_groups, n = spec[:5]
        per = [spec[5 + 4 * t : 9 + 4 * t] for t in range(n)]
        return Exchange.make(
            world,
            batch,
            head_dim,
            dtype,
            [p[0] for p in per],
            [p[1] for p in per],
            [(p[2], p[3]) for p in per],
            head_groups,
        )

    @property
    def groups(self) -> int:
        return len(self.sizes)

    @property
    def tensors(self) -> int:
        return len(self.rows)

    def head_offset(self, g: int, t: int) -> int:
        return sum(self.sizes[i][t] for i in range(g))

    def piece_elems(self, g: int, t: int) -> int:
        """Elements of tensor t's [b, s_t, h_tg, d] piece for one peer in group g."""
        return self.batch * self.rows[t] * self.sizes[g][t] * self.head_dim


_exchanges: Dict[tuple, Exchange] = {}


def _exchange(*key) -> Exchange:
    if key not in _exchanges:
        _exchanges[key] = Exchange(*key)
    return _exchanges[key]


def pieces_from_heads(x: torch.Tensor, world: int, offset: int, count: int) -> torch.Tensor:
    """Heads p * H/P + offset .. + count of `x` ([b, H, s, d], any strides) for every rank p, as a
    [P, b, s, count, d] view: one group's piece of one tensor, in `pack_group`'s order."""
    b, heads, s, d = x.shape
    return x.view(b, world, heads // world, s, d)[:, :, offset : offset + count].permute(1, 0, 3, 2, 4)


def pieces_from_grouped(x: torch.Tensor, world: int) -> torch.Tensor:
    """A group's tensor computed on its own as [b, s, P * h, d] (heads ordered by rank) as the
    [P, b, s, h, d] view `pack_group` takes."""
    b, s, heads, d = x.shape
    return x.view(b, s, world, heads // world, d).permute(2, 0, 1, 3, 4)


def pack_group(pieces: Sequence[torch.Tensor]) -> torch.Tensor:
    """A head group's packed tensor: per peer, every tensor's [b, s_t, h_tg, d] piece in turn.

    `pieces[t]` is [P, b, s_t, h_tg, d] with any strides. Traced by torch.compile, this is
    one `cat` that Inductor writes straight from the kernels producing the pieces; eager,
    one copy per tensor into the result.
    """
    P = pieces[0].shape[0]
    if torch.compiler.is_compiling():
        return torch.cat([p.reshape(P, -1) for p in pieces], dim=1).reshape(-1)
    sizes = [p[0].numel() for p in pieces]
    out = torch.empty(P * sum(sizes), dtype=pieces[0].dtype, device=pieces[0].device)
    rows = out.view(P, -1)
    start = 0
    for piece, n in zip(pieces, sizes):
        rows[:, start : start + n].view(piece.shape).copy_(piece)
        start += n
    return out


class _Rccl:
    """transport.Group's start/wait over RCCL: the all-to-all runs at `start`."""

    def __init__(self, process_group: dist.ProcessGroup):
        self.process_group = process_group
        self.received: Dict[tuple, torch.Tensor] = {}

    def start(self, key: tuple, x: torch.Tensor) -> None:
        out = self.received[key] = torch.empty_like(x)
        dist.all_to_all_single(out, x, group=self.process_group)

    def wait(self, key: tuple, like: torch.Tensor) -> torch.Tensor:
        return self.received.pop(key)


class _Call:
    """One open call on a Ulysses group, from `begin` to `finish`."""

    def __init__(self, exchange: Exchange, a2a):
        self.exchange, self.a2a = exchange, a2a
        # Per head group, its parts sent: (tensors, packed). The packed tensors are held until
        # `finish` has waited for them.
        self.parts: List[List[Tuple[Tuple[int, ...], torch.Tensor]]] = [[] for _ in range(exchange.groups)]
        self.sent = 0  # head groups sent completely
        self.next_tensor = 0  # the first tensor of the current group not sent yet


# Open calls, keyed by the group's ranks.
_open: Dict[tuple, Optional[_Call]] = {}
_side_streams: Dict[torch.device, List[torch.cuda.Stream]] = {}


def _key(process_group: dist.ProcessGroup) -> tuple:
    return tuple(dist.get_process_group_ranks(process_group))


def _streams(device: torch.device, count: int) -> List[torch.cuda.Stream]:
    """`count` side streams on `device`, the first at high priority."""
    streams = _side_streams.setdefault(device, [])
    while len(streams) < count:
        streams.append(torch.cuda.Stream(device=device, priority=-1 if not streams else 0))
    return streams


def begin(process_group: dist.ProcessGroup, exchange: Exchange, rccl: bool = False) -> None:
    """Open a call on `process_group`. With `rccl`, or while a CUDA graph is being captured (MoRI's
    waits block the host, which a capture does not allow), its all-to-alls run with RCCL."""
    key = _key(process_group)
    if _open.get(key) is not None:
        raise RuntimeError("an SDMA Ulysses call is already open on this group")
    world = dist.get_world_size(process_group)
    if exchange.world != world:
        raise ValueError(f"exchange for {exchange.world} ranks on a group of {world}")
    if rccl or torch.cuda.is_current_stream_capturing():
        a2a = _Rccl(process_group)
    else:
        a2a = transport.group_for(process_group)
    _open[key] = _Call(exchange, a2a)


def send(
    process_group: dist.ProcessGroup, gi: int, packed: torch.Tensor, tensors: Optional[Sequence[int]] = None
) -> None:
    """Start the all-to-all of head group `gi`'s packed tensor (`pack_group`) once the stream has
    produced it.

    `tensors` sends only those of the group's tensors, the next ones in order (by default all that
    are left): a model can send Q and K as soon as they exist and V after it is projected.
    """
    c = _open.get(_key(process_group))
    if c is None or gi != c.sent:
        raise RuntimeError(f"SDMA Ulysses send of group {gi} out of order")
    ex = c.exchange
    tensors = tuple(range(c.next_tensor, ex.tensors)) if tensors is None else tuple(tensors)
    if not tensors or tensors != tuple(range(c.next_tensor, c.next_tensor + len(tensors))) or tensors[-1] >= ex.tensors:
        raise ValueError(f"group {gi}: send its tensors in order from {c.next_tensor}; got {tensors}")
    try:
        c.a2a.start(("in", gi, tensors[0]), packed)
    except BaseException:
        # The call cannot complete; later calls must still be able to open.
        _open[_key(process_group)] = None
        raise
    c.parts[gi].append((tensors, packed))
    last = tensors[-1] == ex.tensors - 1
    c.next_tensor = 0 if last else tensors[-1] + 1
    c.sent += last


def _received(c: _Call, gi: int, local) -> List[torch.Tensor]:
    """Head group gi's tensors as [b, h, rows, d] views of bshd over the whole sequence."""
    ex, P, b, d = c.exchange, c.exchange.world, c.exchange.batch, c.exchange.head_dim
    views = []
    for tensors, packed in c.parts[gi]:
        rows = c.a2a.wait(("in", gi, tensors[0]), packed).view(P, -1)
        start = 0
        for t in tensors:
            h, s = ex.sizes[gi][t], ex.rows[t]
            n = ex.piece_elems(gi, t)
            x = rows[:, start : start + n].view(P, b, s, h, d).permute(1, 0, 2, 3, 4).reshape(b, P * s, h, d)
            before, after = ex.reserved[t]
            if local is not None and local[t] is not None and before + after:
                own = local[t][:, ex.head_offset(gi, t) : ex.head_offset(gi, t) + h].transpose(1, 2)
                x = torch.cat([own[:, :before], x, own[:, before:]], dim=1)
            views.append(x.transpose(1, 2))
            start += n
    return views


def finish(
    process_group: dist.ProcessGroup,
    attend: Attend,
    local: Optional[Sequence[Optional[torch.Tensor]]] = None,
    concurrent: bool = False,
) -> torch.Tensor:
    """Attend group by group and exchange the output back; returns this rank's rows for every head
    as a contiguous [b, s_q, H, d].

    `local[t]`, if given, is this rank's own rows for tensor t's reserved rows,
    [b, heads[t], before + after, d], heads split by group like the exchanged ones. With
    `concurrent`, each head group attends on a stream of its own (the first at high priority), so
    a group whose data has landed fills the GPU while the one before it drains.
    """
    key = _key(process_group)
    c = _open.get(key)
    if c is None or c.sent != c.exchange.groups:
        raise RuntimeError("SDMA Ulysses finish before every group was sent")
    _open[key] = None
    ex, P, b, d = c.exchange, c.exchange.world, c.exchange.batch, c.exchange.head_dim
    s = ex.rows[0]
    stream = torch.cuda.current_stream()
    if concurrent:
        ready = torch.cuda.Event()
        ready.record(stream)
        side = _streams(stream.device, ex.groups)
    back = []
    for gi in range(ex.groups):
        group_stream = stream
        if concurrent:
            group_stream = side[gi]
            group_stream.wait_event(ready)
        with torch.cuda.stream(group_stream):
            out = attend(*_received(c, gi, local)).transpose(1, 2).to(ex.dtype)
            h = ex.sizes[gi][0]
            back.append(out.reshape(b, P, s, h, d).permute(1, 0, 2, 3, 4).contiguous())
            c.a2a.start(("out", gi), back[gi])
    if concurrent:
        for gi in range(ex.groups):
            stream.wait_stream(side[gi])
    result = torch.empty(b, s, P, ex.heads[0], d, dtype=ex.dtype, device=back[0].device)
    for gi in range(ex.groups):
        off, h = ex.head_offset(gi, 0), ex.sizes[gi][0]
        result[:, :, :, off : off + h].copy_(c.a2a.wait(("out", gi), back[gi]).permute(1, 2, 0, 3, 4))
    return result.view(b, s, P * ex.heads[0], d)


def attention(
    tensors: Sequence[torch.Tensor],
    attend: Attend,
    process_group: dist.ProcessGroup,
    head_groups: int = 2,
    concurrent: bool = False,
    reserved: Optional[Sequence[Tuple[int, int]]] = None,
    local: Optional[Sequence[Optional[torch.Tensor]]] = None,
    rccl: bool = False,
) -> torch.Tensor:
    """Ulysses attention over `process_group`: begin, pack and send every group, finish.

    `tensors` are this rank's [b, H_t, s_t, d] (any strides; Q first), all with one batch, head
    dim and dtype; every H_t must divide by the group size. Returns [b, s_q, H_q, d].
    """
    world = dist.get_world_size(process_group)
    q = tensors[0]
    ex = Exchange.make(
        world,
        q.shape[0],
        q.shape[3],
        q.dtype,
        [x.shape[2] for x in tensors],
        [x.shape[1] // world for x in tensors],
        reserved or [(0, 0)] * len(tensors),
        head_groups,
    )
    begin(process_group, ex, rccl)
    for gi in range(ex.groups):
        pieces = [pieces_from_heads(x, world, ex.head_offset(gi, t), ex.sizes[gi][t]) for t, x in enumerate(tensors)]
        send(process_group, gi, pack_group(pieces))
    return finish(process_group, attend, local, concurrent)
