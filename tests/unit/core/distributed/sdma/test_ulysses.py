"""The Ulysses exchange's all-to-alls and their order, on a fake all-to-all on CPU."""

from types import SimpleNamespace

import pytest
import torch

from xfuser.core.distributed.sdma import ulysses


class _A2A:
    """Two ranks that hold the same tensors: each receives from both what it sends to itself. That
    is what a real all-to-all delivers for the inputs; the outputs carry each sender's own heads,
    which tests/integration/accelerator/core/distributed/test_sdma_ulysses.py checks."""

    def __init__(self, rank):
        self.rank, self.log, self.sent = rank, [], {}

    def start(self, key, x):
        self.log.append(("start", key))
        mine = x.reshape(2, -1)[self.rank]
        self.sent[key] = torch.stack([mine, mine]).view(x.shape)

    def wait(self, key, like):
        self.log.append(("wait", key))
        return self.sent.pop(key)


@pytest.fixture
def fake(monkeypatch):
    def install(rank):
        a2a = _A2A(rank)
        group = object()
        monkeypatch.setattr(
            ulysses, "dist", SimpleNamespace(get_world_size=lambda g: 2, get_process_group_ranks=lambda g: [0, 1])
        )
        monkeypatch.setattr(ulysses, "transport", SimpleNamespace(group_for=lambda g: a2a))
        monkeypatch.setattr(torch.cuda, "current_stream", lambda *args, **kwargs: None)
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
        monkeypatch.setattr(ulysses, "_open", {})
        return a2a, group

    return install


def _bhsd(b, h, s, d, seed):
    return torch.randn(b, h, s, d, generator=torch.Generator().manual_seed(seed))


def test_each_group_attends_once_its_data_is_waited_for_and_sends_its_output_back(fake):
    a2a, group = fake(0)
    q, k, v = (_bhsd(1, 8, 3, 2, seed) for seed in range(3))
    attended = []

    def attend(q, k, v):
        attended.append(len(a2a.log))
        return q

    ulysses.attention([q, k, v], attend, group, head_groups=2)
    assert a2a.log == [
        ("start", ("in", 0, 0)),
        ("start", ("in", 1, 0)),
        ("wait", ("in", 0, 0)),
        ("start", ("out", 0)),
        ("wait", ("in", 1, 0)),
        ("start", ("out", 1)),
        ("wait", ("out", 0)),
        ("wait", ("out", 1)),
    ]
    # Group 0 attends after its wait, before its output starts; group 1 likewise.
    assert attended == [3, 5]


def test_joint_rows_and_a_split_send(fake):
    # One head group sent as Q/K and then V; K/V have one joint row of this rank's in front.
    a2a, group = fake(1)
    ex = ulysses.Exchange.make(2, 1, 2, torch.float32, [3, 2, 2], [2, 1, 1], [(0, 0), (1, 0), (1, 0)], 1)
    q, k, v = _bhsd(1, 4, 3, 2, 0), _bhsd(1, 2, 2, 2, 1), _bhsd(1, 2, 2, 2, 2)
    joint_k, joint_v = _bhsd(1, 1, 1, 2, 3), _bhsd(1, 1, 1, 2, 4)
    ulysses.begin(group, ex)
    ulysses.send(
        group, 0, ulysses.pack_group([ulysses.pieces_from_heads(x, 2, 0, h) for x, h in ((q, 2), (k, 1))]), (0, 1)
    )
    ulysses.send(group, 0, ulysses.pack_group([ulysses.pieces_from_heads(v, 2, 0, 1)]), (2,))
    seen = {}

    def attend(q, k, v):
        seen.update(k=k, v=v)
        return q

    ulysses.finish(group, attend, [None, joint_k, joint_v])
    assert [entry[1] for entry in a2a.log[:2]] == [("in", 0, 0), ("in", 0, 2)]
    # K: the joint row, then rank 0's and rank 1's rows (the same here): rank 1's heads of K.
    mine = k[:, 1:2]
    assert torch.equal(seen["k"], torch.cat([joint_k, mine, mine], dim=2))
    assert torch.equal(seen["v"], torch.cat([joint_v, v[:, 1:2], v[:, 1:2]], dim=2))


def test_a_send_out_of_order_leaves_the_call_open(fake):
    a2a, group = fake(0)
    exchange = ulysses.Exchange.make(2, 1, 2, torch.float32, [2, 2, 2], [2, 2, 2], [(0, 0)] * 3, 1)
    ulysses.begin(group, exchange)
    with pytest.raises(ValueError):
        ulysses.send(group, 0, torch.zeros(16), (1,))
    assert a2a.log == []
    ulysses.send(group, 0, torch.zeros(48))
    ulysses.finish(group, lambda q, *rest: q)
    assert a2a.log[-1] == ("wait", ("out", 0))
