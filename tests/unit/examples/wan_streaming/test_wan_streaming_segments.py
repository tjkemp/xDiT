import asyncio
import threading

import pytest

from examples.wan_streaming.segments import SegmentStore


def _publish(store: SegmentStore, clip_index: int):
    return store.publish(
        data=bytes([clip_index]),
        codec="avc1.640028",
        duration=5.0,
        clip_index=clip_index,
        prompt="forest",
        generation_seconds=1.0,
        restarted=False,
    )


def test_a_new_viewer_joins_at_the_newest_segment():
    store = SegmentStore(capacity=4)
    assert store.next_after(-1) is None
    for clip_index in range(3):
        _publish(store, clip_index)

    assert store.next_after(-1).id == 2


def test_a_viewer_gets_segments_in_order_then_waits():
    store = SegmentStore(capacity=4)
    for clip_index in range(3):
        _publish(store, clip_index)

    assert store.next_after(0).id == 1
    assert store.next_after(1).id == 2
    assert store.next_after(2) is None


def test_a_viewer_behind_the_retained_window_skips_to_the_newest():
    store = SegmentStore(capacity=2)
    for clip_index in range(5):
        _publish(store, clip_index)

    assert store.next_after(0).id == 4
    assert store.next_after(2).id == 3


def test_segments_carry_the_stream_id_that_changes_per_store():
    first, second = SegmentStore(), SegmentStore()

    assert _publish(first, 0).stream_id == first.stream_id
    assert first.stream_id != second.stream_id


def test_a_waiting_viewer_wakes_when_another_thread_publishes():
    store = SegmentStore()

    async def wait_for_first_segment():
        store.attach_loop(asyncio.get_running_loop())
        threading.Timer(0.05, _publish, args=(store, 0)).start()
        return await store.wait_after(-1, timeout=5.0)

    segment = asyncio.run(wait_for_first_segment())

    assert segment is not None and segment.id == 0


def test_waiting_times_out_without_a_new_segment():
    store = SegmentStore()

    async def wait_briefly():
        store.attach_loop(asyncio.get_running_loop())
        return await store.wait_after(-1, timeout=0.05)

    assert asyncio.run(wait_briefly()) is None


def test_capacity_must_be_positive():
    with pytest.raises(ValueError):
        SegmentStore(capacity=0)
