import numpy as np

from examples.wan_streaming.encoder import EncodedSegment
from examples.wan_streaming.segments import SegmentStore
from examples.wan_streaming.session import PlaybackPacer, StreamSession
from examples.wan_streaming.stream_loop import StreamLoop

FPS = 16
NUM_FRAMES = 9


class _Generator:
    def generate(self, request):
        return np.zeros((NUM_FRAMES, 4, 6, 3), dtype=np.uint8)


def _encode(frames: np.ndarray) -> EncodedSegment:
    return EncodedSegment(data=b"mp4", codec="avc1.640028", duration=len(frames) / FPS, num_frames=len(frames))


def _loop(encode=_encode) -> StreamLoop:
    return StreamLoop(
        generator=_Generator(),
        session=StreamSession(prompt="forest", base_seed=0),
        store=SegmentStore(),
        pacer=PlaybackPacer(margin_seconds=0.0),
        encode=encode,
        fps=FPS,
    )


def test_each_step_publishes_a_segment_tagged_with_the_prompt_it_was_generated_from():
    loop = _loop()
    try:
        loop.step().result(timeout=5)
        first = loop.store.latest()
        loop.session.set_prompt("city")
        loop.step().result(timeout=5)
        second = loop.store.latest()
    finally:
        loop.stop()

    assert (first.clip_index, first.prompt, first.restarted) == (0, "forest", True)
    assert (second.clip_index, second.prompt, second.restarted) == (1, "city", False)


def test_continuation_segments_drop_the_repeated_first_frame():
    loop = _loop()
    try:
        loop.step().result(timeout=5)
        first = loop.store.latest()
        loop.step().result(timeout=5)
        second = loop.store.latest()
    finally:
        loop.stop()

    assert first.duration == NUM_FRAMES / FPS
    assert second.duration == (NUM_FRAMES - 1) / FPS


def test_an_encode_failure_is_reported_and_publishes_nothing():
    def failing_encode(frames):
        raise RuntimeError("encoder exploded")

    loop = _loop(encode=failing_encode)
    try:
        loop.step().exception(timeout=5)
    finally:
        loop.stop()

    assert loop.store.latest() is None
    assert "encoder exploded" in loop.last_error
