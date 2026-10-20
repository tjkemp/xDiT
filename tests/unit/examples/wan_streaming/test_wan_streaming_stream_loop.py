import numpy as np
import pytest

from examples.wan_streaming.encoder import EncodedSegment
from examples.wan_streaming.segments import SegmentStore
from examples.wan_streaming.session import GeneratedClip, PlaybackPacer, StreamSession
from examples.wan_streaming.stream_loop import StreamLoop

FPS = 16
NUM_FRAMES = 9
SAMPLE_RATE = 48000


class _Generator:
    def __init__(self, sample_rate: int | None = None) -> None:
        self.sample_rate = sample_rate

    def generate(self, request):
        frames = np.zeros((NUM_FRAMES, 4, 6, 3), dtype=np.uint8)
        if self.sample_rate is None:
            return GeneratedClip(frames)
        t = np.arange(round((NUM_FRAMES / FPS + 0.1) * self.sample_rate)) / self.sample_rate
        audio = 0.2 * np.stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 660 * t)]).astype(np.float32)
        return GeneratedClip(frames, audio, self.sample_rate)


def _encode(frames: np.ndarray) -> EncodedSegment:
    return EncodedSegment(data=b"mp4", codec="avc1.640028", duration=len(frames) / FPS, num_frames=len(frames))


def _loop(encode=_encode, sample_rate: int | None = None) -> StreamLoop:
    return StreamLoop(
        generator=_Generator(sample_rate),
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


def test_a_video_only_generator_publishes_segments_without_audio():
    loop = _loop()
    try:
        loop.step().result(timeout=5)
    finally:
        loop.stop()

    segment = loop.store.latest()
    assert segment.audio == b""
    assert segment.metadata()["audio_size"] == 0


def test_soundtracks_are_published_as_one_continuous_track_lined_up_with_the_video():
    pytest.importorskip("av")
    loop = _loop(sample_rate=SAMPLE_RATE)
    segments = []
    try:
        for index in range(3):
            if index == 2:
                loop.session.restart()
            loop.step().result(timeout=10)
            segments.append(loop.store.latest())
    finally:
        loop.stop()

    video_starts = np.cumsum([0.0] + [segment.duration for segment in segments[:-1]])
    audio_starts = [start + segment.audio_offset for start, segment in zip(video_starts, segments)]
    for segment in segments:
        assert segment.audio_codec == "mp4a.40.2"
        assert segment.audio[segment.audio_init_size + 4 : segment.audio_init_size + 8] == b"moof"
        # The end of each clip's audio waits for the next clip, so it starts a little early.
        assert -0.2 < segment.audio_offset <= 0
    # A restart changes the picture but not the timeline, so the audio stays continuous across it too.
    for index in range(1, len(segments)):
        expected = audio_starts[index - 1] + segments[index - 1].audio_duration
        assert audio_starts[index] == pytest.approx(expected, abs=1e-9)
