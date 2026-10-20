import numpy as np
import pytest

from examples.wan_streaming.audio import AudioStitcher

FPS, SAMPLE_RATE, NUM_FRAMES = 24, 48000, 9
SAMPLES_PER_FRAME = SAMPLE_RATE // FPS


def _clip(value_offset: float, overhang: int = 500) -> np.ndarray:
    """Stereo audio whose every sample is its own index plus ``value_offset``, so positions can be traced."""
    samples = np.arange(NUM_FRAMES * SAMPLES_PER_FRAME + overhang, dtype=np.float32) + value_offset
    return np.stack([samples, -samples])


def test_each_clip_continues_exactly_where_the_previous_one_ended():
    stitcher = AudioStitcher(FPS, SAMPLE_RATE, channels=2)

    starts, ends = [], []
    for index in range(4):
        start, samples = stitcher.push(_clip(0.0), NUM_FRAMES, continuation=index > 0)
        starts.append(start)
        ends.append(start + len(samples))

    assert starts[0] == 0
    assert starts[1:] == ends[:-1]
    # Everything up to the last emitted frame is out; that frame's audio waits for the next clip.
    emitted_frames = NUM_FRAMES + 3 * (NUM_FRAMES - 1)
    assert stitcher.emitted_frames == emitted_frames
    assert ends[-1] == (emitted_frames - 1) * SAMPLES_PER_FRAME


def test_a_continuation_crossfades_the_repeated_frame_and_then_plays_from_its_second_frame():
    stitcher = AudioStitcher(FPS, SAMPLE_RATE, channels=2)
    stitcher.push(_clip(0.0), NUM_FRAMES, continuation=False)

    start, samples = stitcher.push(_clip(100_000.0), NUM_FRAMES, continuation=True)

    held_back = (NUM_FRAMES - 1) * SAMPLES_PER_FRAME
    assert start == held_back
    # The crossfade begins on the previous clip's held-back audio...
    assert samples[0, 0] == pytest.approx(held_back, rel=1e-2)
    # ...and the body starts at the new clip's frame 1.
    assert samples[SAMPLES_PER_FRAME, 0] == 100_000.0 + SAMPLES_PER_FRAME
    assert samples[SAMPLES_PER_FRAME, 1] == -(100_000.0 + SAMPLES_PER_FRAME)


def test_a_restart_fades_the_old_audio_out_and_the_new_audio_in_from_its_first_frame():
    stitcher = AudioStitcher(FPS, SAMPLE_RATE, channels=2)
    stitcher.push(np.ones((2, NUM_FRAMES * SAMPLES_PER_FRAME), np.float32), NUM_FRAMES, continuation=False)

    start, samples = stitcher.push(np.ones((2, NUM_FRAMES * SAMPLES_PER_FRAME), np.float32), NUM_FRAMES, False)

    assert start == (NUM_FRAMES - 1) * SAMPLES_PER_FRAME
    fade_out, body = samples[:SAMPLES_PER_FRAME, 0], samples[SAMPLES_PER_FRAME:, 0]
    assert fade_out[0] == 1.0 and fade_out[-1] == 0.0
    assert body[0] == 0.0 and body[SAMPLES_PER_FRAME - 1] == 1.0 and body[-1] == 1.0
    # A restart shows every frame, so the timeline grows by the whole clip.
    assert stitcher.emitted_frames == 2 * NUM_FRAMES


def test_audio_shorter_than_its_frames_is_padded_with_silence():
    stitcher = AudioStitcher(FPS, SAMPLE_RATE, channels=1)

    _, samples = stitcher.push(np.ones((1, SAMPLES_PER_FRAME), np.float32), NUM_FRAMES, continuation=False)

    assert len(samples) == (NUM_FRAMES - 1) * SAMPLES_PER_FRAME
    assert samples[:SAMPLES_PER_FRAME].min() == 1.0
    assert samples[SAMPLES_PER_FRAME:].max() == 0.0


def test_a_channel_count_change_is_rejected():
    stitcher = AudioStitcher(FPS, SAMPLE_RATE, channels=2)

    with pytest.raises(ValueError, match="channels|samples"):
        stitcher.push(np.zeros((1, 1000), np.float32), NUM_FRAMES, continuation=False)
