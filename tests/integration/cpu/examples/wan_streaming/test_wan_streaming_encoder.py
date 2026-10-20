import io
import re

import numpy as np
import pytest

av = pytest.importorskip("av")

from examples.wan_streaming.encoder import AacEncoder, encode_fragmented_mp4  # noqa: E402

SAMPLE_RATE = 48000


def _frames(num_frames: int, height: int = 48, width: int = 64) -> np.ndarray:
    ramp = np.linspace(0, 255, width, dtype=np.uint8)
    frames = np.empty((num_frames, height, width, 3), dtype=np.uint8)
    for index in range(num_frames):
        frames[index] = np.roll(ramp, index * 3)[None, :, None]
    return frames


def _top_level_boxes(data: bytes) -> list[bytes]:
    boxes, offset = [], 0
    while offset + 8 <= len(data):
        size = int.from_bytes(data[offset : offset + 4], "big")
        boxes.append(data[offset + 4 : offset + 8])
        if size < 8:
            break
        offset += size
    return boxes


def test_a_segment_is_a_fragmented_mp4_that_decodes_to_every_frame():
    segment = encode_fragmented_mp4(_frames(20), fps=16)

    boxes = _top_level_boxes(segment.data)
    with av.open(io.BytesIO(segment.data)) as container:
        decoded = sum(1 for _ in container.decode(video=0))

    assert boxes[:2] == [b"ftyp", b"moov"]
    assert b"moof" in boxes
    assert decoded == 20
    assert segment.duration == 20 / 16
    assert re.fullmatch(r"avc1\.[0-9a-f]{6}", segment.codec)


def test_odd_dimensions_are_rejected():
    with pytest.raises(ValueError, match="even"):
        encode_fragmented_mp4(_frames(4, height=47), fps=16)


def _tone(start: int, count: int) -> np.ndarray:
    t = (start + np.arange(count)) / SAMPLE_RATE
    return (0.3 * np.stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 550 * t)], axis=1)).astype(np.float32)


def test_audio_fragments_from_one_encoder_follow_on_without_gaps():
    encoder = AacEncoder(SAMPLE_RATE, channels=2)
    chunk = 246_000  # 5.125 s, not a multiple of the 1024-sample AAC frame
    fragments = [encoder.encode(index * chunk, _tone(index * chunk, chunk)) for index in range(3)]

    # The first packet carries the encoder's 1024-sample priming, before stream time zero.
    assert fragments[0].start == pytest.approx(-1024 / SAMPLE_RATE)
    for previous, current in zip(fragments, fragments[1:]):
        assert current.start == pytest.approx(previous.start + previous.duration)
    # The encoder holds back its delay, one frame of lookahead and the incomplete
    # frame, about 60 ms, for the next call.
    held_back = 3 * chunk - (fragments[-1].start + fragments[-1].duration) * SAMPLE_RATE
    assert 0 <= held_back < 3 * 1024

    for fragment in fragments:
        assert fragment.codec == "mp4a.40.2"
        assert _top_level_boxes(fragment.data[: fragment.init_size]) == [b"ftyp", b"moov"]
        assert _top_level_boxes(fragment.data[fragment.init_size :])[0] == b"moof"
        with av.open(io.BytesIO(fragment.data)) as container:
            decoded = sum(frame.samples for frame in container.decode(audio=0))
        assert decoded == round(fragment.duration * SAMPLE_RATE)
    # Every fragment shares one init segment, so a player only needs the first.
    assert len({fragment.data[: fragment.init_size] for fragment in fragments}) == 1


def test_audio_must_be_contiguous():
    encoder = AacEncoder(SAMPLE_RATE, channels=2)
    encoder.encode(0, _tone(0, 4096))

    with pytest.raises(ValueError, match="contiguous"):
        encoder.encode(5000, _tone(5000, 4096))
