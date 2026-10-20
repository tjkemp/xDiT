import io
import re

import numpy as np
import pytest

av = pytest.importorskip("av")

from examples.wan_streaming.encoder import encode_fragmented_mp4  # noqa: E402


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
