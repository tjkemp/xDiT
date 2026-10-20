from __future__ import annotations

import io
from dataclasses import dataclass

import av
import numpy as np

# One fragment per segment, readable by Media Source Extensions without a
# separate init request: the moov comes first and every segment carries its own.
_FRAGMENTED_MP4_FLAGS = "frag_keyframe+empty_moov+default_base_moof"


@dataclass(frozen=True)
class EncodedSegment:
    data: bytes
    codec: str
    duration: float
    num_frames: int


def _avc_codec_string(extradata: bytes | None) -> str:
    # avcC: version, profile_idc, profile_compatibility, level_idc, ...
    if not extradata or len(extradata) < 4 or extradata[0] != 1:
        return "avc1.640028"
    return f"avc1.{extradata[1]:02x}{extradata[2]:02x}{extradata[3]:02x}"


def encode_fragmented_mp4(
    frames: np.ndarray,
    fps: int,
    crf: int = 23,
    preset: str = "veryfast",
) -> EncodedSegment:
    """Encode ``[F, H, W, 3]`` uint8 RGB frames into a self-contained fragmented MP4.

    B-frames are disabled so the segment has no composition offset; with
    ``SourceBuffer.mode = "sequence"`` consecutive segments then butt up
    against each other without gaps.
    """
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise ValueError(f"Expected [F, H, W, 3] uint8 frames, got {frames.shape} {frames.dtype}.")
    num_frames, height, width, _ = frames.shape
    if num_frames == 0:
        raise ValueError("Cannot encode an empty segment.")
    if height % 2 or width % 2:
        raise ValueError(f"yuv420p needs even dimensions, got {width}x{height}.")

    buffer = io.BytesIO()
    container = av.open(buffer, mode="w", format="mp4", options={"movflags": _FRAGMENTED_MP4_FLAGS})
    stream = container.add_stream("libx264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.options = {
        "preset": preset,
        "crf": str(crf),
        "profile": "high",
        "x264-params": f"bframes=0:keyint={num_frames}:min-keyint={num_frames}:scenecut=0",
    }

    for frame in frames:
        video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        for packet in stream.encode(video_frame):
            container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)
    codec = _avc_codec_string(stream.codec_context.extradata)
    container.close()

    return EncodedSegment(
        data=buffer.getvalue(),
        codec=codec,
        duration=num_frames / fps,
        num_frames=num_frames,
    )
