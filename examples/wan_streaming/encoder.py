from __future__ import annotations

import io
from dataclasses import dataclass
from fractions import Fraction

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


@dataclass(frozen=True)
class EncodedAudio:
    # A fragmented MP4: the first ``init_size`` bytes are the init segment
    # (ftyp + moov), the rest is one fragment.
    data: bytes
    init_size: int
    codec: str
    # Stream time of the first sample in the fragment, and how long it plays.
    start: float
    duration: float


def _init_segment_size(data: bytes) -> int:
    offset = 0
    while offset + 8 <= len(data):
        size = int.from_bytes(data[offset : offset + 4], "big")
        offset += size
        if data[offset - size + 4 : offset - size + 8] == b"moov":
            return offset
        if size < 8:
            break
    raise ValueError("Fragmented MP4 has no moov box.")


class AacEncoder:
    """One AAC-LC encoder for the whole stream, packaged as one fragment per call.

    Encoding each segment on its own would prime the encoder again and pad
    its last frame, putting a gap at every seam. A single encoder keeps the
    packets continuous. It holds back the last couple of thousand samples
    until the next call, so a fragment ends slightly before the audio it was
    given; the next fragment picks up exactly there.
    """

    def __init__(self, sample_rate: int, channels: int, bit_rate: int = 128_000) -> None:
        if channels not in (1, 2):
            raise ValueError(f"Only mono and stereo audio are supported, got {channels} channels.")
        self.sample_rate = sample_rate
        self.channels = channels
        self.bit_rate = bit_rate
        self._layout = "stereo" if channels == 2 else "mono"
        self._time_base = Fraction(1, sample_rate)
        self._codec = av.CodecContext.create("aac", "w")
        self._codec.sample_rate = sample_rate
        self._codec.layout = self._layout
        self._codec.format = "fltp"
        self._codec.bit_rate = bit_rate
        self._codec.time_base = self._time_base
        self._codec.open()
        self._next_sample: int | None = None

    def encode(self, start_sample: int, samples: np.ndarray) -> EncodedAudio | None:
        """Encode float32 ``[samples, channels]`` audio starting at stream sample ``start_sample``.

        Returns None while the encoder has not produced a packet yet.
        """
        if self._next_sample is not None and start_sample != self._next_sample:
            raise ValueError(f"Audio must be contiguous: expected sample {self._next_sample}, got {start_sample}.")
        self._next_sample = start_sample + len(samples)
        planar = np.ascontiguousarray(np.asarray(samples, dtype=np.float32).T)
        frame = av.AudioFrame.from_ndarray(planar, format="fltp", layout=self._layout)
        frame.sample_rate = self.sample_rate
        frame.time_base = self._time_base
        frame.pts = start_sample
        packets = list(self._codec.encode(frame))
        if not packets:
            return None

        buffer = io.BytesIO()
        container = av.open(buffer, mode="w", format="mp4", options={"movflags": _FRAGMENTED_MP4_FLAGS})
        stream = container.add_stream("aac", rate=self.sample_rate)
        stream.codec_context.layout = self._layout
        stream.codec_context.bit_rate = self.bit_rate
        stream.time_base = self._time_base
        first = packets[0].pts
        end = packets[-1].pts + packets[-1].duration
        for packet in packets:
            # Each fragment starts at zero; the player places it from ``start``.
            packet.pts -= first
            packet.dts -= first
            packet.stream = stream
            container.mux(packet)
        container.close()
        data = buffer.getvalue()
        return EncodedAudio(
            data=data,
            init_size=_init_segment_size(data),
            codec="mp4a.40.2",
            start=first / self.sample_rate,
            duration=(end - first) / self.sample_rate,
        )
