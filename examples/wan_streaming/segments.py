from __future__ import annotations

import asyncio
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Segment:
    # Changes when the server restarts, so viewers know segment ids started over.
    stream_id: str
    id: int
    data: bytes
    codec: str
    duration: float
    clip_index: int
    prompt: str
    generation_seconds: float
    restarted: bool
    created_at: float = field(default_factory=time.time)
    # A fragmented MP4 sent after the video: its first ``audio_init_size``
    # bytes are the init segment, the rest one fragment. ``audio_offset`` is
    # where the fragment starts relative to this segment's first video frame;
    # it is negative because the encoder and the seam crossfade hold back the
    # end of each clip's audio until the next clip.
    audio: bytes = b""
    audio_codec: str | None = None
    audio_init_size: int = 0
    audio_offset: float = 0.0
    audio_duration: float = 0.0

    def metadata(self) -> dict:
        return {
            "stream_id": self.stream_id,
            "id": self.id,
            "codec": self.codec,
            "duration": self.duration,
            "clip_index": self.clip_index,
            "prompt": self.prompt,
            "generation_seconds": self.generation_seconds,
            "restarted": self.restarted,
            "size": len(self.data) + len(self.audio),
            "video_size": len(self.data),
            "audio_size": len(self.audio),
            "audio_codec": self.audio_codec,
            "audio_init_size": self.audio_init_size,
            "audio_offset": self.audio_offset,
            "audio_duration": self.audio_duration,
            "created_at": self.created_at,
        }

    def payload(self) -> bytes:
        """The bytes sent to a viewer: the video, then the audio."""
        return self.data + self.audio


class SegmentStore:
    """Recent segments, published from a worker thread and awaited from asyncio.

    Only the last ``capacity`` segments are kept. A viewer that falls further
    behind than that is moved to the newest segment rather than replaying
    stale video.
    """

    def __init__(self, capacity: int = 6) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.stream_id = uuid.uuid4().hex[:12]
        self._segments: deque[Segment] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._next_id = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._published: asyncio.Event | None = None

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._published = asyncio.Event()

    def publish(self, **fields) -> Segment:
        with self._lock:
            segment = Segment(stream_id=self.stream_id, id=self._next_id, **fields)
            self._next_id += 1
            self._segments.append(segment)
        if self._loop is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._wake_waiters)
        return segment

    def _wake_waiters(self) -> None:
        published, self._published = self._published, asyncio.Event()
        published.set()

    def latest(self) -> Segment | None:
        with self._lock:
            return self._segments[-1] if self._segments else None

    def next_after(self, after_id: int) -> Segment | None:
        """The segment a viewer that last saw ``after_id`` should get next.

        ``after_id < 0`` means a new viewer, which joins at the newest segment.
        """
        with self._lock:
            if not self._segments:
                return None
            newest = self._segments[-1]
            if after_id < 0 or after_id >= newest.id:
                return newest if after_id < 0 else None
            oldest = self._segments[0]
            if after_id + 1 < oldest.id:
                return newest
            return self._segments[after_id + 1 - oldest.id]

    async def wait_after(self, after_id: int, timeout: float) -> Segment | None:
        if self._published is None:
            raise RuntimeError("attach_loop must be called before waiting")
        deadline = time.monotonic() + timeout
        while True:
            published = self._published
            segment = self.next_after(after_id)
            if segment is not None:
                return segment
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                await asyncio.wait_for(published.wait(), remaining)
            except asyncio.TimeoutError:
                return None
