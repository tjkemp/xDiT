from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Protocol

import numpy as np

from examples.wan_streaming.encoder import EncodedSegment
from examples.wan_streaming.segments import SegmentStore
from examples.wan_streaming.session import GenerationRequest, PlaybackPacer, StreamSession

logger = logging.getLogger(__name__)


class Generator(Protocol):
    def generate(self, request: GenerationRequest) -> np.ndarray: ...


class StreamLoop:
    """Generates clips back to back on a worker thread and publishes them as segments.

    Encoding runs on its own thread so it overlaps the next generation; with
    the real model every rank waits on the output rank before each clip, so
    encode time would otherwise come straight out of the real-time budget.
    """

    def __init__(
        self,
        generator: Generator,
        session: StreamSession,
        store: SegmentStore,
        pacer: PlaybackPacer,
        encode: Callable[[np.ndarray], EncodedSegment],
        fps: int,
    ) -> None:
        self.generator = generator
        self.session = session
        self.store = store
        self.pacer = pacer
        self.fps = fps
        self._encode = encode
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._encoder = ThreadPoolExecutor(max_workers=1, thread_name_prefix="segment-encoder")
        self.last_generation_seconds: float | None = None
        self.last_encode_seconds: float | None = None
        self.last_error: str | None = None

    def start(self) -> None:
        """Run the loop on a background thread."""
        self._thread = threading.Thread(target=self.run_forever, name="stream-loop", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
        self._encoder.shutdown(wait=True, cancel_futures=True)

    def step(self) -> Future:
        """Generate one clip and queue it for encoding and publishing."""
        request = self.session.next_request()
        started = time.monotonic()
        frames = self.generator.generate(request)
        self.last_generation_seconds = time.monotonic() - started
        self.pacer.record_generation(self.last_generation_seconds)
        shown = self.session.complete(request, frames)
        self.pacer.reserve(len(shown) / self.fps)
        future = self._encoder.submit(self._encode_and_publish, request, shown, self.last_generation_seconds)
        future.add_done_callback(self._report_encode_failure)
        return future

    def _report_encode_failure(self, future: Future) -> None:
        if future.cancelled() or future.exception() is None:
            return
        logger.error("Segment encoding failed", exc_info=future.exception())
        self.last_error = repr(future.exception())

    def _encode_and_publish(self, request: GenerationRequest, frames: np.ndarray, generation_seconds: float) -> None:
        started = time.monotonic()
        encoded = self._encode(frames)
        self.last_encode_seconds = time.monotonic() - started
        self.store.publish(
            data=encoded.data,
            codec=encoded.codec,
            duration=encoded.duration,
            clip_index=request.clip_index,
            prompt=request.prompt,
            generation_seconds=generation_seconds,
            restarted=request.restarted,
        )

    def run_forever(self) -> None:
        """Generate clips on the calling thread until ``stop`` is called."""
        while not self._stop.is_set():
            wait = self.pacer.seconds_until_next_start()
            if wait > 0:
                self._stop.wait(min(wait, 0.25))
                continue
            try:
                self.step()
                self.last_error = None
            except Exception as error:
                logger.exception("Clip generation failed")
                self.last_error = repr(error)
                self._stop.wait(1.0)
