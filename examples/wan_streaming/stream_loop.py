from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Protocol

import numpy as np

from examples.wan_streaming.audio import AudioStitcher
from examples.wan_streaming.encoder import AacEncoder, EncodedSegment
from examples.wan_streaming.segments import SegmentStore
from examples.wan_streaming.session import GeneratedClip, GenerationRequest, PlaybackPacer, StreamSession

logger = logging.getLogger(__name__)


class Generator(Protocol):
    def generate(self, request: GenerationRequest) -> GeneratedClip: ...


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
        audio_bit_rate: int = 128_000,
    ) -> None:
        self.generator = generator
        self.session = session
        self.store = store
        self.pacer = pacer
        self.fps = fps
        self._encode = encode
        self._audio_bit_rate = audio_bit_rate
        # Both follow the stream from its first clip and are never reset: a
        # restart changes the picture, not the stream's timeline.
        self._stitcher: AudioStitcher | None = None
        self._audio_encoder: AacEncoder | None = None
        self._video_frames = 0
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
        clip = self.generator.generate(request)
        self.last_generation_seconds = time.monotonic() - started
        self.pacer.record_generation(self.last_generation_seconds)
        shown = self.session.complete(request, clip.frames)
        self.pacer.reserve(len(shown) / self.fps)
        audio = self._stitch_audio(request, clip)
        video_start = self._video_frames / self.fps
        self._video_frames += len(shown)
        future = self._encoder.submit(
            self._encode_and_publish, request, shown, audio, video_start, self.last_generation_seconds
        )
        future.add_done_callback(self._report_encode_failure)
        return future

    def _stitch_audio(self, request: GenerationRequest, clip: GeneratedClip) -> tuple[int, np.ndarray] | None:
        audio, sample_rate = clip.audio, clip.sample_rate
        if audio is None:
            if self._stitcher is None:
                return None
            # Keep the audio timeline in step with the video.
            sample_rate = self._stitcher.sample_rate
            audio = np.zeros((self._stitcher.channels, round(len(clip.frames) * sample_rate / self.fps)), np.float32)
        if self._stitcher is None:
            if self._video_frames:
                raise ValueError("Audio appeared partway through a stream that started without it.")
            self._stitcher = AudioStitcher(self.fps, sample_rate, channels=audio.shape[0])
        elif sample_rate != self._stitcher.sample_rate:
            raise ValueError(f"Audio sample rate changed from {self._stitcher.sample_rate} to {sample_rate}.")
        return self._stitcher.push(audio, len(clip.frames), continuation=not request.restarted)

    def _report_encode_failure(self, future: Future) -> None:
        if future.cancelled() or future.exception() is None:
            return
        logger.error("Segment encoding failed", exc_info=future.exception())
        self.last_error = repr(future.exception())

    def _encode_and_publish(
        self,
        request: GenerationRequest,
        frames: np.ndarray,
        audio: tuple[int, np.ndarray] | None,
        video_start: float,
        generation_seconds: float,
    ) -> None:
        started = time.monotonic()
        # Audio first: the encoder only takes contiguous audio, so a failed
        # video encode must not skip this clip's samples.
        audio_fields = {}
        if audio is not None:
            start_sample, samples = audio
            if self._audio_encoder is None:
                self._audio_encoder = AacEncoder(
                    self._stitcher.sample_rate, self._stitcher.channels, bit_rate=self._audio_bit_rate
                )
            encoded_audio = self._audio_encoder.encode(start_sample, samples)
            if encoded_audio is not None:
                audio_fields = {
                    "audio": encoded_audio.data,
                    "audio_codec": encoded_audio.codec,
                    "audio_init_size": encoded_audio.init_size,
                    "audio_offset": encoded_audio.start - video_start,
                    "audio_duration": encoded_audio.duration,
                }
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
            **audio_fields,
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
