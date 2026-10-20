from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    seed: int
    clip_index: int
    # The frame the clip continues from: the seed image on the first clip and
    # after a restart, otherwise the previous clip's last frame. ``None`` only
    # when no seed image was given, which the fake generator accepts.
    input_frame: np.ndarray | None
    restarted: bool


@dataclass(frozen=True)
class GeneratedClip:
    frames: np.ndarray  # uint8 [F, H, W, 3]
    # float32 [channels, samples] in [-1, 1], usually a little longer than the frames.
    audio: np.ndarray | None = None
    sample_rate: int | None = None


class StreamSession:
    """Viewer-controlled state for the clip chain.

    Changes made while a clip is generating apply to the next clip, never the
    one in flight.
    """

    def __init__(self, prompt: str, base_seed: int, seed_image: np.ndarray | None = None) -> None:
        self._lock = threading.Lock()
        self._prompt = prompt
        self._base_seed = base_seed
        self._seed_image = seed_image
        self._input_frame = seed_image
        self._restart_pending = True
        self._clip_index = 0

    @property
    def prompt(self) -> str:
        with self._lock:
            return self._prompt

    @property
    def clip_index(self) -> int:
        with self._lock:
            return self._clip_index

    def set_prompt(self, prompt: str) -> None:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Prompt cannot be empty.")
        with self._lock:
            self._prompt = prompt

    def restart(self, seed_image: np.ndarray | None = None) -> None:
        """Start the chain again from ``seed_image``, or from the current seed image."""
        with self._lock:
            if seed_image is not None:
                self._seed_image = seed_image
            self._input_frame = self._seed_image
            self._restart_pending = True

    def next_request(self) -> GenerationRequest:
        with self._lock:
            request = GenerationRequest(
                prompt=self._prompt,
                seed=self._base_seed + self._clip_index,
                clip_index=self._clip_index,
                input_frame=self._input_frame,
                restarted=self._restart_pending,
            )
            self._restart_pending = False
            return request

    def complete(self, request: GenerationRequest, frames: np.ndarray) -> np.ndarray:
        """Record a finished clip and return the frames to show.

        A continuation clip starts with the frame it was conditioned on, which
        the viewer has already seen as the previous clip's last frame, so it is
        dropped. A restart shows every frame.
        """
        if len(frames) < 2:
            raise ValueError("A clip needs at least two frames to continue from.")
        with self._lock:
            self._clip_index = request.clip_index + 1
            # A restart requested while this clip was generating already
            # replaced the input frame; keep it.
            if not self._restart_pending:
                self._input_frame = frames[-1].copy()
        return frames if request.restarted else frames[1:]


class PlaybackPacer:
    """Keeps generation just ahead of a live viewer's playback position.

    The server can't see viewers' players, so it models one: published video
    plays back in real time from when it is published, and a stall pushes the
    whole timeline later. The next clip starts generating only when the
    remaining buffered playback is down to the expected generation time plus a
    margin, which keeps a prompt change from waiting behind a long queue.
    """

    def __init__(self, margin_seconds: float, clock: Callable[[], float] = time.monotonic) -> None:
        self._margin = margin_seconds
        self._clock = clock
        self._timeline_end = clock()
        self._estimated_generation: float | None = None

    @property
    def estimated_generation_seconds(self) -> float | None:
        return self._estimated_generation

    def buffered_seconds(self) -> float:
        return max(0.0, self._timeline_end - self._clock())

    def seconds_until_next_start(self) -> float:
        estimate = self._estimated_generation or 0.0
        return max(0.0, self.buffered_seconds() - estimate - self._margin)

    def record_generation(self, seconds: float) -> None:
        if self._estimated_generation is None:
            self._estimated_generation = seconds
        else:
            self._estimated_generation = 0.7 * self._estimated_generation + 0.3 * seconds

    def reserve(self, duration: float) -> None:
        self._timeline_end = max(self._timeline_end, self._clock()) + duration
