from __future__ import annotations

import textwrap
import time
import zlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from examples.wan_streaming.session import GeneratedClip, GenerationRequest

# Like the model's soundtrack, the fake one runs a little past the last frame.
_AUDIO_OVERHANG_SECONDS = 0.1


class FakeGenerator:
    """Stands in for the model so the streaming path can be tested on CPU.

    Draws a scrolling test pattern with the clip index, frame counter and
    prompt burned in, then sleeps so a clip takes ``generation_seconds``.
    Motion continues across clips and resets on a restart, so a seam or a
    dropped segment is visible in the player. A seed image, when given,
    becomes the scrolling background.

    With a ``sample_rate`` it also makes a soundtrack: a tone pulsing once a
    second, its pitch set by the prompt. Like the motion, it continues across
    clips, so a bad seam is audible as a click.
    """

    def __init__(
        self,
        width: int,
        height: int,
        num_frames: int,
        generation_seconds: float,
        fps: int = 24,
        sample_rate: int | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.generation_seconds = generation_seconds
        self.fps = fps
        self.sample_rate = sample_rate
        self._frame_counter = 0
        self._background = self._gradient()
        self._font_large = ImageFont.load_default(size=max(24, height // 14))
        self._font_small = ImageFont.load_default(size=max(16, height // 28))

    def _gradient(self) -> np.ndarray:
        x = np.linspace(0, 1, self.width, dtype=np.float32)[None, :]
        y = np.linspace(0, 1, self.height, dtype=np.float32)[:, None]
        red = 40 + 120 * x + 0 * y
        green = 30 + 80 * y + 0 * x
        blue = 90 + 120 * (1 - x) * (1 - y)
        return np.stack([red, green, blue], axis=-1).clip(0, 255).astype(np.uint8)

    def _render(self, request: GenerationRequest, frame_in_clip: int) -> np.ndarray:
        t = self._frame_counter
        frame = np.roll(self._background, shift=(t * 6) % self.width, axis=1)
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)

        radius = self.height // 10
        span_x = self.width - 2 * radius
        span_y = self.height - 2 * radius
        x = radius + abs((t * 11) % (2 * span_x) - span_x)
        y = radius + abs((t * 7) % (2 * span_y) - span_y)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(250, 220, 60))

        margin = self.height // 24
        draw.text(
            (margin, margin),
            f"clip {request.clip_index}  frame {frame_in_clip:02d}/{self.num_frames}  t={t}",
            font=self._font_large,
            fill=(255, 255, 255),
            stroke_width=3,
            stroke_fill=(0, 0, 0),
        )
        prompt_lines = textwrap.wrap(request.prompt, width=60)[:4]
        line_height = self._font_small.size + 8
        for line_index, line in enumerate(prompt_lines):
            draw.text(
                (margin, self.height - margin - line_height * (len(prompt_lines) - line_index)),
                line,
                font=self._font_small,
                fill=(255, 255, 255),
                stroke_width=2,
                stroke_fill=(0, 0, 0),
            )
        return np.asarray(image)

    def _audio(self, request: GenerationRequest, first_frame: int) -> np.ndarray:
        """``[2, samples]`` of tone starting at frame ``first_frame`` of the motion."""
        num_samples = round((self.num_frames / self.fps + _AUDIO_OVERHANG_SECONDS) * self.sample_rate)
        t = first_frame / self.fps + np.arange(num_samples) / self.sample_rate
        frequency = 220.0 * 2 ** ((zlib.crc32(request.prompt.encode()) % 24) / 12)
        pulse = 0.6 + 0.4 * np.cos(2 * np.pi * t)
        left = np.sin(2 * np.pi * frequency * t)
        right = np.sin(2 * np.pi * 1.5 * frequency * t)
        return (0.2 * pulse * np.stack([left, right])).astype(np.float32)

    def generate(self, request: GenerationRequest) -> GeneratedClip:
        started = time.monotonic()
        if request.restarted:
            self._frame_counter = 0
            if request.input_frame is not None:
                self._background = np.ascontiguousarray(request.input_frame)
        audio = self._audio(request, self._frame_counter) if self.sample_rate else None
        frames = np.empty((self.num_frames, self.height, self.width, 3), dtype=np.uint8)
        for frame_in_clip in range(self.num_frames):
            frames[frame_in_clip] = self._render(request, frame_in_clip)
            self._frame_counter += 1
        # The next clip's first frame duplicates this clip's last one and is dropped.
        self._frame_counter -= 1
        remaining = self.generation_seconds - (time.monotonic() - started)
        if remaining > 0:
            time.sleep(remaining)
        return GeneratedClip(frames, audio, self.sample_rate)
