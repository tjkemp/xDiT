"""Cut each clip's soundtrack onto the stream's video timeline."""

from __future__ import annotations

import numpy as np


def _equal_power_crossfade(tail: np.ndarray, head: np.ndarray) -> np.ndarray:
    ramp = (np.arange(len(tail), dtype=np.float32) + 0.5) / len(tail) * (np.pi / 2)
    return tail * np.cos(ramp)[:, None] + head * np.sin(ramp)[:, None]


def _fade(samples: np.ndarray, start: float, end: float) -> np.ndarray:
    return samples * np.linspace(start, end, len(samples), dtype=np.float32)[:, None]


def _take(audio: np.ndarray, start: int, count: int) -> np.ndarray:
    """``audio[start:start + count]``, zero-padded where the clip's audio runs short."""
    out = np.zeros((count, audio.shape[1]), dtype=np.float32)
    available = audio[start : start + count]
    out[: len(available)] = available
    return out


class AudioStitcher:
    """Turns per-clip soundtracks into one gapless track that stays in sync with the video.

    Video frame ``j`` of the stream owns samples
    ``[round(j * sr / fps), round((j + 1) * sr / fps))``, so the sample count
    stays exact over any number of clips and the audio never drifts from the
    video, even though the model's soundtrack runs a little longer than its
    frames.

    A continuation clip regenerates the previous clip's last frame as its own
    frame 0, which the viewer never sees. The audio under that frame covers the
    same moment as the previous clip's last frame, so the two are equal-power
    crossfaded. That is why the last frame's audio is held back until the next
    clip arrives. A restart is not continuous, so the held-back audio fades out
    and the new clip fades in.
    """

    def __init__(self, fps: int, sample_rate: int, channels: int) -> None:
        self.fps = fps
        self.sample_rate = sample_rate
        self.channels = channels
        self.emitted_frames = 0
        self._tail: np.ndarray | None = None

    def _boundary(self, frame: int) -> int:
        return round(frame * self.sample_rate / self.fps)

    def _samples_per_frame(self, frame: int) -> int:
        return self._boundary(frame + 1) - self._boundary(frame)

    def push(self, audio: np.ndarray, num_frames: int, continuation: bool) -> tuple[int, np.ndarray]:
        """Stitch one clip's ``[channels, samples]`` audio.

        ``continuation`` says whether the clip's first frame was dropped as a
        repeat of the previous clip's last one. Returns the stream sample
        index the returned float32 ``[samples, channels]`` audio starts at; it
        always follows on from the previous call's audio.
        """
        audio = np.ascontiguousarray(np.asarray(audio, dtype=np.float32).T)
        if audio.ndim != 2 or audio.shape[1] != self.channels:
            raise ValueError(f"Expected [{self.channels}, samples] audio, got {audio.T.shape}.")
        first_kept = 1 if continuation and self.emitted_frames > 0 else 0
        start = self.emitted_frames
        end = start + num_frames - first_kept
        pieces = []
        cursor = 0
        fade_in = False

        if self._tail is not None:
            if first_kept:
                pieces.append(_equal_power_crossfade(self._tail, _take(audio, 0, len(self._tail))))
                cursor = len(self._tail)
            else:
                pieces.append(_fade(self._tail, 1.0, 0.0))
                fade_in = True
        start_sample = self._boundary(start - 1) if pieces else self._boundary(start)

        body = _take(audio, cursor, self._boundary(end - 1) - self._boundary(start))
        if fade_in and len(body):
            count = min(len(body), self._samples_per_frame(start))
            body[:count] = _fade(body[:count], 0.0, 1.0)
        pieces.append(body)
        cursor += len(body)

        self._tail = _take(audio, cursor, self._samples_per_frame(end - 1))
        self.emitted_frames = end
        return start_sample, np.concatenate(pieces, axis=0)
