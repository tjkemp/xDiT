import numpy as np
import pytest

from examples.wan_streaming.session import PlaybackPacer, StreamSession


def _clip(first_value: int, num_frames: int = 5) -> np.ndarray:
    return np.stack([np.full((4, 6, 3), first_value + index, dtype=np.uint8) for index in range(num_frames)])


def test_the_first_clip_starts_from_the_seed_image_and_shows_every_frame():
    seed_image = np.full((4, 6, 3), 7, dtype=np.uint8)
    session = StreamSession(prompt="forest", base_seed=100, seed_image=seed_image)

    request = session.next_request()
    frames = _clip(10)

    assert request.restarted
    assert request.input_frame is seed_image
    assert np.array_equal(session.complete(request, frames), frames)


def test_a_continuation_starts_from_the_previous_last_frame_and_drops_the_repeat():
    session = StreamSession(prompt="forest", base_seed=100)
    first = session.next_request()
    first_frames = _clip(10)
    session.complete(first, first_frames)

    second = session.next_request()
    second_frames = _clip(50)

    assert not second.restarted
    assert np.array_equal(second.input_frame, first_frames[-1])
    assert np.array_equal(session.complete(second, second_frames), second_frames[1:])


def test_a_prompt_change_applies_to_the_next_clip_not_the_one_generating():
    session = StreamSession(prompt="forest", base_seed=0)
    in_flight = session.next_request()

    session.set_prompt("  city at night ")
    session.complete(in_flight, _clip(0))

    assert in_flight.prompt == "forest"
    assert session.next_request().prompt == "city at night"


def test_each_clip_gets_the_next_seed():
    session = StreamSession(prompt="forest", base_seed=100)
    seeds = []
    for _ in range(3):
        request = session.next_request()
        seeds.append(request.seed)
        session.complete(request, _clip(0))

    assert seeds == [100, 101, 102]


def test_a_restart_during_generation_wins_over_the_finished_clip():
    session = StreamSession(prompt="forest", base_seed=0, seed_image=np.zeros((4, 6, 3), dtype=np.uint8))
    session.complete(session.next_request(), _clip(0))
    in_flight = session.next_request()

    uploaded = np.full((4, 6, 3), 200, dtype=np.uint8)
    session.restart(uploaded)
    session.complete(in_flight, _clip(30))
    after_restart = session.next_request()

    assert after_restart.restarted
    assert after_restart.input_frame is uploaded


def test_restart_without_an_image_returns_to_the_current_seed_image():
    seed_image = np.full((4, 6, 3), 9, dtype=np.uint8)
    session = StreamSession(prompt="forest", base_seed=0, seed_image=seed_image)
    session.complete(session.next_request(), _clip(0))

    session.restart()

    assert session.next_request().input_frame is seed_image


def test_an_empty_prompt_is_rejected_and_the_old_one_kept():
    session = StreamSession(prompt="forest", base_seed=0)

    with pytest.raises(ValueError):
        session.set_prompt("   ")

    assert session.prompt == "forest"


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_the_pacer_starts_the_next_clip_when_the_buffer_covers_only_generation_and_margin():
    clock = _Clock()
    pacer = PlaybackPacer(margin_seconds=1.0, clock=clock)
    assert pacer.seconds_until_next_start() == 0.0

    pacer.record_generation(2.0)
    pacer.reserve(5.0)
    assert pacer.seconds_until_next_start() == pytest.approx(2.0)

    clock.now += 2.0
    assert pacer.seconds_until_next_start() == pytest.approx(0.0)


def test_after_a_stall_the_pacer_timeline_restarts_from_now():
    clock = _Clock()
    pacer = PlaybackPacer(margin_seconds=0.0, clock=clock)
    pacer.reserve(5.0)

    clock.now += 12.0
    assert pacer.buffered_seconds() == 0.0
    pacer.reserve(5.0)

    assert pacer.buffered_seconds() == pytest.approx(5.0)
