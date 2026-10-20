"""The serving rank's clip requests drive the same pipeline call on every rank."""

import queue
import time
import traceback
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = [pytest.mark.gloo, pytest.mark.slow]

_WORLD_SIZE = 3
_PROMPTS = ("misty forest", "neon city")


class _RecordingRunner:
    """Stands in for xFuserModelRunner and its model: records each call, encodes the seed as brightness.

    Even seeds come back as float frames in [0, 1] like the Wan pipelines', odd
    seeds as uint8 torch frames like MiniMax-H3's.
    """

    def __init__(self) -> None:
        self.calls = []
        self.model = self

    def preprocess_args(self, args: dict) -> dict:
        return dict(args)

    def _run_timed_pipe(self, args: dict):
        import torch

        image = args["input_images"][0]
        self.calls.append((args["prompt"], args["seed"], image.size, image.getpixel((0, 0)), args["fps_hint"]))
        if args["seed"] % 2 == 0:
            video = np.full((3, 4, 6, 3), args["seed"] / 100.0, dtype=np.float32)
        else:
            video = torch.full((3, 4, 6, 3), args["seed"] + 64, dtype=torch.uint8)
        return SimpleNamespace(videos=[video]), 0.0


def _worker(rank, world_size, init_method, result_queue):
    dist = None
    try:
        import torch.distributed as dist

        from examples.wan_streaming.model_generator import DistributedClipGenerator
        from examples.wan_streaming.session import GenerationRequest

        dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=world_size)
        runner = _RecordingRunner()
        generator = DistributedClipGenerator(runner, base_args={"fps_hint": 16})

        if generator.is_server_rank:
            clips = []
            for index, prompt in enumerate(_PROMPTS):
                frame = np.full((4, 6, 3), 10 * (index + 1), dtype=np.uint8)
                request = GenerationRequest(prompt, 40 + index, index, frame, restarted=index == 0)
                frames = generator.generate(request).frames
                clips.append((frames.dtype.str, frames.shape, int(frames[0, 0, 0, 0])))
            generator.shutdown()
            result_queue.put(("returned", rank, runner.calls, clips))
        else:
            generator.serve_forever()
            result_queue.put(("returned", rank, runner.calls, None))
    except Exception:  # noqa: BLE001 - report arbitrary child failures to the parent
        result_queue.put(("error", rank, traceback.format_exc(), None))
    finally:
        if dist is not None and dist.is_initialized():
            dist.destroy_process_group()


def _run_spawned(torch, init_method, timeout):
    context = torch.multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    processes = [
        context.Process(target=_worker, args=(rank, _WORLD_SIZE, init_method, result_queue))
        for rank in range(_WORLD_SIZE)
    ]
    for process in processes:
        process.start()

    deadline = time.monotonic() + timeout
    for process in processes:
        process.join(max(0.0, deadline - time.monotonic()))
    hung = [process.pid for process in processes if process.is_alive()]
    for process in processes:
        if process.is_alive():
            process.kill()
            process.join(5)

    results = []
    while len(results) < _WORLD_SIZE:
        try:
            results.append(result_queue.get(timeout=1))
        except queue.Empty:
            break
    return processes, hung, results


def test_every_rank_runs_each_requested_clip_and_the_server_rank_gets_uint8_frames(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("torch.distributed with the gloo backend is unavailable")

    processes, hung, results = _run_spawned(torch, f"file://{tmp_path / 'clip-generator-init'}", timeout=120)

    assert not hung, f"a rank never left serve_forever; results so far: {results}"
    errors = [result for result in results if result[0] == "error"]
    assert not errors, "\n".join(result[2] for result in errors)
    assert [process.exitcode for process in processes] == [0] * _WORLD_SIZE

    calls = {rank: rank_calls for _, rank, rank_calls, _ in results}
    expected_calls = [
        ("misty forest", 40, (6, 4), (10, 10, 10), 16),
        ("neon city", 41, (6, 4), (20, 20, 20), 16),
    ]
    assert calls == {rank: expected_calls for rank in range(_WORLD_SIZE)}

    clips = {rank: rank_clips for _, rank, _, rank_clips in results}
    assert clips[_WORLD_SIZE - 1] == [("|u1", (3, 4, 6, 3), 102), ("|u1", (3, 4, 6, 3), 105)]
    assert all(clips[rank] is None for rank in range(_WORLD_SIZE - 1))
