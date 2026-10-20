"""Clip generation with an xDiT model runner spread over torchrun ranks."""

from __future__ import annotations

import argparse
import logging
import threading
from datetime import timedelta
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

from examples.wan_streaming.session import GenerationRequest

logger = logging.getLogger("wan_streaming")

# Worker ranks wait in a broadcast between clips, for as long as the demo runs.
_REQUEST_TIMEOUT = timedelta(days=7)

# The last frame of a MiniMax-H3 T2VA clip of the server's default prompt, at 960x544.
DEFAULT_SEED_IMAGE = Path(__file__).parent / "assets" / "seed.png"


def frames_to_uint8(video) -> np.ndarray:
    """``[F, H, W, 3]`` frames, uint8 or floats in ``[0, 1]``, as a uint8 array."""
    if isinstance(video, torch.Tensor):
        video = video.cpu().numpy()
    video = np.asarray(video)
    if video.dtype == np.uint8:
        return video
    return (np.clip(video, 0.0, 1.0) * 255.0).round().astype(np.uint8)


class DistributedClipGenerator:
    """Runs one pipeline call per clip on every rank.

    xDiT leaves the decoded video on the last rank, so that rank serves the
    stream and broadcasts each request; every other rank waits in
    ``serve_forever`` and joins the same pipeline call.
    """

    def __init__(self, runner, base_args: dict, group=None) -> None:
        self.runner = runner
        self.base_args = base_args
        self.rank = dist.get_rank()
        self.source_rank = dist.get_world_size() - 1
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else None
        self._group = group if group is not None else dist.new_group(backend="gloo", timeout=_REQUEST_TIMEOUT)

    @property
    def is_server_rank(self) -> bool:
        return self.rank == self.source_rank

    def _broadcast(self, payload: dict | None) -> dict | None:
        holder = [payload]
        dist.broadcast_object_list(holder, src=self.source_rank, group=self._group)
        return holder[0]

    def generate(self, request: GenerationRequest) -> np.ndarray:
        if not self.is_server_rank:
            raise RuntimeError("Only the last rank requests clips; the others run serve_forever().")
        if request.input_frame is None:
            raise ValueError("Image-to-video needs an input frame; start the server with --seed-image.")
        payload = self._broadcast({"prompt": request.prompt, "seed": request.seed, "input_frame": request.input_frame})
        return self._run(payload)

    def serve_forever(self) -> None:
        while True:
            payload = self._broadcast(None)
            if payload is None:
                return
            try:
                self._run(payload)
            except Exception:
                # Every rank fails the same call, so the ranks stay in step.
                logger.exception("Clip generation failed on rank %d", self.rank)

    def shutdown(self) -> None:
        if self.is_server_rank:
            self._broadcast(None)

    def _run(self, payload: dict) -> np.ndarray | None:
        # The stream loop calls in from its own thread, whose current device
        # would otherwise default to device 0.
        if self.device is not None:
            torch.cuda.set_device(self.device)
        args = dict(
            self.base_args,
            prompt=payload["prompt"],
            seed=payload["seed"],
            input_images=[Image.fromarray(payload["input_frame"])],
        )
        # Not runner.run(): with torch.compile it adds an untimed CUDA graph
        # capture call every time, doubling the time per clip.
        output, _ = self.runner.model._run_timed_pipe(self.runner.preprocess_args(args))
        if not self.is_server_rank:
            return None
        return frames_to_uint8(output.videos[0])


def _load_runner(args: argparse.Namespace, xdit_argv: list[str]):
    from xfuser import xFuserArgs
    from xfuser.config import FlexibleArgumentParser
    from xfuser.runner import xFuserModelRunner

    from examples.wan_streaming.server import fit_image

    parser = FlexibleArgumentParser(description="xDiT runner arguments")
    config = vars(xFuserArgs.add_runner_args(parser).parse_args(xdit_argv))
    config.update(prompt=args.prompt)

    runner = xFuserModelRunner(config)
    defaults = runner.model.default_input_values
    width = config.get("width") or defaults.width
    height = config.get("height") or defaults.height
    # Every later clip starts from a generated frame at this size, so the seed
    # gets it too: one set of shapes, no recompiles mid-stream.
    seed_image = fit_image(Image.open(args.seed_image), width, height)

    base_args = runner.preprocess_args({**config, "input_images": [Image.fromarray(seed_image)]})
    runner.initialize(base_args)
    runner.model._validate_args(base_args)
    runner.model._run_warmup_calls(base_args)
    return runner, base_args, seed_image


def run_model_stream(args: argparse.Namespace, xdit_argv: list[str]) -> None:
    """Load the model on every rank and serve the stream from the last one."""
    import uvicorn

    from examples.wan_streaming.encoder import encode_fragmented_mp4
    from examples.wan_streaming.segments import SegmentStore
    from examples.wan_streaming.server import create_app
    from examples.wan_streaming.session import PlaybackPacer, StreamSession
    from examples.wan_streaming.stream_loop import StreamLoop

    if not args.seed_image:
        args.seed_image = str(DEFAULT_SEED_IMAGE)
        logger.info("No --seed-image given; starting from %s", args.seed_image)

    runner, base_args, seed_image = _load_runner(args, xdit_argv)
    generator = DistributedClipGenerator(runner, base_args)
    try:
        if not generator.is_server_rank:
            generator.serve_forever()
            return

        fps = runner.model.settings.fps
        height, width = seed_image.shape[:2]
        logger.info("Generating one warm-up clip at %dx%d", width, height)
        generator.generate(GenerationRequest(args.prompt, args.seed, 0, seed_image, restarted=True))

        stream = StreamLoop(
            generator=generator,
            session=StreamSession(prompt=args.prompt, base_seed=args.seed, seed_image=seed_image),
            store=SegmentStore(capacity=args.keep_segments),
            pacer=PlaybackPacer(margin_seconds=args.lead_margin),
            encode=partial(encode_fragmented_mp4, fps=fps, crf=args.crf, preset=args.preset),
            fps=fps,
        )
        app = create_app(stream, width=width, height=height, run_stream_loop=False)
        server = uvicorn.Server(
            uvicorn.Config(app, host=args.host, port=args.port, ws_ping_interval=20, ws_ping_timeout=20)
        )
        threading.Thread(target=server.run, name="http", daemon=True).start()
        logger.info("Streaming on http://%s:%d", args.host, args.port)
        # Generation stays on the main thread, where the model was compiled
        # and warmed up; the web server runs beside it.
        try:
            stream.run_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.should_exit = True
            stream.stop()
            generator.shutdown()
    finally:
        runner.cleanup()
