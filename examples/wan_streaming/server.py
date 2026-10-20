"""Live, prompt-steerable video stream served to a browser.

Clips are generated back to back, each continuing from the previous clip's
last frame, and pushed to the page as fragmented MP4 segments that it plays
gaplessly with Media Source Extensions. Segments go out over a WebSocket, or
over HTTP long-polling where a proxy drops WebSockets.

Test the streaming path without a GPU, from the repository root:

    pip install fastapi "uvicorn[standard]" av
    python -m examples.wan_streaming.server --generator fake --port 8000

then open http://<host>:8000 in Chrome or Firefox (append ?transport=http to
force the HTTP fallback).

Stream an image-to-video model with xDiT; everything after ``--`` goes to the
xDiT runner unchanged, and decides the resolution, frame count and parallelism:

    torchrun --nproc_per_node=8 -m examples.wan_streaming.server \\
        --generator model --seed-image seed.png --prompt "..." -- \\
        --model MiniMax-H3 --task i2va \\
        --height 544 --width 960 --num_frames 124 --num_inference_steps 12 \\
        --attention_backend aiter_bf16 --use_fp8_gemms --ulysses_degree 8 \\
        --use_parallel_vae --use_torch_compile --warmup_calls 1

Each clip must generate faster than its new frames play, here
``(124 - 1) / 24 = 5.125`` seconds. The model's audio is not streamed.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import logging
import sys
import time
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from urllib.parse import quote

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from PIL import Image, ImageOps
from pydantic import BaseModel

from examples.wan_streaming.encoder import encode_fragmented_mp4
from examples.wan_streaming.segments import Segment, SegmentStore
from examples.wan_streaming.session import PlaybackPacer, StreamSession
from examples.wan_streaming.stream_loop import StreamLoop

logger = logging.getLogger("wan_streaming")

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_SEED_IMAGE_BYTES = 20 * 1024 * 1024
HTTP_VIEWER_TIMEOUT_SECONDS = 30.0


class PromptUpdate(BaseModel):
    prompt: str


def fit_image(image: Image.Image, width: int, height: int) -> np.ndarray:
    """Scale and centre-crop ``image`` to exactly ``width`` x ``height`` RGB."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    return np.asarray(ImageOps.fit(image, (width, height), Image.Resampling.LANCZOS))


def _segment_headers(segment: Segment) -> dict[str, str]:
    return {
        "X-Stream-Id": segment.stream_id,
        "X-Segment-Id": str(segment.id),
        "X-Created-At": f"{segment.created_at:.3f}",
        "X-Codec": segment.codec,
        "X-Duration": f"{segment.duration:.6f}",
        "X-Clip-Index": str(segment.clip_index),
        "X-Generation-Seconds": f"{segment.generation_seconds:.3f}",
        "X-Restarted": "1" if segment.restarted else "0",
        "X-Prompt": quote(segment.prompt),
        "Cache-Control": "no-store",
        "X-Accel-Buffering": "no",
    }


def create_app(stream_loop: StreamLoop, width: int, height: int, run_stream_loop: bool = True) -> FastAPI:
    """Build the web app. With ``run_stream_loop=False`` the caller drives ``stream_loop``."""
    session = stream_loop.session
    store = stream_loop.store
    websocket_viewers = 0
    http_viewers: dict[str, float] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        store.attach_loop(asyncio.get_running_loop())
        if run_stream_loop:
            stream_loop.start()
        try:
            yield
        finally:
            if run_stream_loop:
                await asyncio.to_thread(stream_loop.stop)

    app = FastAPI(lifespan=lifespan)

    def status() -> dict:
        now = time.monotonic()
        for viewer, seen in list(http_viewers.items()):
            if now - seen > HTTP_VIEWER_TIMEOUT_SECONDS:
                del http_viewers[viewer]
        latest = store.latest()
        return {
            "prompt": session.prompt,
            "next_clip_index": session.clip_index,
            "buffered_seconds": stream_loop.pacer.buffered_seconds(),
            "estimated_generation_seconds": stream_loop.pacer.estimated_generation_seconds,
            "last_generation_seconds": stream_loop.last_generation_seconds,
            "last_encode_seconds": stream_loop.last_encode_seconds,
            "last_error": stream_loop.last_error,
            "latest_segment": latest.metadata() if latest else None,
            "viewers": {"websocket": websocket_viewers, "http": len(http_viewers)},
            "video": {"width": width, "height": height, "fps": stream_loop.fps},
            "server_time": time.time(),
        }

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/api/status")
    async def get_status() -> dict:
        return status()

    @app.get("/api/ping")
    async def ping() -> dict:
        return {"server_time": time.time()}

    @app.post("/api/prompt")
    async def set_prompt(update: PromptUpdate) -> dict:
        try:
            session.set_prompt(update.prompt)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from None
        logger.info("Prompt set to %r", session.prompt)
        return status()

    @app.post("/api/restart")
    async def restart() -> dict:
        session.restart()
        logger.info("Restarting from the seed image")
        return status()

    @app.post("/api/seed_image")
    async def seed_image(request: Request) -> dict:
        body = await request.body()
        if not body:
            raise HTTPException(status_code=400, detail="Send the image file as the request body.")
        if len(body) > MAX_SEED_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="Seed image is larger than 20 MB.")
        try:
            image = fit_image(Image.open(io.BytesIO(body)), width, height)
        except Exception as error:
            raise HTTPException(status_code=400, detail=f"Could not read image: {error}") from None
        session.restart(image)
        logger.info("Restarting from an uploaded %dx%d seed image", width, height)
        return status()

    @app.get("/api/segment")
    async def get_segment(after: int = -1, stream: str = "", timeout: float = 20.0, viewer: str = "") -> Response:
        if viewer:
            http_viewers[viewer] = time.monotonic()
        if stream != store.stream_id:
            after = -1
        segment = await store.wait_after(after, min(max(timeout, 0.0), 30.0))
        if segment is None:
            return Response(status_code=204, headers={"Cache-Control": "no-store"})
        return Response(content=segment.data, media_type="video/mp4", headers=_segment_headers(segment))

    @app.websocket("/api/ws")
    async def stream_websocket(websocket: WebSocket) -> None:
        nonlocal websocket_viewers
        await websocket.accept()
        websocket_viewers += 1
        send_lock = asyncio.Lock()

        async def send_segments() -> None:
            last_id = -1
            while True:
                segment = await store.wait_after(last_id, 15.0)
                async with send_lock:
                    if segment is None:
                        # Keeps idle proxies from closing the connection.
                        await websocket.send_json({"type": "status", **status()})
                        continue
                    await websocket.send_json({"type": "segment", **segment.metadata()})
                    await websocket.send_bytes(segment.data)
                last_id = segment.id

        sender = asyncio.create_task(send_segments())
        try:
            while True:
                receive = asyncio.create_task(websocket.receive_json())
                done, _ = await asyncio.wait({receive, sender}, return_when=asyncio.FIRST_COMPLETED)
                if sender in done:
                    receive.cancel()
                    sender.result()
                    break
                message = receive.result()
                if message.get("type") == "ping":
                    reply = {"type": "pong", "t": message.get("t"), "server_time": time.time()}
                elif message.get("type") == "prompt":
                    try:
                        session.set_prompt(str(message.get("prompt", "")))
                        reply = {"type": "status", **status()}
                    except ValueError as error:
                        reply = {"type": "error", "detail": str(error)}
                else:
                    reply = {"type": "error", "detail": f"Unknown message type {message.get('type')!r}"}
                async with send_lock:
                    await websocket.send_json(reply)
        except WebSocketDisconnect:
            pass
        finally:
            sender.cancel()
            websocket_viewers -= 1

    return app


def build_fake_stream(args: argparse.Namespace) -> StreamLoop:
    from examples.wan_streaming.fake_generator import FakeGenerator

    seed_image = None
    if args.seed_image:
        seed_image = fit_image(Image.open(args.seed_image), args.width, args.height)
    generator = FakeGenerator(
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        generation_seconds=args.fake_generation_seconds,
    )
    return StreamLoop(
        generator=generator,
        session=StreamSession(prompt=args.prompt, base_seed=args.seed, seed_image=seed_image),
        store=SegmentStore(capacity=args.keep_segments),
        pacer=PlaybackPacer(margin_seconds=args.lead_margin),
        encode=partial(encode_fragmented_mp4, fps=args.fps, crf=args.crf, preset=args.preset),
        fps=args.fps,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--generator",
        choices=["fake", "model"],
        default="fake",
        help="fake: CPU test pattern; model: the xDiT model given after --, launched with torchrun",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", default="A drone shot flying slowly over a misty pine forest at sunrise")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-image", help="Image the first clip starts from")
    fake_only = "fake generator only; the model takes it from the xDiT arguments"
    parser.add_argument("--width", type=int, default=960, help=fake_only)
    parser.add_argument("--height", type=int, default=544, help=fake_only)
    parser.add_argument("--fps", type=int, default=24, help=fake_only)
    parser.add_argument(
        "--num-frames", type=int, default=124, help=f"Frames per clip, input frame included ({fake_only})"
    )
    parser.add_argument(
        "--fake-generation-seconds",
        type=float,
        default=3.0,
        help="How long the fake generator takes per clip; above the clip length it simulates a too-slow model",
    )
    parser.add_argument(
        "--lead-margin",
        type=float,
        default=1.5,
        help="Seconds of slack between a clip being published and the previous one running out",
    )
    parser.add_argument("--keep-segments", type=int, default=6)
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument("--preset", default="veryfast")
    return parser.parse_args(argv)


def split_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split ``[demo args] -- [xDiT args]``."""
    if "--" not in argv:
        return argv, []
    index = argv.index("--")
    return argv[:index], argv[index + 1 :]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    demo_argv, xdit_argv = split_argv(sys.argv[1:])
    args = parse_args(demo_argv)

    if args.generator == "model":
        from examples.wan_streaming.model_generator import run_model_stream

        run_model_stream(args, xdit_argv)
        return

    if xdit_argv:
        raise SystemExit("Arguments after -- are only used with --generator model.")
    import uvicorn

    app = create_app(build_fake_stream(args), width=args.width, height=args.height)
    uvicorn.run(app, host=args.host, port=args.port, ws_ping_interval=20, ws_ping_timeout=20)


if __name__ == "__main__":
    main()
