import io
import time
from urllib.parse import unquote

import numpy as np
import pytest

av = pytest.importorskip("av")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from examples.wan_streaming.server import build_fake_stream, create_app, parse_args  # noqa: E402

WIDTH, HEIGHT, FPS, NUM_FRAMES = 64, 48, 30, 6


@pytest.fixture
def client():
    args = parse_args(
        [
            f"--width={WIDTH}",
            f"--height={HEIGHT}",
            f"--fps={FPS}",
            f"--num-frames={NUM_FRAMES}",
            "--fake-generation-seconds=0",
            "--lead-margin=0",
            "--prompt=forest",
        ]
    )
    app = create_app(build_fake_stream(args), width=WIDTH, height=HEIGHT)
    with TestClient(app) as test_client:
        yield test_client


def _next_segment(client, after: int, stream: str):
    response = client.get("/api/segment", params={"after": after, "stream": stream, "timeout": 10})
    assert response.status_code == 200
    return response


def _wait_for_segment(client, predicate, after: int = -1, stream: str = "", attempts: int = 40):
    for _ in range(attempts):
        response = _next_segment(client, after, stream)
        if predicate(response):
            return response
        after, stream = int(response.headers["X-Segment-Id"]), response.headers["X-Stream-Id"]
    raise AssertionError("No matching segment arrived")


def test_http_segments_are_playable_mp4_in_order(client):
    first = _next_segment(client, -1, "")
    stream = first.headers["X-Stream-Id"]
    second = _next_segment(client, int(first.headers["X-Segment-Id"]), stream)

    video = second.content[: int(second.headers["X-Video-Size"])]
    with av.open(io.BytesIO(video)) as container:
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]

    assert first.headers["content-type"] == "video/mp4"
    assert int(second.headers["X-Segment-Id"]) == int(first.headers["X-Segment-Id"]) + 1
    assert frames[0].shape == (HEIGHT, WIDTH, 3)
    assert len(frames) in (NUM_FRAMES, NUM_FRAMES - 1)


def test_http_segments_carry_the_soundtrack_after_the_video(client):
    first = _next_segment(client, -1, "")
    segment = _next_segment(client, int(first.headers["X-Segment-Id"]), first.headers["X-Stream-Id"])

    video_size, audio_size = int(segment.headers["X-Video-Size"]), int(segment.headers["X-Audio-Size"])
    audio = segment.content[video_size:]
    with av.open(io.BytesIO(audio)) as container:
        samples = sum(frame.samples for frame in container.decode(audio=0))

    assert len(segment.content) == video_size + audio_size
    assert segment.headers["X-Audio-Codec"] == "mp4a.40.2"
    assert 0 < int(segment.headers["X-Audio-Init-Size"]) < audio_size
    assert float(segment.headers["X-Audio-Offset"]) <= 0
    assert samples == round(float(segment.headers["X-Audio-Duration"]) * 48000)


def test_a_prompt_set_over_http_reaches_a_later_segment(client):
    response = client.post("/api/prompt", json={"prompt": "neon city at night"})
    assert response.status_code == 200
    assert response.json()["prompt"] == "neon city at night"

    segment = _wait_for_segment(client, lambda r: unquote(r.headers["X-Prompt"]) == "neon city at night")

    assert segment.headers["X-Restarted"] == "0"


def test_an_empty_prompt_is_rejected(client):
    assert client.post("/api/prompt", json={"prompt": "  "}).status_code == 400
    assert client.get("/api/status").json()["prompt"] == "forest"


def test_an_uploaded_seed_image_restarts_the_chain(client):
    buffer = io.BytesIO()
    Image.fromarray(np.full((120, 200, 3), 200, dtype=np.uint8)).save(buffer, format="PNG")
    before = _next_segment(client, -1, "")

    response = client.post("/api/seed_image", content=buffer.getvalue(), headers={"Content-Type": "image/png"})
    assert response.status_code == 200

    segment = _wait_for_segment(
        client,
        lambda r: r.headers["X-Restarted"] == "1",
        after=int(before.headers["X-Segment-Id"]),
        stream=before.headers["X-Stream-Id"],
    )
    assert int(segment.headers["X-Clip-Index"]) > int(before.headers["X-Clip-Index"])


def test_a_non_image_upload_is_rejected(client):
    response = client.post("/api/seed_image", content=b"not an image", headers={"Content-Type": "image/png"})

    assert response.status_code == 400


def test_websocket_delivers_metadata_then_the_segment_and_answers_pings(client):
    with client.websocket_connect("/api/ws") as websocket:
        websocket.send_json({"type": "ping", "t": 12.5})
        seen_pong = seen_segment = False
        deadline = time.monotonic() + 10
        while not (seen_pong and seen_segment) and time.monotonic() < deadline:
            message = websocket.receive_json()
            if message["type"] == "pong":
                assert message["t"] == 12.5
                seen_pong = True
            elif message["type"] == "segment":
                data = websocket.receive_bytes()
                assert len(data) == message["size"] == message["video_size"] + message["audio_size"]
                assert message["codec"].startswith("avc1.")
                seen_segment = True

    assert seen_pong and seen_segment


def test_websocket_prompt_messages_update_the_session(client):
    with client.websocket_connect("/api/ws") as websocket:
        websocket.send_json({"type": "prompt", "prompt": "desert dunes"})
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            message = websocket.receive_json()
            if message["type"] == "segment":
                websocket.receive_bytes()
            elif message["type"] == "status":
                break

    assert message["type"] == "status"
    assert message["prompt"] == "desert dunes"
