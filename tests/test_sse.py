"""Unit tests for the SSE generator helper."""
import json
import queue
import threading
import time

import pytest

from aura.sse import format_sse_stream


def test_initial_frame_is_connected_comment():
    q = queue.Queue()
    q.put(("close", {}))
    frames = list(format_sse_stream(q, heartbeat_seconds=0.1))
    assert frames[0] == ": connected\n\n"


def test_heartbeat_on_empty_queue():
    """With no events for longer than heartbeat_seconds, emit a heartbeat comment."""
    q = queue.Queue()
    gen = format_sse_stream(q, heartbeat_seconds=0.05)
    # First frame is the initial ": connected" comment
    next(gen)
    # Second frame should be a heartbeat (timed out waiting)
    frame = next(gen)
    assert frame == ": heartbeat\n\n"


def test_token_event_framing():
    q = queue.Queue()
    q.put(("token", {"raw": '{"token":"hi"}'}))
    q.put(("close", {}))
    frames = list(format_sse_stream(q, heartbeat_seconds=1.0))
    # Skip the initial ": connected" comment
    token_frame = frames[1]
    assert token_frame.startswith("event: token\n")
    assert token_frame.endswith("\n\n")
    # Extract the data: line and parse it
    data_line = [l for l in token_frame.splitlines() if l.startswith("data:")][0]
    payload = json.loads(data_line[len("data:"):].strip())
    assert payload == {"raw": '{"token":"hi"}'}


def test_close_pill_terminates_generator():
    q = queue.Queue()
    q.put(("token", {"raw": "first"}))
    q.put(("close", {}))
    # Additional items after close should never be consumed
    q.put(("token", {"raw": "after-close"}))
    frames = list(format_sse_stream(q, heartbeat_seconds=1.0))
    # Generator must emit connected + token + close, then stop
    assert len(frames) == 3
    assert frames[2].startswith("event: close\n")
    # The post-close token must still be in the queue
    kind, _ = q.get_nowait()
    assert kind == "token"


def test_multiple_event_kinds_framed_separately():
    q = queue.Queue()
    q.put(("token", {"raw": "t"}))
    q.put(("chunk", {"pcm_base64": "AAAA"}))
    q.put(("error", {"message": "boom"}))
    q.put(("close", {}))
    frames = list(format_sse_stream(q, heartbeat_seconds=1.0))
    kinds = [f.split("\n", 1)[0] for f in frames[1:]]
    assert kinds == ["event: token", "event: chunk", "event: error", "event: close"]


def test_payload_with_unicode_preserved():
    q = queue.Queue()
    q.put(("token", {"raw": "你好"}))
    q.put(("close", {}))
    frames = list(format_sse_stream(q, heartbeat_seconds=1.0))
    data_line = [l for l in frames[1].splitlines() if l.startswith("data:")][0]
    payload = json.loads(data_line[len("data:"):].strip())
    assert payload["raw"] == "你好"
