"""Server-Sent Events helper.

Frames a multiplexed event queue as an SSE byte stream suitable for use
with Flask's `Response(..., mimetype="text/event-stream")`.

Each queue item is a `(kind, payload)` tuple:
- kind is the SSE `event:` name (e.g. "token", "chunk", "error", "close").
- payload is a JSON-serialisable dict emitted as the `data:` field.

The special kind `"close"` is a poison pill — the generator yields the
event (so the browser sees it) and then stops. Use this to tear down a
session cleanly, e.g. from Flask's force_release_session.

Idle periods emit an SSE comment line ": heartbeat\\n\\n" every
`heartbeat_seconds`. Comments are ignored by EventSource but keep
reverse proxies from idle-timing out the connection.
"""
import json
import queue
from typing import Iterator


def format_sse_stream(
    event_queue: "queue.Queue",
    heartbeat_seconds: float = 15.0,
) -> Iterator[str]:
    """Yield SSE-formatted strings until a ('close', _) item arrives.

    Runs as a Flask streaming generator. Each `yield` flushes one SSE
    frame. Blocks on `event_queue.get(timeout=heartbeat_seconds)` when
    idle.
    """
    # Kick the response off so headers flush immediately.
    yield ": connected\n\n"
    while True:
        try:
            kind, payload = event_queue.get(timeout=heartbeat_seconds)
        except queue.Empty:
            yield ": heartbeat\n\n"
            continue

        data_line = json.dumps(payload, ensure_ascii=False)
        yield f"event: {kind}\ndata: {data_line}\n\n"

        if kind == "close":
            return
