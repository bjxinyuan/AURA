"""Integration tests for aura.server_io using real socketpair().

These tests cover the per-session socket fix from arch.md §8.1 — the bug
where a global `active_connection` could leak one session's output to a
different session's socket. Key regression: test_two_sessions_independent.
"""
import asyncio
import json
import socket
import struct
import threading
import time

import pytest

from aura.protocol import (
    HEADER_SIZE,
    unpack_header,
    STREAMING_TOKEN_TYPE,
    ASR_QUERY_ECHO_TYPE,
    TTS_AUDIO_CHUNK_TYPE,
)
from aura.session import StreamingSession
from aura.session_history import SessionHistory
from aura.server_io import (
    send_streaming_token,
    send_asr_query,
    send_audio_chunk,
)


def make_session(conn: socket.socket, session_id: str = "test") -> StreamingSession:
    """Build a minimal StreamingSession backed by a real socket."""
    return StreamingSession(
        session_id=session_id,
        history=SessionHistory(),
        input_queue=asyncio.Queue(),
        output_queue=asyncio.Queue(),
        conn=conn,
    )


def recv_message(sock: socket.socket, timeout: float = 2.0) -> tuple[int, bytes]:
    """Read one full [9-byte header][payload] message from sock."""
    sock.settimeout(timeout)
    header = b""
    while len(header) < HEADER_SIZE:
        chunk = sock.recv(HEADER_SIZE - len(header))
        if not chunk:
            raise RuntimeError("socket closed before header complete")
        header += chunk
    msg_type, payload_len = unpack_header(header)
    payload = b""
    while len(payload) < payload_len:
        chunk = sock.recv(payload_len - len(payload))
        if not chunk:
            raise RuntimeError("socket closed mid-payload")
        payload += chunk
    return msg_type, payload


# ---------------------------------------------------------------------------
# Happy-path framing
# ---------------------------------------------------------------------------

def test_send_streaming_token_writes_correct_frame():
    a, b = socket.socketpair()
    try:
        session = make_session(a)
        send_streaming_token(session, token="hi", response_id="r1", is_start=True)
        msg_type, payload = recv_message(b)
        assert msg_type == STREAMING_TOKEN_TYPE
        data = json.loads(payload.decode("utf-8"))
        assert data == {
            "response_id": "r1",
            "token": "hi",
            "is_final": False,
            "type": "streaming_token",
            "is_start": True,
        }
    finally:
        a.close()
        b.close()


def test_send_asr_query_uses_type_10():
    a, b = socket.socketpair()
    try:
        session = make_session(a)
        send_asr_query(session, "what time is it?")
        msg_type, payload = recv_message(b)
        assert msg_type == ASR_QUERY_ECHO_TYPE
        data = json.loads(payload.decode("utf-8"))
        assert data == {"type": "asr_query", "query": "what time is it?"}
    finally:
        a.close()
        b.close()


def test_send_audio_chunk_payload_layout():
    """Verify the exact binary layout of a Type 9 chunk."""
    a, b = socket.socketpair()
    try:
        session = make_session(a)
        pcm = b"\x00\x01" * 16
        send_audio_chunk(
            session, pcm_bytes=pcm, response_id="r1",
            sentence_idx=2, chunk_idx=3, sample_rate=24000, is_final=True,
        )
        msg_type, payload = recv_message(b)
        assert msg_type == TTS_AUDIO_CHUNK_TYPE
        # [rid_len:1][rid:2][sidx:2][cidx:2][sr:4][is_final:1][pcm]
        rid_len = payload[0]
        assert rid_len == len("r1")
        rid = payload[1:1+rid_len].decode("utf-8")
        assert rid == "r1"
        sidx, cidx, sr, is_final = struct.unpack(">HHIB", payload[1+rid_len:1+rid_len+9])
        assert (sidx, cidx, sr, is_final) == (2, 3, 24000, 1)
        assert payload[1+rid_len+9:] == pcm
    finally:
        a.close()
        b.close()


# ---------------------------------------------------------------------------
# Error isolation — the core bug fix
# ---------------------------------------------------------------------------

def test_send_with_null_conn_is_noop():
    """If session.conn is already None, send returns silently."""
    session = make_session(conn=None)
    # Should not raise:
    send_streaming_token(session, "x", "r1")
    send_asr_query(session, "x")
    send_audio_chunk(session, b"\x00\x00", "r1", 0, 0, 24000)


def test_send_after_broken_socket_drops_conn():
    """When the peer is closed, send must catch OSError and null session.conn."""
    a, b = socket.socketpair()
    session = make_session(a, session_id="sess-a")
    b.close()
    # Give the kernel a moment to mark the local end as unusable
    # First send may still succeed (data goes into send buffer before FIN propagates).
    # Keep sending until we detect the drop or hit a safety cap.
    for _ in range(100):
        send_streaming_token(session, "x" * 128, "r1")
        if session.conn is None:
            break
    assert session.conn is None, "expected session.conn to be nulled after broken-pipe"
    # Subsequent sends must be no-ops
    send_streaming_token(session, "y", "r2")   # does not raise
    a.close()


def test_two_sessions_independent():
    """Regression for arch.md §8.1: closing session A's socket must not affect B."""
    a1, a2 = socket.socketpair()
    b1, b2 = socket.socketpair()
    try:
        sess_a = make_session(a1, session_id="A")
        sess_b = make_session(b1, session_id="B")

        # Close A's peer → writes to sess_a eventually fail
        a2.close()
        for _ in range(100):
            send_streaming_token(sess_a, "x" * 128, "r_a")
            if sess_a.conn is None:
                break

        # Now send on B — it must go through untouched
        send_streaming_token(sess_b, "ping", "r_b")
        msg_type, payload = recv_message(b2)
        assert msg_type == STREAMING_TOKEN_TYPE
        data = json.loads(payload)
        assert data["token"] == "ping"
        # Sess A's drop did not corrupt sess B's conn
        assert sess_b.conn is b1
    finally:
        a1.close()
        b1.close()
        b2.close()


# ---------------------------------------------------------------------------
# Concurrency — conn_lock must serialise writers so messages don't interleave
# ---------------------------------------------------------------------------

def test_conn_lock_serialises_writers():
    """Two threads sending concurrently must produce well-framed messages.

    Without per-session conn_lock, two threads could interleave header + payload
    bytes and the receiver would mis-frame subsequent messages. This test sends
    many messages from 2 threads and verifies every one parses correctly.
    """
    a, b = socket.socketpair()
    a.settimeout(5.0)
    b.settimeout(5.0)
    try:
        session = make_session(a)
        N = 50

        def worker(prefix: str):
            for i in range(N):
                send_streaming_token(session, f"{prefix}{i}", f"r_{prefix}")

        t1 = threading.Thread(target=worker, args=("t1_",))
        t2 = threading.Thread(target=worker, args=("t2_",))
        t1.start(); t2.start()

        received = []
        for _ in range(2 * N):
            msg_type, payload = recv_message(b, timeout=5.0)
            assert msg_type == STREAMING_TOKEN_TYPE
            data = json.loads(payload)    # must parse — proves no interleaving
            received.append(data["token"])

        t1.join(); t2.join()
        assert len(received) == 2 * N
        # Both workers' messages should all be present (order not guaranteed)
        assert sum(1 for tok in received if tok.startswith("t1_")) == N
        assert sum(1 for tok in received if tok.startswith("t2_")) == N
    finally:
        a.close()
        b.close()
