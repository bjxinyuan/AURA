"""aura.server_io — outbound messages from the inference server to the Flask bridge.

Every send targets a specific StreamingSession's socket. Failures are isolated
per-session so one dropped client does not affect other active sessions.
Used from both the asyncio task in handle_client_connection_async and the
TTS worker thread; the session's own conn_lock serialises concurrent writers
on a single socket.

When a socket write fails with an OSError (broken pipe, reset, etc.) the
session's `conn` is nulled so subsequent sends short-circuit as no-ops
instead of repeatedly hammering a dead fd — this fixes the issue called out
in arch.md §8 "send failed should immediately clear active_connection".
"""
import json
import struct

from aura.protocol import (
    pack_header,
    STREAMING_TOKEN_TYPE,
    ASR_QUERY_ECHO_TYPE,
    TTS_AUDIO_CHUNK_TYPE,
)


def _send(session, msg_type: int, payload: bytes) -> bool:
    """Send one framed message to session.conn. Returns True on success."""
    conn = session.conn
    if conn is None:
        return False
    header = pack_header(msg_type, len(payload))
    try:
        with session.conn_lock:
            conn.sendall(header + payload)
        return True
    except OSError as e:
        # Real socket failure — drop the fd so later sends are no-ops.
        print(f"[send] session {session.session_id} socket dropped: {e}")
        session.conn = None
        return False


def send_streaming_token(session, token: str, response_id: str, is_final: bool = False,
                         query: str = None, is_start: bool = False,
                         is_silent: bool = False):
    """Send a streaming token to the client.

    Protocol: Type 8 (STREAMING_TOKEN)
    - Type 7 is reserved for ERROR messages in the client
    """
    if session.conn is None:
        return
    response_data = {
        "response_id": response_id,
        "token": token,
        "is_final": is_final,
        "type": "streaming_token"
    }
    if is_start:
        response_data["is_start"] = True
    if query:
        response_data["query"] = query

    # Mark silent responses so client can handle them appropriately
    if is_silent:
        response_data["is_silent"] = True

    payload = json.dumps(response_data, ensure_ascii=False).encode('utf-8')
    _send(session, STREAMING_TOKEN_TYPE, payload)


def send_asr_query(session, query: str):
    """Send ASR-transcribed query text to client immediately (before model inference).

    Protocol: Type 10 (ASR_QUERY_ECHO) - a dedicated message type so that
    clients won't mistake it for a model streaming response (Type 8).
    This allows the client to display the user's query as soon as ASR finishes,
    without waiting for the model to start generating (Plan 2 optimization).
    """
    if session.conn is None:
        return
    response_data = {
        "type": "asr_query",
        "query": query,
    }
    payload = json.dumps(response_data, ensure_ascii=False).encode('utf-8')
    if _send(session, ASR_QUERY_ECHO_TYPE, payload):
        print(f"📤 [Plan2] Sent ASR query to client immediately: {query[:50]}...")


def send_audio_chunk(session, pcm_bytes: bytes, response_id: str,
                     sentence_idx: int, chunk_idx: int,
                     sample_rate: int, is_final: bool = False):
    """Send a streaming audio chunk to the client (Raw PCM int16).

    Protocol: Type 9 (TTS Audio Chunk)

    This enables true streaming TTS - each chunk is sent as soon as it's generated,
    allowing the client to start playback before the entire sentence is synthesized.

    Payload format:
    - response_id_len (1 byte)
    - response_id (variable)
    - sentence_idx (2 bytes, big-endian)
    - chunk_idx (2 bytes, big-endian)
    - sample_rate (4 bytes, big-endian)
    - is_final (1 byte: 0 or 1)
    - pcm_data (Raw int16 PCM)
    """
    if session.conn is None:
        return
    response_id_bytes = (response_id or "").encode('utf-8')
    response_id_len = len(response_id_bytes)

    payload = (
        struct.pack(">B", response_id_len) +
        response_id_bytes +
        struct.pack(">HHIB", sentence_idx, chunk_idx, sample_rate, 1 if is_final else 0) +
        pcm_bytes
    )
    if _send(session, TTS_AUDIO_CHUNK_TYPE, payload):
        if is_final:
            print(f"🔊 [Chunk] Sent final chunk for sentence {sentence_idx} (chunk {chunk_idx})")
