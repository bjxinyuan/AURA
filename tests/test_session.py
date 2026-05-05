"""Tests for aura.session.StreamingSession — pin dataclass defaults.

The defaults matter because several handlers (especially the R6-refactored
_handle_video_msg / _handle_audio_msg / _reset_session_state) expect
accumulated_video_frames and last_prompt to start empty and be mutable
in place. These tests catch any accidental mutable-default-argument bug
or missing field.
"""
import asyncio

from aura.session import StreamingSession
from aura.session_history import SessionHistory


def _make_session(session_id="test") -> StreamingSession:
    """Minimal StreamingSession for unit tests — no real socket, no cross-turn penalty."""
    return StreamingSession(
        session_id=session_id,
        history=SessionHistory(),
        input_queue=asyncio.Queue(),
        output_queue=asyncio.Queue(),
    )


def test_required_fields_round_trip():
    s = _make_session("abc")
    assert s.session_id == "abc"
    assert isinstance(s.history, SessionHistory)
    assert isinstance(s.input_queue, asyncio.Queue)
    assert isinstance(s.output_queue, asyncio.Queue)


def test_default_flags_are_idle():
    s = _make_session()
    assert s.is_generating is False
    assert s.is_auto_generating is False
    assert s.current_task is None
    assert s.cross_turn_penalty is None
    assert s.conn is None


def test_conn_lock_is_independent_per_session():
    """conn_lock uses default_factory=threading.Lock. If someone changes
    it to a shared class-level instance, two sessions' writes would block
    each other — this test catches that regression."""
    a = _make_session("a")
    b = _make_session("b")
    assert a.conn_lock is not b.conn_lock


def test_accumulated_frames_is_independent_per_session():
    """default_factory=list — same concern as above: a shared list would
    let one client's video frames leak into another's accumulation
    buffer."""
    a = _make_session("a")
    b = _make_session("b")
    assert a.accumulated_video_frames == []
    assert b.accumulated_video_frames == []
    a.accumulated_video_frames.append("frame-a")
    assert b.accumulated_video_frames == []
    assert a.accumulated_video_frames is not b.accumulated_video_frames


def test_last_prompt_default_is_empty_string():
    s = _make_session()
    assert s.last_prompt == ""
    s.last_prompt = "hello"
    other = _make_session("other")
    assert other.last_prompt == ""  # independent
