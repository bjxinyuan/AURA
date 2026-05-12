"""StreamingSession — per-connection state carrier.

Moved out of Qwen3_VL_online_streaming_v2_ContextManaged.py during the
refactor that removed the global `active_connection` / `connection_lock`
pair (arch.md §8.1). Each session now owns its socket and a per-socket
write lock so that outbound sends target the correct connection and
concurrent writers (asyncio task + TTS worker thread) are serialised
without affecting other sessions.
"""
import asyncio
import socket
import threading
from dataclasses import dataclass, field
from typing import Any, List, Optional

from aura.session_history import SessionHistory
from aura.cross_turn_penalty import CrossTurnPenalty


@dataclass
class StreamingSession:
    session_id: str
    history: SessionHistory
    input_queue: asyncio.Queue
    output_queue: asyncio.Queue
    is_generating: bool = False
    is_auto_generating: bool = False
    current_task: Optional[asyncio.Task] = None
    cross_turn_penalty: Optional[CrossTurnPenalty] = None
    conn: Optional[socket.socket] = None
    conn_lock: threading.Lock = field(default_factory=threading.Lock)

    # Per-connection scratch state updated by the TCP handler as messages
    # arrive. accumulated_video_frames holds unpacked numpy arrays waiting
    # for a generation trigger; last_prompt is the most recent typed/text
    # prompt not yet consumed by a video-triggered generation;
    # pending_audio is the most recent decoded audio waveform (waveform,
    # sr) for Qwen3-Omni end-to-end mode — the inference loop feeds it
    # to the model directly and extracts the transcription from the
    # <query>...</query> prefix of the response.
    accumulated_video_frames: List[Any] = field(default_factory=list)
    last_prompt: str = ""
    pending_audio: Optional[tuple] = None
