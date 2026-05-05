# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Qwen3 Omni Streaming Input Example

本文件实现了基于 vLLM V1 流式输入 API 的 Qwen3 Omni 模型实时视频流处理。

基于 Qwen3_VL_online_streaming.py 修改，适配 Qwen3 Omni 模型。

1. 输入模态: whisper 转化后的 text + video
2. 输出模态: 仅文字 (之后利用外部 TTS 将文字回复变成语音)
3. 输出模态: 仅文字 (之后利用外部 TTS 将文字回复变成语音)

关键配置差异 (Qwen3 Omni vs Qwen3 VL):
- Silent Token ID: 151676 (Omni) vs 151669 (VL)
- 架构: Qwen3OmniMoeForConditionalGeneration (需要 trust_remote_code=True)
- <|audio_start|> = 151669 (在 Omni 中，VL 中的 151669 是 silent token)
- <|silent|> = 151676 (在 Omni 中)

架构:
┌─────────────────┐      ┌────────────────────┐      ┌──────────────────┐
│   Web Client    │ ◄──► │  AsyncLLM Engine   │ ◄──► │  GPU Workers     │
│   (Browser)     │      │  (Streaming Input) │      │  (Model)         │
└─────────────────┘      └────────────────────┘      └──────────────────┘

启动方式:
    python Qwen3_omni_online_streaming.py --listen-port 12345

依赖:
    - vllm >= 0.14.0rc2 (支持 StreamingInput)
    - 需要 vLLM V1 引擎
"""

import argparse
import logging
import asyncio
import json
import os
import signal
import socket
import struct
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import aiohttp
import requests
import re  # Added for TTS sentence splitting

from datetime import datetime

# TTS is now a separate service (tts_service.py), no local model import needed

# vLLM V1 imports for streaming input
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM, StreamingInput

# Context management (reuse remove_markdown for TTS)
from aura.text_utils import remove_markdown

# Extracted modules (see aura/)
from aura.session_history import SessionHistory, SILENT_TEXT
from aura.cross_turn_penalty import CrossTurnPenalty
from aura.session import StreamingSession
from aura.server_io import (
    send_streaming_token,
    send_asr_query,
)
from aura.tts import TTSController
from aura.media import downsample_video_to_numpy

# Global configuration
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# ============================================================================
# Data Classes
# ============================================================================

# ============================================================================
# Session History Management
# ============================================================================

# Special token IDs
# Qwen3 Omni 和 Qwen3 VL 的 silent token ID 不同:
#   Qwen3 VL:   silent_token_id = 151669  (<|silent|>)
#   Qwen3 Omni: silent_token_id = 151676  (<|silent|>), 151669 在 Omni 中是 <|audio_start|>
# silent_token_id 在 main() 中根据 --model 路径动态设置 (see ctx.silent_token_id).
IM_END_TOKEN_ID = 151645   # <|im_end|> token id

VISION_START_TOKEN_ID = 151652 # <|vision_start|>
VISION_END_TOKEN_ID = 151653   # <|vision_end|>
VIDEO_PAD_TOKEN_ID = 151656    # <|video_pad|>
IMAGE_PAD_TOKEN_ID = 151655    # <|image_pad|>



logger = logging.getLogger("aura.inference")


# ============================================================================
# Server Context — process-wide mutable state in one container.
# ============================================================================
#
# Everything mutable that used to live as `global FOO` now lives as a field
# on the module-level `ctx` singleton below. Fields can be reassigned
# (`ctx.async_engine = ...`) without a `global` declaration because the
# binding of `ctx` itself never changes — only its attributes do.
#
# Constants (IM_END_TOKEN_ID, VISION_*, VIDEO_DIR, AUDIO_DIR, TTS_OUTPUT_DIR)
# stay as module-level names since they don't mutate at runtime.

@dataclass
class ServerContext:
    silent_token_id: Optional[int] = None
    tts_ctl: TTSController = field(default_factory=TTSController)
    streaming_sessions: dict = field(default_factory=dict)
    session_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    async_engine: Optional[AsyncLLM] = None
    model_tokenizer: object = None
    response_id_counter: int = 0
    response_id_lock: threading.Lock = field(default_factory=threading.Lock)


ctx = ServerContext()


def detect_model_type(model_path: str) -> str:
    """根据模型路径判断模型类型: 'omni' 或 'vl'"""
    name = os.path.basename(model_path.rstrip("/")).lower()
    if "omni" in name:
        return "omni"
    return "vl"


def setup_silent_token_id(model_path: str):
    """根据模型路径设置 ctx.silent_token_id"""
    model_type = detect_model_type(model_path)
    if model_type == "omni":
        ctx.silent_token_id = 151676
    else:
        ctx.silent_token_id = 151669
    logger.info(f"🔧 Model type detected: {model_type} → silent_token_id = {ctx.silent_token_id}")


# SessionHistory has moved to aura/session_history.py (see arch.md §5.1).


# ============================================================================
# Cross-Turn Repetition Penalty (adapted from streaming_client.py)
# ============================================================================

# CrossTurnPenalty has moved to aura/cross_turn_penalty.py (see arch.md §5.3).


# StreamingSession moved to aura/session.py


# Directories
VIDEO_DIR = "real_time_captured_video"
AUDIO_DIR = "real_time_captured_audio"
TTS_OUTPUT_DIR = "tts_results"


def generate_response_id() -> str:
    """Generate a unique response ID."""
    with ctx.response_id_lock:
        ctx.response_id_counter += 1
        return f"resp_{int(time.time() * 1000)}_{ctx.response_id_counter}"


# Video decoding moved to aura/media.py.


# ============================================================================
# ASR (Automatic Speech Recognition)
# ============================================================================

def get_audio_prompt(audio_path: str, asr_url: str) -> str:
    """
    Transcribe audio file to text using ASR service (synchronous version).

    Args:
        audio_path: Path to the audio file (MP3, WAV, etc.)
        asr_url: URL of the ASR service

    Returns:
        Transcribed text, or empty string if failed
    """
    logger.info(f"🎤 Transcribing audio from {audio_path}...")
    try:
        with open(audio_path, 'rb') as f:
            files = {'file': f}
            # Request only ASR, do not trigger vLLM in ASR service
            response = requests.post(asr_url, files=files, params={"run_vllm": "false"}, timeout=30)

        if response.status_code == 200:
            data = response.json()
            text = data.get("text", "")
            logger.info(f"✅ Transcribed: {text!r}")
            return text
        else:
            logger.error(f"❌ ASR failed with status {response.status_code}: {response.text}")
            return ""
    except requests.exceptions.Timeout:
        logger.error("❌ ASR request timeout")
        return ""
    except requests.RequestException as e:
        logger.error(f"❌ ASR error: {e}")
        return ""


async def transcribe_audio_async(audio_path: str, asr_url: str) -> str:
    """
    Transcribe audio file to text using ASR service (asynchronous version).

    Args:
        audio_path: Path to the audio file (MP3, WAV, etc.)
        asr_url: URL of the ASR service

    Returns:
        Transcribed text, or empty string if failed
    """
    logger.info(f"🎤 [Async] Transcribing audio from {audio_path}...")
    try:
        async with aiohttp.ClientSession() as session:
            with open(audio_path, 'rb') as f:
                data = aiohttp.FormData()
                data.add_field('file', f, filename=os.path.basename(audio_path))

                async with session.post(
                    asr_url,
                    data=data,
                    params={"run_vllm": "false"},
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        text = result.get("text", "")
                        logger.info(f"✅ [Async] Transcribed: {text!r}")
                        return text
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ [Async] ASR failed with status {response.status}: {error_text}")
                        return ""
    except asyncio.TimeoutError:
        logger.error("❌ [Async] ASR request timeout")
        return ""
    except aiohttp.ClientError as e:
        logger.error(f"❌ [Async] ASR error: {e}")
        return ""


# TTS pipeline moved to aura/tts.py — see `ctx.tts_ctl` above.


# ============================================================================
# AsyncLLM Engine Management
# ============================================================================

async def init_async_engine(args) -> AsyncLLM:
    """Initialize the AsyncLLM engine with streaming support."""
    # Build engine args dict, only include non-None values
    engine_kwargs = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "max_model_len": args.max_model_len,
        "trust_remote_code": detect_model_type(args.model) == "omni",
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
        "limit_mm_per_prompt": {"image": args.max_images_per_prompt},
        "enable_expert_parallel": args.enable_expert_parallel
    }

    # Add optional advanced parameters if specified
    if args.kv_offloading_size is not None:
        engine_kwargs["kv_offloading_size"] = args.kv_offloading_size
    if args.mm_encoder_attn_backend is not None:
        engine_kwargs["mm_encoder_attn_backend"] = args.mm_encoder_attn_backend
    if args.mm_encoder_tp_mode is not None:
        engine_kwargs["mm_encoder_tp_mode"] = args.mm_encoder_tp_mode
    if args.disable_hybrid_kv_cache_manager:
        engine_kwargs["disable_hybrid_kv_cache_manager"] = True
    if args.block_size is not None:
        engine_kwargs["block_size"] = args.block_size
    if args.cache_dtype is not None:
        engine_kwargs["kv_cache_dtype"] = args.cache_dtype
    if args.prefix_caching_hash_algo is not None:
        engine_kwargs["prefix_caching_hash_algo"] = args.prefix_caching_hash_algo
    if args.max_num_batched_tokens is not None:
        engine_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens

    engine_kwargs["cudagraph_capture_sizes"] = [1,2,4]

    engine_args = AsyncEngineArgs(**engine_kwargs)

    model_label = "Qwen3 Omni" if detect_model_type(args.model) == "omni" else "Qwen3 VL"
    logger.info(f"🚀 Initializing {model_label} AsyncLLM engine with model: {args.model}")
    logger.info(f"   trust_remote_code={engine_kwargs['trust_remote_code']}, silent_token_id={ctx.silent_token_id}")
    ctx.async_engine = AsyncLLM.from_engine_args(engine_args)
    logger.info(f"✅ {model_label} AsyncLLM engine initialized successfully")

    # Store tokenizer on ctx for CrossTurnPenalty
    ctx.model_tokenizer = ctx.async_engine.get_tokenizer()

    # ========== DEBUG: 保存词表到日志文件 ==========
    try:
        tokenizer = ctx.model_tokenizer
        vocab = tokenizer.get_vocab()  # Dict[str, int]: token_str -> token_id

        vocab_log_path = "vocab_debug.log"
        with open(vocab_log_path, "w", encoding="utf-8") as f:
            f.write(f"# Vocabulary Debug Log\n")
            f.write(f"# Model: {args.model}\n")
            f.write(f"# Vocab Size: {len(vocab)}\n")
            f.write(f"# Format: token_id | token_str | repr(token_str)\n")
            f.write("=" * 80 + "\n\n")

            # 按 token_id 排序输出
            sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
            for token_str, token_id in sorted_vocab:
                # 保持格式：ID | 原始字符串 | repr表示（显示转义字符）
                f.write(f"{token_id:>8} | {token_str:<40} | {repr(token_str)}\n")

        logger.info(f"📝 Vocabulary saved to {vocab_log_path} ({len(vocab)} tokens)")
    except OSError as e:
        logger.warning(f"⚠️ Failed to save vocabulary: {e}")
    # ========== END DEBUG ==========

    # Note: No need to load transformers processor!
    # vLLM handles multimodal processing internally.
    # We use the pattern from test_qwen2_5_vl.py:
    # - Build prompt string with placeholders
    # - Pass images via multi_modal_data

    return ctx.async_engine


async def generate_response_with_video(
    session: StreamingSession,
    video_tuple: tuple,
    prompt: str,
    sampling_params: SamplingParams,
    args = None,
):
    """
    Generate response for a single turn using video input.
    Qwen3-VL expects video as (numpy_array, metadata_dict) tuple.

    Args:
        video_tuple: Tuple of (numpy_array, metadata_dict)
                    numpy_array shape: (num_frames, height, width, 3)
                    metadata_dict: {"fps": float, "duration": float, ...}
    """
    logger.info(f"==== Calling generate_response_with_video() ====")

    if ctx.async_engine is None:
        raise RuntimeError("AsyncLLM engine not initialized")

    if video_tuple is None or video_tuple[0] is None:
        logger.warning("⚠️ No valid video for generation")
        session.is_generating = False
        return

    # Qwen3-VL requires at least 2 frames (temporal_factor=2)
    # Defensive check: if only 1 frame, duplicate it
    video_array_check = video_tuple[0]
    if video_array_check.shape[0] < 2:
        import numpy as np
        logger.warning(f"⚠️ Video has only {video_array_check.shape[0]} frame(s), duplicating to meet Qwen3-VL minimum (2 frames)")
        duplicated_array = np.concatenate([video_array_check] * 2, axis=0)[:2]
        video_metadata = video_tuple[1].copy() if video_tuple[1] else {}
        video_metadata["total_num_frames"] = 2
        video_metadata["duration"] = 2 / video_metadata.get("fps", 2.0)
        video_tuple = (duplicated_array, video_metadata)

    request_id = ""
    streaming_started = False
    try:
        # Add current turn to history with video tuple
        session.history.add_user_message(prompt, video_tuple=video_tuple)

        # Get vLLM inputs (with history for Prefix Caching)
        # t_get_inputs_start = time.time()
        vllm_inputs = session.history.get_vllm_inputs()
        # t_get_inputs_end = time.time()
        # logger.info(f"⏱️ [TIMING] get_vllm_inputs() took {(t_get_inputs_end - t_get_inputs_start)*1000:.1f}ms")

        # Generate unique request ID for this turn
        request_id = generate_response_id()

        session.history.save_context_debug(request_id=request_id)

        video_array, video_metadata = video_tuple
        logger.info(f"🎬 [Session {session.session_id}] Starting generation (request_id={request_id})")
        logger.info(f"📥 Input: {video_array.shape[0]} video frames ({video_array.shape}), prompt='{prompt}'")

        full_response = ""
        previous_text = ""
        is_silent_response = False
        ttft = 0

        tts_enabled_for_this_response = args and args.enable_tts and ctx.tts_ctl.enabled

        if tts_enabled_for_this_response and prompt:
            ctx.tts_ctl.clear_queue(new_response_id=request_id)

        # Timing measurements
        generation_start_time = time.time()
        first_token_time = None
        token_count = 0
        logger.info(f"[TTFT_DEBUG] stream generate_start request_id={request_id} t={generation_start_time:.6f}")
        logger.info(f"⏱️ [TIMING] prefill_submit_time={generation_start_time:.6f} (engine.generate called, prefill starts)")

        # Incremental sentence buffer for streaming TTS
        _tts_sentence_buf = ""
        _tts_sentence_idx = 0
        _SENT_ENDS = frozenset("。！？；.!?;\n")
        _COMMA_ENDS = frozenset("，,")
        _TTS_MIN_CHARS = 10
        streaming_started = False

        # ===== Streaming generation: send tokens to frontend as they arrive =====
        async for response in ctx.async_engine.generate(
            prompt=vllm_inputs,
            sampling_params=sampling_params,
            request_id=request_id,
        ):
            if first_token_time is None:
                first_token_time = time.time()
                ttft = first_token_time - generation_start_time
                logger.info(f"[TTFT_DEBUG] stream first_token request_id={request_id} t={first_token_time:.6f} ttft_ms={ttft*1000:.1f}")

            token_count += 1
            if response.outputs:
                output = response.outputs[0]

                if len(output.token_ids) > 0 and output.token_ids[0] in (ctx.silent_token_id, IM_END_TOKEN_ID):
                    is_silent_response = True
                    first_tid = output.token_ids[0]
                    tag = "SILENT" if first_tid == ctx.silent_token_id else "IM_END"
                    logger.info(f"🔇 [Session {session.session_id}] {tag} as first token → silent response "
                          f"(first_token_id={first_tid}, token_ids={list(output.token_ids[:5])}...)")
                    break

                if hasattr(output, 'text') and output.text:
                    current_text = output.text
                    if len(current_text) > len(previous_text):
                        delta = current_text[len(previous_text):]
                        previous_text = current_text
                        full_response += delta

                        # --- Stream delta to frontend ---
                        if not streaming_started:
                            send_streaming_token(session, delta, request_id, is_start=True)
                            streaming_started = True
                        else:
                            send_streaming_token(session, delta, request_id)

                        # --- Incremental TTS sentence detection ---
                        if tts_enabled_for_this_response:
                            _tts_sentence_buf += delta
                            while _tts_sentence_buf:
                                split_pos = -1
                                for i, ch in enumerate(_tts_sentence_buf):
                                    if ch in _SENT_ENDS:
                                        split_pos = i + 1
                                        break
                                    if ch in _COMMA_ENDS and i + 1 >= _TTS_MIN_CHARS:
                                        split_pos = i + 1
                                        break
                                if split_pos < 0:
                                    break
                                sentence = _tts_sentence_buf[:split_pos]
                                _tts_sentence_buf = _tts_sentence_buf[split_pos:]
                                if sentence.strip():
                                    ctx.tts_ctl.enqueue_sentence(sentence, session, request_id, _tts_sentence_idx, args)
                                    _tts_sentence_idx += 1

        # ===== Generation finished — decide: silent / send =====
        generation_end_time = time.time()
        total_time = generation_end_time - generation_start_time

        logger.info(f"✅ [Session {session.session_id}] Generation finished")
        logger.info(f"⏱️ [TIMING] Time to first token (TTFT): {ttft*1000:.1f}ms, timestamp: {time.time()}")
        logger.info(f"⏱️ [TIMING] TTFT avg. by {video_array.shape[0]} frames: {(ttft*1000/video_array.shape[0]):.1f}ms")
        logger.info(f"⏱️ [TIMING] Total generation time: {total_time*1000:.1f}ms")
        logger.info(f"⏱️ [TIMING] Tokens generated: {token_count}")
        if token_count > 1 and first_token_time is not None:
            decode_only_ms = (generation_end_time - first_token_time) * 1000
            avg_decode_per_token = decode_only_ms / (token_count - 1)
            logger.info(f"⏱️ [TIMING] Decode phase: {decode_only_ms:.1f}ms for {token_count-1} tokens, "
                  f"avg={avg_decode_per_token:.1f}ms/token ({1000/avg_decode_per_token:.1f} tokens/s)")

        logger.info(f"📋 [DECISION] request_id={request_id} | is_silent={is_silent_response} | "
              f"full_response({len(full_response)} chars)='{full_response[:80]}'")

        if is_silent_response:
            logger.info(f"🔇 [DECISION] → MODEL_SILENT (first token was silent/im_end)")
            session.history.add_assistant_message(SILENT_TEXT)
            send_streaming_token(session, SILENT_TEXT, request_id, is_final=True, is_silent=True)
            if session.cross_turn_penalty is not None:
                session.cross_turn_penalty.record(None)

        else:
            # Send final marker to frontend (content already streamed)
            send_streaming_token(session, "", request_id, is_final=True)
            session.history.add_assistant_message(full_response)
            if session.cross_turn_penalty is not None:
                session.cross_turn_penalty.record(full_response)

            # Flush remaining TTS sentence buffer
            if tts_enabled_for_this_response and _tts_sentence_buf.strip():
                ctx.tts_ctl.enqueue_sentence(_tts_sentence_buf, session, request_id, _tts_sentence_idx, args)
                _tts_sentence_idx += 1

            if tts_enabled_for_this_response:
                logger.info(f"🎤 [Queue] Streamed {_tts_sentence_idx} sentences to TTS")

    except asyncio.CancelledError:
        logger.info(f"⏹ [Session {session.session_id}] Generation cancelled (context reset)")
        if streaming_started and request_id:
            send_streaming_token(session, "", request_id, is_final=True)
    except Exception as e:
        # Top-level catch for the async generation task: any uncaught exception
        # here would otherwise silently kill the asyncio task and strand the
        # client waiting for a final token. Keep broad; log stack for diagnosis.
        logger.error(f"❌ [Session {session.session_id}] Generation error: {e}")
        import traceback
        traceback.print_exc()
        if request_id:
            send_streaming_token(session, "", request_id, is_final=True)
    finally:
        # CRITICAL: Always reset the generating flags
        session.is_generating = False
        session.is_auto_generating = False
        session.current_task = None
        logger.info(f"🔓 [Session {session.session_id}] Released generation lock")


# ============================================================================
# Socket Server for Client Communication
# ============================================================================

# 4 send_* functions moved to aura/server_io.py — per-session, fail-fast.


def recv_exactly(conn, n: int, timeout: float = 30.0) -> bytes:
    """Receive exactly n bytes from socket, blocking."""
    conn.settimeout(timeout)
    data = b""
    while len(data) < n:
        try:
            chunk = conn.recv(n - len(data))
            if not chunk:
                raise ConnectionError("Connection closed")
            data += chunk
        except socket.timeout:
            raise TimeoutError(f"Timeout waiting for {n} bytes")
    return data


async def _read_header(conn) -> Optional[tuple]:
    """Read a 9-byte protocol header. Returns (file_type, file_len) or None
    when the peer disconnects or the socket errors out. Idle timeouts
    simply return a sentinel ("timeout",) so the caller can continue
    waiting without treating it as an error."""
    try:
        header = await asyncio.get_event_loop().run_in_executor(
            None, recv_exactly, conn, 9, 5.0
        )
    except TimeoutError:
        return ("timeout",)
    except ConnectionError:
        logger.info("🔌 Client disconnected")
        return None
    except OSError as e:
        logger.error(f"❌ Header read error: {e}")
        return None

    file_type, file_len = struct.unpack(">BQ", header)
    logger.info(f"📩 Received: type={file_type}, length={file_len}, time={datetime.now().strftime('%H:%M:%S.%f')}")
    return file_type, file_len


async def _read_payload(conn, file_len: int) -> Optional[bytes]:
    """Read `file_len` bytes of payload. Returns the bytes, or None when
    the socket errors out. On a timeout returns the sentinel ``b"\\x00TIMEOUT"``
    wrapper? Simpler: return empty bytes on timeout so caller can skip
    the message without breaking the loop."""
    try:
        return await asyncio.get_event_loop().run_in_executor(
            None, recv_exactly, conn, file_len, 30.0
        )
    except TimeoutError:
        logger.warning(f"⚠ Timeout reading {file_len} bytes, skipping")
        return b""
    except ConnectionError:
        logger.info("🔌 Client disconnected during data read")
        return None
    except OSError as e:
        logger.error(f"❌ Data read error: {e}")
        return None


def _reset_session_state(session: StreamingSession, reason: str):
    """Cancel any running generation task and clear per-connection scratch
    state. Shared by the Clear-Context (Type 4) and Start-Camera (Type 6)
    branches — they differ only in log message."""
    logger.info(reason)
    if session.current_task and not session.current_task.done():
        logger.info("⏹ Cancelling running generation task...")
        session.current_task.cancel()
        session.is_generating = False
        session.is_auto_generating = False
    session.history._reset()
    if session.cross_turn_penalty is not None:
        session.cross_turn_penalty.reset()
    session.accumulated_video_frames = []
    session.last_prompt = ""


def _decode_webm_to_frames(file_data: bytes, args) -> Optional[tuple]:
    """Write a WebM blob to a temp file, decode to (frames, metadata), and
    clean up. Returns (video_array, metadata) or (None, None) on failure.
    Returns the sentinel ("too_small",) when the payload is clearly
    truncated so the caller can log once and skip."""
    MIN_WEBM_SIZE = 1000  # 1KB minimum for a valid EBML header
    if len(file_data) < MIN_WEBM_SIZE:
        logger.warning(f"⚠️ Video data too small ({len(file_data)} bytes < {MIN_WEBM_SIZE}), skipping corrupted/incomplete data")
        return ("too_small",)

    logger.info("🎥 Processing video data...")
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(file_data)
        input_path = tmp.name

    try:
        return downsample_video_to_numpy(input_path, target_fps=args.target_fps)
    finally:
        try:
            os.remove(input_path)
        except OSError:
            pass


def _maybe_launch_generation(session: StreamingSession, args):
    """If we have enough accumulated frames (or a pending user prompt),
    concatenate them, build sampling params, and launch the generation
    background task. Clears accumulated frames once the task is scheduled."""
    total_frames = sum(arr.shape[0] for arr in session.accumulated_video_frames)

    should_process = False
    # Priority trigger: pending user prompt.
    if session.last_prompt and total_frames > 0:
        logger.info(f"⚡ Triggering immediate generation for user prompt (frames={total_frames})")
        should_process = True
    # Background trigger: have enough frames and the task is idle.
    elif total_frames >= 2 and not session.is_generating:
        logger.info(f"⚡ Triggering background generation (frames={total_frames})")
        should_process = True

    if not should_process:
        return
    if session.is_generating:
        if session.last_prompt:
            logger.info("⏳ Waiting for previous generation to finish before processing prompt...")
        return

    # Concatenate all accumulated frames.
    all_frames = np.concatenate(session.accumulated_video_frames, axis=0)

    # Qwen3-VL requires at least 2 frames (temporal_factor=2). Duplicate
    # a single frame if necessary.
    if all_frames.shape[0] == 1:
        logger.warning(f"⚠️ Only 1 frame, duplicating to meet Qwen3-VL minimum requirement (2 frames)")
        all_frames = np.concatenate([all_frames, all_frames], axis=0)

    # Limit to max 16 frames to avoid OOM.
    if all_frames.shape[0] > 16:
        all_frames = all_frames[-16:]

    video_metadata = {
        "fps": args.target_fps,
        "duration": all_frames.shape[0] / args.target_fps,
        "total_num_frames": all_frames.shape[0],
        "frames_indices": list(range(all_frames.shape[0])),
        "video_backend": "opencv",
        "do_sample_frames": False,
    }
    video_tuple = (all_frames, video_metadata)
    session.accumulated_video_frames = []

    # Mark as generating IMMEDIATELY to prevent double trigger.
    session.is_generating = True

    current_prompt = session.last_prompt
    session.last_prompt = ""
    session.is_auto_generating = (current_prompt == "")

    penalty_kwargs = {}
    if session.cross_turn_penalty is not None:
        penalty_kwargs = session.cross_turn_penalty.build_sampling_kwargs()

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        **penalty_kwargs,
    )

    session.current_task = asyncio.create_task(generate_response_with_video(
        session,
        video_tuple,
        current_prompt,
        sampling_params,
        args
    ))


def _handle_video_msg(file_data: bytes, session: StreamingSession, args):
    """Decode a WebM chunk, accumulate frames on the session, and
    opportunistically launch generation."""
    result = _decode_webm_to_frames(file_data, args)
    if result == ("too_small",):
        return
    video_array, _metadata = result
    if video_array is None:
        logger.error("❌ Video processing failed - no frames extracted (possible codec incompatibility with iOS Chrome)")
        return

    session.accumulated_video_frames.append(video_array)
    total_frames = sum(arr.shape[0] for arr in session.accumulated_video_frames)
    logger.info(f"📹 Got {video_array.shape[0]} frames, total accumulated: {total_frames}")

    _maybe_launch_generation(session, args)


async def _handle_audio_msg(file_data: bytes, session: StreamingSession, args):
    """Persist audio, run ASR, and set session.last_prompt. If a previous
    auto-generation is still running, cancel it so the user prompt can
    take precedence."""
    audio_path = os.path.join(AUDIO_DIR, "latest.mp3")
    os.makedirs(AUDIO_DIR, exist_ok=True)
    with open(audio_path, "wb") as f:
        f.write(file_data)
    logger.info(f"🎤 Saved audio to {audio_path}")

    _asr_start = time.time()
    if args.asr_sync:
        loop = asyncio.get_event_loop()
        transcribed_text = await loop.run_in_executor(
            None, get_audio_prompt, audio_path, args.asr_url
        )
    else:
        transcribed_text = await transcribe_audio_async(audio_path, args.asr_url)
    _asr_end = time.time()
    logger.info(f"⏱️ [TIMING] ASR latency: {(_asr_end - _asr_start)*1000:.1f}ms")

    if not transcribed_text:
        logger.warning("⚠ ASR returned empty, will use default prompt")
        return

    session.last_prompt = transcribed_text
    logger.info(f"📝 Set prompt from ASR: {session.last_prompt[:50]}...")

    # Plan 2: ASR query is sent to client immediately; model inference
    # result will be sent separately later.
    send_asr_query(session, transcribed_text)

    # Optimization: try to trigger immediately if we have any video frames.
    if session.accumulated_video_frames:
        logger.info("🚀 Audio arrived, attempting immediate trigger...")
        if session.is_generating and session.is_auto_generating:
            if session.current_task and not session.current_task.done():
                logger.info("🛑 Interrupting auto-generation for user prompt (from Audio event)!")
                session.current_task.cancel()


async def handle_client_connection_async(conn, addr, args):
    """Handle client connection with async support."""
    logger.info(f"================================================")
    logger.info(f"✅ Connected by {addr} with SUYI")

    # Set socket to blocking mode with timeout
    conn.setblocking(True)
    conn.settimeout(1.0)  # 1 second timeout for initial reads

    # Create a streaming session for this client
    session_id = f"client-{addr[0]}-{addr[1]}-{int(time.time())}"

    # Build cross-turn penalty manager (if enabled)
    penalty_mgr = None
    if getattr(args, "cross_turn_penalty", 0) > 0 and ctx.model_tokenizer is not None:
        penalty_mgr = CrossTurnPenalty(
            tokenizer=ctx.model_tokenizer,
            window=getattr(args, "cross_turn_lookback", 2),
            logit_penalty=args.cross_turn_penalty,
            ngram_sizes=getattr(args, "cross_turn_ngram_sizes", [3, 4, 5]),
        )
        logger.info(f"🔧 [Session {session_id}] CrossTurnPenalty enabled: "
              f"penalty={args.cross_turn_penalty}, window={args.cross_turn_lookback}, "
              f"ngram_sizes={args.cross_turn_ngram_sizes}")

    # Session state with History
    session = StreamingSession(
        session_id=session_id,
        history=SessionHistory(
            max_rounds=args.max_rounds,
            num_rounds_keep=args.num_rounds_keep,
            pruning_enabled=args.enable_pruning,
            debug_context_file=args.debug_context_file if args.debug_context else None,
            max_context_qas=args.max_context_qas,
        ),
        input_queue=asyncio.Queue(),
        output_queue=asyncio.Queue(),
        is_generating=False,
        cross_turn_penalty=penalty_mgr,
    )
    session.conn = conn

    async with ctx.session_lock:
        ctx.streaming_sessions[session_id] = session

    try:
        while True:
            header_result = await _read_header(conn)
            if header_result is None:
                break
            if header_result[0] == "timeout":
                continue
            file_type, file_len = header_result

            # Sanity check for length (prevent memory issues)
            if file_len > 100 * 1024 * 1024:  # 100MB max
                logger.warning(f"⚠ Invalid length {file_len}, skipping message")
                continue

            file_data = await _read_payload(conn, file_len)
            if file_data is None:
                break
            if file_data == b"":
                # Timeout on payload — skip, keep loop alive.
                continue

            # Yield immediately so other tasks (e.g. generate() waiting for
            # first token) can run. Avoids event loop starvation that causes
            # ~800ms TTFT delay.
            await asyncio.sleep(0)

            if file_type == 1:  # Video (WebM)
                _handle_video_msg(file_data, session, args)
            elif file_type == 2:  # Audio
                await _handle_audio_msg(file_data, session, args)
            elif file_type == 4:  # Clear Context
                _reset_session_state(session, "🗑 Clearing context...")
            elif file_type == 6:  # Start Camera
                _reset_session_state(session, "📷 Camera started, resetting state...")

    except Exception as e:
        # Top-level catch for the per-connection handler: anything uncaught
        # would drop the connection without a clean finally; keep broad.
        logger.error(f"❌ Connection error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.info(f"👋 Connection closed by {addr}")

        # Cleanup
        async with ctx.session_lock:
            if session_id in ctx.streaming_sessions:
                del ctx.streaming_sessions[session_id]

        # session.conn may already be None if server_io dropped it on a write error
        session.conn = None
        conn.close()


async def run_accept_loop(server_sock, args):
    """
    Run accept loop in the same event loop as the engine.
    This ensures handle_client_connection_async and engine.generate() share one loop,
    fixing TTFT delay caused by two separate loops (output put in one loop, generate() awaiting in another).
    """
    os.makedirs(VIDEO_DIR, exist_ok=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)
    loop = asyncio.get_event_loop()
    while True:
        conn, addr = await loop.run_in_executor(None, server_sock.accept)
        asyncio.create_task(handle_client_connection_async(conn, addr, args))


def _create_listen_socket(port: int):
    """Create, bind and listen on a TCP socket. Call from main thread before starting accept loop."""
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("0.0.0.0", port))
    server_sock.listen(1)
    return server_sock


# ============================================================================
# Main Entry Point
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3 omni Streaming Input Server")

    # Server configuration
    parser.add_argument("--host", type=str, default="localhost",
                        help="vLLM API server host (for HTTP mode)")
    parser.add_argument("--port", type=int, default=8000,
                        help="vLLM API server port (for HTTP mode)")
    parser.add_argument("--listen-port", type=int, default=12345,
                        help="Port to listen for client connections")

    # Model configuration
    parser.add_argument("--model", type=str,
                        default=os.environ.get("AURA_MODEL_PATH"),
                        help="Model name or path (env: AURA_MODEL_PATH)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--pipeline-parallel-size", type=int, default=1,
                        help="Number of GPUs for pipeline parallelism")
    parser.add_argument("--max-model-len", type=int, default=256*1024,
                        help="Maximum model context length")
    parser.add_argument("--max-seq-len", type=int, default=256*1024,
                        help="Maximum sequence length (256k tokens)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85,
                        help="GPU memory utilization")
    parser.add_argument("--enforce-eager", action="store_true",
                        help="Enforce eager execution (disable CUDA graphs)")
    parser.add_argument("--max-images-per-prompt", type=int, default=600,
                        help="Maximum images per prompt (vLLM engine limit)")
    parser.add_argument("--max-streaming-images", type=int, default=600,
                        help="Maximum images for streaming input session")
    parser.add_argument("--max-num-batched-tokens", type=int, default=None,
                        help="Maximum number of batched tokens (SchedulerConfig)")
    parser.add_argument("--enable-expert-parallel", action="store_true", default=False,
                        help="Enable expert parallel (default: False)")

    # Advanced vLLM engine options
    parser.add_argument("--kv-offloading-size", type=int, default=None,
                        help="KV cache offloading size (in GB)")
    parser.add_argument("--mm-encoder-attn-backend", type=str, default=None,
                        choices=["FLASH_ATTN", "XFORMERS", "TORCH_SDPA"],
                        help="Multimodal encoder attention backend")
    parser.add_argument("--mm-encoder-tp-mode", type=str, default=None,
                        choices=["data", "model"],
                        help="Multimodal encoder tensor parallelism mode")
    parser.add_argument("--disable-hybrid-kv-cache-manager", action="store_true",
                        help="Disable hybrid KV cache manager")
    parser.add_argument("--block-size", type=int, default=None,
                        choices=[1, 8, 16, 32, 64, 128, 256],
                        help="KV cache block size in tokens (vLLM CacheConfig)")
    parser.add_argument("--cache-dtype", type=str, default="auto",
                        choices=["auto", "fp8"],
                        help="Cache dtype (vLLM CacheConfig)")
    parser.add_argument("--prefix-caching-hash-algo", type=str, default=None,
                        choices=["sha256", "sha256_cbor", "xxhash", "xxhash_cbor"],
                        help="Hash algorithm for prefix caching (vLLM CacheConfig)")

    # Sampling parameters
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="Sampling temperature for generation (default: 0.9)")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Maximum number of tokens to generate (default: 512)")

    # Mode selection
    parser.add_argument("--use-http-api", action="store_true",
                        help="Use HTTP API instead of embedded engine")

    # ASR configuration
    parser.add_argument("--asr-url", type=str, default="http://localhost:8001/asr",
                        help="ASR service URL")
    parser.add_argument("--asr-sync", action="store_true",
                        help="Use synchronous ASR (default: async). Use sync mode if async has issues.")

    # Streaming configuration
    parser.add_argument("--frame-buffer-size", type=int, default=8,
                        help="Number of frames to buffer before processing")
    parser.add_argument("--stream-interval", type=float, default=0.5,
                        help="Interval between streaming inputs (seconds)")
    parser.add_argument("--target-fps", type=float, default=2.0,
                        help="Target FPS for frame extraction (default: 2.0)")
    parser.add_argument("--min-yield-interval", type=float, default=3.0,
                        help="Minimum seconds between yielding StreamingInputs (throttling)")
    parser.add_argument("--video-resize", action="store_true",
                        help="Enable video frame resize (use full resolution, slower TTFT)")
    parser.add_argument("--enable-pruning", action="store_true",
                        help="Enable video frame pruning")
    parser.add_argument("--max-rounds", type=int, default=60,
                        help="Maximum number of rounds to keep in history")
    parser.add_argument("--num-rounds-keep", type=int, default=15,
                        help="Number of rounds to keep in sliding window after pruning")
    parser.add_argument("--max-context-qas", type=int, default=10,
                        help="Maximum number of QAs to keep in context history")
    parser.add_argument("--cross-turn-penalty", type=float, default=0.0,
                        help="Cross-turn repetition penalty strength "
                             "(0=disabled, 2.0~3.0 recommended). "
                             "Combines soft logit_bias + hard n-gram blocking.")
    parser.add_argument("--cross-turn-lookback", type=int, default=2,
                        help="Number of recent assistant responses to penalize (window size)")
    parser.add_argument("--cross-turn-ngram-sizes", type=int, nargs="*", default=[3, 4, 5],
                        help="N-gram sizes for bad_words hard blocking (default: 3 4 5, pass empty to disable)")
    parser.add_argument("--debug-context", action="store_true",
                        help="Enable debug: save inference context to JSONL file")
    parser.add_argument("--debug-context-file", type=str, default="context_debug.jsonl",
                        help="JSONL file path for context debug output")

    # TTS configuration (TTS is now a remote service)
    parser.add_argument("--enable-tts", action="store_true", help="Enable TTS")
    parser.add_argument("--tts-service-url", type=str, default="http://localhost:8002",
                        help="URL of the standalone TTS service")
    parser.add_argument("--tts-speaker", type=str, default="Vivian",
                        help="TTS speaker name")
    parser.add_argument("--tts-language", type=str, default="Chinese", choices=["Chinese", "English"],
                        help="TTS language")
    parser.add_argument("--tts-instruct", type=str, default="",
                        help="TTS instruction")
    parser.add_argument("--tts-output-dir", type=str, default="tts_results",
                        help="TTS output directory")

    return parser.parse_args()


async def main_async(args):
    """Main async entry point."""
    # 服务启动时清空 debug context 文件
    if args.debug_context:
        try:
            with open(args.debug_context_file, "w", encoding="utf-8") as f:
                pass
            logger.info(f"🗑 [Debug] Cleared context file on server start: {args.debug_context_file}")
        except OSError as e:
            logger.warning(f"⚠️ [Debug] Failed to clear context file: {e}")

    # 根据 model path 动态设置 ctx.silent_token_id
    setup_silent_token_id(args.model)

    model_type = detect_model_type(args.model)
    logger.info("=" * 60)
    logger.info(f"Qwen3 {'Omni' if model_type == 'omni' else 'VL'} Streaming Input Server")
    logger.info("=" * 60)
    logger.info(f"Mode: {'HTTP API' if args.use_http_api else 'Embedded Engine'}")
    logger.info(f"Model: {args.model} (type: {model_type})")
    logger.info(f"Listen Port: {args.listen_port}")
    logger.info("-" * 60)
    logger.info("Streaming Configuration:")
    logger.info(f"  Target FPS: {args.target_fps} (frames extracted per second)")
    logger.info(f"  Max Streaming Images: {args.max_streaming_images}")
    logger.info(f"  Max Images Per Prompt: {args.max_images_per_prompt}")
    logger.info(f"  Min Yield Interval: {args.min_yield_interval}s (throttling)")
    logger.info(f"  Note: Client records at 15fps, server extracts at {args.target_fps}fps")
    logger.info(f"  Capacity: ~{int(args.max_streaming_images / args.target_fps / 60)} minutes of video")
    logger.info(f"  Video resize: {'ENABLED (1/8 resolution)' if args.video_resize else 'DISABLED (full res)'}")
    logger.info(f"  Enable pruning: {'ENABLED' if args.enable_pruning else 'DISABLED'}")
    logger.info(f"  Num rounds keep: {args.num_rounds_keep}")
    logger.info(f"  Max rounds: {args.max_rounds}")
    logger.info(f"  Max context QAs: {args.max_context_qas}")
    logger.info(f"  Enable expert parallel: {'ENABLED' if args.enable_expert_parallel else 'DISABLED'}")
    if args.kv_offloading_size is not None:
        logger.info(f"  KV Offloading Size: {args.kv_offloading_size} GB")
    if args.mm_encoder_attn_backend is not None:
        logger.info(f"  MM Encoder Attention Backend: {args.mm_encoder_attn_backend}")
    if args.mm_encoder_tp_mode is not None:
        logger.info(f"  MM Encoder TP Mode: {args.mm_encoder_tp_mode}")
    if args.disable_hybrid_kv_cache_manager:
        logger.info(f"  Hybrid KV Cache Manager: DISABLED")
    if args.block_size is not None:
        logger.info(f"  Block size: {args.block_size}")
    if args.cache_dtype is not None:
        logger.info(f"  Cache dtype: {args.cache_dtype}")
    if args.prefix_caching_hash_algo is not None:
        logger.info(f"  Prefix caching hash algo: {args.prefix_caching_hash_algo}")
    if args.max_num_batched_tokens is not None:
        logger.info(f"  Max num batched tokens: {args.max_num_batched_tokens}")
    if args.enable_tts:
        logger.info(f"  TTS Enabled: service at {args.tts_service_url}")
    logger.info("=" * 60)

    # Initialize TTS if enabled
    if args.enable_tts:
        if ctx.tts_ctl.configure(args):
            ctx.tts_ctl.start_worker(args)
        else:
            logger.warning("⚠ TTS initialization failed, TTS will be disabled")
    # logger.warning("⚠ TTS initialization failed, TTS will be disabled")

    if not args.use_http_api:
        # Initialize embedded engine
        await init_async_engine(args)

    # Run TCP accept loop in the same event loop as the engine (fixes TTFT ~800ms delay
    # caused by engine and connection handler living in different loops).
    server_sock = _create_listen_socket(args.listen_port)
    logger.info(f"🌐 Server listening on port {args.listen_port}")
    asyncio.create_task(run_accept_loop(server_sock, args))

    logger.info("✅ Server started. Press Ctrl+C to exit.")

    # Keep main thread alive
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("\n👋 Shutting down...")


def main():
    args = parse_args()

    # Configure root logging once at entry. Other modules (aura.*) that
    # use `logging.getLogger(...)` inherit this setup; keep format
    # compatible with the Flask bridge in realtime_capture_video_audio_streaming.py.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Handle signals
    def signal_handler(sig, frame):
        logger.info("\n👋 Received shutdown signal")
        exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run async main
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=1,2,3,4,5 python Qwen3_VL_online_streaming.py --listen-port 12345 --model $AURA_MODEL_PATH --tensor-parallel-size 4 --max-model-len 128000 --gpu-memory-utilization 0.85 --asr-url http://localhost:8001/asr --kv-offloading-size 300 --disable-hybrid-kv-cache-manager --mm-encoder-attn-backend FLASH_ATTN --mm-encoder-tp-mode data --enable-tts --tts-gpu 5 --tts-model Qwen/Qwen3-TTS-12Hz-1.7B-Base --tts-language Chinese --tts-ref-audio test_query.mp3 --tts-ref-text "仔细观察当前你看到的画面，并且结合之前你看到的画面，仔细描述你看到了什么" --tts-output-dir tts_results

# ssh -L 5003:<remote-host>:5003 <user>@<gateway>