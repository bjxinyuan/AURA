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
import asyncio
import base64
import json
import os
import signal
import socket
import struct
import threading
import time
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import aiohttp
import requests
import re  # Added for TTS sentence splitting
import sys

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
    send_audio,
    send_audio_chunk,
)
from aura.tts import TTSController
from aura.media import downsample_video_to_numpy, _decode_video_sync

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
#   Qwen3 VL:   SILENT_TOKEN_ID = 151669  (<|silent|>)
#   Qwen3 Omni: SILENT_TOKEN_ID = 151676  (<|silent|>), 151669 在 Omni 中是 <|audio_start|>
# SILENT_TOKEN_ID 在 main() 中根据 --model 路径动态设置
SILENT_TOKEN_ID = None  # 由 detect_model_type() 设置
IM_END_TOKEN_ID = 151645   # <|im_end|> token id

VISION_START_TOKEN_ID = 151652 # <|vision_start|>
VISION_END_TOKEN_ID = 151653   # <|vision_end|>
VIDEO_PAD_TOKEN_ID = 151656    # <|video_pad|>
IMAGE_PAD_TOKEN_ID = 151655    # <|image_pad|>


def detect_model_type(model_path: str) -> str:
    """根据模型路径判断模型类型: 'omni' 或 'vl'"""
    name = os.path.basename(model_path.rstrip("/")).lower()
    if "omni" in name:
        return "omni"
    return "vl"


def setup_silent_token_id(model_path: str):
    """根据模型路径设置全局 SILENT_TOKEN_ID"""
    global SILENT_TOKEN_ID
    model_type = detect_model_type(model_path)
    if model_type == "omni":
        SILENT_TOKEN_ID = 151676
    else:
        SILENT_TOKEN_ID = 151669
    print(f"🔧 Model type detected: {model_type} → SILENT_TOKEN_ID = {SILENT_TOKEN_ID}")


# SessionHistory has moved to aura/session_history.py (see arch.md §5.1).


# ============================================================================
# Cross-Turn Repetition Penalty (adapted from streaming_client.py)
# ============================================================================

# CrossTurnPenalty has moved to aura/cross_turn_penalty.py (see arch.md §5.3).


# StreamingSession moved to aura/session.py

# ============================================================================
# Global State
# ============================================================================

# TTSController owns all TTS state (sentence queue, worker thread, cancel flag).
tts_ctl = TTSController()

# Streaming session state
streaming_sessions: dict[str, StreamingSession] = {}
session_lock = asyncio.Lock()

# Directories
VIDEO_DIR = "real_time_captured_video"
AUDIO_DIR = "real_time_captured_audio"
TTS_OUTPUT_DIR = "tts_results"

# Engine instance
async_engine: Optional[AsyncLLM] = None
model_tokenizer = None

# Response ID counter
response_id_counter = 0
response_id_lock = threading.Lock()


def generate_response_id() -> str:
    """Generate a unique response ID."""
    global response_id_counter
    with response_id_lock:
        response_id_counter += 1
        return f"resp_{int(time.time() * 1000)}_{response_id_counter}"


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
    print(f"🎤 Transcribing audio from {audio_path}...", flush=True)
    try:
        with open(audio_path, 'rb') as f:
            files = {'file': f}
            # Request only ASR, do not trigger vLLM in ASR service
            response = requests.post(asr_url, files=files, params={"run_vllm": "false"}, timeout=30)

        if response.status_code == 200:
            data = response.json()
            text = data.get("text", "")
            print(f"✅ Transcribed: {text!r}", flush=True)
            return text
        else:
            print(f"❌ ASR failed with status {response.status_code}: {response.text}")
            return ""
    except requests.exceptions.Timeout:
        print("❌ ASR request timeout")
        return ""
    except Exception as e:
        print(f"❌ ASR error: {e}")
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
    print(f"🎤 [Async] Transcribing audio from {audio_path}...", flush=True)
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
                        print(f"✅ [Async] Transcribed: {text!r}", flush=True)
                        return text
                    else:
                        error_text = await response.text()
                        print(f"❌ [Async] ASR failed with status {response.status}: {error_text}")
                        return ""
    except asyncio.TimeoutError:
        print("❌ [Async] ASR request timeout")
        return ""
    except Exception as e:
        print(f"❌ [Async] ASR error: {e}")
        return ""


# TTS pipeline moved to aura/tts.py — see `tts_ctl` above.


# ============================================================================
# AsyncLLM Engine Management
# ============================================================================

async def init_async_engine(args) -> AsyncLLM:
    """Initialize the AsyncLLM engine with streaming support."""
    global async_engine

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
    print(f"🚀 Initializing {model_label} AsyncLLM engine with model: {args.model}")
    print(f"   trust_remote_code={engine_kwargs['trust_remote_code']}, SILENT_TOKEN_ID={SILENT_TOKEN_ID}")
    async_engine = AsyncLLM.from_engine_args(engine_args)
    print(f"✅ {model_label} AsyncLLM engine initialized successfully")

    # Store tokenizer globally for CrossTurnPenalty
    global model_tokenizer
    model_tokenizer = async_engine.get_tokenizer()

    # ========== DEBUG: 保存词表到日志文件 ==========
    try:
        tokenizer = model_tokenizer
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

        print(f"📝 Vocabulary saved to {vocab_log_path} ({len(vocab)} tokens)")
    except Exception as e:
        print(f"⚠️ Failed to save vocabulary: {e}")
    # ========== END DEBUG ==========

    # Note: No need to load transformers processor!
    # vLLM handles multimodal processing internally.
    # We use the pattern from test_qwen2_5_vl.py:
    # - Build prompt string with placeholders
    # - Pass images via multi_modal_data

    return async_engine


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
    print(f"==== Calling generate_response_with_video() ====")
    global async_engine

    if async_engine is None:
        raise RuntimeError("AsyncLLM engine not initialized")

    if video_tuple is None or video_tuple[0] is None:
        print("⚠️ No valid video for generation")
        session.is_generating = False
        return

    # Qwen3-VL requires at least 2 frames (temporal_factor=2)
    # Defensive check: if only 1 frame, duplicate it
    video_array_check = video_tuple[0]
    if video_array_check.shape[0] < 2:
        import numpy as np
        print(f"⚠️ Video has only {video_array_check.shape[0]} frame(s), duplicating to meet Qwen3-VL minimum (2 frames)")
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
        # print(f"⏱️ [TIMING] get_vllm_inputs() took {(t_get_inputs_end - t_get_inputs_start)*1000:.1f}ms")

        # Generate unique request ID for this turn
        request_id = generate_response_id()

        session.history.save_context_debug(request_id=request_id)

        video_array, video_metadata = video_tuple
        print(f"🎬 [Session {session.session_id}] Starting generation (request_id={request_id})")
        print(f"📥 Input: {video_array.shape[0]} video frames ({video_array.shape}), prompt='{prompt}'")

        full_response = ""
        previous_text = ""
        is_silent_response = False
        ttft = 0

        tts_enabled_for_this_response = args and args.enable_tts and tts_ctl.enabled

        if tts_enabled_for_this_response and prompt:
            tts_ctl.clear_queue(new_response_id=request_id)

        # Timing measurements
        generation_start_time = time.time()
        first_token_time = None
        token_count = 0
        print(f"[TTFT_DEBUG] stream generate_start request_id={request_id} t={generation_start_time:.6f}")
        print(f"⏱️ [TIMING] prefill_submit_time={generation_start_time:.6f} (engine.generate called, prefill starts)")

        # Incremental sentence buffer for streaming TTS
        _tts_sentence_buf = ""
        _tts_sentence_idx = 0
        _SENT_ENDS = frozenset("。！？；.!?;\n")
        _COMMA_ENDS = frozenset("，,")
        _TTS_MIN_CHARS = 10
        streaming_started = False

        # ===== Streaming generation: send tokens to frontend as they arrive =====
        async for response in async_engine.generate(
            prompt=vllm_inputs,
            sampling_params=sampling_params,
            request_id=request_id,
        ):
            if first_token_time is None:
                first_token_time = time.time()
                ttft = first_token_time - generation_start_time
                print(f"[TTFT_DEBUG] stream first_token request_id={request_id} t={first_token_time:.6f} ttft_ms={ttft*1000:.1f}")

            token_count += 1
            if response.outputs:
                output = response.outputs[0]

                if len(output.token_ids) > 0 and output.token_ids[0] in (SILENT_TOKEN_ID, IM_END_TOKEN_ID):
                    is_silent_response = True
                    first_tid = output.token_ids[0]
                    tag = "SILENT" if first_tid == SILENT_TOKEN_ID else "IM_END"
                    print(f"🔇 [Session {session.session_id}] {tag} as first token → silent response "
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
                                    tts_ctl.enqueue_sentence(sentence, session, request_id, _tts_sentence_idx, args)
                                    _tts_sentence_idx += 1

        # ===== Generation finished — decide: silent / send =====
        generation_end_time = time.time()
        total_time = generation_end_time - generation_start_time

        print(f"✅ [Session {session.session_id}] Generation finished")
        print(f"⏱️ [TIMING] Time to first token (TTFT): {ttft*1000:.1f}ms, timestamp: {time.time()}")
        print(f"⏱️ [TIMING] TTFT avg. by {video_array.shape[0]} frames: {(ttft*1000/video_array.shape[0]):.1f}ms")
        print(f"⏱️ [TIMING] Total generation time: {total_time*1000:.1f}ms")
        print(f"⏱️ [TIMING] Tokens generated: {token_count}")
        if token_count > 1 and first_token_time is not None:
            decode_only_ms = (generation_end_time - first_token_time) * 1000
            avg_decode_per_token = decode_only_ms / (token_count - 1)
            print(f"⏱️ [TIMING] Decode phase: {decode_only_ms:.1f}ms for {token_count-1} tokens, "
                  f"avg={avg_decode_per_token:.1f}ms/token ({1000/avg_decode_per_token:.1f} tokens/s)")

        print(f"📋 [DECISION] request_id={request_id} | is_silent={is_silent_response} | "
              f"full_response({len(full_response)} chars)='{full_response[:80]}'")

        if is_silent_response:
            print(f"🔇 [DECISION] → MODEL_SILENT (first token was silent/im_end)")
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
                tts_ctl.enqueue_sentence(_tts_sentence_buf, session, request_id, _tts_sentence_idx, args)
                _tts_sentence_idx += 1

            if tts_enabled_for_this_response:
                print(f"🎤 [Queue] Streamed {_tts_sentence_idx} sentences to TTS")

    except asyncio.CancelledError:
        print(f"⏹ [Session {session.session_id}] Generation cancelled (context reset)")
        if streaming_started and request_id:
            send_streaming_token(session, "", request_id, is_final=True)
    except Exception as e:
        print(f"❌ [Session {session.session_id}] Generation error: {e}")
        import traceback
        traceback.print_exc()
        if request_id:
            send_streaming_token(session, "", request_id, is_final=True)
    finally:
        # CRITICAL: Always reset the generating flags
        session.is_generating = False
        session.is_auto_generating = False
        session.current_task = None
        print(f"🔓 [Session {session.session_id}] Released generation lock")


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


async def handle_client_connection_async(conn, addr, args):
    """Handle client connection with async support."""
    print(f"================================================")
    print(f"✅ Connected by {addr} with SUYI")

    # Set socket to blocking mode with timeout
    conn.setblocking(True)
    conn.settimeout(1.0)  # 1 second timeout for initial reads

    # Create a streaming session for this client
    session_id = f"client-{addr[0]}-{addr[1]}-{int(time.time())}"

    # Build cross-turn penalty manager (if enabled)
    penalty_mgr = None
    if getattr(args, "cross_turn_penalty", 0) > 0 and model_tokenizer is not None:
        penalty_mgr = CrossTurnPenalty(
            tokenizer=model_tokenizer,
            window=getattr(args, "cross_turn_lookback", 2),
            logit_penalty=args.cross_turn_penalty,
            ngram_sizes=getattr(args, "cross_turn_ngram_sizes", [3, 4, 5]),
        )
        print(f"🔧 [Session {session_id}] CrossTurnPenalty enabled: "
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

    async with session_lock:
        streaming_sessions[session_id] = session

    # Generation task tracking (not used for long-running loop anymore)
    generation_task = None
    accumulated_video_frames: list = []  # List of numpy arrays (each shape: num_frames, H, W, 3)
    last_prompt = ""

    try:
        while True:
            # Read header: [Type 1 byte] [Length 8 bytes] = 9 bytes total
            try:
                header = await asyncio.get_event_loop().run_in_executor(
                    None, recv_exactly, conn, 9, 5.0
                )
            except TimeoutError:
                # No data received, continue waiting
                continue
            except ConnectionError:
                print("🔌 Client disconnected")
                break
            except Exception as e:
                print(f"❌ Header read error: {e}")
                break

            file_type, file_len = struct.unpack(">BQ", header)
            print(f"📩 Received: type={file_type}, length={file_len}, time={datetime.now().strftime('%H:%M:%S.%f')}")

            # Sanity check for length (prevent memory issues)
            if file_len > 100 * 1024 * 1024:  # 100MB max
                print(f"⚠ Invalid length {file_len}, skipping message")
                continue

            # Read file content
            try:
                file_data = await asyncio.get_event_loop().run_in_executor(
                    None, recv_exactly, conn, file_len, 30.0
                )
            except TimeoutError:
                print(f"⚠ Timeout reading {file_len} bytes, skipping")
                continue
            except ConnectionError:
                print("🔌 Client disconnected during data read")
                break
            except Exception as e:
                print(f"❌ Data read error: {e}")
                break

            # Yield immediately so other tasks (e.g. generate() waiting for first token) can run.
            # Avoids event loop starvation that causes ~800ms TTFT delay.
            await asyncio.sleep(0)

            if file_type == 1:  # Video (WebM)
                # Sanity check: WebM files need a minimum size for valid EBML header
                # A valid WebM file is typically at least 1KB even for short clips
                MIN_WEBM_SIZE = 1000  # 1KB minimum
                if len(file_data) < MIN_WEBM_SIZE:
                    print(f"⚠️ Video data too small ({len(file_data)} bytes < {MIN_WEBM_SIZE}), skipping corrupted/incomplete data")
                    continue

                # Downsample video to target FPS and get as numpy array with metadata
                print("🎥 Processing video data...")
                timestamp = int(time.time() * 1000)
                input_path = f"/tmp/video_{timestamp}_input.webm"
                
                with open(input_path, "wb") as f:
                    f.write(file_data)
                
                # Downsample video and get (numpy_array, metadata) tuple
                video_array, metadata = downsample_video_to_numpy(input_path, target_fps=args.target_fps)
                
                # Clean up input file
                try:
                    os.remove(input_path)
                except:
                    pass

                # # Run video decode in executor to avoid blocking the event loop (prevents TTFT starvation).
                # print("🎥 Processing video data...")
                # timestamp = int(time.time() * 1000)
                # input_path = f"/tmp/video_{timestamp}_input.webm"
                # print("New......")
                # loop = asyncio.get_event_loop()
                # video_array, metadata = await loop.run_in_executor(
                #     None,
                #     _decode_video_sync,
                #     file_data,
                #     input_path,
                #     args.target_fps,
                #     not args.no_video_resize,
                # )

                if video_array is not None:
                    # Accumulate video frames (as numpy arrays)
                    accumulated_video_frames.append(video_array)
                    total_frames = sum(arr.shape[0] for arr in accumulated_video_frames)
                    print(f"📹 Got {video_array.shape[0]} frames, total accumulated: {total_frames}")

                    # Process when we have frames OR if we have a pending prompt
                    should_process = False

                    # Priority trigger: Pending user prompt
                    if last_prompt and total_frames > 0:
                        print(f"⚡ Triggering immediate generation for user prompt (frames={total_frames})")
                        should_process = True
                    # Background trigger: Have enough frames AND idle
                    elif total_frames >= 2 and not session.is_generating:
                        print(f"⚡ Triggering background generation (frames={total_frames})")
                        should_process = True

                    if should_process:
                        if session.is_generating and last_prompt:
                             print("⏳ Waiting for previous generation to finish before processing prompt...")
                             pass

                        if not session.is_generating:
                            import numpy as np

                            # Concatenate all accumulated frames
                            all_frames = np.concatenate(accumulated_video_frames, axis=0)

                            # Qwen3-VL requires at least 2 frames (temporal_factor=2)
                            # If we only have 1 frame, duplicate it to meet the minimum requirement
                            if all_frames.shape[0] == 1:
                                print(f"⚠️ Only 1 frame, duplicating to meet Qwen3-VL minimum requirement (2 frames)")
                                all_frames = np.concatenate([all_frames, all_frames], axis=0)

                            # Limit to max 16 frames to avoid OOM
                            if all_frames.shape[0] > 16:
                                all_frames = all_frames[-16:]

                            # Create metadata for the combined video
                            video_metadata = {
                                "fps": args.target_fps,
                                "duration": all_frames.shape[0] / args.target_fps,
                                "total_num_frames": all_frames.shape[0],
                                "frames_indices": list(range(all_frames.shape[0])),
                                "video_backend": "opencv",
                                "do_sample_frames": False,
                            }
                            video_tuple = (all_frames, video_metadata)

                            accumulated_video_frames = []

                            # Mark as generating IMMEDIATELY to prevent double trigger
                            session.is_generating = True

                            # Default prompt if none
                            current_prompt = last_prompt if last_prompt else ""
                            last_prompt = ""  # Clear prompt after using

                            # Track if this is an auto-generation (no user prompt)
                            session.is_auto_generating = (current_prompt == "")

                            penalty_kwargs = {}
                            if session.cross_turn_penalty is not None:
                                penalty_kwargs = session.cross_turn_penalty.build_sampling_kwargs()

                            sampling_params = SamplingParams(
                                temperature=args.temperature,
                                max_tokens=args.max_tokens,
                                **penalty_kwargs,
                            )

                            # Launch background task with video tuple
                            session.current_task = asyncio.create_task(generate_response_with_video(
                                session,
                                video_tuple,
                                current_prompt,
                                sampling_params,
                                args
                            ))
                else:
                    print("❌ Video processing failed - no frames extracted (possible codec incompatibility with iOS Chrome)")

            elif file_type == 2:  # Audio
                # Save audio for ASR
                audio_path = os.path.join(AUDIO_DIR, "latest.mp3")
                os.makedirs(AUDIO_DIR, exist_ok=True)
                with open(audio_path, "wb") as f:
                    f.write(file_data)
                print(f"🎤 Saved audio to {audio_path}")

                # Call ASR service to transcribe audio
                _asr_start = time.time()
                if args.asr_sync:
                    # Synchronous version
                    loop = asyncio.get_event_loop()
                    transcribed_text = await loop.run_in_executor(
                        None, get_audio_prompt, audio_path, args.asr_url
                    )
                else:
                    # Asynchronous version (default)
                    transcribed_text = await transcribe_audio_async(audio_path, args.asr_url)
                _asr_end = time.time()
                print(f"⏱️ [TIMING] ASR latency: {(_asr_end - _asr_start)*1000:.1f}ms")

                if transcribed_text:
                    last_prompt = transcribed_text
                    print(f"📝 Set prompt from ASR: {last_prompt[:50]}...")

                    # Plan 2: ASR query is sent to client immediately,
                    # model inference result will be sent separately later.
                    send_asr_query(session, transcribed_text)

                    # Optimization: Try to trigger immediately if we have ANY video frames
                    if accumulated_video_frames:
                        print("🚀 Audio arrived, attempting immediate trigger...")
                        if session.is_generating and session.is_auto_generating:
                            if session.current_task and not session.current_task.done():
                                print("🛑 Interrupting auto-generation for user prompt (from Audio event)!")
                                session.current_task.cancel()
                else:
                    print("⚠ ASR returned empty, will use default prompt")

                # Note: Generation is triggered by Video frame loop when frames arrive
                # If frames are already there, we could trigger here, but to avoid race conditions
                # we let the video loop handle it.

            elif file_type == 4:  # Clear Context
                print("🗑 Clearing context...")
                # Cancel any running generation task first
                if session.current_task and not session.current_task.done():
                    print("⏹ Cancelling running generation task...")
                    session.current_task.cancel()
                    session.is_generating = False
                    session.is_auto_generating = False
                session.history._reset()
                if session.cross_turn_penalty is not None:
                    session.cross_turn_penalty.reset()
                accumulated_video_frames = []
                last_prompt = ""

            elif file_type == 6:  # Start Camera
                print("📷 Camera started, resetting state...")
                # Cancel any running generation task first
                if session.current_task and not session.current_task.done():
                    print("⏹ Cancelling running generation task...")
                    session.current_task.cancel()
                    session.is_generating = False
                    session.is_auto_generating = False
                session.history._reset()
                if session.cross_turn_penalty is not None:
                    session.cross_turn_penalty.reset()
                accumulated_video_frames = []
                last_prompt = ""

    except Exception as e:
        print(f"❌ Connection error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"👋 Connection closed by {addr}")

        # Cleanup
        # No streaming input to stop

        async with session_lock:
            if session_id in streaming_sessions:
                del streaming_sessions[session_id]

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
    parser.add_argument("--dedup-threshold", type=float, default=0.0,
                        help="(deprecated, no longer used)")
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
            print(f"🗑 [Debug] Cleared context file on server start: {args.debug_context_file}")
        except Exception as e:
            print(f"⚠️ [Debug] Failed to clear context file: {e}")

    # 根据 model path 动态设置 SILENT_TOKEN_ID
    setup_silent_token_id(args.model)

    model_type = detect_model_type(args.model)
    print("=" * 60)
    print(f"Qwen3 {'Omni' if model_type == 'omni' else 'VL'} Streaming Input Server")
    print("=" * 60)
    print(f"Mode: {'HTTP API' if args.use_http_api else 'Embedded Engine'}")
    print(f"Model: {args.model} (type: {model_type})")
    print(f"Listen Port: {args.listen_port}")
    print("-" * 60)
    print("Streaming Configuration:")
    print(f"  Target FPS: {args.target_fps} (frames extracted per second)")
    print(f"  Max Streaming Images: {args.max_streaming_images}")
    print(f"  Max Images Per Prompt: {args.max_images_per_prompt}")
    print(f"  Min Yield Interval: {args.min_yield_interval}s (throttling)")
    print(f"  Note: Client records at 15fps, server extracts at {args.target_fps}fps")
    print(f"  Capacity: ~{int(args.max_streaming_images / args.target_fps / 60)} minutes of video")
    print(f"  Video resize: {'ENABLED (1/8 resolution)' if args.video_resize else 'DISABLED (full res)'}")
    print(f"  Enable pruning: {'ENABLED' if args.enable_pruning else 'DISABLED'}")
    print(f"  Num rounds keep: {args.num_rounds_keep}")
    print(f"  Max rounds: {args.max_rounds}")
    print(f"  Max context QAs: {args.max_context_qas}")
    print(f"  Enable expert parallel: {'ENABLED' if args.enable_expert_parallel else 'DISABLED'}")
    if args.kv_offloading_size is not None:
        print(f"  KV Offloading Size: {args.kv_offloading_size} GB")
    if args.mm_encoder_attn_backend is not None:
        print(f"  MM Encoder Attention Backend: {args.mm_encoder_attn_backend}")
    if args.mm_encoder_tp_mode is not None:
        print(f"  MM Encoder TP Mode: {args.mm_encoder_tp_mode}")
    if args.disable_hybrid_kv_cache_manager:
        print(f"  Hybrid KV Cache Manager: DISABLED")
    if args.block_size is not None:
        print(f"  Block size: {args.block_size}")
    if args.cache_dtype is not None:
        print(f"  Cache dtype: {args.cache_dtype}")
    if args.prefix_caching_hash_algo is not None:
        print(f"  Prefix caching hash algo: {args.prefix_caching_hash_algo}")
    if args.max_num_batched_tokens is not None:
        print(f"  Max num batched tokens: {args.max_num_batched_tokens}")
    if args.enable_tts:
        print(f"  TTS Enabled: service at {args.tts_service_url}")
    print("=" * 60)

    # Initialize TTS if enabled
    if args.enable_tts:
        if tts_ctl.configure(args):
            tts_ctl.start_worker(args)
        else:
            print("⚠ TTS initialization failed, TTS will be disabled")
    # print("⚠ TTS initialization failed, TTS will be disabled")

    if not args.use_http_api:
        # Initialize embedded engine
        await init_async_engine(args)

    # Run TCP accept loop in the same event loop as the engine (fixes TTFT ~800ms delay
    # caused by engine and connection handler living in different loops).
    server_sock = _create_listen_socket(args.listen_port)
    print(f"🌐 Server listening on port {args.listen_port}")
    asyncio.create_task(run_accept_loop(server_sock, args))

    print("✅ Server started. Press Ctrl+C to exit.")

    # Keep main thread alive
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")


def main():
    args = parse_args()

    # Handle signals
    def signal_handler(sig, frame):
        print("\n👋 Received shutdown signal")
        exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run async main
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=1,2,3,4,5 python Qwen3_VL_online_streaming.py --listen-port 12345 --model $AURA_MODEL_PATH --tensor-parallel-size 4 --max-model-len 128000 --gpu-memory-utilization 0.85 --asr-url http://localhost:8001/asr --kv-offloading-size 300 --disable-hybrid-kv-cache-manager --mm-encoder-attn-backend FLASH_ATTN --mm-encoder-tp-mode data --enable-tts --tts-gpu 5 --tts-model Qwen/Qwen3-TTS-12Hz-1.7B-Base --tts-language Chinese --tts-ref-audio test_query.mp3 --tts-ref-text "仔细观察当前你看到的画面，并且结合之前你看到的画面，仔细描述你看到了什么" --tts-output-dir tts_results

# ssh -L 5003:<remote-host>:5003 <user>@<gateway>