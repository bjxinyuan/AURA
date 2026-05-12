# AURA Architecture (Qwen3-Omni E2E mode)

A real-time multimodal streaming system: live video + voice in, streamed text + voice out. Built around a **Qwen3-Omni** end-to-end multimodal model served by vLLM, with a separate streaming TTS service.

> **Migration note**: AURA originally ran AURA-8B (a Qwen3-VL fork) plus a standalone `Qwen3-ASR` service. The current architecture replaces both with Qwen3-Omni-30B-A3B (FP8 quantized for single-A800 deployment), which natively ingests audio. Type 2 of the wire protocol still carries the audio blob, but the inference server now feeds it to Omni directly instead of POSTing to an ASR service. The transcription is recovered from a `<query>...</query>` prefix that the model is prompted to emit. The two architectures are not interoperable.

## 1. Top-level component map

```
        ┌─────────────┐  HTTP (Flask)         ┌──────────────────────┐
Browser │ Chrome page │◄─────────────────────►│ Flask bridge         │
(camera │ (HTML+JS)   │  /api/video           │ realtime_capture_*py │
 + mic) │             │  /api/audio           │  - per-user lock     │
        │   SSE       │  /api/events          │  - SSE fan-out       │
        │  /api/events│  (POST + EventSource) │  - TCP framing       │
        └─────────────┘                       └──────────┬───────────┘
                                                         │ TCP :12345
                                                         │ (9-byte header)
                                                         ▼
                                              ┌──────────────────────┐
                                              │ Inference server     │
                                              │ Qwen3_VL_..._CM.py   │
                                              │  - asyncio TCP       │
                                              │  - vLLM AsyncLLM     │
                                              │  - SessionHistory    │
                                              │  - QueryExtractor    │
                                              │  - TTSController     │
                                              └──────────────┬───────┘
                                                             │ HTTP :8002
                                                             ▼
                                                    ┌────────────┐
                                                    │ TTS svc    │
                                                    │ Qwen3-TTS  │
                                                    │ FastAPI    │
                                                    └────────────┘
```

Three independent processes (down from four), started in order by `start_all.sh`:

| Process | File | Port | Purpose |
|---|---|---|---|
| TTS | `tts_service.py` | 8002 | Text → streaming PCM (Qwen3-TTS, voice clone) |
| Inference | `Qwen3_VL_online_streaming_v2_ContextManaged.py` | 12345 (TCP) | vLLM AsyncLLM serving Qwen3-Omni + session orchestration |
| Flask bridge | `realtime_capture_video_audio_streaming.py` | 5003 | Browser ↔ inference TCP socket adapter |

Browser is a single page (`templates/index_streaming.html` + `static/*.js`). Talks only to the Flask bridge — never directly to inference or TTS. Audio recorded in the browser flows: browser → Flask bridge → inference server → Omni model → `<query>` echo back to browser.

## 2. Wire protocol (Flask bridge ↔ inference)

Defined once in `aura/protocol.py`. Every message is `[type:1][length:8][payload:length]` (big-endian, 9-byte header).

| Type | Direction | Name | Payload |
|---|---|---|---|
| 1 | C→S | `VIDEO` | WebM blob (browser MediaRecorder output) |
| 2 | C→S | `AUDIO` | WebM/Opus blob — fed to Omni directly |
| 4 | C→S | `CLEAR_CONTEXT` | empty — wipes session history |
| 6 | C→S | `START_CAMERA` | empty — same reset; emitted on camera-start |
| 7 | S→C | `ERROR` | UTF-8 error message |
| 8 | S→C | `STREAMING_TOKEN` | JSON: `{response_id, token, is_final, is_start?, is_silent?, query?}` |
| 9 | S→C | `TTS_AUDIO_CHUNK` | binary: `[id_len:1][response_id][sentence_idx:2][chunk_idx:2][sample_rate:4][is_final:1][int16 PCM…]` |
| 10 | S→C | `ASR_QUERY_ECHO` | JSON: `{type:"asr_query", query}` — emitted when Omni completes the `<query>...</query>` prefix |

Inbound on the bridge is multiplexed onto a `queue.Queue` and re-emitted to the browser as Server-Sent Events on `/api/events`. The SSE wrapper lives in `aura/sse.py` and supports `("close", _)` as a poison pill plus periodic heartbeats.

**Type 10 timing change**: in the AURA-8B era, the bridge received Type 10 within ~300-500 ms of the browser sending audio (independent ASR call ran in parallel with video processing). In Omni E2E mode, Type 10 is emitted by the inference server only after the model has decoded `</query>` (typically a few hundred ms into generation). Net effect on UX is small but measurable.

## 3. Inference server internals

`Qwen3_VL_online_streaming_v2_ContextManaged.py` is the orchestrator. All process-wide mutable state lives on a single `ServerContext` dataclass (`ctx`).

### 3.1 Per-connection lifecycle

`handle_client_connection_async(conn, addr, args)` runs once per accepted TCP connection in the same asyncio event loop as the vLLM engine (sharing the loop is intentional — split loops added ~800 ms TTFT).

It builds a `StreamingSession` (`aura/session.py`) holding:
- `history: SessionHistory` — chat memory
- `cross_turn_penalty: CrossTurnPenalty | None`
- `conn: socket` + `conn_lock: threading.Lock` — outbound writes serialised per session
- `accumulated_video_frames: list[np.ndarray]` — frames waiting for a generation trigger
- `last_prompt: str` — typed/text prompt (rarely used in Omni mode; kept for backward-compat)
- **`pending_audio: tuple | None`** — decoded `(waveform, sample_rate)` for the next turn
- `is_generating`, `is_auto_generating`, `current_task` — scheduling flags

The handler loops on `_read_header` / `_read_payload` (each running blocking `recv` in a thread executor with timeouts) and dispatches:

- **Type 1 (video)** → `_handle_video_msg` → decode WebM via `aura.media.downsample_video_to_numpy` (OpenCV; iOS-Chrome-compatible sequential-read fallback) → append to `accumulated_video_frames` → `_maybe_launch_generation`.
- **Type 2 (audio)** → `_handle_audio_msg` → save WebM to disk → `aura.audio_media.decode_audio_to_numpy` (librosa) decodes to mono float32 @ 16 kHz → `session.pending_audio = (waveform, sr)` → cancel any in-flight *auto-generation* so the user voice turn takes precedence. **No external service call.**
- **Type 4 / Type 6** → `_reset_session_state`: cancel current task, `history._reset()`, reset penalty, drop accumulated frames, clear `pending_audio`.

### 3.2 Generation trigger logic (`_maybe_launch_generation`)

Two triggers, in priority order:
1. **User input** (audio or typed prompt) AND any frames buffered → launch immediately.
2. **Background ("auto-generation")**: ≥2 frames buffered and idle → launch with no audio, no prompt.

If already generating, the new request waits — except in the audio handler, which proactively cancels an in-flight auto-generation. `is_auto_generating` is true only when both `pending_audio` is None and `last_prompt` is empty.

The trigger concatenates buffered frames (capped at 16, enforces ≥2 for Qwen3-Omni's `temporal_factor=2`), passes both `video_tuple` and `audio_tuple=current_audio` to `generate_response_with_video`, builds `SamplingParams` (with cross-turn-penalty `logit_bias` + `bad_words` if enabled), and creates the asyncio task.

### 3.3 Generation loop (`generate_response_with_video`)

1. `session.history.add_user_message(prompt, video_tuple, audio_tuple)` — appends to sliding window with both modalities.
2. `session.history.get_vllm_inputs()` — flattens entire history into one prompt string with `<|vision_start|><|video_pad|><|vision_end|>` and `<|audio_start|><|audio_pad|><|audio_end|>` placeholders, plus aggregated `multi_modal_data`. **Critical for prefix caching**: text and media must line up across turns so vLLM's APC reuses prior KV.
3. **`QueryExtractor` is instantiated** if `audio_tuple is not None`. Video-only auto-generation skips it (the model isn't expected to emit `<query>` for these turns).
4. `async for response in ctx.async_engine.generate(...)` — vLLM streams tokens.
5. **Silence shortcut**: if the *first* output token is `silent_token_id` or `<|im_end|>`, the response is silent — record `<|silent|>` in history, drop any pending audio entry via `replace_last_user_audio_with_text("")`, emit `is_silent=True` final marker.
6. **Streaming**: each delta is fed to `QueryExtractor.feed(delta)`. When `result.query_complete` fires:
   - `send_asr_query(session, result.query_text)` (Type 10 echo to browser)
   - `session.history.replace_last_user_audio_with_text(result.query_text)` — the audio waveform in the most recent user turn is replaced by the transcription text. **This is critical for prefix-cache friendliness and context-budget bounding** — keeping the waveform would re-feed the same audio on every subsequent turn.
   - `result.passthrough_delta` is forwarded to the streaming token channel + TTS sentence buffer via the local `_emit_passthrough` closure.
7. **Sentence splitting** (inside `_emit_passthrough`): post-`<query>` deltas are buffered and split on sentence terminators (`。！？；.!?;\n`) or commas after ≥10 chars; each completed sentence is enqueued on `ctx.tts_ctl` for streaming TTS.
8. **Finalisation**: if the stream ended without ever emitting `</query>`, `extractor.finalize()` returns a fallback echo; this still drives the Type 10 send + history rewrite. Then append assistant text (post-query passthrough only, **not** including the `<query>...</query>` prefix) to history, record into `cross_turn_penalty`, flush remaining sentence buffer, send empty `is_final=True` token.
9. `finally:` always clears `is_generating` flags.

### 3.4 TTS pipeline (`aura/tts.py`)

Unchanged from the AURA-8B era. `TTSController` owns:
- `_sentence_queue: queue.Queue` — sentences from the generation loop
- A daemon worker thread (`_worker_loop`) that POSTs each sentence to the remote TTS service's `/v1/tts/stream` and streams PCM chunks back over the per-session socket as Type 9 frames
- A latency log (`tts_latency.log`) with first-chunk latency, total latency, RTF
- `clear_queue` is called between user-driven turns to drop pending sentences

Pipeline parallelism: model decodes sentence N+1 while TTS synthesises sentence N. First-chunk-out happens before the model finishes generation.

## 4. Memory: two-tier history (`aura/session_history.py`)

The system runs continuously and accumulates ~128K tokens of context (down from 256K to fit the FP8 Omni weights + TTS on a single A800). Pruning is unchanged from the AURA-8B era:

```
history = [system_prompt]
        ++ context_history (compressed, text-only QAs, max 10)
        ++ sliding_window  (recent rounds, full multimedia)
```

Rewrite rules A-E (see source for details) move stale rounds from sliding window to context history while stripping multimedia. **Audio entries are stripped just like video** — extended Rule A.

The system prompt now instructs the model to wrap user speech in `<query>...</query>`:

> "You are receiving a live video stream where the final frame is the present moment, along with the user's voice. When the user speaks, begin your response by wrapping your best transcription of what they said in `<query>...</query>`, then immediately produce your reply in Chinese..."

### 4.1 Audio in history: write-once-then-rewrite

`add_user_message(text, video_tuple, audio_tuple)` stores the audio waveform as `{"type": "audio", "audio": (np.ndarray, sr)}`. **This is transient state** — within the same generation, `replace_last_user_audio_with_text(transcription)` rewrites that same user message:

- Audio entry deleted from `content`
- Transcription added as a `{"type": "text"}` entry (merged with any pre-existing text)
- The shared dict in `self.history` mutates in-place so the next `get_vllm_inputs()` call sees text only

This ensures: (a) prefix cache hits across turns (same text → same hash), and (b) bounded context (one second of audio costs many tokens, accumulating quickly).

The rewrite happens at three points in the generation loop:
- When `</query>` is parsed: rewrite to the extracted query.
- When the stream ends without `</query>`: rewrite to the fallback echo.
- When the response is silent (first token is silent/im_end): rewrite to empty string (still drops the audio).

## 5. `<query>` extraction (`aura/query_extractor.py`)

Pure streaming state machine for parsing the `<query>...</query>` prefix that Omni emits. No tokenizer dependency, no I/O — fully unit-testable.

API:
```python
extractor = QueryExtractor(max_chars_before_fallback=80)
for delta in stream:
    result = extractor.feed(delta)
    if result.query_complete:
        # echo + history rewrite
    if result.passthrough_delta:
        # forward to client + TTS
final = extractor.finalize()  # if stream ended mid-query, force-emit echo
```

Three failure modes handled:
- **Cross-delta tag split**: `<que` then `ry>...` then `</que` then `ry>rest` — state machine accumulates internally and only reports completion on the delta containing `</query>`.
- **Missing open tag**: model forgets `<query>` but emits `</query>` — everything before it is treated as the query.
- **No closing tag within `max_chars_before_fallback`** (default 80 chars): force-complete with the accumulated buffer as the query, all subsequent deltas pass through. Tunable via `--query-echo-max-chars`.

17 test cases in `tests/test_query_extractor.py` cover normal path, format violations, fallback, finalize, and unicode.

## 6. Cross-turn repetition penalty (`aura/cross_turn_penalty.py`)

Unchanged. The `<query>` prefix is excluded from `record(full_response)` because `full_response` only accumulates passthrough deltas, not the parsed-out query. So the penalty operates only on the assistant's actual reply, which is what we want.

## 7. The `aura/` package

| Module | Lines | Role |
|---|---|---|
| `protocol.py` | ~30 | Pack/unpack the 9-byte header. Type 2 is now E2E audio. |
| `session.py` | ~45 | `StreamingSession` dataclass (now with `pending_audio`). |
| `session_history.py` | ~580 | Two-tier history with audio-aware add + rewrite. |
| `cross_turn_penalty.py` | ~200 | logit_bias + bad_words computation. |
| `media.py` | ~150 | WebM video → numpy frames (OpenCV, iOS fallback). |
| `audio_media.py` | ~55 | WebM/Opus audio → mono float32 @ 16 kHz (librosa). |
| `query_extractor.py` | ~165 | Streaming `<query>...</query>` parser. |
| `tts.py` | ~440 | `TTSController` (queue + worker thread + latency log). |
| `server_io.py` | ~125 | Outbound senders. Per-session, fail-fast. |
| `sse.py` | ~50 | Generic SSE framing for Flask. |
| `text_utils.py` | ~20 | `remove_markdown` for TTS-safe text. |

Tests under `tests/` (no GPU, no model load, no vLLM) cover all of the above. **165+ test cases** including the new `test_query_extractor.py` (17) and `test_audio_media.py` (7, may skip in dev environments lacking soundfile).

## 8. Flask bridge (`realtime_capture_video_audio_streaming.py`)

Unchanged. Single user lock, persistent TCP socket, daemon receive thread demultiplexing onto `event_queue`, `/api/events` SSE response. The bridge is **agnostic to the inference backend** — it never knew about ASR and doesn't need to know it's gone.

## 9. Browser side (`templates/`, `static/`)

Unchanged. Single SSE stream, single set of POST routes. Type 10 echo arrives slightly later than before (after model `</query>` instead of after standalone ASR), routed identically through `handleStreamingToken`.

## 10. Configuration

- `.env.example` — `AURA_MODEL_PATH` (required, points to FP8 Omni weights) plus port overrides (`AURA_FLASK_PORT`, `AURA_INFER_PORT`, `AURA_TTS_PORT`, `AURA_INFER_HOST`). **`AURA_ASR_PORT` removed.**
- `start_all.sh` — port + GPU allocation (`GPU_TTS`, `GPU_INFERENCE`). Defaults both to `0` (single-A800 co-resident). **ASR launch block removed.** Health-checks TTS before launching inference.
- `Qwen3_VL_online_streaming_v2_CM.sh` — argparse for the inference server, FP8-tuned: `--max-model-len 131072`, `--gpu-memory-utilization 0.80`, `--enable-expert-parallel`, `--kv-offloading-size 20`, `--temperature 0.5`, `--max-tokens 128`, `--cross-turn-penalty 1`, `--cross-turn-lookback 10`, `--enable-pruning`, `--max-rounds 45`, `--num-rounds-keep 30`, `--query-echo-max-chars 80`.

## 11. Single-A800 deployment notes

| Item | Value | Notes |
|---|---|---|
| Model | Qwen3-Omni-30B-A3B-FP8 | ~30 GB weights vs. 60 GB BF16 |
| TTS | Qwen3-TTS-12Hz-1.7B-Base | ~4 GB BF16, co-resident |
| Multimodal encoder | BF16 (not quantized) | ~2-4 GB |
| vLLM overhead | ~3-5 GB | CUDA graphs, scheduler |
| Free for KV | ~35-40 GB | Supports 128K context comfortably |
| `--cache-dtype` | `auto` | Don't combine FP8 weights + FP8 KV until validated separately |

### Pre-deployment checklist

The FP8 Omni image (`marksverdhei/Qwen3-Omni-30B-A3B-FP8`) is community-quantized. Before relying on the new pipeline, validate in this order:

1. **vLLM loads it**: bare offline `LLM(model=...)` smoke test, no TCP, no TTS. Confirms FP8 + Omni + vLLM 0.17+ compatibility.
2. **`nvidia-smi` budget**: actual VRAM after load. Adjust `--gpu-memory-utilization` and `--max-model-len` accordingly.
3. **Quality eyeball**: 3-5 typical AURA queries (e.g., the demo videos in README), compare BF16 vs. FP8 responses.
4. **`<query>` format adherence**: prompt the model on a held-out audio sample and check whether it emits well-formed `<query>...</query>`. If adherence is low, consider few-shot prefixing or fine-tuning. The `QueryExtractor.fallback_triggered` rate in production telemetry is the running indicator.

## 12. Latency design notes

Three latency budgets, same as AURA-8B but with shifted constants:

1. **TTFT for video-only auto-generation** — measured by `[TTFT_DEBUG]` log markers. Omni 30B prefill is heavier than 8B; expect ~1-1.5× the AURA-8B numbers despite FP8.
2. **ASR-to-display** — replaced by **audio-to-`</query>` time**. No longer dominated by an external HTTP call; instead dominated by Omni prefill on the audio + video and decoding ~10-30 chars of `<query>` text. Net: similar order of magnitude to the old ASR call (~300-500 ms), removes one network hop.
3. **First TTS chunk** — sentence-level streaming starts after `</query>` is seen (TTS doesn't speak the query). Slightly later than AURA-8B's first-sentence point but the gap is in the tens of ms.

Things that protect these budgets (unchanged):
- vLLM engine and TCP accept loop share one event loop.
- `get_vllm_inputs` is deterministic across turns to maximise APC hit rate. Critical that `replace_last_user_audio_with_text` runs in-place so the prompt for turn N+1 contains the same text as it did when the model first read the transcription.
- KV offloading to CPU (`--kv-offloading-size 20` GB) lets long sessions fit in memory.
- Sliding-window pruning bounds context length.
- Auto-generation is preemptable: an incoming audio turn cancels the running task immediately.

New caveat: **the audio waveform is large**. A 5-second user utterance at 16 kHz float32 = 320 KB plus encoder overhead. Holding many waveforms in `multi_modal_data` would be expensive; the rewrite-to-text protocol prevents that by design.

## 13. Benchmarks

`AURA_bench_eval/` is a separate evaluation harness with its own `requirements.txt`. It still targets the original AURA-8B model class and may need updating for Omni — out of scope for this migration.
