"""aura.tts — Streaming TTS pipeline controller and helpers.

Wraps the previously module-level TTS state (sentence queue, cancel
flag, worker thread, latency log) into a single TTSController instance.
The main inference service keeps a `tts_ctl` instance and hands it to
whichever asyncio task or thread needs to enqueue sentences.

Algorithm, logging, and timing are preserved verbatim from the pre-refactor
Qwen3_VL_online_streaming_v2_ContextManaged.py.
"""
import datetime
import queue
import re
import struct
import threading
import time as _time
import traceback

import requests
from aura.server_io import send_audio_chunk
from aura.text_utils import remove_markdown


# ---------------------------------------------------------------
# Pure helpers (stateless, unit-testable)
# ---------------------------------------------------------------

SENTENCE_TERMINATORS = "。！？；.!?;，,"
SENTENCE_TERMINATORS_SET = set(SENTENCE_TERMINATORS)


def split_text_to_sentences(text: str) -> list:
    """Split text into sentences, preserving punctuation."""
    if not text or not text.strip():
        return []

    # Split by sentence terminators
    sentence_pattern = r'([^。！？；.!?;]+[。！？；.!?;]?)'
    raw_sentences = re.findall(sentence_pattern, text)

    sentences = []
    for s in raw_sentences:
        s = s.strip()
        if not s:
            continue

        # Split long sentences at commas
        if len(s) > 50:
            sub_pattern = r'([^，,]+[，,]?)'
            sub_sentences = re.findall(sub_pattern, s)
            for sub in sub_sentences:
                sub = sub.strip()
                if sub:
                    sentences.append(sub)
        else:
            sentences.append(s)

    if not sentences and text.strip():
        sentences = [text.strip()]

    return sentences


def merge_short_sentences(sentences: list, min_chars: int = 15) -> list:
    """Merge short sentences to reduce TTS calls and improve naturalness."""
    if not sentences:
        return sentences

    merged = []
    buffer = ""

    for s in sentences:
        if len(buffer) + len(s) < min_chars * 3:  # Allow merging up to ~45 chars
            buffer = (buffer + s) if buffer else s
        else:
            if buffer:
                merged.append(buffer)
            buffer = s

    if buffer:
        merged.append(buffer)

    return merged


def _text_to_speech_generator(service_url: str,
                              text: str,
                              language: str = "Chinese",
                              speaker: str = "Vivian",
                              instruct: str = ""):
    """Stream PCM chunks from the remote TTS service.

    The service returns a binary stream where each chunk is:
        [sample_rate : 4 bytes big-endian uint32]
        [pcm_length  : 4 bytes big-endian uint32]
        [pcm_data    : pcm_length bytes, int16 LE]
    """
    print(f"🎤 [Remote] Requesting TTS: {text[:40]}...")

    try:
        resp = requests.post(
            f"{service_url}/v1/tts/stream",
            json={"text": text, "language": language, "speaker": speaker, "instruct": instruct},
            stream=True,
            timeout=(5, 120),
        )
        resp.raise_for_status()

        buf = b""
        for raw_chunk in resp.iter_content(chunk_size=8192):
            buf += raw_chunk
            while len(buf) >= 8:
                sr, pcm_len = struct.unpack(">II", buf[:8])
                if len(buf) < 8 + pcm_len:
                    break
                pcm_data = buf[8 : 8 + pcm_len]
                buf = buf[8 + pcm_len :]
                yield pcm_data, sr

    except requests.RequestException as e:
        print(f"TTS remote call error: {e}")
    except struct.error as e:
        print(f"TTS stream parse error (malformed chunk): {e}")


# ---------------------------------------------------------------
# TTSController — owns all TTS mutable state
# ---------------------------------------------------------------


class TTSController:
    """Owner of the TTS sentence queue, cancel flags, and worker thread.

    Lifecycle:
        ctl = TTSController()
        if ctl.configure(args):      # health-checks the remote service
            ctl.start_worker(args)
        # ... during generation ...
        ctl.enqueue_sentence(sentence, session, response_id, idx, args)
        ctl.clear_queue()            # between responses
    """

    def __init__(self, latency_log_path: str = "tts_latency.log"):
        # Configuration (set by configure())
        self.enabled = False
        self.streaming = False
        self.service_url = None

        # Sentence queue
        self._sentence_queue: queue.Queue = queue.Queue()
        self._sentence_queue_lock = threading.Lock()

        # Pending task slot (kept for parity with original module API)
        self._pending_task = None
        self._pending_lock = threading.RLock()

        # Worker + cancel state
        self._worker_running = False
        self._current_response_id = None
        self._cancel_flag = False

        # Latency log
        self._latency_log_path = latency_log_path
        self._latency_log_lock = threading.Lock()

    # ---- configuration --------------------------------------------------

    def configure(self, args) -> bool:
        """Health-check the remote TTS service and flip self.enabled.

        Returns True if the service is reachable and configured.
        """
        url = getattr(args, "tts_service_url", None)
        if not url:
            print("⚠️ --tts-service-url not specified")
            self.enabled = False
            return False

        self.service_url = url.rstrip("/")
        print(f"🔊 Checking TTS service at {self.service_url} ...")

        try:
            resp = requests.get(f"{self.service_url}/v1/tts/health", timeout=10)
            resp.raise_for_status()
            info = resp.json()
            print(f"✓ TTS service connected: {info}")
            self.streaming = True
            self.enabled = True
            return True
        except requests.RequestException as e:
            print(f"⚠️ TTS service unreachable ({self.service_url}): {e}")
            self.enabled = False
            return False

    def start_worker(self, args):
        """Idempotently start the background worker thread."""
        if not self._worker_running:
            self._worker_running = True
            t = threading.Thread(target=self._worker_loop, args=(args,), daemon=True)
            t.start()

    # ---- queue API ------------------------------------------------------

    def enqueue_sentence(self, sentence: str, session, response_id: str,
                         sentence_idx: int, args):
        """Add a sentence to the TTS queue for processing.

        This enables pipeline parallelism: model generates the next sentence
        while TTS processes the current one.
        """
        if not sentence or not sentence.strip():
            return

        clean_text = remove_markdown(sentence)
        if not clean_text.strip():
            return

        task = {
            "session": session,
            "response_id": response_id,
            "sentence_idx": sentence_idx,
            "text": clean_text,
            "language": args.tts_language if args else "Chinese",
            "speaker": args.tts_speaker if args else "Vivian",
            "instruct": args.tts_instruct if args else "",
            "output_dir": args.tts_output_dir if args else "tts_results",
        }

        self._sentence_queue.put(task)
        print(f"🎤 [Queue] Enqueued sentence {sentence_idx}: "
              f"{clean_text[:30]}... (queue size: {self._sentence_queue.qsize()})")

    def clear_queue(self, new_response_id: str = None):
        """Drain pending sentences. Does NOT cancel the currently running TTS."""
        with self._sentence_queue_lock:
            cleared = 0
            while not self._sentence_queue.empty():
                try:
                    self._sentence_queue.get_nowait()
                    cleared += 1
                except queue.Empty:
                    break

            if cleared > 0:
                print(f"🗑 Cleared {cleared} pending TTS sentences (current TTS continues)")

    def _get_task(self, timeout: float = 0.1):
        """Block for up to `timeout` seconds waiting for the next sentence task."""
        try:
            return self._sentence_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # ---- cancel / pending API -------------------------------------------

    def should_cancel(self) -> bool:
        with self._pending_lock:
            return self._cancel_flag

    def clear_cancel(self):
        with self._pending_lock:
            self._cancel_flag = False

    def set_pending(self, task: dict):
        """Replace the pending task slot and mark the current TTS for cancellation."""
        with self._pending_lock:
            old_task = self._pending_task
            self._pending_task = task

            if self._current_response_id is not None:
                self._cancel_flag = True
                print(f"⏭ Cancelling current TTS (id={self._current_response_id})")

            if old_task is not None:
                print(f"⏭ Dropping pending TTS task (id={old_task.get('response_id', 'unknown')})")

    def get_pending(self):
        with self._pending_lock:
            task = self._pending_task
            self._pending_task = None
            return task

    # ---- latency log ----------------------------------------------------

    def _log_latency(self, response_id: str, sentence_idx: int, text: str,
                     first_chunk_latency: float, total_latency: float,
                     audio_duration: float, num_chunks: int,
                     model_type: str = "unknown"):
        """Append one TTS latency record to disk + stdout."""
        # Calculate RTF (Real-Time Factor) — lower is better
        rtf = total_latency / audio_duration if audio_duration > 0 else float('inf')

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        text_preview = text[:50].replace('\n', ' ') + ('...' if len(text) > 50 else '')

        log_entry = (
            f"[{timestamp}] "
            f"response_id={response_id} | "
            f"sentence={sentence_idx} | "
            f"model={model_type} | "
            f"first_chunk={first_chunk_latency*1000:.1f}ms | "
            f"total={total_latency*1000:.1f}ms | "
            f"audio={audio_duration:.2f}s | "
            f"RTF={rtf:.3f} | "
            f"chunks={num_chunks} | "
            f"text=\"{text_preview}\"\n"
        )

        with self._latency_log_lock:
            try:
                with open(self._latency_log_path, "a", encoding="utf-8") as f:
                    f.write(log_entry)
            except OSError as e:
                print(f"⚠️ Failed to write TTS latency log: {e}")

        print(f"📊 [TTS Latency] first_chunk={first_chunk_latency*1000:.1f}ms, "
              f"total={total_latency*1000:.1f}ms, audio={audio_duration:.2f}s, RTF={rtf:.3f}")

    # ---- worker loop (runs in its own thread) ---------------------------

    def _worker_loop(self, args):
        """TTS worker thread loop - streams Type 9 (TTS Audio Chunk) raw PCM
        chunks for each sentence, one empty final marker per sentence.

        The CustomVoice / Type 5 (complete WAV) path was removed: the Flask
        bridge no longer routes Type 5 messages, and no production TTS
        deployment still uses that path.
        """
        # Probe remote service so startup logs show what we're connected to;
        # health check is informational only, the main pipeline is the same
        # regardless of model type.
        model_type = "base"
        try:
            resp = requests.get(f"{self.service_url}/v1/tts/health", timeout=5)
            if resp.ok:
                model_type = resp.json().get("model_type", "base")
        except (requests.RequestException, ValueError):
            # ValueError covers JSONDecodeError when the service returns non-JSON
            pass

        print(f"🔊 TTS Worker started (Remote service, Model: {model_type})")

        while True:
            task = self._get_task(timeout=0.1)
            if task is None:
                continue

            try:
                sentence_start = _time.time()
                first_chunk_sent = False

                session = task.get("session")
                if session is None:
                    continue  # tasks enqueued without a session (shouldn't happen post-refactor)
                response_id = task.get("response_id", "")
                sentence_idx = task.get("sentence_idx", 0)
                text = task.get("text", "")
                language = task.get("language", "Chinese")
                speaker = task.get("speaker", "Vivian")
                instruct = task.get("instruct", "")

                if not text.strip():
                    continue

                with self._pending_lock:
                    self._current_response_id = response_id
                    self._cancel_flag = False

                print(f"🎤 [TTS] Processing sentence {sentence_idx}: {text[:40]}...")

                chunk_idx = 0
                sr = 24000
                total_samples = 0
                first_chunk_latency = 0.0

                for audio_bytes, sample_rate in _text_to_speech_generator(
                    self.service_url, text, language, speaker, instruct
                ):
                    if self.should_cancel():
                        print(f"⏹ TTS cancelled for sentence {sentence_idx}")
                        break

                    sr = sample_rate
                    total_samples += len(audio_bytes) // 2  # int16 = 2 bytes

                    send_audio_chunk(
                        session,
                        pcm_bytes=audio_bytes,
                        response_id=response_id,
                        sentence_idx=sentence_idx,
                        chunk_idx=chunk_idx,
                        sample_rate=sr,
                        is_final=False,
                    )

                    if not first_chunk_sent:
                        first_chunk_latency = _time.time() - sentence_start
                        print(f"🚀 [TTS] First chunk sent in {first_chunk_latency:.3f}s")
                        first_chunk_sent = True

                    chunk_idx += 1

                if chunk_idx > 0:
                    sentence_time = _time.time() - sentence_start
                    audio_duration = total_samples / sr if sr > 0 else 0

                    self._log_latency(
                        response_id=response_id,
                        sentence_idx=sentence_idx,
                        text=text,
                        first_chunk_latency=first_chunk_latency,
                        total_latency=sentence_time,
                        audio_duration=audio_duration,
                        num_chunks=chunk_idx,
                        model_type=model_type,
                    )

                    if not self.should_cancel():
                        send_audio_chunk(
                            session,
                            pcm_bytes=b'',  # Empty data for final marker
                            response_id=response_id,
                            sentence_idx=sentence_idx,
                            chunk_idx=chunk_idx,
                            sample_rate=sr,
                            is_final=True,
                        )
                        print(f"🔊 [TTS] Sentence {sentence_idx} complete: "
                              f"{chunk_idx} chunks, {audio_duration:.1f}s audio "
                              f"in {sentence_time:.2f}s")
                    else:
                        print(f"⏹ [TTS] Sentence {sentence_idx} cancelled after "
                              f"{chunk_idx} chunks")

                with self._pending_lock:
                    self._current_response_id = None

            except Exception as e:
                # Worker-thread top-level catch: a narrow except would risk
                # killing the thread and silently stopping all TTS output.
                # Keep broad; log stack for diagnosis.
                print(f"TTS Worker error: {e}")
                traceback.print_exc()
