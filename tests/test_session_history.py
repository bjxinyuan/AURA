"""Unit tests for SessionHistory.

Behaviour-preservation tests extracted from the refactor of
Qwen3_VL_online_streaming_v2_ContextManaged.py. None of these require
GPU, vLLM, or a real tokenizer.
"""
import json
import tempfile
from pathlib import Path

import pytest

from aura.session_history import SessionHistory, SILENT_TEXT


def test_initial_state_has_system_prompt():
    h = SessionHistory()
    out = h.get_vllm_inputs()
    assert "prompt" in out and "multi_modal_data" in out
    assert "<|im_start|>system" in out["prompt"]
    # No images or videos before any user input
    assert out["multi_modal_data"] == {}
    assert h.current_rounds == 0


def test_add_user_text_only_message_appends_round():
    h = SessionHistory()
    h.add_user_message("what do you see?")
    assert h.current_rounds == 1
    out = h.get_vllm_inputs()
    assert "what do you see?" in out["prompt"]
    assert "<|im_start|>user" in out["prompt"]


def test_add_user_without_text_images_or_video_is_noop():
    """add_user_message with no content at all must not append a round."""
    h = SessionHistory()
    h.add_user_message("", images=None, video_tuple=None)
    assert h.current_rounds == 0


def test_add_user_with_video_tuple_emits_video_tokens():
    h = SessionHistory()
    fake_video = ("numpy_ndarray_placeholder", {"fps": 2.0})
    h.add_user_message("", video_tuple=fake_video)
    out = h.get_vllm_inputs()
    assert "<|vision_start|><|video_pad|><|vision_end|>" in out["prompt"]
    # Video payload preserved in multi_modal_data
    assert out["multi_modal_data"].get("video") == [fake_video]


def test_add_user_with_images_emits_image_tokens():
    h = SessionHistory()
    h.add_user_message("describe", images=["img1_placeholder", "img2_placeholder"])
    out = h.get_vllm_inputs()
    # Two image placeholders in the prompt
    assert out["prompt"].count("<|image_pad|>") == 2
    assert out["multi_modal_data"]["image"] == ["img1_placeholder", "img2_placeholder"]


def test_add_assistant_message_appears_in_prompt():
    h = SessionHistory()
    h.add_user_message("q1")
    h.add_assistant_message("a1")
    out = h.get_vllm_inputs()
    assert "<|im_start|>assistant" in out["prompt"]
    assert "a1" in out["prompt"]


def test_reset_clears_history():
    h = SessionHistory()
    h.add_user_message("q1")
    h.add_assistant_message("a1")
    h.add_user_message("q2")
    assert h.current_rounds == 2
    h._reset()
    assert h.current_rounds == 0
    out = h.get_vllm_inputs()
    assert "q1" not in out["prompt"]
    assert "q2" not in out["prompt"]
    # System prompt survives reset
    assert "<|im_start|>system" in out["prompt"]


def test_pruning_disabled_by_default_keeps_all_rounds():
    """Without --enable-pruning, history grows unbounded."""
    h = SessionHistory(max_rounds=2, num_rounds_keep=1, pruning_enabled=False)
    for i in range(5):
        h.add_user_message(f"q{i}")
        h.add_assistant_message(f"a{i}")
    assert h.current_rounds == 5
    # All 5 rounds visible in prompt
    prompt = h.get_vllm_inputs()["prompt"]
    for i in range(5):
        assert f"q{i}" in prompt and f"a{i}" in prompt


def test_pruning_migrates_old_rounds_to_context_history():
    """With pruning on and sw_rounds > max_rounds, oldest rounds move to context_history."""
    h = SessionHistory(
        max_rounds=3, num_rounds_keep=2, pruning_enabled=True, max_context_qas=10
    )
    for i in range(5):
        h.add_user_message(f"question{i}")
        h.add_assistant_message(f"answer{i}")
    # After 5 rounds with max=3, oldest should be pruned to context_history
    assert len(h._context_history) >= 1
    # Sliding window retains the most recent rounds
    assert h.current_rounds == len(h._context_history) * 0 + h._sw_round_count()
    assert h._sw_round_count() <= 3


def test_silent_rounds_dropped_when_moving_to_context_history():
    """Rule B: assistant=<|silent|> rounds are removed during migration."""
    h = SessionHistory(
        max_rounds=2, num_rounds_keep=1, pruning_enabled=True, max_context_qas=10
    )
    # Create a pattern: q0/a0, q1/SILENT, q2/a2, q3/a3 -> triggers pruning
    h.add_user_message("question0")
    h.add_assistant_message("answer0")
    h.add_user_message("question1")
    h.add_assistant_message(SILENT_TEXT)
    h.add_user_message("question2")
    h.add_assistant_message("answer2")
    h.add_user_message("question3")
    h.add_assistant_message("answer3")
    # Silent turn should not appear in context_history
    ch_flat = [m for qa in h._context_history for m in qa]
    for msg in ch_flat:
        assert msg["content"] != SILENT_TEXT, f"silent round leaked into context_history: {msg}"


def test_video_stripped_when_moving_to_context_history():
    """Rule A: videos removed from context_history; only text QA retained."""
    h = SessionHistory(
        max_rounds=2, num_rounds_keep=1, pruning_enabled=True, max_context_qas=10
    )
    for i in range(4):
        # Each round has both text and video
        h.add_user_message(f"q{i}", video_tuple=(f"np_frames_{i}", {"fps": 2.0}))
        h.add_assistant_message(f"a{i}")

    # Flatten context_history and ensure none of the content is a list containing video
    for qa in h._context_history:
        for msg in qa:
            content = msg.get("content")
            assert isinstance(content, str), \
                f"context_history should hold plain text, got {type(content).__name__}: {content!r}"


def test_context_history_fifo_cap():
    """§3.3: context_history capacity ≤ max_context_qas."""
    h = SessionHistory(
        max_rounds=2, num_rounds_keep=1, pruning_enabled=True, max_context_qas=2
    )
    # Each text-bearing QA stays as its own entry in context_history. Add many.
    for i in range(10):
        h.add_user_message(f"q{i}")
        h.add_assistant_message(f"a{i}")
    assert len(h._context_history) <= 2


def test_save_context_debug_writes_jsonl(tmp_path):
    debug_file = tmp_path / "debug.jsonl"
    h = SessionHistory(debug_context_file=str(debug_file))
    h.add_user_message("hello")
    h.add_assistant_message("world")
    h.save_context_debug(request_id="req-1")
    h.add_user_message("second")
    h.save_context_debug(request_id="req-2")

    lines = debug_file.read_text().splitlines()
    assert len(lines) == 2
    rec1 = json.loads(lines[0])
    assert rec1["request_id"] == "req-1"
    assert "history" in rec1 and len(rec1["history"]) > 0
    assert rec1["history"][0]["role"] == "system"


def test_save_context_debug_noop_when_file_unset():
    """If debug_context_file is None, save_context_debug is a no-op."""
    h = SessionHistory(debug_context_file=None)
    h.add_user_message("x")
    h.save_context_debug(request_id="ignored")  # must not raise


# ---------------------------------------------------------------------------
# Omni E2E audio support
# ---------------------------------------------------------------------------

def test_add_user_with_audio_tuple_emits_audio_tokens():
    h = SessionHistory()
    fake_audio = ("waveform_placeholder", 16000)
    h.add_user_message("", audio_tuple=fake_audio)
    out = h.get_vllm_inputs()
    assert "<|audio_start|><|audio_pad|><|audio_end|>" in out["prompt"]
    assert out["multi_modal_data"].get("audio") == [fake_audio]
    assert h.current_rounds == 1


def test_add_user_with_video_and_audio_emits_both():
    h = SessionHistory()
    fake_video = ("video_placeholder", {"fps": 2.0})
    fake_audio = ("waveform_placeholder", 16000)
    h.add_user_message("", video_tuple=fake_video, audio_tuple=fake_audio)
    out = h.get_vllm_inputs()
    assert "<|vision_start|><|video_pad|><|vision_end|>" in out["prompt"]
    assert "<|audio_start|><|audio_pad|><|audio_end|>" in out["prompt"]
    assert out["multi_modal_data"].get("video") == [fake_video]
    assert out["multi_modal_data"].get("audio") == [fake_audio]


def test_replace_audio_with_text_drops_audio_from_multi_modal():
    """After rewrite, the user message no longer contributes an audio entry."""
    h = SessionHistory()
    h.add_user_message("", audio_tuple=("wav", 16000))
    h.replace_last_user_audio_with_text("今天天气怎么样")

    out = h.get_vllm_inputs()
    assert "audio" not in out["multi_modal_data"]
    assert "<|audio_start|>" not in out["prompt"]
    assert "今天天气怎么样" in out["prompt"]


def test_replace_audio_preserves_video_in_same_turn():
    """A turn with both video + audio: rewrite drops only audio."""
    h = SessionHistory()
    fake_video = ("video", {"fps": 2.0})
    h.add_user_message("", video_tuple=fake_video, audio_tuple=("wav", 16000))
    h.replace_last_user_audio_with_text("hello")

    out = h.get_vllm_inputs()
    # Video survives.
    assert "<|video_pad|>" in out["prompt"]
    assert out["multi_modal_data"]["video"] == [fake_video]
    # Audio gone, text added.
    assert "audio" not in out["multi_modal_data"]
    assert "hello" in out["prompt"]


def test_replace_audio_noop_when_no_audio():
    """Calling rewrite on a video-only turn does nothing harmful."""
    h = SessionHistory()
    h.add_user_message("", video_tuple=("v", {"fps": 2.0}))
    h.replace_last_user_audio_with_text("transcribed")

    out = h.get_vllm_inputs()
    # Transcription should NOT have been appended since there was no audio.
    assert "transcribed" not in out["prompt"]


def test_replace_audio_with_empty_transcription_just_drops_audio():
    """Empty transcription = drop audio, don't insert empty text item."""
    h = SessionHistory()
    h.add_user_message("", audio_tuple=("wav", 16000))
    h.replace_last_user_audio_with_text("")

    out = h.get_vllm_inputs()
    assert "audio" not in out["multi_modal_data"]
    assert "<|audio_start|>" not in out["prompt"]


def test_pruning_strips_audio_when_moved_to_context_history():
    """Audio entries that survive into a pruned round must be stripped
    (rewrite rule A extended)."""
    h = SessionHistory(max_rounds=2, num_rounds_keep=1, pruning_enabled=True)
    # Round 1: user with audio (intentionally NOT rewritten — simulating
    # the case where the rewrite was skipped, e.g. silent response path).
    h.add_user_message("", audio_tuple=("wav1", 16000))
    h.add_assistant_message("a1")
    # Round 2: another audio turn
    h.add_user_message("", audio_tuple=("wav2", 16000))
    h.add_assistant_message("a2")
    # Round 3: triggers pruning (sw_rounds=3 > max_rounds=2)
    h.add_user_message("", audio_tuple=("wav3", 16000))
    h.add_assistant_message("a3")

    # context_history should not contain any audio entries — Rule A strips
    # all multimedia placeholders. (Existing _extract_user_text returns "" for
    # audio-only content; that's fine, the round either becomes truncated and
    # gets merged, or just becomes a "" user message.)
    for qa in h._context_history:
        for msg in qa:
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    assert not (isinstance(item, dict) and item.get("type") == "audio")
