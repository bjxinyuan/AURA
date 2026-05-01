"""Unit tests for aura.tts — pure helpers + TTSController state machine.

We do not test _worker_loop (requires HTTP + a live TTS service) or
configure() (requires the remote /v1/tts/health endpoint). Those are
integration-level and covered by end-to-end smoke on the deployment
machine.
"""
import queue
import threading
import time

import pytest

from aura.tts import (
    TTSController,
    split_text_to_sentences,
    merge_short_sentences,
)


# ---------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------

class TestSplitTextToSentences:
    def test_empty_string(self):
        assert split_text_to_sentences("") == []
        assert split_text_to_sentences("   ") == []

    def test_chinese_punctuation(self):
        result = split_text_to_sentences("你好。世界！再见？")
        assert result == ["你好。", "世界！", "再见？"]

    def test_english_punctuation(self):
        result = split_text_to_sentences("Hello. World! Bye?")
        assert result == ["Hello.", "World!", "Bye?"]

    def test_long_sentence_split_at_commas(self):
        """Sentences longer than 50 chars are sub-split at commas."""
        long_text = "This is a really very quite extremely long sentence, with many parts, split by commas."
        result = split_text_to_sentences(long_text)
        # The long sentence (~90 chars) should be broken up
        assert len(result) > 1

    def test_no_terminator_keeps_original(self):
        """Text without terminators falls back to a single entry."""
        assert split_text_to_sentences("hello world") == ["hello world"]


class TestMergeShortSentences:
    def test_empty_input(self):
        assert merge_short_sentences([]) == []

    def test_short_fragments_merged(self):
        # min_chars=15 => merge budget 45 chars
        result = merge_short_sentences(["ab", "cd", "ef"], min_chars=15)
        assert result == ["abcdef"]

    def test_long_sentence_not_merged(self):
        long_s = "x" * 50
        result = merge_short_sentences([long_s, "short"], min_chars=15)
        # Long sentence passes through; "short" tries to merge but buffer is full
        assert long_s in result

    def test_trailing_short_fragment_kept(self):
        """A trailing short buffer is appended even if not merged with anything."""
        result = merge_short_sentences(["final bit"], min_chars=15)
        assert result == ["final bit"]


# ---------------------------------------------------------------
# TTSController state machine
# ---------------------------------------------------------------

class TestControllerState:
    def test_initial_state(self):
        ctl = TTSController()
        assert ctl.enabled is False
        assert ctl.streaming is False
        assert ctl.service_url is None
        assert ctl.should_cancel() is False
        assert ctl.get_pending() is None

    def test_configure_without_url_disables(self):
        class Args: tts_service_url = None
        ctl = TTSController()
        assert ctl.configure(Args()) is False
        assert ctl.enabled is False


class TestControllerQueue:
    def test_enqueue_requires_non_empty_text(self):
        ctl = TTSController()

        class Args:
            tts_language = "Chinese"
            tts_speaker = "Vivian"
            tts_instruct = ""
            tts_output_dir = "tts_results"

        # whitespace-only sentence is silently dropped
        ctl.enqueue_sentence("   ", session=object(), response_id="r1",
                             sentence_idx=0, args=Args())
        assert ctl._sentence_queue.empty()

    def test_enqueue_and_get(self):
        ctl = TTSController()

        class Args:
            tts_language = "Chinese"
            tts_speaker = "Vivian"
            tts_instruct = ""
            tts_output_dir = "tts_results"

        session_sentinel = object()
        ctl.enqueue_sentence("hello", session=session_sentinel, response_id="r1",
                             sentence_idx=3, args=Args())
        task = ctl._get_task(timeout=0.01)
        assert task is not None
        assert task["session"] is session_sentinel
        assert task["response_id"] == "r1"
        assert task["sentence_idx"] == 3
        assert task["text"] == "hello"

    def test_get_task_returns_none_on_empty(self):
        ctl = TTSController()
        assert ctl._get_task(timeout=0.01) is None

    def test_clear_queue_drains_pending(self):
        ctl = TTSController()

        class Args:
            tts_language = "Chinese"
            tts_speaker = "Vivian"
            tts_instruct = ""
            tts_output_dir = "tts_results"

        for i in range(5):
            ctl.enqueue_sentence(f"sent{i}", session=object(), response_id="r1",
                                 sentence_idx=i, args=Args())
        assert ctl._sentence_queue.qsize() == 5
        ctl.clear_queue()
        assert ctl._sentence_queue.empty()


class TestControllerCancel:
    def test_cancel_flag_lifecycle(self):
        ctl = TTSController()
        assert ctl.should_cancel() is False
        # set_pending triggers cancel only if a response is currently in-flight.
        # Without that, should_cancel stays False.
        ctl.set_pending({"response_id": "r1"})
        assert ctl.should_cancel() is False  # no current_response_id yet

    def test_cancel_raised_when_current_set(self):
        ctl = TTSController()
        # simulate worker entering a sentence
        with ctl._pending_lock:
            ctl._current_response_id = "r1"
        ctl.set_pending({"response_id": "r2"})
        assert ctl.should_cancel() is True

    def test_clear_cancel(self):
        ctl = TTSController()
        with ctl._pending_lock:
            ctl._cancel_flag = True
        assert ctl.should_cancel() is True
        ctl.clear_cancel()
        assert ctl.should_cancel() is False


class TestControllerPending:
    def test_pending_slot_set_and_get(self):
        ctl = TTSController()
        task = {"response_id": "r1"}
        ctl.set_pending(task)
        # get_pending consumes the slot
        assert ctl.get_pending() is task
        assert ctl.get_pending() is None
