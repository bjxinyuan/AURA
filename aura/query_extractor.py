"""Streaming parser for <query>...</query> prefix in model output.

Qwen3-Omni-E2E mode: the model is prompted to wrap the user's transcribed
speech in <query>...</query> at the start of its response, then produce
the normal reply. This module separates the query prefix from the reply
as tokens stream in, so the caller can:

  1. Echo the query to the frontend (Type 10 ASR_QUERY_ECHO) as soon as
     </query> is seen.
  2. Rewrite SessionHistory's last user turn from raw audio to the
     transcription (critical for prefix-cache friendliness and context
     budget — see arch.md §5).
  3. Forward only the post-</query> tokens to the streaming token channel
     and TTS sentence buffer.

The parser is deliberately a pure string state machine — no tokenizer,
no I/O, no vLLM dependency — so it can be unit-tested without loading a
model.

Fallback contract: if the model fails to emit </query> within
``max_chars_before_fallback`` characters, everything accumulated so far
is treated as the query and subsequent deltas are passed through. This
keeps a format-violating model from stalling the echo + history rewrite
indefinitely.
"""
from dataclasses import dataclass


DEFAULT_OPEN_TAG = "<query>"
DEFAULT_CLOSE_TAG = "</query>"


@dataclass
class FeedResult:
    """Result of feeding one streaming delta to the extractor.

    - query_complete: True exactly on the feed() call that completes the
      query (either via </query> or via fallback). False on all other
      calls, including subsequent feeds after completion.
    - query_text: The extracted query, set on the completing call and
      also returned on later finalize() if still pending. None otherwise.
    - passthrough_delta: Text to forward to frontend + TTS. Empty while
      still inside the query tag; non-empty once the query has
      completed (may include post-</query> content from the same delta).
    - fallback_triggered: True when query_complete was reached via the
      max_chars fallback rather than seeing </query>.
    """
    query_complete: bool = False
    query_text: str | None = None
    passthrough_delta: str = ""
    fallback_triggered: bool = False


class QueryExtractor:
    """Streaming state machine for <query>...</query> prefix extraction.

    Usage:
        extractor = QueryExtractor()
        for delta in model_stream:
            result = extractor.feed(delta)
            if result.query_complete:
                send_asr_query(session, result.query_text)
                history.replace_last_user_audio_with_text(result.query_text)
            if result.passthrough_delta:
                send_streaming_token(session, result.passthrough_delta, ...)
                tts_buffer.append(result.passthrough_delta)
        # After decode end:
        final = extractor.finalize()
        if final.query_complete and not already_echoed:
            # Rare: model ended without ever emitting </query>; fall back
            # to echoing whatever accumulated (may be empty).
            ...
    """

    def __init__(
        self,
        max_chars_before_fallback: int = 80,
        open_tag: str = DEFAULT_OPEN_TAG,
        close_tag: str = DEFAULT_CLOSE_TAG,
    ):
        self.max_chars_before_fallback = max_chars_before_fallback
        self.open_tag = open_tag
        self.close_tag = close_tag

        # Accumulated raw output before </query> is seen. This is also
        # the buffer we draw the query text from.
        self._buf = ""
        # True once we've emitted the query (by </query> or fallback).
        # After this, feed() just passes through.
        self._query_emitted = False
        # Cached extracted query for finalize() to return if caller asks.
        self._query_text: str | None = None
        # Whether we've consumed an <open_tag> (if present). Used to decide
        # where the query starts inside _buf. None = not yet decided,
        # int = start offset of query content within _buf.
        self._query_start: int | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed(self, delta: str) -> FeedResult:
        """Consume one streaming delta. See FeedResult for return shape."""
        if not delta:
            return FeedResult()

        # Fast path: query already done, just pass through.
        if self._query_emitted:
            return FeedResult(passthrough_delta=delta)

        self._buf += delta

        # Decide where query content starts, if we haven't yet.
        # Only look once: the first <query> tag wins; any later ones are
        # treated as literal content (no nesting support).
        if self._query_start is None:
            idx = self._buf.find(self.open_tag)
            if idx >= 0:
                self._query_start = idx + len(self.open_tag)
            # If no open tag seen yet, we allow the model to skip it —
            # _query_start stays None, and on </query> we'll treat the
            # whole buffer up to </query> as the query.

        # Look for </query>.
        close_idx = self._buf.find(self.close_tag)
        if close_idx >= 0:
            return self._complete_via_close_tag(close_idx)

        # Fallback: too much accumulated without seeing </query>.
        if len(self._buf) >= self.max_chars_before_fallback:
            return self._complete_via_fallback()

        # Still buffering, nothing to pass through yet.
        return FeedResult()

    def finalize(self) -> FeedResult:
        """Call once the underlying stream has ended (decode complete).

        If the query never completed, force completion now so the caller
        has something to echo. If it already completed, this is a no-op
        that just reports the cached query_text.
        """
        if self._query_emitted:
            return FeedResult(query_text=self._query_text)

        # Stream ended mid-query. Best-effort: treat whatever we have as
        # the query. This may be empty string if the model produced no
        # text at all (pure silence response); caller decides whether to
        # echo an empty query.
        query = self._extract_query_from_buf(end=len(self._buf))
        self._query_emitted = True
        self._query_text = query
        return FeedResult(
            query_complete=True,
            query_text=query,
            passthrough_delta="",  # nothing left in the stream
            fallback_triggered=True,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _extract_query_from_buf(self, end: int) -> str:
        """Slice query text out of _buf[0:end], respecting the open tag
        position if one was seen."""
        start = self._query_start if self._query_start is not None else 0
        return self._buf[start:end].strip()

    def _complete_via_close_tag(self, close_idx: int) -> FeedResult:
        query = self._extract_query_from_buf(end=close_idx)
        passthrough = self._buf[close_idx + len(self.close_tag):]
        self._query_emitted = True
        self._query_text = query
        # Drop the buffer now that it's fully consumed; keeps memory
        # bounded if the caller lets the extractor live across turns
        # (which they shouldn't, but defensive).
        self._buf = ""
        return FeedResult(
            query_complete=True,
            query_text=query,
            passthrough_delta=passthrough,
            fallback_triggered=False,
        )

    def _complete_via_fallback(self) -> FeedResult:
        # Treat the entire accumulated buffer as the query. There's no
        # post-</query> content by definition, so passthrough is empty.
        # Subsequent feed() calls will pass through everything.
        query = self._extract_query_from_buf(end=len(self._buf))
        self._query_emitted = True
        self._query_text = query
        self._buf = ""
        return FeedResult(
            query_complete=True,
            query_text=query,
            passthrough_delta="",
            fallback_triggered=True,
        )
