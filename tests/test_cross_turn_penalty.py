"""Unit tests for CrossTurnPenalty.

Uses a minimal fake tokenizer — no transformers / HF / GPU required.
"""
import pytest
from aura.cross_turn_penalty import CrossTurnPenalty, _PENALTY_PUNCT


class FakeTokenizer:
    """Character-level tokenizer.

    - encode(): each character -> its Unicode code point
    - decode(): inverse
    - all_special_ids: none
    - Deterministic, lossless round-trip for ASCII + CJK test inputs.
    """
    all_special_ids = []

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def test_empty_history_returns_empty_kwargs():
    p = CrossTurnPenalty(tokenizer=FakeTokenizer())
    assert p.build_sampling_kwargs() == {}


def test_single_turn_no_logit_bias():
    """Need >=2 spoken turns to have cross-turn overlap."""
    p = CrossTurnPenalty(tokenizer=FakeTokenizer(), window=5)
    p.record("hello world")
    kwargs = p.build_sampling_kwargs()
    # With only 1 spoken turn, logit_bias must be empty (but bad_words may populate)
    assert "logit_bias" not in kwargs or not kwargs["logit_bias"]


def test_two_spoken_turns_with_overlap_build_logit_bias():
    """Tokens appearing in 2+ spoken turns should get negative logit_bias."""
    p = CrossTurnPenalty(tokenizer=FakeTokenizer(), window=5, logit_penalty=2.0)
    p.record("abcdef")
    p.record("abcxyz")  # 'a','b','c' overlap
    kwargs = p.build_sampling_kwargs()
    assert "logit_bias" in kwargs
    bias = kwargs["logit_bias"]
    # Overlapping tokens 'a' 'b' 'c' (code points 97-99) should all appear with negative bias
    for ch in "abc":
        assert ord(ch) in bias, f"expected '{ch}' ({ord(ch)}) in logit_bias"
        assert bias[ord(ch)] < 0


def test_non_overlapping_tokens_excluded_from_bias():
    p = CrossTurnPenalty(tokenizer=FakeTokenizer(), window=5)
    p.record("abcdef")
    p.record("abcxyz")
    kwargs = p.build_sampling_kwargs()
    bias = kwargs.get("logit_bias", {})
    # 'd' 'e' 'f' appear only in turn 1; 'x' 'y' 'z' only in turn 2 -> excluded
    for ch in "defxyz":
        assert ord(ch) not in bias, f"non-overlapping '{ch}' leaked into bias"


def test_punctuation_not_penalized():
    p = CrossTurnPenalty(tokenizer=FakeTokenizer(), window=5)
    # Both turns are pure punctuation from the penalty set — should be filtered out
    p.record("...,,")
    p.record("...,,")
    kwargs = p.build_sampling_kwargs()
    bias = kwargs.get("logit_bias", {})
    for ch in "...,,":
        assert ord(ch) not in bias, \
            f"punct '{ch}' should be excluded (_PENALTY_PUNCT contains it: {ch in _PENALTY_PUNCT})"


def test_digits_not_penalized():
    """_is_penalizable excludes pure-digit decodes."""
    p = CrossTurnPenalty(tokenizer=FakeTokenizer(), window=5)
    p.record("a1 b2 c3")
    p.record("a1 b2 c3")
    bias = p.build_sampling_kwargs().get("logit_bias", {})
    for digit in "123":
        assert ord(digit) not in bias, f"digit '{digit}' should not be penalised"


def test_window_limits_turn_retention():
    """Turns beyond `window` are evicted (FIFO)."""
    p = CrossTurnPenalty(tokenizer=FakeTokenizer(), window=2)
    p.record("alpha")
    p.record("beta")
    p.record("gamma")
    # Only last 2 ("beta", "gamma") retained
    assert p._history == ["beta", "gamma"]


def test_silent_turn_recorded_as_none_not_biased():
    """record(None) stores None; _spoken_history() filters it out."""
    p = CrossTurnPenalty(tokenizer=FakeTokenizer(), window=5)
    p.record("hello")
    p.record(None)              # silent
    p.record("hello world")
    assert p._history.count(None) == 1
    spoken = p._spoken_history()
    assert spoken == ["hello", "hello world"]


def test_reset_clears_history():
    p = CrossTurnPenalty(tokenizer=FakeTokenizer())
    p.record("hello")
    p.record("hello")
    assert p.build_sampling_kwargs()  # non-empty
    p.reset()
    assert p._history == []
    assert p.build_sampling_kwargs() == {}


def test_bad_words_contain_repeated_phrases():
    """N-grams from spoken history populate bad_words."""
    p = CrossTurnPenalty(tokenizer=FakeTokenizer(), window=5, ngram_sizes=[3])
    p.record("abcde")  # n-grams: abc, bcd, cde
    p.record("fghij")
    kwargs = p.build_sampling_kwargs()
    assert "bad_words" in kwargs
    # At least one n-gram from each spoken turn should show up
    assert any(bw in ("abc", "bcd", "cde") for bw in kwargs["bad_words"])


def test_max_bad_ngrams_capped():
    p = CrossTurnPenalty(
        tokenizer=FakeTokenizer(),
        window=5,
        ngram_sizes=[3],
        max_bad_ngrams=5,
    )
    # Generate far more than 5 distinct trigrams
    long_text = "".join(chr(ord("a") + i) for i in range(26))  # abcd...z
    p.record(long_text)
    kwargs = p.build_sampling_kwargs()
    assert len(kwargs.get("bad_words", [])) <= 5


def test_max_bias_tokens_capped():
    """max_bias_tokens clamps the returned logit_bias dict."""
    p = CrossTurnPenalty(
        tokenizer=FakeTokenizer(),
        window=5,
        max_bias_tokens=3,
    )
    # Build two turns that share many tokens — far more than 3
    shared = "abcdefghijklmnop"
    p.record(shared)
    p.record(shared)
    kwargs = p.build_sampling_kwargs()
    assert len(kwargs.get("logit_bias", {})) <= 3
