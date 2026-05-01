"""CrossTurnPenalty — soft logit bias + hard bad-word n-gram blocking.

Extracted verbatim from Qwen3_VL_online_streaming_v2_ContextManaged.py
as part of the refactor described in arch.md. Behavior must be identical
to the pre-refactor implementation.
"""
from collections import Counter


_PENALTY_PUNCT = frozenset(
    ".,!?;:，。！？；：、'\"()[]{}""''…—–\n\t\r /-_@#$%^&*+=<>~`|\\（）【】《》"
)


class CrossTurnPenalty:
    """Cross-turn repetition penalty for embedded vLLM engine.

    Two complementary mechanisms:
    1. logit_bias  — soft penalty on content tokens from recent responses
    2. bad_words   — hard n-gram blocking via logits processor

    All penalty data is pre-computed before generation to minimize per-token
    overhead.  The logits processor itself only does tensor indexing (logit_bias)
    and a few dict lookups (bad_words) per step.
    """

    def __init__(
        self,
        tokenizer,
        window: int = 2,
        logit_penalty: float = 2.0,
        ngram_sizes: list[int] | None = None,
        max_bad_ngrams: int = 200,
        max_bias_tokens: int = 500,
    ):
        self.tokenizer = tokenizer
        self.window = window
        self.logit_penalty = logit_penalty
        self.ngram_sizes = ngram_sizes if ngram_sizes is not None else [3, 4, 5]
        self.max_bad_ngrams = max_bad_ngrams
        self.max_bias_tokens = max_bias_tokens
        self._history: list[str | None] = []   # None = silent turn, str = spoken turn
        self._special_ids = set(self.tokenizer.all_special_ids)
        self._penalizable_cache: dict[int, bool] = {}

    def _is_penalizable(self, token_id: int) -> bool:
        cached = self._penalizable_cache.get(token_id)
        if cached is not None:
            return cached
        if token_id in self._special_ids:
            self._penalizable_cache[token_id] = False
            return False
        decoded = self.tokenizer.decode([token_id]).strip()
        if not decoded or all(c in _PENALTY_PUNCT for c in decoded) or decoded.isdigit():
            self._penalizable_cache[token_id] = False
            return False
        self._penalizable_cache[token_id] = True
        return True

    def _spoken_history(self) -> list[str]:
        """Return only spoken (non-silent) entries from _history."""
        return [text for text in self._history if text is not None]

    def _build_logit_bias(self) -> dict[int, float]:
        spoken = self._spoken_history()
        if len(spoken) < 2:
            return {}
        n = len(spoken)

        # Phase 1: find tokens that appear in 2+ distinct spoken responses
        token_presence: dict[int, int] = {}
        for text in spoken:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            for tid in set(ids):
                token_presence[tid] = token_presence.get(tid, 0) + 1
        cross_turn_tids = {tid for tid, cnt in token_presence.items() if cnt >= 2}

        if not cross_turn_tids:
            return {}

        # Phase 2: compute penalty only for cross-turn tokens
        bias: dict[int, float] = {}
        for idx, text in enumerate(spoken):
            recency = (idx + 1) / n
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            freq = Counter(ids)
            for tid, cnt in freq.items():
                if tid not in cross_turn_tids:
                    continue
                if not self._is_penalizable(tid):
                    continue
                p = self.logit_penalty * min(cnt, 3) * recency
                bias[tid] = bias.get(tid, 0.0) + p

        penalized_details = ", ".join(
            f"'{self.tokenizer.decode([tid]).strip()}'({-val:.1f})"
            for tid, val in sorted(bias.items(), key=lambda kv: kv[1], reverse=True)[:20]
        )
        total_turns = len(self._history)
        print(f"🔧 [CrossTurnPenalty] window: {total_turns} actual turns "
              f"({len(spoken)} spoken, {total_turns - len(spoken)} silent)")
        print(f"🔧 [CrossTurnPenalty] cross-turn tokens: {len(cross_turn_tids)} "
              f"(penalizable: {len(bias)}) out of {len(token_presence)} total unique tokens")
        print(f"🔧 [CrossTurnPenalty] penalized: [{penalized_details}]")

        if len(bias) > self.max_bias_tokens:
            items = sorted(bias.items(), key=lambda kv: kv[1], reverse=True)
            bias = dict(items[: self.max_bias_tokens])
        return {k: min(v, 100.0) for k, v in bias.items()}

    def _build_bad_ngram_map(self) -> dict[tuple, set]:
        """prefix (n-1 token IDs) → set of blocked completing token IDs."""
        spoken = self._spoken_history()
        if not spoken:
            return {}
        prefix_map: dict[tuple, set] = {}
        seen: set[tuple] = set()
        count = 0
        for text in reversed(spoken):
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            for ng_size in self.ngram_sizes:
                if len(ids) < ng_size:
                    continue
                for i in range(len(ids) - ng_size + 1):
                    ngram = tuple(ids[i : i + ng_size])
                    if ngram in seen:
                        continue
                    phrase = self.tokenizer.decode(list(ngram)).strip()
                    if not phrase or all(c in _PENALTY_PUNCT for c in phrase):
                        continue
                    seen.add(ngram)
                    prefix = ngram[:-1]
                    if prefix not in prefix_map:
                        prefix_map[prefix] = set()
                    prefix_map[prefix].add(ngram[-1])
                    count += 1
                    if count >= self.max_bad_ngrams:
                        return prefix_map
        return prefix_map

    def build_sampling_kwargs(self) -> dict:
        """Return kwargs for SamplingParams: logit_bias and bad_words.

        Uses SamplingParams-native logit_bias (dict[int, float]) to softly
        penalise repeated content tokens, and bad_words (list[str]) to hard-
        block previously seen n-grams.  This replaces the old logits_processors
        approach which is no longer supported by vLLM V1.
        """
        raw_bias = self._build_logit_bias()
        bad_ngram_map = self._build_bad_ngram_map()

        if not raw_bias and not bad_ngram_map:
            return {}

        kwargs: dict = {}

        if raw_bias:
            kwargs["logit_bias"] = {tid: -val for tid, val in raw_bias.items()}

        if bad_ngram_map:
            bad_words: list[str] = []
            seen: set[tuple] = set()
            for prefix, blocked_set in bad_ngram_map.items():
                for last_tok in blocked_set:
                    ngram = prefix + (last_tok,)
                    if ngram in seen:
                        continue
                    seen.add(ngram)
                    phrase = self.tokenizer.decode(list(ngram))
                    if phrase.strip():
                        bad_words.append(phrase)
            if bad_words:
                kwargs["bad_words"] = bad_words

        bias_count = len(raw_bias)
        bw_count = len(kwargs.get("bad_words", []))
        spoken_count = len(self._spoken_history())
        total_turns = len(self._history)
        print(
            f"🔧 [CrossTurnPenalty] logit_bias: {bias_count} tokens | "
            f"bad_words: {bw_count} phrases | "
            f"window: {total_turns} turns ({spoken_count} spoken, "
            f"{total_turns - spoken_count} silent)"
        )

        return kwargs

    def record(self, response_text: str | None = None):
        """Call after every assistant turn (both spoken and silent).

        Args:
            response_text: The response text for spoken turns, or None for silent turns.
        """
        if response_text and response_text.strip():
            self._history.append(response_text)
        else:
            self._history.append(None)
        if len(self._history) > self.window:
            self._history.pop(0)

    def reset(self):
        self._history.clear()

