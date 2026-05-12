"""Tests for aura.query_extractor — the streaming <query>...</query> parser."""
from aura.query_extractor import QueryExtractor, FeedResult


def feed_all(extractor: QueryExtractor, chunks: list[str]) -> list[FeedResult]:
    """Feed each chunk in turn, return list of per-call results."""
    return [extractor.feed(c) for c in chunks]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_single_delta_full_tag():
    """Whole response fits in one delta."""
    ext = QueryExtractor()
    r = ext.feed("<query>今天天气</query>好像不错")
    assert r.query_complete is True
    assert r.query_text == "今天天气"
    assert r.passthrough_delta == "好像不错"
    assert r.fallback_triggered is False


def test_cross_delta_split():
    """Tag straddles delta boundaries."""
    ext = QueryExtractor()
    results = feed_all(ext, ["<qu", "ery>a", "bc</que", "ry>xyz"])
    # Only the delta containing </query> reports completion.
    completed = [r for r in results if r.query_complete]
    assert len(completed) == 1
    assert completed[0].query_text == "abc"
    assert completed[0].passthrough_delta == "xyz"


def test_passthrough_after_complete():
    """Feeds after completion pass through unchanged."""
    ext = QueryExtractor()
    ext.feed("<query>hi</query>first ")
    r = ext.feed("second")
    assert r.query_complete is False  # not re-reporting
    assert r.passthrough_delta == "second"
    r2 = ext.feed(" third")
    assert r2.passthrough_delta == " third"


# ---------------------------------------------------------------------------
# Format tolerance
# ---------------------------------------------------------------------------

def test_missing_open_tag():
    """Model forgot <query> but emitted </query> — take everything before it."""
    ext = QueryExtractor()
    r = ext.feed("天气怎么样</query>还行")
    assert r.query_complete is True
    assert r.query_text == "天气怎么样"
    assert r.passthrough_delta == "还行"


def test_nested_open_tags_no_nesting():
    """Second <query> inside is treated as literal content; first </query> closes."""
    ext = QueryExtractor()
    r = ext.feed("<query>a<query>b</query>rest")
    assert r.query_complete is True
    # First <query> opens; content up to first </query> is the query.
    assert r.query_text == "a<query>b"
    assert r.passthrough_delta == "rest"


def test_whitespace_stripped_from_query():
    """Leading/trailing whitespace in query content is stripped."""
    ext = QueryExtractor()
    r = ext.feed("<query>  hello  </query>world")
    assert r.query_text == "hello"
    assert r.passthrough_delta == "world"


# ---------------------------------------------------------------------------
# Fallback path
# ---------------------------------------------------------------------------

def test_fallback_when_exceeds_max_chars():
    """No </query> after max_chars_before_fallback → force-complete."""
    ext = QueryExtractor(max_chars_before_fallback=10)
    # "<query>" is 7 chars; add 5 more to exceed 10.
    r = ext.feed("<query>abcdef")
    assert r.query_complete is True
    assert r.fallback_triggered is True
    assert r.query_text == "abcdef"
    assert r.passthrough_delta == ""
    # Subsequent deltas pass through.
    r2 = ext.feed("ghi")
    assert r2.passthrough_delta == "ghi"
    assert r2.query_complete is False


def test_fallback_preserves_model_output_via_passthrough():
    """After fallback, nothing is lost — post-fallback deltas are passthrough."""
    ext = QueryExtractor(max_chars_before_fallback=8)
    r1 = ext.feed("<query>longer")  # 13 chars, triggers fallback
    assert r1.fallback_triggered is True
    # The "longer" was captured as query. Subsequent content is passthrough.
    r2 = ext.feed(" tail")
    assert r2.passthrough_delta == " tail"


# ---------------------------------------------------------------------------
# finalize()
# ---------------------------------------------------------------------------

def test_finalize_before_close_tag():
    """Stream ended mid-query — finalize returns best-effort query."""
    ext = QueryExtractor()
    ext.feed("<query>partial")
    fin = ext.finalize()
    assert fin.query_complete is True
    assert fin.fallback_triggered is True
    assert fin.query_text == "partial"


def test_finalize_after_normal_completion():
    """finalize() after </query> already seen reports the cached query."""
    ext = QueryExtractor()
    ext.feed("<query>done</query>reply")
    fin = ext.finalize()
    # query_complete defaults False on "already completed" finalize — we
    # only signal completion once. But query_text is still available.
    assert fin.query_text == "done"


def test_finalize_on_empty_stream():
    """Model produced no text at all — finalize gives empty query."""
    ext = QueryExtractor()
    fin = ext.finalize()
    assert fin.query_complete is True
    assert fin.query_text == ""
    assert fin.fallback_triggered is True


def test_finalize_with_no_tags_any():
    """Model ignored the format entirely — treat all output as query."""
    ext = QueryExtractor(max_chars_before_fallback=1000)
    ext.feed("just a reply with no tags")
    fin = ext.finalize()
    assert fin.query_complete is True
    assert fin.query_text == "just a reply with no tags"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_feed_is_noop():
    ext = QueryExtractor()
    r = ext.feed("")
    assert r.query_complete is False
    assert r.passthrough_delta == ""


def test_close_tag_immediately_at_start():
    """Pathological: </query> with nothing before it."""
    ext = QueryExtractor()
    r = ext.feed("</query>response")
    assert r.query_complete is True
    assert r.query_text == ""  # empty query
    assert r.passthrough_delta == "response"


def test_passthrough_includes_content_from_same_delta():
    """When </query> and reply appear together, reply is in passthrough."""
    ext = QueryExtractor()
    r = ext.feed("<query>Q</query>R")
    assert r.query_text == "Q"
    assert r.passthrough_delta == "R"


def test_unicode_query_content():
    """CJK + emoji in query content."""
    ext = QueryExtractor()
    r = ext.feed("<query>你好 🌍 world</query>回答")
    assert r.query_text == "你好 🌍 world"
    assert r.passthrough_delta == "回答"


def test_custom_tags():
    """Non-default tag configuration."""
    ext = QueryExtractor(open_tag="[[", close_tag="]]")
    r = ext.feed("[[hi]]rest")
    assert r.query_text == "hi"
    assert r.passthrough_delta == "rest"
