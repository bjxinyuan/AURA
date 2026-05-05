"""Tests for aura.text_utils.remove_markdown.

Covers the handful of Markdown constructs the TTS feed sees so that
someone editing a regex doesn't silently change what the speech engine
pronounces.
"""
from aura.text_utils import remove_markdown


def test_bold_double_asterisk():
    assert remove_markdown("hello **world** !") == "hello world !"


def test_italic_single_asterisk():
    assert remove_markdown("a *b* c") == "a b c"


def test_bold_double_underscore():
    assert remove_markdown("__strong__ text") == "strong text"


def test_italic_single_underscore():
    assert remove_markdown("_em_ text") == "em text"


def test_heading_only_stripped_at_line_start():
    assert remove_markdown("# Title") == "Title"
    assert remove_markdown("## Subheading\nbody") == "Subheading\nbody"
    # Not at line start → kept.
    assert remove_markdown("abc # not-a-heading") == "abc # not-a-heading"


def test_list_bullet_stripped_at_line_start():
    assert remove_markdown("- item one\n- item two") == "item one\nitem two"
    assert remove_markdown("  * nested") == "nested"
    # Mid-line asterisk isn't a list marker → kept untouched.
    assert remove_markdown("word * word") == "word * word"


def test_link_keeps_anchor_text_drops_url():
    assert remove_markdown("[click](https://example.com)") == "click"
    assert remove_markdown("See [docs](foo) for more.") == "See docs for more."


def test_fenced_code_block_removed():
    assert remove_markdown("before\n```\nx = 1\n```\nafter") == "before\n\nafter"


def test_inline_code_unwrapped():
    assert remove_markdown("call `foo()`") == "call foo()"


def test_plain_text_passes_through():
    assert remove_markdown("Just some plain prose.") == "Just some plain prose."


def test_empty_input():
    assert remove_markdown("") == ""


def test_combined_constructs():
    src = "**bold** and *em* and `code` and [link](url)"
    assert remove_markdown(src) == "bold and em and code and link"
