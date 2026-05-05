"""Tests for aura.protocol — header pack/unpack and constant values.

Guards against silent drift: the Flask middle layer imports these same
constants, so a wire-format change on either side must update both.
"""
import struct

import pytest

from aura.protocol import (
    HEADER_FMT,
    HEADER_SIZE,
    pack_header,
    unpack_header,
    VIDEO_TYPE,
    AUDIO_TYPE,
    CLEAR_CONTEXT_TYPE,
    START_CAMERA_TYPE,
    ERROR_TYPE,
    STREAMING_TOKEN_TYPE,
    TTS_AUDIO_CHUNK_TYPE,
    ASR_QUERY_ECHO_TYPE,
)


def test_header_format_is_1byte_type_plus_8byte_length():
    """Wire format is [type:1 byte][length:8 bytes big-endian uint64]."""
    assert HEADER_FMT == ">BQ"
    assert HEADER_SIZE == 9
    assert struct.calcsize(HEADER_FMT) == HEADER_SIZE


def test_constant_values_are_stable():
    """These ints are part of the client-server wire protocol; changing
    any of them is a breaking change. Pin them explicitly."""
    assert VIDEO_TYPE == 1
    assert AUDIO_TYPE == 2
    assert CLEAR_CONTEXT_TYPE == 4
    assert START_CAMERA_TYPE == 6
    assert ERROR_TYPE == 7
    assert STREAMING_TOKEN_TYPE == 8
    assert TTS_AUDIO_CHUNK_TYPE == 9
    assert ASR_QUERY_ECHO_TYPE == 10


def test_pack_returns_nine_bytes():
    hdr = pack_header(VIDEO_TYPE, 0)
    assert isinstance(hdr, bytes) and len(hdr) == HEADER_SIZE


@pytest.mark.parametrize("msg_type", [
    VIDEO_TYPE, AUDIO_TYPE, CLEAR_CONTEXT_TYPE, START_CAMERA_TYPE,
    ERROR_TYPE, STREAMING_TOKEN_TYPE, TTS_AUDIO_CHUNK_TYPE, ASR_QUERY_ECHO_TYPE,
])
@pytest.mark.parametrize("payload_len", [0, 1, 255, 65536, 2**32 - 1, 2**63 - 1])
def test_roundtrip_every_type_every_size(msg_type, payload_len):
    """pack/unpack roundtrip must be lossless for every defined type and
    across the full uint64 length range."""
    hdr = pack_header(msg_type, payload_len)
    t_back, l_back = unpack_header(hdr)
    assert t_back == msg_type
    assert l_back == payload_len


def test_pack_rejects_type_out_of_byte_range():
    """Type is a single byte; values outside 0..255 should raise."""
    with pytest.raises(struct.error):
        pack_header(256, 0)
    with pytest.raises(struct.error):
        pack_header(-1, 0)


def test_pack_rejects_length_out_of_uint64():
    with pytest.raises(struct.error):
        pack_header(VIDEO_TYPE, -1)
    with pytest.raises(struct.error):
        pack_header(VIDEO_TYPE, 2**64)


def test_unpack_rejects_wrong_size():
    """unpack_header requires exactly 9 bytes."""
    with pytest.raises(struct.error):
        unpack_header(b"\x00" * 8)
    with pytest.raises(struct.error):
        unpack_header(b"\x00" * 10)
