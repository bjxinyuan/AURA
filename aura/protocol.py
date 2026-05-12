"""AURA TCP protocol between Flask bridge and main inference service.

9-byte header: [type:1][length:8] big-endian.
Used in both Qwen3_VL_online_streaming_v2_ContextManaged.py and
realtime_capture_video_audio_streaming.py.

Type 2 semantics changed with the Qwen3-Omni E2E migration: the
payload is now the raw audio blob fed directly to the model (decoded
to a waveform in aura.audio_media). Previously this was POSTed to a
separate ASR service to obtain a transcription. Older AURA-8B clients
will still send the same wire format but talk to a different backend
expectation — these versions are not interoperable.
"""
import struct

# Client -> Server
VIDEO_TYPE = 1
AUDIO_TYPE = 2          # End-to-end audio (fed to Omni directly, no ASR call)
CLEAR_CONTEXT_TYPE = 4
START_CAMERA_TYPE = 6

# Server -> Client
ERROR_TYPE = 7
STREAMING_TOKEN_TYPE = 8
TTS_AUDIO_CHUNK_TYPE = 9       # streaming raw PCM
ASR_QUERY_ECHO_TYPE = 10       # transcription extracted from <query>...</query>

HEADER_FMT = ">BQ"
HEADER_SIZE = 9


def pack_header(msg_type: int, payload_len: int) -> bytes:
    return struct.pack(HEADER_FMT, msg_type, payload_len)


def unpack_header(header: bytes) -> tuple[int, int]:
    msg_type, length = struct.unpack(HEADER_FMT, header)
    return msg_type, length
