"""aura.audio_media — browser audio blob → numpy waveform for Qwen3-Omni.

Mirror of aura/media.py but for audio. Browser's MediaRecorder typically
produces WebM/Opus; iOS Chrome may send MP4/AAC. librosa (backed by
audioread + ffmpeg) handles both uniformly.

Qwen3-Omni's audio encoder expects mono float32 at 16 kHz. We resample
and downmix here so the inference server doesn't have to know the
on-wire format.

Extracted verbatim from Qwen3_asr_serve.py's _read_audio_from_file at
the time of the ASR→E2E migration. The soundfile fallback was dropped:
browser-recorded WebM/MP4 isn't decodable by soundfile, and the Omni
deployment already requires librosa.
"""
import os
from typing import Optional

import numpy as np

try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False


OMNI_TARGET_SR = 16000


def decode_audio_to_numpy(
    input_path: str,
    target_sr: int = OMNI_TARGET_SR,
) -> tuple[Optional[np.ndarray], Optional[int]]:
    """Decode a browser audio blob to (mono float32, sample_rate).

    Args:
        input_path: Path to the saved audio file. Any format ffmpeg can
            read via librosa/audioread: webm, mp4, mp3, wav, flac, ogg.
        target_sr: Resample to this rate. Omni expects 16 kHz.

    Returns:
        (audio_array, sample_rate) on success, or (None, None) on any
        decode failure. audio_array is float32, mono, shape (n_samples,).
    """
    if not _HAS_LIBROSA:
        # Fail loudly — Omni deployments require librosa. Don't silently
        # degrade to something that can't decode WebM.
        raise RuntimeError(
            "librosa is required for audio decoding in Omni E2E mode. "
            "Install it via `pip install librosa`."
        )

    if not os.path.exists(input_path):
        return None, None

    try:
        wav, sr = librosa.load(input_path, sr=target_sr, mono=True)
    except (OSError, ValueError, RuntimeError) as e:
        # librosa raises ValueError on malformed input and RuntimeError
        # when the ffmpeg backend fails. Both indicate a bad blob from
        # the browser — log and skip, don't crash the session.
        print(f"❌ audio_media: decode failed for {input_path}: {e}")
        return None, None

    if wav.size == 0:
        print(f"⚠️ audio_media: empty waveform from {input_path}")
        return None, None

    return np.asarray(wav, dtype=np.float32), int(sr)
