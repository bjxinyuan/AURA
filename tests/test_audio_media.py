"""Unit tests for aura.audio_media.

We synthesise tiny WAV files into tmp_path with soundfile (a transitive
dep of librosa, always present in this environment). The goal is to
verify the librosa decode path + resampling, not to test librosa itself.
"""
from pathlib import Path

import numpy as np
import pytest

try:
    import soundfile as sf
    _HAS_SF = True
except ImportError:
    _HAS_SF = False

from aura.audio_media import decode_audio_to_numpy


pytestmark = pytest.mark.skipif(
    not _HAS_SF, reason="soundfile not available; cannot synthesise test WAVs"
)


def _synthesise_wav(path: Path, duration_s: float, sr: int,
                    freq: float = 440.0, channels: int = 1) -> None:
    """Write a mono/stereo sine-wave WAV with the given sample rate."""
    n = int(duration_s * sr)
    t = np.linspace(0, duration_s, n, endpoint=False, dtype=np.float32)
    tone = 0.1 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    if channels == 2:
        tone = np.stack([tone, tone], axis=1)
    sf.write(str(path), tone, sr, subtype="PCM_16")


def test_missing_file_returns_none():
    arr, sr = decode_audio_to_numpy("/tmp/does_not_exist_aura_audio_test.wav")
    assert arr is None and sr is None


def test_basic_wav_decode(tmp_path):
    """16 kHz mono WAV passes through unchanged (no resample needed)."""
    path = tmp_path / "mono.wav"
    _synthesise_wav(path, duration_s=0.5, sr=16000, channels=1)

    arr, sr = decode_audio_to_numpy(str(path))
    assert arr is not None
    assert sr == 16000
    assert arr.dtype == np.float32
    assert arr.ndim == 1
    # 0.5s * 16000 = 8000 samples (librosa may be off by <=1)
    assert abs(arr.shape[0] - 8000) <= 2


def test_resample_from_44100_to_16000(tmp_path):
    """Non-target sample rate is resampled to 16 kHz."""
    path = tmp_path / "high.wav"
    _synthesise_wav(path, duration_s=1.0, sr=44100, channels=1)

    arr, sr = decode_audio_to_numpy(str(path), target_sr=16000)
    assert arr is not None
    assert sr == 16000
    # 1 second at 16k → 16000 samples (tolerance for librosa filter)
    assert abs(arr.shape[0] - 16000) <= 100


def test_stereo_downmixed_to_mono(tmp_path):
    """2-channel input is collapsed to mono by librosa (mono=True)."""
    path = tmp_path / "stereo.wav"
    _synthesise_wav(path, duration_s=0.5, sr=16000, channels=2)

    arr, sr = decode_audio_to_numpy(str(path))
    assert arr is not None
    assert arr.ndim == 1   # mono
    assert sr == 16000


def test_custom_target_sr(tmp_path):
    """target_sr parameter is respected."""
    path = tmp_path / "src.wav"
    _synthesise_wav(path, duration_s=0.5, sr=16000, channels=1)

    arr, sr = decode_audio_to_numpy(str(path), target_sr=8000)
    assert arr is not None
    assert sr == 8000
    # 0.5s * 8000 = 4000 samples
    assert abs(arr.shape[0] - 4000) <= 50


def test_corrupted_file_returns_none(tmp_path):
    """Garbage bytes → decoder fails gracefully, returns (None, None)."""
    path = tmp_path / "bad.wav"
    path.write_bytes(b"\x00\x01\x02not a valid audio file" * 10)

    arr, sr = decode_audio_to_numpy(str(path))
    assert arr is None and sr is None


def test_empty_file_returns_none(tmp_path):
    path = tmp_path / "empty.wav"
    path.write_bytes(b"")

    arr, sr = decode_audio_to_numpy(str(path))
    assert arr is None and sr is None
