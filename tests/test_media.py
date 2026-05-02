"""Unit tests for aura.media video decoding.

We synthesise small, valid videos with cv2.VideoWriter into tmp_path
rather than relying on any shipped test asset. This keeps the tests
hermetic and reasonably fast (~100ms per test).
"""
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from aura.media import _decode_video_sync, downsample_video_to_numpy


def _synthesise_video(path: Path, num_frames: int, fps: int,
                      width: int = 64, height: int = 48) -> None:
    """Write a tiny solid-colour MJPG video. MJPG is widely supported by
    the OpenCV wheels we depend on; avoids pulling codec packs."""
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))
    assert writer.isOpened(), f"VideoWriter could not open {path}"
    for i in range(num_frames):
        # Each frame a slightly different shade so we could distinguish them
        frame = np.full((height, width, 3), i * 3 % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_missing_file_returns_none_tuple():
    arr, meta = downsample_video_to_numpy("/tmp/does_not_exist_aura_test.mp4")
    assert arr is None and meta is None


def test_basic_decode(tmp_path):
    """1-second 10fps video → target_fps=2 → step=5 → ≈2 frames."""
    video = tmp_path / "basic.avi"
    _synthesise_video(video, num_frames=10, fps=10)
    arr, meta = downsample_video_to_numpy(str(video), target_fps=2.0)
    assert arr is not None
    assert arr.ndim == 4 and arr.shape[-1] == 3   # (N, H, W, 3)
    assert arr.dtype == np.uint8
    assert meta["fps"] == 2.0
    assert meta["video_backend"] == "opencv"
    assert meta["do_sample_frames"] is False
    assert meta["total_num_frames"] == arr.shape[0]
    # With step = floor(10 / 2) = 5, over 10 frames we should get ~2 samples
    assert arr.shape[0] in (2, 3)


def test_target_fps_controls_step(tmp_path):
    """Doubling target_fps roughly doubles extracted frame count."""
    video = tmp_path / "step.avi"
    _synthesise_video(video, num_frames=30, fps=30)

    arr_low, _ = downsample_video_to_numpy(str(video), target_fps=2.0)   # step=15
    arr_high, _ = downsample_video_to_numpy(str(video), target_fps=6.0)  # step=5

    assert arr_low is not None and arr_high is not None
    assert arr_high.shape[0] > arr_low.shape[0]


def test_frames_indices_populated(tmp_path):
    """frame_indices records the source-video indices that were sampled."""
    video = tmp_path / "indices.avi"
    _synthesise_video(video, num_frames=20, fps=20)

    _, meta = downsample_video_to_numpy(str(video), target_fps=2.0)   # step=10
    indices = meta["frames_indices"]
    assert len(indices) == meta["total_num_frames"]
    assert indices == sorted(indices)       # monotonically increasing
    # Step size between consecutive indices should be roughly 10
    if len(indices) >= 2:
        assert indices[1] - indices[0] == 10


def test_resize_halves_dimensions(tmp_path):
    """With resize=True the output is 1/8 the input dimensions.

    Regression: the old code passed float dsize into cv2.resize and would
    crash. Verify the integer-divide fix works.
    """
    video = tmp_path / "resize.avi"
    _synthesise_video(video, num_frames=5, fps=5, width=80, height=64)

    arr, _ = downsample_video_to_numpy(str(video), target_fps=5.0, resize=True)
    assert arr is not None
    # Original 80x64 → 1/8 → 10x8
    assert arr.shape[1:3] == (8, 10)   # (H, W)


def test_decode_video_sync_writes_and_cleans_up(tmp_path):
    """_decode_video_sync: writes file_data, decodes, removes file."""
    # Build a tiny video in memory: encode with writer, then read bytes
    src_video = tmp_path / "src.avi"
    _synthesise_video(src_video, num_frames=10, fps=10)
    file_data = src_video.read_bytes()

    dest_path = tmp_path / "sync_tmp.avi"
    arr, meta = _decode_video_sync(
        file_data, str(dest_path), target_fps=2.0, resize=False,
    )
    assert arr is not None and meta is not None
    # Temp file should have been cleaned up
    assert not dest_path.exists(), "_decode_video_sync should remove the input file"


def test_decode_video_sync_cleans_up_on_decode_failure(tmp_path):
    """Even when downsample fails, _decode_video_sync still removes the temp file."""
    dest_path = tmp_path / "bad.avi"
    garbage = b"this is not a valid video stream"
    arr, meta = _decode_video_sync(
        garbage, str(dest_path), target_fps=2.0, resize=False,
    )
    assert arr is None and meta is None
    assert not dest_path.exists(), "temp file must be removed even when decode fails"
