"""aura.media — video frame decoding for the AURA inference server.

Wraps the OpenCV path that turns a browser-uploaded WebM blob into the
(numpy_array, metadata) tuple that Qwen3-VL expects as its video input.

`downsample_video_to_numpy` is the main entry point; `_decode_video_sync`
is the thin wrapper used from `run_in_executor` so the asyncio handler
can offload file write + decode + cleanup to a thread.

Extracted verbatim from Qwen3_VL_online_streaming_v2_ContextManaged.py,
with two exceptions called out inline:
- cv2.resize() now receives integer dsize (was `w / 8, h / 8` which raises
  at runtime — dead path in production, but corrected while moving).
- numpy is top-level imported here since the whole module is OpenCV/numpy.
"""
import os

import cv2
import numpy as np


def _decode_video_sync(
    file_data: bytes,
    input_path: str,
    target_fps: float,
    resize: bool,
) -> tuple:
    """Sync helper for run_in_executor: write file, downsample, remove.

    Returns (video_array, metadata) or (None, None).
    """
    with open(input_path, "wb") as f:
        f.write(file_data)
    try:
        return downsample_video_to_numpy(input_path, target_fps=target_fps, resize=resize)
    finally:
        try:
            os.remove(input_path)
        except OSError:
            pass


def downsample_video_to_numpy(
    input_path: str,
    target_fps: float = 2.0,
    resize: bool = False,
) -> tuple:
    """
    Downsample a video to target FPS and return as (numpy_array, metadata_dict) tuple.
    This format is compatible with Qwen3-VL's video input in vLLM.

    Args:
        input_path: Path to input video file
        target_fps: Target frame rate (default: 2 fps)
        resize: If True, resize frames to 1/8 resolution to reduce tokens and TTFT

    Returns:
        Tuple of (numpy_array, metadata_dict) or (None, None) if failed
        numpy_array shape: (num_frames, height, width, 3), dtype=uint8
        metadata_dict: {"fps": float, "duration": float, "total_num_frames": int, ...}
    """
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            file_size = os.path.getsize(input_path) if os.path.exists(input_path) else 0
            print(f"❌ Cannot open video: {input_path} (file size: {file_size} bytes)")
            print(f"   This may be due to unsupported codec (iOS Chrome often uses different codecs)")
            return None, None

        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Debug: print video properties for diagnosis
        need_sequential_read = False
        if original_fps <= 0 or total_frames <= 0:
            print(f"⚠️ Video metadata issue: fps={original_fps}, frames={total_frames}, size={width}x{height}")
            need_sequential_read = True
            # Try to read frames manually if metadata is invalid (iOS Chrome compatibility)
            if original_fps <= 0:
                original_fps = 15.0  # Assume 15fps as fallback
                print(f"⚠️ Using fallback fps: {original_fps}")

        # Calculate frame step
        step = max(1, int(original_fps / target_fps))

        frames = []
        frame_indices = []

        if need_sequential_read:
            # iOS Chrome compatibility: read ALL frames sequentially, then subsample
            # Some codecs don't support random frame access (cap.set doesn't work)
            cap.release()
            cap = cv2.VideoCapture(input_path)  # Reopen to reset position

            all_frames = []
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                all_frames.append((frame_idx, frame))
                frame_idx += 1

            print(f"⚠️ Sequential read: got {len(all_frames)} frames, subsampling with step={step}")

            # Subsample frames
            for idx, frame in all_frames[::step]:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                frame_indices.append(idx)
        else:
            # Normal mode: use frame seeking (works for most codecs)
            duration = total_frames / original_fps
            frame_idx = 0

            while frame_idx < total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                    frame_indices.append(frame_idx)
                frame_idx += step

        cap.release()

        if not frames:
            print(f"❌ No frames extracted from video")
            return None, None

        # Stack frames into numpy array: (num_frames, height, width, 3)
        video_array = np.stack(frames, axis=0)

        # Resize to 1/8 resolution to reduce token count and TTFT.
        # BUG FIX: dsize must be int — `w / 8, h / 8` raised OpenCV's "wrong type"
        # error at runtime. Dead path in production (default resize=False and
        # neither external caller sets it), but corrected while moving.
        if resize:
            h, w = video_array.shape[1], video_array.shape[2]
            video_array = np.stack([
                cv2.resize(video_array[i], (w // 8, h // 8), interpolation=cv2.INTER_AREA)
                for i in range(video_array.shape[0])
            ], axis=0)

        # Calculate original duration based on extracted frame indices
        if frame_indices:
            original_frame_count = frame_indices[-1] + 1 if need_sequential_read else total_frames
            duration = original_frame_count / original_fps
        else:
            duration = len(frames) / target_fps

        # Create metadata dict required by Qwen3-VL
        metadata = {
            "fps": target_fps,
            "duration": len(frames) / target_fps,
            "total_num_frames": len(frames),
            "frames_indices": frame_indices,
            "video_backend": "opencv",
            "do_sample_frames": False,  # Already sampled
        }

        print(f"📹 Video downsampled: {duration:.1f}s @ {original_fps:.0f}fps → {len(frames)} frames @ {target_fps}fps")

        return video_array, metadata

    except Exception as e:
        print(f"❌ Error downsampling video: {e}")
        return None, None
