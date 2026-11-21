"""Decode only the prescribed real frames, sequentially, with no seek/repeat fallback."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from numbers import Integral
import math
import cv2
import numpy as np


@dataclass(frozen=True)
class DecodedFrames:
    rgb: tuple[np.ndarray, ...]
    frame_indices: tuple[int, ...]
    timestamps_seconds: tuple[float, ...]
    fps: float
    parent_num_frames: int
    width: int
    height: int
    backend: str


def decode_exact_frames(path: Path | str, frame_indices, *, expected_num_frames=None, expected_fps=None) -> DecodedFrames:
    requested = list(frame_indices)
    if not requested or any(not isinstance(k, Integral) or isinstance(k, bool) for k in requested):
        raise ValueError("Expected a nonempty list of integer real frame indices")
    requested = [int(k) for k in requested]
    if requested != sorted(set(requested)) or requested[0] < 0:
        raise ValueError("Frame indices must be unique, nonnegative and in temporal order")
    capture = cv2.VideoCapture(str(Path(path)), cv2.CAP_FFMPEG,
                               [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE])
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        n = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if n <= 0 or not math.isfinite(fps) or fps <= 0 or requested[-1] >= n:
            raise ValueError("Invalid media metadata or out-of-range requested frame")
        if expected_num_frames is not None and n != expected_num_frames:
            raise ValueError("Decoded parent frame count differs from release-v1")
        if expected_fps is not None and abs(fps - float(expected_fps)) > 1e-6:
            raise ValueError("Decoded FPS differs from release-v1")
        selected = set(requested)
        frames, timestamps = [], []
        for k in range(requested[-1] + 1):
            if not capture.grab():
                raise RuntimeError(f"Missing actual frame {k}: {path}")
            if k in selected:
                ok, bgr = capture.retrieve()
                if not ok or bgr is None or bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
                    raise RuntimeError(f"Invalid decoded RGB input at frame {k}: {path}")
                timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000
                if not math.isfinite(timestamp) or abs(timestamp - k / fps) > 1e-6:
                    raise RuntimeError(f"Timestamp grid mismatch at frame {k}: {timestamp} versus {k/fps}")
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                timestamps.append(timestamp)
        if len(frames) != len(requested) or len({x.shape for x in frames}) != 1:
            raise RuntimeError("Incomplete selection or changing spatial dimensions within clip")
        h, w = frames[0].shape[:2]
        return DecodedFrames(tuple(frames), tuple(requested), tuple(timestamps), fps, n, w, h, capture.getBackendName())
    finally:
        capture.release()
