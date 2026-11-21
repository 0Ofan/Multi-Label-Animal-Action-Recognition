"""Real-frame interval sampling and the unchanged release full-frame transform."""
import math
from fractions import Fraction
from pathlib import Path
import cv2
import torch
from animalact2.phase1_5_1.video_decoder import decode_exact_frames
from animalact2.phase1_5_1.transforms import FullFrameTransform
SUPPORTED = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}
MAX_BYTES = 200 * 1024**2

def inspect_video(path):
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED:
        raise ValueError("Unsupported video format")
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_BYTES:
        raise ValueError("Video is empty, missing or exceeds 200 MiB")
    # Direct developer/CLI use is also prohibited from opening dataset Test paths.
    if any(x.lower() == "test" for x in path.resolve().parts):
        raise ValueError("Test media is disabled in this Dashboard")
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE])
    try:
        if not cap.isOpened(): raise ValueError("Video cannot be decoded; check format and codec")
        n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=float(cap.get(cv2.CAP_PROP_FPS))
        w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if n<1 or not math.isfinite(fps) or fps<=0 or min(w,h)<1: raise ValueError("Invalid video metadata")
        if n/fps>600 or n>100000: raise ValueError("Maximum duration is 600 seconds / 100000 source frames")
        return dict(frames=n, fps=fps, width=w, height=h, duration=n/fps)
    finally: cap.release()

def sample_indices(n, fps, start=0., end=None):
    start=float(start); end=n/fps if end is None else float(end)
    if not all(math.isfinite(x) for x in (start,end)) or not 0<=start<end<=n/fps+1e-8:
        raise ValueError("Window must satisfy 0 <= start < end <= video duration")
    # Frame timestamps are k/fps, exactly as the release decoder.
    available=[k for k in range(n) if start <= k/fps < end]
    if not available: raise ValueError("The selected window has no actual frame timestamp")
    indices=available if len(available)<=16 else [available[round(Fraction(j*(len(available)-1),15))] for j in range(16)]
    return indices, [start,end]

def prepare_video(path, start=0., end=None):
    meta=inspect_video(path)
    indices,window=sample_indices(meta["frames"],meta["fps"],start,end)
    decoded=decode_exact_frames(path,indices,expected_num_frames=meta["frames"],expected_fps=meta["fps"])
    transform=FullFrameTransform()
    pixels=torch.stack([transform(frame) for frame in decoded.rgb])[None]
    lt=(len(indices)+1)//2; h,w=pixels.shape[-2:]
    tokens=lt*((h+15)//16)*((w+15)//16)
    if tokens>4096: raise ValueError("Full-frame aspect ratio produces more than 4096 tokens; choose a less extreme video. No automatic crop is applied")
    return pixels,dict(**meta, sampled_frame_indices=indices, sampled_timestamps=list(decoded.timestamps_seconds),
        window_seconds=window, window_policy="One supplied clip/window; no automatic action localization or sliding windows",
        real_length=len(indices), tubelet_length=lt, resized_hw=[h,w], token_count=tokens)

class VideoInputError(ValueError):
    def __init__(self,message,code="invalid_video"):
        super().__init__(message);self.code=code
