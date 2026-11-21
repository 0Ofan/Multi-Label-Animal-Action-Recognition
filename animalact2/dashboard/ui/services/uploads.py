"""Only fresh owned temp files are removed; no dataset/project video writes."""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import tempfile

import cv2

from animalact2.dashboard.video import SUPPORTED as SUPPORTED_EXTENSIONS, VideoInputError

MAX_BYTES = 200 * 1024**2
TEMP_ROOT = Path(tempfile.gettempdir()) / "animalact2_dashboard_uploads"


def checked_name(name, content):
    name = Path(str(name).replace("\\", "/")).name
    if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise VideoInputError("Unsupported format. Choose MP4, AVI, MOV, MKV, WEBM, M4V, MPG or MPEG.", "unsupported_format")
    if not content:
        raise VideoInputError("The uploaded video is empty.", "unreadable_video")
    if len(content) > MAX_BYTES:
        raise VideoInputError("Video exceeds the 200 MiB upload limit.", "upload_too_large")
    return name


@contextmanager
def temporary_video(name, content):
    name = checked_name(name, content)
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(prefix="clip_", suffix=Path(name).suffix.lower(), dir=TEMP_ROOT, delete=False)
    path = Path(handle.name)
    try:
        handle.write(content); handle.close()
        yield path
    finally:
        handle.close()
        # Windows-safe, single-file cleanup, restricted to the file we created.
        if path.resolve().parent != TEMP_ROOT.resolve() or not path.name.startswith("clip_"):
            raise RuntimeError("Upload cleanup path escaped its owned temporary directory")
        path.unlink(missing_ok=True)


def fingerprint(name, content):
    return hashlib.sha256(name.encode("utf-8") + b"\0" + content).hexdigest()


def probe(name, content):
    with temporary_video(name, content) as path:
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise VideoInputError("This file could not be opened. It may be damaged or use an unsupported codec.", "unreadable_video")
            frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)); fps = float(capture.get(cv2.CAP_PROP_FPS))
            width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if frames < 1 or fps <= 0 or width < 1 or height < 1:
                raise VideoInputError("The video has invalid frame or duration information.", "unreadable_video")
            return {"frames": frames, "fps": fps, "width": width, "height": height, "duration": frames/fps, "bytes": len(content)}
        finally:
            capture.release()
