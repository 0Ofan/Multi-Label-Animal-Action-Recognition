"""Load immutable release-v1 records and their real frames; no label reconstruction."""
from __future__ import annotations
import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from .transforms import FullFrameTransform, PREPROCESSING_ID, MEAN, STD
from .video_decoder import decode_exact_frames

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RELEASE = PROJECT_ROOT / "animalact2/release_v1"
RELEASE_ID = "animalact2-2.0.0-release-v1"


def _sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class ReleaseClipDataset(Dataset):
    """Train/Valid only in Phase1.5.1. Frames are decoded lazily in __getitem__."""
    def __init__(self, split="train", *, release_root=DEFAULT_RELEASE, project_root=PROJECT_ROOT, clip_ids=None):
        if split not in ("train", "valid"):
            raise ValueError("Phase1.5.1 only enables Train/Valid; Test is not a model-validation source")
        self.release_root, self.project_root = Path(release_root).resolve(), Path(project_root).resolve()
        self.split = split
        # Verify only the development files actually used; never open the Test manifest.
        checksums = dict(line.split("  ", 1)[::-1] for line in (self.release_root / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines())
        names = ("version.json", "development_index.json", "class_mapping.json", "preprocessing.json", f"{split}/manifest.jsonl")
        for name in names:
            if _sha(self.release_root / name) != checksums[name]:
                raise ValueError(f"Release integrity failure: {name}")
        version = json.loads((self.release_root / "version.json").read_text(encoding="utf-8"))
        if version["release_id"] != RELEASE_ID or version["status"] != "released_local":
            raise ValueError("Requires sealed release-v1")
        config = json.loads((self.release_root / "preprocessing.json").read_text(encoding="utf-8"))
        if (config["spatial_preprocessing_id"] != PREPROCESSING_ID or config["resize_short_edge"] != 224
                or config["center_crop"] or config["letterbox"] or not config["preserve_aspect_ratio"]
                or config["normalization_mean"] != list(MEAN) or config["normalization_std"] != list(STD)):
            raise ValueError("Unsupported preprocessing; do not silently alter release rules")
        self.class_names = tuple(c["name"] for c in json.loads((self.release_root / "class_mapping.json").read_text(encoding="utf-8"))["classes"])
        with (self.release_root / split / "manifest.jsonl").open(encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        if clip_ids is not None:
            wanted = list(clip_ids)
            if len(wanted) != len(set(wanted)):
                raise ValueError("Duplicate requested Clip ID")
            by_id = {r["clip_id"]: r for r in records}
            if not set(wanted) <= set(by_id):
                raise KeyError("Unknown Clip ID or Clip belongs to another split")
            records = [by_id[cid] for cid in wanted]
        for record in records:
            self._validate(record)
        self.records = records
        self.transform = FullFrameTransform()
        self.manifest_sha256 = checksums[f"{split}/manifest.jsonl"]

    def _validate(self, r):
        if r["release_id"] != RELEASE_ID or r["split"] != self.split or r["mapping_status"] != "mapped" or r["usable"] is not True:
            raise ValueError("Only mapped records of the requested fixed split are allowed")
        idx, length = r["frame_indices"], r["effective_length"]
        if not (1 <= length <= 16 and length == len(idx) and idx == sorted(set(idx)) and all(type(k) is int for k in idx)):
            raise ValueError("Invalid real frame sequence")
        if r["valid_frame_mask"] != [True] * length:
            raise ValueError("Manifest must contain real frames only")
        s, e, fps = (Fraction(str(r[k])) for k in ("effective_start_exact", "effective_end_exact", "fps"))
        if not all(0 <= k < r["parent_num_frames"] and s <= Fraction(k, 1) / fps < e for k in idx):
            raise ValueError("Frame outside approved effective half-open interval")
        classes = r["class_ids"]
        if (len(r["target_119"]) != 119 or r["target_119"] != [int(i in classes) for i in range(119)]
                or classes != sorted(set(classes)) or not classes or min(classes) < 0 or max(classes) >= 119):
            raise ValueError("Invalid release label mapping")
        if {x["class_id"] for x in r["label_evidence"]} != set(classes) or not all(x["contributors"] and x["dense_support_count"] > 0 for x in r["label_evidence"]):
            raise ValueError("Missing original label provenance or dense support")
        if r["spatial_preprocessing_id"] != PREPROCESSING_ID:
            raise ValueError("Different preprocessing policy")
        expected = self.project_root / "AnimalAct/videos" / self.split / (r["parent_video_id"] + ".mp4")
        if (self.project_root / r["video_path"]).resolve() != expected.resolve():
            raise ValueError("Video path does not match official parent/split")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        decoded = decode_exact_frames(self.project_root / record["video_path"], record["frame_indices"],
                                      expected_num_frames=record["parent_num_frames"], expected_fps=record["fps"])
        frames = torch.stack([self.transform(rgb) for rgb in decoded.rgb])
        if tuple(frames.shape[1:]) != tuple(record["spatial_output_shape"]) or not torch.isfinite(frames).all():
            raise ValueError("Input shape/numerical values differ from published contract")
        return dict(frames=frames, target=torch.tensor(record["target_119"], dtype=torch.float32),
                    frame_indices=torch.tensor(decoded.frame_indices, dtype=torch.int64),
                    effective_length=record["effective_length"], valid_frame_mask=torch.ones(len(frames), dtype=torch.bool),
                    clip_id=record["clip_id"], parent_video_id=record["parent_video_id"], split=self.split,
                    metadata=copy.deepcopy(record), label_sources=copy.deepcopy(record["label_evidence"]),
                    decode_info=dict(fps=decoded.fps, parent_num_frames=decoded.parent_num_frames, width=decoded.width,
                                     height=decoded.height, backend=decoded.backend, timestamps_seconds=list(decoded.timestamps_seconds)))
