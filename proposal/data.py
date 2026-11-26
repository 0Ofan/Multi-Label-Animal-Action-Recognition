"""Read and audit the fixed AnimalAct student dataset directly from its ZIP."""

import hashlib
import itertools
import json
import math
from collections import Counter


PREFIX = "AnimalAct/"
EXPECTED_SPLITS = {"train": 2075, "valid": 446, "test": 447}
MIN_OVERLAP = 0.25


def load_dataset(archive):
    def read_json(path):
        return json.loads(archive.read(PREFIX + path))

    manifest_bytes = archive.read(PREFIX + "manifest.json")
    manifest = json.loads(manifest_bytes)
    vocabulary = read_json("action_classes.json")
    primary = read_json("main_evaluation_classes.json")
    entries = sorted(vocabulary["classes"], key=lambda entry: entry["class_id"])
    classes = [entry["name"] for entry in entries]
    action_ids = {name: index for index, name in enumerate(classes)}
    main_ids = primary["class_ids"]
    main_set = set(main_ids)
    if (
        len(action_ids) != 119
        or [entry["class_id"] for entry in entries] != list(range(119))
        or len(main_set) != 31
        or [classes[index] for index in main_ids] != primary["class_names"]
        or len({doc["split_version"] for doc in (manifest, vocabulary, primary)}) != 1
    ):
        raise ValueError("Inconsistent vocabulary, main class list, or split version")

    audit = {split: Counter() for split in EXPECTED_SPLITS}
    supports = {split: Counter() for split in EXPECTED_SPLITS}
    rows = {split: [] for split in EXPECTED_SPLITS}
    id_set = set()
    group_splits = {}
    files = set(archive.namelist())
    for sample in manifest["samples"]:
        video_id, split = sample["video_id"], sample["split"]
        group = sample["split_group_id"]
        video = sample["video_path"]
        annotation = sample["annotation_path"]
        if (
            video_id in id_set
            or group_splits.setdefault(group, split) != split
            or video != f"videos/{split}/{video_id}.mp4"
            or annotation != f"annotations/{split}/{video_id}.json"
            or PREFIX + video not in files
        ):
            raise ValueError(f"Duplicate ID, shared group across splits, or invalid path: {video_id}")
        id_set.add(video_id)
        raw = archive.read(PREFIX + annotation)
        if hashlib.sha256(raw).hexdigest() != sample["annotation_sha256"]:
            raise ValueError(f"Annotation checksum mismatch: {video_id}")
        record = json.loads(raw)
        if record["video_id"] != video_id:
            raise ValueError(f"Annotation ID mismatch: {video_id}")

        intervals = [
            (action_ids[item["action"]], float(item["start"]), float(item["end"]))
            for animal in record["annotations"]
            for item in animal["actions"]
        ]
        tolerance = 1 / sample["fps"] + 1e-6
        if len(intervals) != sample["action_count"] or any(
            not (math.isfinite(start) and math.isfinite(end))
            or not (-1e-6 <= start < end <= sample["duration"] + tolerance)
            for _, start, end in intervals
        ):
            raise ValueError(f"Invalid interval count or boundary: {video_id}")

        labels = sorted({action for action, _, _ in intervals})
        primary_intervals = [item for item in intervals if item[0] in main_set]
        concurrent = set()
        for (first, start, end), (second, other_start, other_end) in itertools.combinations(
            primary_intervals, 2
        ):
            if first != second and min(end, other_end) - max(start, other_start) >= MIN_OVERLAP:
                concurrent.update((first, second))
        audit[split].update({
            "videos": 1,
            "intervals": len(intervals),
            "multi_label_videos": int(len(labels) >= 2),
            "concurrent_main_action_videos": int(bool(concurrent)),
        })
        supports[split].update(labels)
        rows[split].append({
            "id": video_id, "path": PREFIX + video, "labels": labels,
            "concurrent": sorted(concurrent),
        })

    declared_splits = {
        split: {sample["video_id"] for sample in read_json(f"splits/{split}.json")["samples"]}
        for split in rows
    }
    for split, samples in rows.items():
        if (
            len(samples) != EXPECTED_SPLITS[split]
            or {row["id"] for row in samples} != declared_splits[split]
            or any(supports[split][index] == 0 for index in main_ids)
        ):
            raise ValueError(f"Unexpected split membership or missing main class in {split}")
        samples.sort(key=lambda row: row["id"])
    report = {
        "dataset_revision": manifest["dataset_revision"],
        "split_version": manifest["split_version"],
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "total_videos": len(id_set),
        "total_intervals": sum(item["intervals"] for item in audit.values()),
        "classes": classes,
        "main_class_ids": main_ids,
        "min_concurrent_overlap_seconds": MIN_OVERLAP,
        "split_groups": len(group_splits),
        "splits": {
            split: {
                **counts,
                "positive_videos_by_class": [supports[split][i] for i in range(len(classes))],
            }
            for split, counts in audit.items()
        },
    }
    return rows, report
