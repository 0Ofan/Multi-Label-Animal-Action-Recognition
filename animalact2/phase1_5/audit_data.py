"""All Train/Valid clips: independent labels/indices and one dense decode per parent."""
from __future__ import annotations
import csv
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
import hashlib
import json
import math
import time
import traceback
import cv2
import numpy as np
import torch
from animalact2.phase1_5_1.dataset import ReleaseClipDataset, RELEASE_ID
from animalact2.phase1_5_1.transforms import FullFrameTransform
from .safety import ROOT, OUT, enable_guard, sha, write_json


def frac(v):
    return Fraction(str(v))


def points(s, e, n, fps):
    return set(range(max(0, math.ceil(s * fps)), min(n, math.ceil(e * fps))))


def save_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit_parent(clips, names):
    first = clips[0]
    vid, split = first["parent_video_id"], first["split"]
    n, fps = first["parent_num_frames"], frac(first["fps"])
    duration = Fraction(n, 1) / fps
    vpath, apath = ROOT / first["video_path"], ROOT / first["annotation_path"]
    assert vpath.is_file() and apath.is_file()
    assert sha(vpath) == first["video_sha256"] and sha(apath) == first["annotation_sha256"]
    raw = json.loads(apath.read_text(encoding="utf-8"))
    assert raw["video_id"] == vid
    sources = []
    for ai, animal in enumerate(raw["annotations"]):
        for ii, action in enumerate(animal["actions"]):
            pointer = f"/annotations/{ai}/actions/{ii}"
            uid = "ann_" + hashlib.sha256(json.dumps([first["annotation_sha256"], pointer], sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()[:20]
            s, e = frac(action["start"]), frac(action["end"])
            es, ee = (s, min(e, duration)) if s < e and s < duration else (s, e)
            sources.append(dict(uid=uid, cid=names[action["action"]], s=es, e=ee, os=s, oe=e,
                                ref=first["annotation_path"] + "#" + pointer, animal=animal["animal"]))
    by_uid = {a["uid"]: a for a in sources}
    rows, needed = [], set()
    for c in clips:
        assert c["split"] == split and c["parent_video_id"] == vid
        s, e = frac(c["effective_start_exact"]), frac(c["effective_end_exact"])
        assert 0 <= s < e <= duration
        available = sorted(points(s, e, n, fps))
        selected = available if len(available) <= 16 else [available[round(Fraction(j * (len(available)-1), 15))] for j in range(16)]
        assert selected == c["frame_indices"] and len(available) == c["available_strict_frame_count"]
        assert c["effective_length"] == len(selected) and c["valid_frame_mask"] == [True] * len(selected)
        label_support, label_sources, overlaps = defaultdict(set), defaultdict(set), {}
        for a in sources:
            lo, hi = max(s, a["s"]), min(e, a["e"])
            if lo < hi:
                label_support[a["cid"]].update(points(lo, hi, n, fps))
                label_sources[a["cid"]].add(a["uid"])
                overlaps[a["uid"]] = lo, hi
        ids = sorted(label_sources)
        assert c["class_ids"] == ids and c["target_119"] == [int(i in ids) for i in range(119)]
        assert set(c["all_positive_label_source_uids"]) == set(overlaps)
        warnings = []
        for ev in c["label_evidence"]:
            cid = ev["class_id"]
            support = label_support[cid]
            assert support and ev["dense_support_count"] == len(support)
            hits = [k for k in selected if k in support]
            assert ev["sampled_support_indices"] == hits and ev["sampled_support_count"] == len(hits)
            assert {a["annotation_uid"] for a in ev["contributors"]} == label_sources[cid]
            for a in ev["contributors"]:
                uid = a["annotation_uid"]
                assert a["annotation_ref"] == by_uid[uid]["ref"]
                assert (frac(a["overlap_start_exact"]), frac(a["overlap_end_exact"])) == overlaps[uid]
            if not hits:
                warnings.append(cid)
        assert warnings == c["sampled_label_coverage_warnings"]
        for window in c["original_windows"]:
            os_, oe = frac(window["original_start_exact"]), frac(window["original_end_exact"])
            assert os_ < oe and os_ < duration and (os_, min(oe, duration)) == (s, e)
            for uid in window["source_annotation_uids"]:
                assert (by_uid[uid]["os"], by_uid[uid]["oe"]) == (os_, oe)
        needed.update(selected)
        rows.append(dict(clip_id=c["clip_id"], parent_video_id=vid, split=split, status="passed", effective_length=len(selected),
                         frame_indices=json.dumps(selected, separators=(",", ":")), class_ids=json.dumps(ids, separators=(",", ":")),
                         target_SHA256=hashlib.sha256(bytes(c["target_119"])).hexdigest(), labels_verified=True, provenance_verified=True,
                         sampled_coverage_warnings=json.dumps(warnings), decoded_frames_verified=True))
    capture = cv2.VideoCapture(str(vpath), cv2.CAP_FFMPEG, [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE])
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == n
        assert abs(capture.get(cv2.CAP_PROP_FPS) - float(fps)) < 1e-6
        transform, checked, max_error = FullFrameTransform(), set(), 0.
        pixel_hash = hashlib.sha256()
        for k in range(n):
            assert capture.grab(), f"decode missing frame {k}"
            if k in needed:
                ok, bgr = capture.retrieve()
                assert ok and bgr is not None and bgr.shape == (360, 640, 3)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                tensor = transform(rgb)
                assert tensor.shape == (3, 224, 398) and torch.isfinite(tensor).all()
                observed = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
                err = abs(observed - k / float(fps))
                assert np.isfinite(observed) and err < 1e-6
                max_error = max(max_error, err)
                pixel_hash.update(k.to_bytes(8, "little"))
                pixel_hash.update(rgb.tobytes())
                checked.add(k)
        assert not capture.grab(), "Actual video longer than manifest"
        assert checked == needed
    finally:
        capture.release()
    return dict(parent_video_id=vid, split=split, status="passed", clips=len(clips), parent_frames_decoded=n,
                unique_selected_frames_checked=len(needed), maximum_timestamp_error_seconds=max_error,
                selected_RGB_stream_SHA256=pixel_hash.hexdigest(), video_SHA_verified=True, annotation_SHA_verified=True), rows


def main():
    started = time.perf_counter()
    enable_guard()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    destination = OUT / "data_audit"
    if (destination / "summary.json").exists():
        raise FileExistsError("Preserve completed full audit")
    datasets = {s: ReleaseClipDataset(s) for s in ("train", "valid")}
    assert {s: len(d) for s, d in datasets.items()} == dict(train=4667, valid=979)
    groups = defaultdict(list)
    for d in datasets.values():
        for c in d.records:
            groups[(c["split"], c["parent_video_id"])].append(c)
    names = {name: i for i, name in enumerate(datasets["train"].class_names)}
    parents, clip_rows, errors = [], [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(audit_parent, cc, names): key for key, cc in groups.items()}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                parent, rows = future.result()
                parents.append(parent)
                clip_rows.extend(rows)
            except Exception as exc:
                split, vid = futures[future]
                errors.append(dict(split=split, parent_video_id=vid, error=str(exc), traceback=traceback.format_exc()))
            if i % 100 == 0 or i == len(futures):
                print(json.dumps(dict(parents_checked=i, total=len(futures), errors=len(errors), elapsed_seconds=round(time.perf_counter()-started, 2))), flush=True)
    parents.sort(key=lambda x: (x["split"], x["parent_video_id"]))
    clip_rows.sort(key=lambda x: (x["split"], x["clip_id"]))
    save_csv(destination / "parent_video_checks.csv", parents)
    save_csv(destination / "clip_checks.csv", clip_rows)
    write_json(destination / "errors.json", errors)
    main_ids = json.loads((ROOT / "animalact2/release_v1/main_evaluation_classes.json").read_text(encoding="utf-8"))["class_ids"]
    class_rows = []
    for name, cid in names.items():
        row = dict(class_id=cid, action=name, main31=cid in main_ids)
        for s, d in datasets.items():
            cc = [c for c in d.records if cid in c["class_ids"]]
            row[s+"_positive_clips"] = len(cc)
            row[s+"_positive_parents"] = len({c["parent_video_id"] for c in cc})
        row["learnable_from_Train"] = row["train_positive_clips"] > 0
        class_rows.append(row)
    save_csv(destination / "class_support_119.csv", class_rows)
    save_csv(destination / "main31_support.csv", [r for r in class_rows if r["main31"]])
    result = dict(status="passed" if not errors and len(clip_rows) == 5646 else "failed", release_id=RELEASE_ID,
                  clips_checked=len(clip_rows), expected_clips=5646, split_clips={s: len(d) for s,d in datasets.items()},
                  parent_videos_checked=len(parents), expected_parent_videos=2505, failures=len(errors),
                  source_frames_dense_decoded=sum(p["parent_frames_decoded"] for p in parents),
                  unique_selected_frames_RGB_resize_checked=sum(p["unique_selected_frames_checked"] for p in parents),
                  maximum_timestamp_error_seconds=max(p["maximum_timestamp_error_seconds"] for p in parents),
                  unsupported_Train_class_ids=[r["class_id"] for r in class_rows if not r["learnable_from_Train"]],
                  all_main31_have_Train_Valid_positives=all(r["train_positive_clips"] and r["valid_positive_clips"] for r in class_rows if r["main31"]),
                  sampling_warning_pairs={s: sum(len(c["sampled_label_coverage_warnings"]) for c in d.records) for s,d in datasets.items()},
                  elapsed_seconds=time.perf_counter()-started, workers=4, source_videos_modified=False, RGB_or_feature_cache_written=False, Test_opened=False)
    write_json(destination / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if result["status"] != "passed":
        raise RuntimeError("Full data validation failed; keep errors and do not launch training")


if __name__ == "__main__":
    main()
