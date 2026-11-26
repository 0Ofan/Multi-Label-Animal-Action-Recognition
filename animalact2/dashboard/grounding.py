"""Web adapter for the original trained template-query ranker, without retraining."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def model_info():
    from deployment.final_artifacts import manifest
    info = json.loads((ROOT / "deployment/grounding_metadata.json").read_text(encoding="utf-8"))
    expected = manifest()["files"][info["checkpoint_path"]]["SHA256"]
    if info["checkpoint_SHA256"] != expected:
        raise ValueError("Grounding metadata and released checkpoint differ")
    return info


def prepare_query(query):
    from proposal.predict import parse_query
    query = " ".join(str(query or "").split())
    if not query or len(query) > 300:
        raise ValueError("Enter an English query using A {animal} is {action}. (maximum 300 characters)")
    info = model_info()
    animal_id, action_id = parse_query(query, info["animals"], info["classes"])
    return query, animal_id, action_id


def validate_result(result):
    if result.get("task") != "template_temporal_grounding":
        raise ValueError("Expected a query-based grounding result")
    duration = result["duration"]
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid grounding video duration")
    intervals = result["intervals"]
    if not 1 <= len(intervals) <= 3:
        raise ValueError("Expected a top candidate and at most two alternatives")
    for rank, row in enumerate(intervals, 1):
        start, end, score = row["start"], row["end"], row["match_score"]
        # Original float32 candidate endpoints may differ slightly from FPS-derived duration.
        if row["rank"] != rank or not all(math.isfinite(v) for v in (start, end, score)):
            raise ValueError("Invalid grounding candidate")
        if not 0 <= start < end <= duration + 1e-5 or not 0 <= score <= 1:
            raise ValueError("Invalid grounding candidate bounds or score")
    if any(intervals[i]["match_score"] < intervals[i + 1]["match_score"] for i in range(len(intervals) - 1)):
        raise ValueError("Grounding candidates are not ranked")
    return result


def predict(predictor, path, query):
    query, animal_id, action_id = prepare_query(query)
    info = model_info()
    # Ensure the UI vocabulary is taken from the same bytes the original parser uses.
    checkpoint = ROOT / info["checkpoint_path"]
    with checkpoint.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != info["checkpoint_SHA256"]:
            raise ValueError("Grounding checkpoint checksum mismatch")
    raw = predictor.predict(path, kind=info["classification_kind"], query=query)
    original = raw["grounding"]
    if original["unknown_animal"] != (animal_id == 0):
        raise ValueError("Grounding query vocabulary mismatch")
    result = dict(
        task="template_temporal_grounding", schema_version=1, query=query,
        model_id="resnet_bigru_attention_query_ranker", classification_kind=info["classification_kind"],
        checkpoint_SHA256=info["checkpoint_SHA256"], classification_SHA256=info["classification_SHA256"],
        best_epoch=info["best_epoch"], device=predictor.device, action_id=action_id,
        action=info["classes"][action_id], animal_id=animal_id, unknown_animal=animal_id == 0,
        duration=original["time_resolution_seconds"] * info["frames"],
        time_resolution_seconds=original["time_resolution_seconds"], input_frames=info["frames"],
        inference_time=raw["inference_time"], experimental=True, calibrated=False,
        score_meaning=original["score_meaning"], absence_detection_supported=False,
        alternatives_are_separate_events=False,
        intervals=[dict(rank=i + 1, start=row["interval"][0], end=row["interval"][1], match_score=row["score"])
                   for i, row in enumerate(original["intervals"])],
    )
    return validate_result(result)
