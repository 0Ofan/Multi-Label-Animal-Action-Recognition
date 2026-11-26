"""View transforms and downloads; never changes prediction/threshold decisions."""
import csv
import io
import json


def score_rows(result, query="", scope="All 119 classes", order="Highest probability", limit="All 119", query_ids=None, relevance=None):
    rows = list(result["class_scores"])
    if query_ids is not None:
        keep = set(query_ids)
        rows = [r for r in rows if r["class_id"] in keep]
    if query: rows = [r for r in rows if query.casefold() in r["name"].casefold()]
    if scope == "Predicted only": rows = [r for r in rows if r["selected"]]
    elif scope == "Main 31 classes": rows = [r for r in rows if r["main_evaluation"]]
    elif scope == "Extension classes": rows = [r for r in rows if not r["main_evaluation"]]
    if order == "Search relevance":
        rank = relevance or {}
        rows.sort(key=lambda r: (rank.get(r["class_id"], 10**9), -r["score"], r["class_id"]))
    elif order == "Highest probability": rows.sort(key=lambda r: (-r["score"], r["class_id"]))
    elif order == "Lowest probability": rows.sort(key=lambda r: (r["score"], r["class_id"]))
    elif order == "Action name": rows.sort(key=lambda r: r["name"])
    else: rows.sort(key=lambda r: r["class_id"])
    return rows[:int(limit.split()[1])] if limit.startswith("Top ") else rows


def export_csv(envelope):
    result = envelope["result"]; stream = io.StringIO(newline="")
    fields = ["filename", "video_id", "model_id", "action_id", "action_name", "probability", "threshold", "predicted", "main_evaluation", "inference_time", "eligible_for_prediction", "above_threshold"]
    writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
    for row in result["class_scores"]:
        writer.writerow(dict(zip(fields, [envelope["filename"], result["video_id"], result["model_id"], row["class_id"], row["name"], row["score"], result["threshold"], row["selected"], row["main_evaluation"], result["inference_time"], row["eligible_for_prediction"], row["above_threshold"]])))
    return ("\ufeff"+stream.getvalue()).encode("utf-8")


def export_json(envelope):
    return json.dumps({"filename": envelope["filename"], "upload_sha256": envelope["fingerprint"], **envelope["result"]}, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")


def export_intervals_csv(envelope):
    stream = io.StringIO(newline="")
    fields = ["filename", "class_id", "action", "start_seconds", "end_seconds", "score"]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in envelope["result"].get("temporal_detection", {}).get("segments", []):
        writer.writerow(dict(zip(fields, [envelope["filename"], row["class_id"], row["action"], row["start"], row["end"], row["score"]])))
    return ("\ufeff" + stream.getvalue()).encode("utf-8")
