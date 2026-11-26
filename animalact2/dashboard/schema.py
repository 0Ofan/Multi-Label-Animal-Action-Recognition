"""Validate every output including the unsupported class emission policy."""
import math
import struct

def validate_prediction(r):
    rows=r["class_scores"]
    if len(rows)!=119 or [x["class_id"] for x in rows]!=list(range(119)) or len(r["logits"])!=119:
        raise ValueError("Expected 119 ordered classes/logits")
    t=struct.unpack("f",struct.pack("f",r["threshold"]))[0]
    unsupported = set(r.get("unsupported_class_ids", [23]))
    if unsupported != ({23} if r.get("evaluation_unit") != "whole_video" else set()):
        raise ValueError("Unexpected model class policy")
    for row in rows:
        if not math.isfinite(row["score"]) or not 0<=row["score"]<=1: raise ValueError("Invalid score")
        if row["above_threshold"]!=(row["score"]>=t) or row["selected"]!=(row["score"]>=t and row["class_id"] not in unsupported):
            raise ValueError("Threshold/unsupported policy mismatch")
    selected=[x["name"] for x in sorted(rows,key=lambda x:(-x["score"],x["class_id"])) if x["selected"]]
    if r["predicted_labels"]!=selected: raise ValueError("Multilabel selection mismatch")
    temporal = r.get("temporal_detection")
    if temporal is not None:
        if temporal.get("status") not in ("ready", "unavailable", "unsupported_window"):
            raise ValueError("Invalid temporal detection status")
        if temporal["status"] != "ready" and temporal.get("segments"):
            raise ValueError("Unavailable detector cannot emit segments")
        if temporal["status"] == "ready":
            duration = temporal["duration"]
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError("Invalid detection duration")
            for segment in temporal["segments"]:
                cid = segment["class_id"]
                if not isinstance(cid, int) or not 0 <= cid < 119 or segment["action"] != rows[cid]["name"]:
                    raise ValueError("Invalid detection vocabulary")
                start, end, score = segment["start"], segment["end"], segment["score"]
                if not all(math.isfinite(x) for x in (start, end, score)) or not 0 <= start < end <= duration or not 0 <= score <= 1:
                    raise ValueError("Invalid action time interval")
    return r
