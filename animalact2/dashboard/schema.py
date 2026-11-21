"""Validate every output including the unsupported class emission policy."""
import math
import struct

def validate_prediction(r):
    rows=r["class_scores"]
    if len(rows)!=119 or [x["class_id"] for x in rows]!=list(range(119)) or len(r["logits"])!=119:
        raise ValueError("Expected 119 ordered classes/logits")
    t=struct.unpack("f",struct.pack("f",r["threshold"]))[0]
    for row in rows:
        if not math.isfinite(row["score"]) or not 0<=row["score"]<=1: raise ValueError("Invalid score")
        if row["above_threshold"]!=(row["score"]>=t) or row["selected"]!=(row["score"]>=t and row["class_id"]!=23):
            raise ValueError("Threshold/unsupported policy mismatch")
    selected=[x["name"] for x in sorted(rows,key=lambda x:(-x["score"],x["class_id"])) if x["selected"]]
    if r["predicted_labels"]!=selected: raise ValueError("Multilabel selection mismatch")
    return r
