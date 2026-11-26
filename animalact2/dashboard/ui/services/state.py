"""Session state transitions independent from widgets, testable with a dict."""
from animalact2.dashboard.schema import validate_prediction
from .uploads import checked_name, fingerprint, probe, temporary_video


def clear(state):
    for key in ("upload", "recognition", "recognition_error", "running"):
        state.pop(key, None)
    invalidate_grounding(state)
    state["grounding_running"] = False
    state["uploader_revision"] = state.get("uploader_revision", 0)+1


def accept_upload(state, filename, content):
    key = fingerprint(filename, content)
    if state.get("upload", {}).get("fingerprint") == key:
        return False
    state.pop("recognition", None); state.pop("recognition_error", None)
    invalidate_grounding(state)
    state["grounding_running"] = False
    state["running"] = False
    for field in ("window_start_input","window_end_input"):
        state.pop(field,None)
    state["upload"] = {"filename": filename, "content": content, "fingerprint": key, "metadata": None}
    try:
        state["upload"]["filename"] = checked_name(filename, content)
        state["upload"]["metadata"] = probe(filename, content)
    except Exception as exc:
        state["recognition_error"] = str(exc)
    return True


def recognize(state, runtime, device):
    upload = state.get("upload")
    if not upload or not upload.get("metadata"):
        raise ValueError("Choose a readable video before running recognition.")
    identity=recognition_identity(state)
    if state.get("recognition", {}).get("identity") == identity:
        return state["recognition"]
    if state.get("running"):
        raise RuntimeError("Recognition is already in progress.")
    state.pop("recognition", None); state.pop("recognition_error", None)
    state["running"] = True
    try:
        with temporary_video(upload["filename"], upload["content"]) as path:
            result = runtime.predict(path, device=device, top_k=119, model=state.get("model_key","attention"),start=state.get("window_start",0.),end=state.get("window_end"))
        validate_prediction(result)
        envelope = {"filename": upload["filename"], "fingerprint": upload["fingerprint"], "result": result, "identity": identity}
        state["recognition"] = envelope
        return envelope
    except Exception as exc:
        state.pop("recognition", None)
        state["recognition_error"] = str(exc)
        raise
    finally:
        state["running"] = False


def recognition_identity(state):
    from deployment.final_artifacts import manifest
    model_release = manifest()["archive"]["SHA256"]
    return (state.get("upload", {}).get("fingerprint"), state.get("model_key", "attention"),
            state.get("window_start", 0.), state.get("window_end"), "automatic-actions-v1", model_release)


def invalidate_grounding(state):
    for key in ("grounding_result", "grounding_error"):
        state.pop(key, None)


def grounding_identity(state):
    from animalact2.dashboard.grounding import model_info
    info = model_info()
    return (state.get("upload", {}).get("fingerprint"),
            " ".join(str(state.get("grounding_query", "")).split()).casefold(),
            info["checkpoint_SHA256"], info["classification_SHA256"])


def ground(state, runtime):
    from animalact2.dashboard.grounding import prepare_query, validate_result
    upload = state.get("upload")
    if not upload or not upload.get("metadata"):
        raise ValueError("Choose a readable video before running query localization.")
    identity = grounding_identity(state)
    if state.get("grounding_result", {}).get("identity") == identity:
        return state["grounding_result"]
    if state.get("grounding_running"):
        raise RuntimeError("Query localization is already in progress.")
    invalidate_grounding(state)
    state["grounding_running"] = True
    try:
        query, _, _ = prepare_query(state.get("grounding_query", ""))
        with temporary_video(upload["filename"], upload["content"]) as path:
            result = validate_result(runtime.ground(path, query, device="cpu"))
        if grounding_identity(state) != identity:
            raise RuntimeError("The video or query changed. Run query localization again.")
        envelope = dict(filename=upload["filename"], fingerprint=upload["fingerprint"], result=result, identity=identity)
        state["grounding_result"] = envelope
        return envelope
    except Exception as exc:
        state["grounding_error"] = str(exc)
        raise
    finally:
        state["grounding_running"] = False
