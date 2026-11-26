"""Session state transitions independent from widgets, testable with a dict."""
from animalact2.dashboard.schema import validate_prediction
from .uploads import checked_name, fingerprint, probe, temporary_video


def clear(state):
    for key in ("upload", "recognition", "recognition_error", "running"):
        state.pop(key, None)
    state["uploader_revision"] = state.get("uploader_revision", 0)+1


def accept_upload(state, filename, content):
    key = fingerprint(filename, content)
    if state.get("upload", {}).get("fingerprint") == key:
        return False
    state.pop("recognition", None); state.pop("recognition_error", None)
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
