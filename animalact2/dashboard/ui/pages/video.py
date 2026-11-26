from animalact2.dashboard.ui.i18n import t, display_format, display_data, option_formatter
import streamlit as st

from animalact2.dashboard.ui.components.ui import header, metrics, tags, probability_chart
from animalact2.dashboard.ui.services import resources
from animalact2.dashboard.ui.services.state import accept_upload, recognize, clear, recognition_identity
from animalact2.dashboard.video import SUPPORTED as SUPPORTED_EXTENSIONS


def upload_changed(key):
    if st.session_state.get(key) is None:
        clear(st.session_state)


def render():
    model=st.selectbox(t("Recognition method"), resources.available_models(), key="model_key", format_func=display_format(resources.model_label))
    whole_video = model in resources.RESNET_MODELS or model == "resnet_ensemble"
    info = resources.model_info(model)
    header("VIDEO INTELLIGENCE", "AnimalAct 2.0 Recognition", "Upload an animal video and let the AI model recognize multiple actions in the video.")
    st.caption(t("Method progression: ResNet-18 → mean pooling → BiGRU → temporal attention; VideoMAE extends the encoder."))
    metrics([("Current model", info["head"]), ("Backbone", info.get("backbone", "VideoMAE")), ("Action classes", "119"), ("Real input frames", "16" if whole_video else "1–16")])
    if model == "resnet_ensemble":
        st.info(t("Weighted scores: 20% BiGRU Mean + 80% Attention. Weights and threshold selected on Valid; time detection remains a separate original model."))
    st.write(t(""))
    left, right = st.columns([1.65, 1], gap="large")
    with left, st.container(border=True):
        st.subheader(t("Upload your video"))
        st.caption(t("Drop a file to begin · Up to 200 MiB · Processed on this app server"))
        upload_key = f"video_upload_{st.session_state.get('uploader_revision', 0)}"
        upload = st.file_uploader(t("Choose an animal video"), type=sorted(e.lstrip(".") for e in SUPPORTED_EXTENSIONS),
                                  key=upload_key, on_change=upload_changed, args=(upload_key,))
        if upload is not None:
            accept_upload(st.session_state, upload.name, upload.getvalue())
        current = st.session_state.get("upload")
        if current:
            st.markdown(t(f"**{current['filename']}** · {len(current['content'])/1024**2:.2f} MiB"))
            if current.get("metadata"):
                st.video(current["content"])
                m = current["metadata"]
                st.caption(t(f"{m['duration']:.2f} seconds · {m['width']} × {m['height']} · {m['fps']:.2f} FPS · {m['frames']} frames"))
                if not whole_video and m["duration"] > 60:
                    st.info(t("Classification summarizes the supplied video; automatic time detection is evaluated separately."))
                if not whole_video and m["frames"] < 16:
                    st.caption(t("Short clip: all real frames are kept. No repeated frames and no padding to 16; an odd last frame gets one internal zero slot for tubelet alignment."))
            st.caption(t("MP4/H.264 is recommended for browser playback. Other supported formats may still be recognized if their codec is available."))
            st.button(t("Clear video"), on_click=clear, args=(st.session_state,), width="stretch", key='ui_video_41')
        else:
            st.markdown(t('<div style="padding:48px 20px;text-align:center;color:#8696ae"><div style="font-size:42px;color:#abc0e0">▷</div><b>Your video preview will appear here</b><br><small>Animals in motion. More than one action.</small></div>'), unsafe_allow_html=True)
    with right:
        with st.container(border=True):
            st.subheader(t("Recognition settings"))
            options = ["CUDA", "CPU"] if resources.cuda_available() and model != "resnet_ensemble" else ["CPU"]
            device = st.selectbox(t("Processing device"), options, key="recognition_device", format_func=option_formatter())
            st.caption(t("CUDA is available." if resources.cuda_available() else "CUDA is unavailable. Recognition will run on the CPU."))
            st.caption(t("Actions and time intervals are detected automatically; no query is required."))
            if resources.detection_checkpoint() is None:
                st.info(t("Automatic time detection is not ready yet. Classification remains available; no timestamps will be invented."))
            st.session_state["window_start"] = 0.
            st.session_state["window_end"] = None
            identity=recognition_identity(st.session_state)
            if st.session_state.get("recognition") and st.session_state["recognition"].get("identity") != identity:
                st.session_state.pop("recognition",None)
            st.markdown(t(f"**Decision threshold**　`{info['threshold']:.2f}`"))
            st.caption(t("Selected on validation data and kept fixed. Top-ranked actions below this value are not selected predictions."))
            current = st.session_state.get("upload")
            done = bool(current and st.session_state.get("recognition", {}).get("fingerprint") == current["fingerprint"])
            disabled = not current or not current.get("metadata") or done or st.session_state.get("running", False)
            run = st.button(t("Recognition complete" if done else "Run Recognition"), type="primary", disabled=disabled, width="stretch", key='ui_video_65')
            if run:
                try:
                    with st.status(t("Recognizing actions…"), expanded=True) as status:
                        st.write(t("Preparing the video and loading the frozen model if needed."))
                        st.write(t("Whole-video ResNet-18 uses 16 uniform samples and trains all 119 classes. Short videos may reuse frames.") if whole_video else t("Sequential real-frame decode → full-frame resize → aligned variable-length VideoMAE → selected BiGRU head."))
                        recognize(st.session_state, resources.get_runtime(), device.lower())
                        status.update(label=t("Recognition complete"), state="complete", expanded=False)
                    st.rerun()
                except Exception as exc:
                    st.error(t(f"Recognition failed: {exc}"))
            if done: st.success(t("Prediction saved for this video. Open Results for all 119 classes."))
        with st.container(border=True):
            st.subheader(t("One clip. Multiple actions."))
            st.write(t("Method progression: ResNet-18 → mean pooling → BiGRU → temporal attention; VideoMAE extends the encoder."))
            st.caption(t("Classification summarizes the supplied video; automatic time detection is evaluated separately."))
            if not whole_video: st.caption(t("119 raw class scores are retained. Defensive Rearing has no Train positive and cannot be emitted as a learned action. Scores are uncalibrated; errors remain possible."))
    if st.session_state.get("recognition_error"):
        st.error(t(st.session_state["recognition_error"]))
    envelope = st.session_state.get("recognition")
    if envelope:
        r = envelope["result"]
        st.subheader(t("Recognition preview"))
        metrics([("Predicted actions", str(len(r["predicted_labels"]))), ("Inference time", f"{r['inference_time']:.2f} s"), ("Threshold", f"{r['threshold']:.2f}")])
        if r["predicted_labels"]: tags(r["predicted_labels"])
        else: st.info(t("No action score reached the fixed threshold. Top-K remains available for inspection."))
        probability_chart(sorted(r["class_scores"], key=lambda row: -row["score"])[:10], r["threshold"], "video_preview_chart")
        def results():
            st.session_state["nav"] = "Results"; st.query_params["page"] = "results"
        st.button(t("View detailed results →"), on_click=results, width="stretch", key='ui_video_94')
