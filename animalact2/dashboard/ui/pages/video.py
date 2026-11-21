import streamlit as st

from animalact2.dashboard.ui.components.ui import header, metrics, tags, probability_chart
from animalact2.dashboard.ui.services import resources
from animalact2.dashboard.ui.services.state import accept_upload, recognize, clear
from animalact2.dashboard.video import SUPPORTED as SUPPORTED_EXTENSIONS


def upload_changed(key):
    if st.session_state.get(key) is None:
        clear(st.session_state)


def render():
    model=st.selectbox("Classification head",["attention","bigru_mean"],key="model_key",format_func=lambda x: "BiGRU + Temporal Attention (primary)" if x=="attention" else "BiGRU Mean")
    info = resources.model_info(model)
    header("VIDEO INTELLIGENCE", "AnimalAct 2.0 Recognition", "Upload an animal video and let the AI model recognize multiple actions in the video.")
    metrics([("Current model", info["head"]), ("Backbone", "VideoMAE"), ("Action classes", "119"), ("Real input frames", "1–16")])
    st.write("")
    left, right = st.columns([1.65, 1], gap="large")
    with left, st.container(border=True):
        st.subheader("Upload your video")
        st.caption("Drop a file to begin · Up to 200 MiB · Your video stays local")
        upload_key = f"video_upload_{st.session_state.get('uploader_revision', 0)}"
        upload = st.file_uploader("Choose an animal video", type=sorted(e.lstrip(".") for e in SUPPORTED_EXTENSIONS),
                                  key=upload_key, on_change=upload_changed, args=(upload_key,))
        if upload is not None:
            accept_upload(st.session_state, upload.name, upload.getvalue())
        current = st.session_state.get("upload")
        if current:
            st.markdown(f"**{current['filename']}** · {len(current['content'])/1024**2:.2f} MiB")
            if current.get("metadata"):
                st.video(current["content"])
                m = current["metadata"]
                st.caption(f"{m['duration']:.2f} seconds · {m['width']} × {m['height']} · {m['fps']:.2f} FPS · {m['frames']} frames")
                if m["duration"] > 60:
                    st.info("The whole video is treated as ONE supplied clip. You may select a time window; it is not an automatically localized action interval.")
                if m["frames"] < 16:
                    st.caption("Short clip: all real frames are kept. No repeated frames and no padding to 16; an odd last frame gets one internal zero slot for tubelet alignment.")
            st.caption("MP4/H.264 is recommended for browser playback. Other supported formats may still be recognized if their codec is available.")
            st.button("Clear video", on_click=clear, args=(st.session_state,), width="stretch")
        else:
            st.markdown('<div style="padding:48px 20px;text-align:center;color:#8696ae"><div style="font-size:42px;color:#abc0e0">▷</div><b>Your video preview will appear here</b><br><small>Animals in motion. More than one action.</small></div>', unsafe_allow_html=True)
    with right:
        with st.container(border=True):
            st.subheader("Recognition settings")
            options = ["CUDA", "CPU"] if resources.cuda_available() else ["CPU"]
            device = st.selectbox("Processing device", options, key="recognition_device")
            st.caption("CUDA is available." if resources.cuda_available() else "CUDA is unavailable. Recognition will run on the CPU.")
            use_window=st.checkbox("Select a clip window",key="use_window")
            duration=st.session_state.get("upload",{}).get("metadata",{}).get("duration",1.) if st.session_state.get("upload",{}).get("metadata") else 1.
            if use_window:
                start=st.number_input("Window start (seconds)",min_value=0.,max_value=duration,value=0.,step=.1,key="window_start_input")
                end=st.number_input("Window end (seconds)",min_value=0.,max_value=duration,value=duration,step=.1,key="window_end_input")
            else: start,end=0.,None
            st.session_state["window_start"]=start; st.session_state["window_end"]=end
            identity=(st.session_state.get("upload",{}).get("fingerprint"),model,start,end)
            if st.session_state.get("recognition") and st.session_state["recognition"].get("identity") != identity:
                st.session_state.pop("recognition",None)
            st.markdown(f"**Decision threshold**　`{info['threshold']:.2f}`")
            st.caption("Selected on validation data and kept fixed. Top-ranked actions below this value are not selected predictions.")
            current = st.session_state.get("upload")
            done = bool(current and st.session_state.get("recognition", {}).get("fingerprint") == current["fingerprint"])
            disabled = not current or not current.get("metadata") or done or st.session_state.get("running", False)
            run = st.button("Recognition complete" if done else "Run Recognition", type="primary", disabled=disabled, width="stretch")
            if run:
                try:
                    with st.status("Recognizing actions…", expanded=True) as status:
                        st.write("Preparing the video and loading the frozen model if needed.")
                        st.write("Sequential real-frame decode → full-frame resize → aligned variable-length VideoMAE → selected BiGRU head.")
                        recognize(st.session_state, resources.get_runtime(), device.lower())
                        status.update(label="Recognition complete", state="complete", expanded=False)
                    st.rerun()
                except Exception as exc:
                    st.error(f"Recognition failed: {exc}")
            if done: st.success("Prediction saved for this video. Open Results for all 119 classes.")
        with st.container(border=True):
            st.subheader("One clip. Multiple actions.")
            st.write("A frozen VideoMAE encoder feeds the selected BiGRU head. Results describe the supplied clip, not automatically detected action intervals.")
            st.caption("No action onset/offset is inferred. The window bounds are input bounds only. Predictions can be empty or multi-label.")
            st.caption("119 raw class scores are retained. Defensive Rearing has no Train positive and cannot be emitted as a learned action. Scores are uncalibrated; errors remain possible.")
    if st.session_state.get("recognition_error"):
        st.error(st.session_state["recognition_error"])
    envelope = st.session_state.get("recognition")
    if envelope:
        r = envelope["result"]
        st.subheader("Recognition preview")
        metrics([("Predicted actions", str(len(r["predicted_labels"]))), ("Inference time", f"{r['inference_time']:.2f} s"), ("Threshold", f"{r['threshold']:.2f}")])
        if r["predicted_labels"]: tags(r["predicted_labels"])
        else: st.info("No action score reached the fixed threshold. Top-K remains available for inspection.")
        probability_chart(sorted(r["class_scores"], key=lambda row: -row["score"])[:10], r["threshold"], "video_preview_chart")
        def results():
            st.session_state["nav"] = "Results"; st.query_params["page"] = "results"
        st.button("View detailed results →", on_click=results, width="stretch")
