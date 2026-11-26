"""Independent Bonus query controls and results, using the existing query ranker."""
import plotly.graph_objects as go
import streamlit as st

from animalact2.dashboard.grounding import model_info, prepare_query
from animalact2.dashboard.ui.i18n import t
from animalact2.dashboard.ui.services import resources
from animalact2.dashboard.ui.services.state import ground, grounding_identity, invalidate_grounding
from animalact2.dashboard.ui.services.presentation import export_grounding_csv, export_grounding_json
from .ui import metrics, table


def _changed():
    invalidate_grounding(st.session_state)


def _edited():
    st.session_state["grounding_query"] = st.session_state["grounding_query_input"]
    _changed()


def _use_template(animal, action):
    st.session_state["grounding_query"] = f"A {animal.lower()} is {action.lower()}."
    st.session_state["grounding_query_input"] = st.session_state["grounding_query"]
    _changed()


def render_input():
    info = model_info()
    # Keep the durable query separate from Streamlit's page-owned widget key.
    if "grounding_query_input" not in st.session_state:
        st.session_state["grounding_query_input"] = st.session_state.get("grounding_query", "")
    if st.session_state.get("grounding_result", {}).get("identity") not in (None, grounding_identity(st.session_state)):
        _changed()
    with st.container(border=True):
        st.subheader(t("Query-based temporal grounding (Bonus)"))
        st.write(t("Ask when a specified animal behavior occurs in the uploaded video."))
        st.caption(t("Independent query model: ResNet-18 + BiGRU window ranker, epoch 3, 16 frames. The classification model selector does not change this model."))
        with st.expander(t("Build an English template query")):
            left, right = st.columns(2)
            animal = left.selectbox(t("Animal name from the training vocabulary"), info["animals"],
                index=info["animals"].index("Goat"), key="grounding_builder_animal")
            action = right.selectbox(t("Target action"), info["classes"],
                index=info["classes"].index("Walking"), key="grounding_builder_action")
            st.button(t("Use this query"), on_click=_use_template, args=(animal, action), key="grounding_use_template")
        query = st.text_input(t("Behavior query (English template)"), placeholder="A goat is walking.",
            key="grounding_query_input", max_chars=300, on_change=_edited)
        st.caption(t("Format: A {animal} is {action}. Use an exact action name; the builder supplies supported names."))
        valid = False
        if query.strip():
            try:
                _, animal_id, _ = prepare_query(query)
                valid = True
                if animal_id == 0:
                    st.warning(t("This animal name is outside the training vocabulary. The original model uses its unknown-animal embedding; localization may be less reliable."))
            except ValueError as exc:
                st.warning(t(str(exc)))
        upload = st.session_state.get("upload", {})
        if not upload.get("metadata"):
            st.info(t("Upload a readable video above to run query localization. Classification is optional."))
        done = st.session_state.get("grounding_result", {}).get("identity") == grounding_identity(st.session_state)
        disabled = not valid or not upload.get("metadata") or done or st.session_state.get("grounding_running", False) or st.session_state.get("running", False)
        if st.button(t("Query localization complete" if done else "Run query localization"), key="run_grounding",
                     disabled=disabled, type="primary"):
            try:
                with st.status(t("Localizing the query on CPU…"), expanded=True) as status:
                    ground(st.session_state, resources.get_runtime())
                    status.update(label=t("Query localization complete"), state="complete", expanded=False)
                st.rerun()
            except Exception:
                pass  # The original error is retained below; classification is untouched.
        if st.session_state.get("grounding_error"):
            st.error(t("Query localization failed: ") + st.session_state["grounding_error"])
    render_result("video")


def render_result(location="results"):
    envelope = st.session_state.get("grounding_result")
    if not envelope:
        return False
    if envelope.get("identity") != grounding_identity(st.session_state):
        _changed()
        return False
    result = envelope["result"]
    with st.container(border=True):
        st.subheader(t("Query localization result"))
        st.write(result["query"])
        st.caption(envelope["filename"])
        top = result["intervals"][0]
        metrics([(t("Top interval (seconds)"), f"{top['start']:.3f} – {top['end']:.3f}"),
                 (t("Match score"), f"{top['match_score']:.4f}"),
                 (t("Inference time"), f"{result['inference_time']:.2f} s")])
        st.warning(t("Experimental query localization: long-window bias remains. The model always ranks candidates and cannot confirm that the queried behavior is absent."))
        st.caption(t("The first interval is the top prediction. Other intervals are alternatives, not separate detected events. Scores are uncalibrated matching scores."))
        if result["unknown_animal"]:
            st.warning(t("This animal name is outside the training vocabulary. The original model uses its unknown-animal embedding; localization may be less reliable."))
        rows = [{"Rank": row["rank"], "Start (seconds)": row["start"], "End (seconds)": row["end"],
                 "Match score": row["match_score"]} for row in result["intervals"]]
        table(rows)
        candidates = result["intervals"]
        labels = [t("Top prediction") if row["rank"] == 1 else t("Alternative") + f" {row['rank'] - 1}" for row in candidates]
        fig = go.Figure(go.Bar(y=labels, x=[row["end"] - row["start"] for row in candidates],
            base=[row["start"] for row in candidates], orientation="h",
            marker_color=["#2563eb"] + ["#a8b8d5"] * (len(candidates) - 1),
            customdata=[[row["start"], row["end"], row["match_score"]] for row in candidates],
            hovertemplate="%{y}<br>%{customdata[0]:.3f}–%{customdata[1]:.3f} s<br>%{customdata[2]:.4f}<extra></extra>"))
        fig.update_layout(height=260, xaxis=dict(title=t("Time (seconds)"), range=[0, result["duration"]]),
                          yaxis=dict(autorange="reversed"), showlegend=False)
        st.plotly_chart(fig, width="stretch", key=location + "_query_grounding_timeline")
        st.caption(t("Input video duration and candidate resolution (seconds): ") +
                   f"{result['duration']:.3f} / {result['time_resolution_seconds']:.3f}")
        left, right = st.columns(2)
        left.download_button(t("Download query intervals CSV"), export_grounding_csv(envelope),
            "animalact_query_intervals.csv", "text/csv", key=location + "_query_csv", on_click="ignore")
        right.download_button(t("Download query result JSON"), export_grounding_json(envelope),
            "animalact_query_grounding.json", "application/json", key=location + "_query_json", on_click="ignore")
    return True
