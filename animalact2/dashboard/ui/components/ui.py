from animalact2.dashboard.ui.i18n import t, display_format, display_data, option_formatter
import html
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


def header(eyebrow, title, subtitle):
    st.markdown(f'<div class="eyebrow">{html.escape(t(eyebrow))}</div>', unsafe_allow_html=True)
    st.title(t(title)); st.markdown(f'<p class="page-description">{html.escape(t(subtitle))}</p>', unsafe_allow_html=True)


def metrics(items):
    for col, (label, value) in zip(st.columns(len(items)), items):
        with col: st.metric(t(label), value)


def tags(names):
    st.markdown(t('<div class="action-tags">'+''.join(f'<span>{html.escape(n)}</span>' for n in names)+'</div>'), unsafe_allow_html=True)


def table(rows, **kwargs):
    st.dataframe(display_data(pd.DataFrame(rows)), hide_index=True, width="stretch", **kwargs)


def probability_chart(rows, threshold, key):
    if not rows:
        st.info(t("No categories match the current filters.")); return
    fig = go.Figure(go.Bar(x=[r["score"] for r in rows], y=[r["name"] for r in rows], orientation="h",
        marker_color=["#2563eb" if r["selected"] else "#b9c9e1" for r in rows],
        customdata=[r["class_id"] for r in rows], hovertemplate="%{y}<br>Score: %{x:.4f}<br>Global ID: %{customdata}<extra></extra>"))
    fig.add_vline(x=threshold, line_dash="dash", line_color="#e19026", annotation_text=f"Threshold {threshold:.2f}")
    fig.update_layout(height=max(300, len(rows)*25+90), margin=dict(l=0,r=20,t=35,b=20),
        xaxis=dict(title="Independent sigmoid score", range=[0,1]), yaxis=dict(autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch", key=key, config={"displayModeBar": False})


def empty_results():
    with st.container(border=True):
        st.subheader(t("Your next discovery starts with a video"))
        st.write(t("Upload a clip on the Video page and run recognition. Your latest prediction will appear here, with all 119 action scores."))
        st.caption(t("Changing pages never starts recognition."))


def action_timeline(envelope):
    from animalact2.dashboard.ui.services.presentation import export_intervals_csv
    st.subheader(t("Detected actions and time intervals"))
    detection = envelope["result"].get("temporal_detection", {})
    if detection.get("status") != "ready":
        if detection.get("error"):
            st.warning(t("Time detection failed; classification results are retained."))
            st.caption(str(detection["error"]))
            return
        st.info(t("Automatic time detection is not ready yet. Classification remains available; no timestamps will be invented."))
        return
    st.caption(t("Experimental automatic detection. Scores are uncalibrated; sub-second boundary accuracy is not guaranteed."))
    segments = detection["segments"]
    if not segments:
        st.info(t("No action interval passed the detector thresholds. This does not prove that no action occurred."))
        return
    table([{"Action": row["action"], "Start (seconds)": row["start"], "End (seconds)": row["end"],
            "Duration (seconds)": row["end"] - row["start"], "Match score": row["score"]} for row in segments])
    fig = go.Figure(go.Bar(y=[row["action"] for row in segments],
        x=[row["end"] - row["start"] for row in segments], base=[row["start"] for row in segments],
        orientation="h", customdata=[[row["start"], row["end"], row["score"]] for row in segments],
        hovertemplate="%{y}<br>%{customdata[0]:.3f}–%{customdata[1]:.3f} s<br>%{customdata[2]:.3f}<extra></extra>"))
    fig.update_layout(barmode="overlay", height=max(260, len({row["class_id"] for row in segments}) * 30 + 80),
        xaxis=dict(title=t("Time (seconds)"), range=[0, detection["duration"]]), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, key="automatic_action_timeline", width="stretch")
    st.download_button(t("Download action intervals CSV"), export_intervals_csv(envelope),
                       "animalact_action_intervals.csv", "text/csv", key="action_intervals_csv")
