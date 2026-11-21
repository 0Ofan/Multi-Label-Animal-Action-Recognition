import html
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


def header(eyebrow, title, subtitle):
    st.markdown(f'<div class="eyebrow">{html.escape(eyebrow)}</div>', unsafe_allow_html=True)
    st.title(title); st.markdown(f'<p class="page-description">{html.escape(subtitle)}</p>', unsafe_allow_html=True)


def metrics(items):
    for col, (label, value) in zip(st.columns(len(items)), items):
        with col: st.metric(label, value)


def tags(names):
    st.markdown('<div class="action-tags">'+''.join(f'<span>{html.escape(n)}</span>' for n in names)+'</div>', unsafe_allow_html=True)


def table(rows, **kwargs):
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", **kwargs)


def probability_chart(rows, threshold, key):
    if not rows:
        st.info("No categories match the current filters."); return
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
        st.subheader("Your next discovery starts with a video")
        st.write("Upload a clip on the Video page and run recognition. Your latest prediction will appear here, with all 119 action scores.")
        st.caption("Changing pages never starts recognition.")
