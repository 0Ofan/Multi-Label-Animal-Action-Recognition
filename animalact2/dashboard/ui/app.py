from animalact2.dashboard.ui.i18n import t, display_format, display_data, option_formatter
"""Run with .venv/Scripts/python.exe -m streamlit run frontend/app.py."""
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

import streamlit as st
from animalact2.dashboard.ui.services.resources import model_info,get_runtime,available_models
from animalact2.dashboard.ui.pages import video,results,evaluation

if "ui_language" not in st.session_state:
    st.session_state["ui_language"] = "zh" if st.query_params.get("lang") == "zh" else "en"
st.set_page_config(page_title=t("AnimalAct 2.0 · Research Dashboard"),page_icon="🐾",layout="wide",initial_sidebar_state="expanded")
st.markdown(t('<style>'+ (Path(__file__).parent/'styles/dashboard.css').read_text(encoding='utf-8')+'</style>'),unsafe_allow_html=True)
if "model_key" not in st.session_state:
    st.session_state["model_key"]="resnet_ensemble" if "resnet_ensemble" in available_models() else "resnet_bigru_attention"
try: info=model_info(st.session_state["model_key"])
except Exception as exc:
    st.error(t(f"AnimalAct 2.0 backend unavailable: {exc}"))
    st.code(".\\.venv\\Scripts\\python.exe -B -m animalact2.dashboard.api")
    st.stop()
if "nav" not in st.session_state:
    selected=st.query_params.get("page","video").title()
    if selected == "Proposal": selected = "Evaluation"
    st.session_state["nav"]=selected if selected in ("Video","Results","Evaluation") else "Video"
with st.sidebar:
    def language_changed():
        st.query_params["lang"] = st.session_state["ui_language"]
    st.selectbox("Language / 语言", ["en", "zh"], key="ui_language",
                 format_func=lambda value: "简体中文" if value == "zh" else "English",
                 on_change=language_changed)
    st.markdown(t('<div class="brand"><div class="brand-icon">Aa</div><div><div class="brand-name">AnimalAct 2.0</div><div class="brand-sub">AI RESEARCH WORKSPACE</div></div></div>'),unsafe_allow_html=True)
    st.caption(t("DASHBOARD"))
    def sync(): st.query_params["page"]=st.session_state["nav"].lower()
    nav=st.radio(t("Dashboard"),["Video","Results","Evaluation"],key="nav",on_change=sync,label_visibility="collapsed", format_func=option_formatter())
    st.write(t(""))
    with st.expander(t("Model Information"),expanded=True):
        st.markdown("**" + info.get("backbone", "VideoMAE") + " · " + t(info["head"]) + "**")
        st.write(t(f"Seed **{info['seed']}** · Best epoch **{info['best_epoch']}**"))
        st.write(t(f"Threshold **{info['threshold']:.2f}** · 119 action classes"))
        st.caption(t("Whole-video classification · 119 trainable classes") if info.get("evaluation_unit") == "whole_video" else t("1–16 unique real frames · full aspect ratio · no crop"))
    runtime=get_runtime()
    status="Loaded · "+", ".join(d.upper() for d in sorted(runtime.loaded_devices)) if runtime.loaded_devices else "Ready · loads on demand"
    st.markdown(t(f'<div class="status-card"><span class="status-dot">●</span> <b>{status}</b><br><span style="color:#8290a6">App server processing · Fixed model</span></div>'),unsafe_allow_html=True)
    st.caption(t("COMP6116 · Stage 2\n\nAnimal action recognition"))
try:
    {"Video":video.render,"Results":results.render,"Evaluation":evaluation.render}[nav]()
except Exception as exc:
    st.error(t(f"This view could not be loaded: {exc}"))
    st.caption(t("No placeholder predictions or evaluation values have been substituted."))
footer = t("Whole-video classification · 119 trainable classes") if info.get("evaluation_unit") == "whole_video" else t("VideoMAE · action-interval clips · Valid evaluation")
st.markdown(f'<div class="app-footer">AnimalAct · COMP6116 &nbsp; / &nbsp; {footer}</div>',unsafe_allow_html=True)
