"""Run with .venv/Scripts/python.exe -m streamlit run frontend/app.py."""
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

import streamlit as st
from animalact2.dashboard.ui.services.resources import model_info,get_runtime
from animalact2.dashboard.ui.pages import video,results,evaluation

st.set_page_config(page_title="AnimalAct 2.0 · Research Dashboard",page_icon="🐾",layout="wide",initial_sidebar_state="expanded")
st.markdown('<style>'+ (Path(__file__).parent/'styles/dashboard.css').read_text(encoding='utf-8')+'</style>',unsafe_allow_html=True)
if "model_key" not in st.session_state: st.session_state["model_key"]="attention"
try: info=model_info(st.session_state["model_key"])
except Exception as exc:
    st.error(f"AnimalAct 2.0 backend unavailable: {exc}")
    st.code(".\\.venv\\Scripts\\python.exe -B -m animalact2.dashboard.api")
    st.stop()
if "nav" not in st.session_state:
    selected=st.query_params.get("page","video").title()
    st.session_state["nav"]=selected if selected in ("Video","Results","Evaluation") else "Video"
with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-icon">Aa</div><div><div class="brand-name">AnimalAct 2.0</div><div class="brand-sub">AI RESEARCH WORKSPACE</div></div></div>',unsafe_allow_html=True)
    st.caption("DASHBOARD")
    def sync(): st.query_params["page"]=st.session_state["nav"].lower()
    nav=st.radio("Dashboard",["Video","Results","Evaluation"],key="nav",on_change=sync,label_visibility="collapsed")
    st.write("")
    with st.expander("Model Information",expanded=True):
        st.markdown("**AnimalAct 2.0 · VideoMAE Base**")
        st.caption("Frozen encoder · BiGRU · selectable head")
        st.write(f"Seed **{info['seed']}** · Best epoch **{info['best_epoch']}**")
        st.write(f"Threshold **{info['threshold']:.2f}** · 119 action classes")
        st.caption("1–16 unique real frames · full aspect ratio · no crop")
    runtime=get_runtime()
    status="Loaded · "+", ".join(d.upper() for d in sorted(runtime.loaded_devices)) if runtime.loaded_devices else "Ready · loads on demand"
    st.markdown(f'<div class="status-card"><span class="status-dot">●</span> <b>{status}</b><br><span style="color:#8290a6">Local processing · Fixed model</span></div>',unsafe_allow_html=True)
    st.caption("COMP6116 · Stage 2\n\nAnimal action recognition")
try:
    {"Video":video.render,"Results":results.render,"Evaluation":evaluation.render}[nav]()
except Exception as exc:
    st.error(f"This view could not be loaded: {exc}")
    st.caption("No placeholder predictions or evaluation values have been substituted.")
st.markdown(f'<div class="app-footer">AnimalAct · COMP6116 &nbsp; / &nbsp; AnimalAct 2.0 · seed {info["seed"]} · epoch {info["best_epoch"]} &nbsp; / &nbsp; Clip classification · 119 outputs · 118 learnable classes · Valid-only evaluation</div>',unsafe_allow_html=True)
