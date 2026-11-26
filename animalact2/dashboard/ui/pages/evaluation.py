from animalact2.dashboard.ui.i18n import t, display_format, display_data, option_formatter
"""Existing v2 artifacts only. No Test selector, predictions or inference."""
import json
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from animalact2.dashboard.ui.components.ui import header,table
from ..services.resources import evaluation, whole_video_results

def frame(kind,model="model3",seed=42):
    payload=evaluation(kind,model,seed)
    st.caption(t("Source: "+payload["source"]+" · "+payload["split"]))
    return pd.DataFrame(payload["rows"])

def numeric(df,cols):
    for col in cols:
        if col in df: df[col]=pd.to_numeric(df[col],errors="coerce")
    return df

def render_clip_extension():
    st.info(t("Evaluation unit: action-interval clip. 31 fixed main classes; BCE118 excludes unsupported Defensive Rearing. BCE119 remains a diagnostic. Multi-seed means are not ensembles."))
    tabs=st.tabs([t("Overview"),t("Per-class AP"),t("Confusion"),t("Training"),t("Generalization")])
    with tabs[0]:
        st.subheader(t("Three heads · three seeds · Valid mean ± sample std"))
        df=frame("summary")
        for metric in ("Valid_mAP","Valid_Macro_F1","Valid_BCE118"):
            means=pd.to_numeric(df[metric+"_mean"]); std=pd.to_numeric(df[metric+"_std"])
            fig=go.Figure(go.Bar(x=df["head"],y=means,error_y=dict(type="data",array=std)))
            fig.update_layout(title=metric,height=280,yaxis_tickformat=".1%" if metric!="Valid_BCE118" else ".3f")
            st.plotly_chart(fig,key=metric,width="stretch")
        st.dataframe(display_data(df),hide_index=True,width="stretch")
        st.subheader(t("ResNet18 vs VideoMAE"))
        st.caption(t("ResNet18 has one Seed42 run; VideoMAE has N=3. Different pretraining sources and feature dimensions; this is not an isolated architecture effect."))
        st.dataframe(display_data(frame("backbones")),hide_index=True,width="stretch")
        st.subheader(t("All nine runs · fixed best checkpoints"))
        st.dataframe(display_data(frame("runs")),hide_index=True,width="stretch")
    with tabs[1]:
        df=numeric(frame("per_class"),["seed","AP","class_id"])
        split=st.selectbox(t("AP split"),["valid","train"], format_func=option_formatter(), key='ui_evaluation_39')
        # Source uses capitalized Train/Valid in some schemas; compare case-insensitively.
        df=df[df["split"].str.lower()==split]
        st.dataframe(display_data(df),hide_index=True,width="stretch")
        st.subheader(t("Cross-seed per-class stability"))
        st.dataframe(display_data(frame("stability")),hide_index=True,width="stretch")
    with tabs[2]:
        model=st.selectbox(t("Confusion model"),["model3","model2","model1"], format_func=option_formatter(), key='ui_evaluation_46')
        seed=st.selectbox(t("Confusion seed"),[42,123,2026], format_func=option_formatter(), key='ui_evaluation_47')
        df=numeric(frame("confusion",model,seed),["TP","FP","TN","FN","Precision","Recall"])
        run=evaluation("runs")["rows"]
        row=next(x for x in run if x["model"]==model and int(x["seed"])==seed)
        st.caption(t(f"Valid-only one-vs-rest matrices · threshold {row['threshold']} selected on Valid · 31 separate 2×2 matrices, not a multiclass 31×31 matrix."))
        st.dataframe(display_data(df),hide_index=True,width="stretch")
        cid=st.selectbox(t("Confusion class"),df["name"].tolist(), format_func=option_formatter(), key='ui_evaluation_53')
        r=df[df["name"]==cid].iloc[0]
        fig=px.imshow([[r["TN"],r["FP"]],[r["FN"],r["TP"]]],text_auto=True,x=["Pred negative","Pred positive"],y=["True negative","True positive"],color_continuous_scale="Blues")
        st.plotly_chart(fig,width="stretch")
    with tabs[3]:
        model=st.selectbox(t("Training model"),["model3","model2","model1"], format_func=option_formatter(), key='ui_evaluation_58')
        seed=st.selectbox(t("Training seed"),[42,123,2026], format_func=option_formatter(), key='ui_evaluation_59')
        df=frame("history",model,seed)
        df=numeric(df,list(df.columns))
        st.caption(t("Train curves are online pre-update metrics; Valid curves use the fixed model at epoch end. Curve F1 uses 0.5, unlike final Valid-selected thresholds."))
        for columns in (["train_online_BCE118","valid_BCE118"],["train_online_BCE119","valid_BCE119"],["train_online_mAP","valid_mAP"]):
            fig=px.line(df,x="epoch",y=columns)
            st.plotly_chart(fig,width="stretch",key="curve_"+columns[0])
        st.dataframe(display_data(df),hide_index=True,width="stretch")
    with tabs[4]:
        st.subheader(t("Fixed-checkpoint Train/Valid gap"))
        st.caption(t("Each checkpoint is evaluated on both splits using its unchanged Valid-selected threshold. These are distinct from online Train curves."))
        df=numeric(frame("runs"),["Train_mAP","Valid_mAP","Train_Macro_F1","Valid_Macro_F1","mAP_gap","F1_gap"])
        st.dataframe(display_data(df),hide_index=True,width="stretch")
        st.warning(t("Train mAP is approximately 97–100% while Valid mAP is 43–47%. Strong overfitting is reproducible across seeds; this dashboard does not resolve model accuracy limits."))


def render():
    header("REPRODUCIBLE EXPERIMENTS", "AnimalAct 2.0 Evaluation", "From the proposed ResNet-18 baseline to temporal modeling and VideoMAE extensions.")
    st.caption(t("Method progression: ResNet-18 → mean pooling → BiGRU → temporal attention; VideoMAE extends the encoder."))
    fusion, base, extension, grounding = st.tabs([t("Weighted ensemble optimization"), t("Whole-video classification"), t("Clip-based extensions"), t("Template-query temporal grounding")])
    with fusion:
        render_fusion()
    with base:
        path = Path(__file__).resolve().parents[4] / "proposal/results/results.json"
        if path.exists():
            result = whole_video_results()
            comparison = []
            for name, item in result["models"].items():
                test = item["test"]
                comparison.append({"Model": name, "Valid mAP": item["validation"]["map"],
                    "Test mAP": test["map"], "Test Macro-F1": test["macro_f1"],
                    "Concurrent recall": test["concurrent_subset"]["recall"],
                    "Concurrent precision": test["concurrent_subset"]["precision"], "Threshold": item["threshold"]})
            st.dataframe(display_data(pd.DataFrame(comparison)), hide_index=True, width="stretch")
            st.caption(t("2,075 Train / 446 Valid / 447 Test videos. All 119 classes trained. Primary metrics use the fixed 31 classes. Concurrent Test subset contains 108 videos. Checkpoints and thresholds were selected on Valid."))
            kind = st.selectbox(t("Classification head"), list(result["models"]), format_func=option_formatter(), key="evaluation_whole_head")
            st.subheader(t("Test per-class AP and confusion counts"))
            st.dataframe(display_data(pd.DataFrame(result["models"][kind]["test"]["per_class"])), hide_index=True, width="stretch")
            st.download_button(t("Download classification evaluation"), path.read_bytes(), "whole_video_results.json", "application/json")
        else:
            st.info(t("Whole-video experiment artifacts are unavailable in this installation."))
    with extension:
        st.caption(t("Different sample units, pretraining and training policies: these scores are separate experiments, not a controlled backbone ablation."))
        render_clip_extension()
    with grounding:
        root = Path(__file__).resolve().parents[4] / "proposal"
        path = root / "grounding_rank_results/results.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            st.json(data["test"]["all"])
            st.warning(t("Experimental localization. The learned scorer has not improved over returning the entire video."))
            st.caption(t("Grounding Test scores are exploratory: Test was inspected across development rounds. The latest dense model was developed on Valid only."))
            st.caption(t("Query format: A {animal} is {action}. The highest-ranked interval is evaluated against all reference intervals using maximum temporal IoU."))
            st.write(t("Queries whose best whole-video baseline IoU is below 0.5"))
            st.json(data["test"]["temporally_selective"])
            st.download_button(t("Download grounding evaluation"), path.read_bytes(), "grounding_results.json", "application/json")
        else:
            st.info(t("The grounding checkpoint is still being prepared."))

def render_fusion():
    from pathlib import Path
    path=Path(__file__).resolve().parents[4]/"ensemble_integration/experiment/results.json"
    data=json.loads(path.read_text(encoding='utf-8'))
    st.subheader(t("Weighted ensemble optimization"))
    st.caption(t("Original models are retained. This module adds score fusion, Valid-only selection and comparison."))
    before=data['methods']['bigru_attention']['test'];after=data['methods']['weighted_fusion']['test']
    left,right=st.columns(2)
    left.metric(t("Test mAP"),f"{100*after['map']:.2f}%",f"+{100*(after['map']-before['map']):.2f} pp")
    right.metric(t("Test Macro-F1"),f"{100*after['macro_f1']:.2f}%",f"+{100*(after['macro_f1']-before['macro_f1']):.2f} pp")
    rows=[]
    for key in ['bigru_attention','equal_average','weighted_fusion']:
        m=data['methods'][key]
        rows.append({t('Model'):key,t('Valid mAP'):100*m['valid']['map'],t('Test mAP'):100*m['test']['map'],t('Test Macro-F1'):100*m['test']['macro_f1'],t('Threshold'):m['threshold']})
    st.dataframe(pd.DataFrame(rows).round(2),hide_index=True,width='stretch')
    st.info(t("Equal averaging has higher Test mAP, but the weighted policy was selected on Valid. No policy was reselected using Test."))
    st.caption(t("F1 gains reflect fusion and threshold selection together. Exploratory results on a previously evaluated Test split; statistical significance is not established."))
    st.download_button(t("Download fusion experiment"),path.read_bytes(),'ensemble_results.json','application/json')
