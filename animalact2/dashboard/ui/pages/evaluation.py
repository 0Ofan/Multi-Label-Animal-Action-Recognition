"""Existing v2 artifacts only. No Test selector, predictions or inference."""
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from animalact2.dashboard.ui.components.ui import header,table
from ..services.resources import evaluation

def frame(kind,model="model3",seed=42):
    payload=evaluation(kind,model,seed)
    st.caption("Source: "+payload["source"]+" · "+payload["split"])
    return pd.DataFrame(payload["rows"])

def numeric(df,cols):
    for col in cols:
        if col in df: df[col]=pd.to_numeric(df[col],errors="coerce")
    return df

def render():
    header("REPRODUCIBLE EXPERIMENTS","AnimalAct 2.0 Evaluation","Read-only real Train/Valid results. Test has not been evaluated for this release.")
    st.info("Evaluation unit: action-interval clip. 31 fixed main classes; BCE118 excludes unsupported Defensive Rearing. BCE119 remains a diagnostic. Multi-seed means are not ensembles.")
    tabs=st.tabs(["Overview","Per-class AP","Confusion","Training","Generalization"])
    with tabs[0]:
        st.subheader("Three heads · three seeds · Valid mean ± sample std")
        df=frame("summary")
        for metric in ("Valid_mAP","Valid_Macro_F1","Valid_BCE118"):
            means=pd.to_numeric(df[metric+"_mean"]); std=pd.to_numeric(df[metric+"_std"])
            fig=go.Figure(go.Bar(x=df["head"],y=means,error_y=dict(type="data",array=std)))
            fig.update_layout(title=metric,height=280,yaxis_tickformat=".1%" if metric!="Valid_BCE118" else ".3f")
            st.plotly_chart(fig,key=metric,width="stretch")
        st.dataframe(df,hide_index=True,width="stretch")
        st.subheader("ResNet18 vs VideoMAE")
        st.caption("ResNet18 has one Seed42 run; VideoMAE has N=3. Different pretraining sources and feature dimensions; this is not an isolated architecture effect.")
        st.dataframe(frame("backbones"),hide_index=True,width="stretch")
        st.subheader("All nine runs · fixed best checkpoints")
        st.dataframe(frame("runs"),hide_index=True,width="stretch")
    with tabs[1]:
        df=numeric(frame("per_class"),["seed","AP","class_id"])
        split=st.selectbox("AP split",["valid","train"])
        # Source uses capitalized Train/Valid in some schemas; compare case-insensitively.
        df=df[df["split"].str.lower()==split]
        st.dataframe(df,hide_index=True,width="stretch")
        st.subheader("Cross-seed per-class stability")
        st.dataframe(frame("stability"),hide_index=True,width="stretch")
    with tabs[2]:
        model=st.selectbox("Confusion model",["model3","model2","model1"])
        seed=st.selectbox("Confusion seed",[42,123,2026])
        df=numeric(frame("confusion",model,seed),["TP","FP","TN","FN","Precision","Recall"])
        run=evaluation("runs")["rows"]
        row=next(x for x in run if x["model"]==model and int(x["seed"])==seed)
        st.caption(f"Valid-only one-vs-rest matrices · threshold {row['threshold']} selected on Valid · 31 separate 2×2 matrices, not a multiclass 31×31 matrix.")
        st.dataframe(df,hide_index=True,width="stretch")
        cid=st.selectbox("Confusion class",df["name"].tolist())
        r=df[df["name"]==cid].iloc[0]
        fig=px.imshow([[r["TN"],r["FP"]],[r["FN"],r["TP"]]],text_auto=True,x=["Pred negative","Pred positive"],y=["True negative","True positive"],color_continuous_scale="Blues")
        st.plotly_chart(fig,width="stretch")
    with tabs[3]:
        model=st.selectbox("Training model",["model3","model2","model1"])
        seed=st.selectbox("Training seed",[42,123,2026])
        df=frame("history",model,seed)
        df=numeric(df,list(df.columns))
        st.caption("Train curves are online pre-update metrics; Valid curves use the fixed model at epoch end. Curve F1 uses 0.5, unlike final Valid-selected thresholds.")
        for columns in (["train_online_BCE118","valid_BCE118"],["train_online_BCE119","valid_BCE119"],["train_online_mAP","valid_mAP"]):
            fig=px.line(df,x="epoch",y=columns)
            st.plotly_chart(fig,width="stretch",key="curve_"+columns[0])
        st.dataframe(df,hide_index=True,width="stretch")
    with tabs[4]:
        st.subheader("Fixed-checkpoint Train/Valid gap")
        st.caption("Each checkpoint is evaluated on both splits using its unchanged Valid-selected threshold. These are distinct from online Train curves.")
        df=numeric(frame("runs"),["Train_mAP","Valid_mAP","Train_Macro_F1","Valid_Macro_F1","mAP_gap","F1_gap"])
        st.dataframe(df,hide_index=True,width="stretch")
        st.warning("Train mAP is approximately 97–100% while Valid mAP is 43–47%. Strong overfitting is reproducible across seeds; this dashboard does not resolve model accuracy limits.")
