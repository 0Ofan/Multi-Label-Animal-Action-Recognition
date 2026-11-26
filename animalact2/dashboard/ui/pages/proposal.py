from animalact2.dashboard.ui.i18n import t, display_format, display_data, option_formatter
"""Results and inference for the submitted whole-video proposal."""

import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[4] / "proposal"


@st.cache_resource
def predictor():
    from proposal.predict import ProposalPredictor
    return ProposalPredictor("cpu")


def render():
    st.title(t("Proposal experiments"))
    st.caption(t("ResNet-18 · whole-video multi-label classification · fixed official splits"))
    results_path = ROOT / "results/results.json"
    if not results_path.exists():
        st.info(t("The complete whole-video experiment is running. Results appear after evaluation completes."))
        return
    result = json.loads(results_path.read_text())
    comparison = []
    for name, item in result["models"].items():
        test = item["test"]
        comparison.append({
            "Model": name, "Valid mAP": item["validation"]["map"],
            "Test mAP": test["map"], "Test Macro-F1": test["macro_f1"],
            "Concurrent recall": test["concurrent_subset"]["recall"],
            "Concurrent precision": test["concurrent_subset"]["precision"],
            "Threshold": item["threshold"],
        })
    st.dataframe(display_data(pd.DataFrame(comparison)), hide_index=True, width="stretch")
    st.caption(t("2,075 Train / 446 Valid / 447 Test videos. All 119 classes trained. "
               "Primary metrics use the fixed 31 classes. Concurrent Test subset contains 108 videos. "
               "Checkpoints and thresholds were selected on Valid."))
    kind = st.selectbox(t("Proposal classification model"), list(result["models"]), format_func=option_formatter(), key='ui_proposal_41')
    per_class = result["models"][kind]["test"]["per_class"]
    st.subheader(t("Test per-class AP and confusion counts"))
    st.dataframe(display_data(pd.DataFrame(per_class)), hide_index=True, width="stretch")
    st.download_button(t("Download proposal evaluation"), results_path.read_bytes(),
                       "proposal_classification_results.json", "application/json", key='ui_proposal_45')
    ground_path = ROOT / "grounding_rank_results/results.json"
    if ground_path.exists():
        ground = json.loads(ground_path.read_text())
        st.subheader(t("Template-query temporal grounding"))
        st.json(ground["test"]["all"])
        st.warning(t("Experimental localization. The learned scorer has not improved over returning the entire video."))
        st.caption(t("Query format: A {animal} is {action}. The highest-ranked interval is evaluated "
                   "against all reference intervals using maximum temporal IoU."))
        st.write(t("Queries whose best whole-video baseline IoU is below 0.5"))
        st.json(ground["test"]["temporally_selective"])
        st.download_button(t("Download grounding evaluation"), ground_path.read_bytes(),
                           "proposal_grounding_results.json", "application/json", key='ui_proposal_57')
    st.subheader(t("Run the proposal model"))
    examples = sorted((ROOT / "examples").glob("*.mp4"))
    selected = st.selectbox(t("Unseen Test example"), ["Upload a video"] + [path.stem for path in examples], format_func=option_formatter(), key='ui_proposal_61')
    uploaded = st.file_uploader(t("Video for whole-video classification"), type=["mp4", "avi", "mov", "mkv"], key='ui_proposal_62')
    query = st.text_input(t("Optional template query"), placeholder="A honey bee is flying.", key='ui_proposal_63')
    example = next((path for path in examples if path.stem == selected), None)
    if example and uploaded is None:
        st.video(str(example))
        known = json.loads(example.with_suffix(".json").read_text())
        st.caption(t("Example query: " + known["grounding"]["query"]))
    if st.button(t("Run proposal inference"), disabled=uploaded is None and example is None, key='ui_proposal_69'):
        if query and not ground_path.exists():
            st.error(t("The grounding checkpoint is still being prepared."))
            return
        with st.spinner(t("Running ResNet-18 on CPU")), tempfile.TemporaryDirectory(prefix="proposal-upload-") as tmp:
            path = example
            if uploaded is not None:
                path = Path(tmp) / ("video" + Path(uploaded.name).suffix)
                path.write_bytes(uploaded.getvalue())
            output = predictor().predict(path, kind, query or None)
        st.dataframe(display_data(pd.DataFrame(sorted(output["scores"], key=lambda row: -row["score"]))),
                     hide_index=True, width="stretch")
        if "grounding" in output:
            st.json(output["grounding"])
            st.caption(t("The first interval is the top-ranked prediction. The remaining intervals are alternatives."))
        st.download_button(t("Download proposal prediction"), json.dumps(output, indent=2),
                           "proposal_prediction.json", "application/json", key='ui_proposal_84')
