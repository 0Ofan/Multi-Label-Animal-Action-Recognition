import streamlit as st

from animalact2.dashboard.ui.components.ui import header, metrics, tags, probability_chart, table, empty_results
from animalact2.dashboard.ui.services.presentation import score_rows, export_csv, export_json


def render():
    header("PREDICTION EXPLORER", "Recognition Results", "Explore every action score from your latest video. No additional inference is run on this page.")
    envelope = st.session_state.get("recognition")
    if not envelope:
        empty_results(); return
    r = envelope["result"]
    st.markdown(f"**{envelope['filename']}** · {r['model_name']} · {r['device'].upper()}")
    metrics([("Predicted actions", str(len(r["predicted_labels"]))), ("Highest score", f"{max(x['score'] for x in r['class_scores']):.2%}"),
             ("Inference time", f"{r['inference_time']:.2f} s"), ("Fixed threshold", f"{r['threshold']:.2f}")])
    if st.session_state.get("upload"): st.video(st.session_state["upload"]["content"])
    st.caption(f"Input window [start,end): {r['video']['window_seconds']} seconds · {r['video']['real_length']} unique real frames · feature shape {r['feature_shape']}. These are input bounds, not predicted action times.")
    st.warning("Defensive Rearing (ID23): raw score is shown for traceability, but it has no Train positives and is excluded from emitted labels.")
    st.write("")
    with st.container(border=True):
        st.subheader("Predicted actions")
        selected = [x for x in sorted(r["class_scores"], key=lambda x: -x["score"]) if x["selected"]]
        if selected:
            tags(r["predicted_labels"])
            table([{"Action": x["name"], "Probability": x["score"], "Threshold": r["threshold"], "Prediction": "Selected"} for x in selected],
                  column_config={"Probability": st.column_config.ProgressColumn(min_value=0,max_value=1,format="%.4f")})
        else: st.info("No class reached the validation-selected threshold. A low score does not establish that an action is absent.")
        st.caption("Scores are independent sigmoid outputs, not calibrated certainty. No ground truth is provided for this upload, so accuracy, precision and recall are not shown.")
    with st.container(border=True):
        st.subheader("Probability distribution")
        a,b,c,d = st.columns([1.4,1.2,1.1,1])
        query = a.text_input("Search actions", placeholder="e.g. Walking", key="result_search")
        scope = b.selectbox("Categories", ["All 119 classes","Predicted only","Main 31 classes","Extension classes"], key="result_scope")
        order = c.selectbox("Sort", ["Highest probability","Lowest probability","Action name","Class ID"], key="result_sort")
        limit = d.selectbox("Display", ["Top 10","Top 20","All 119"], index=0, key="result_limit")
        rows = score_rows(r, query, scope, order, limit)
        st.caption(f"Showing {len(rows)} of 119 class scores. Filters affect the view only; exports always contain all 119 classes.")
        probability_chart(rows, r["threshold"], "results_probability_chart")
        table([{"ID":x["class_id"],"Action":x["name"],"Probability":x["score"],"Above threshold":x["above_threshold"],"Eligible":x["eligible_for_prediction"],"Predicted":x["selected"],"Main class":x["main_evaluation"]} for x in rows],
              column_config={"Probability":st.column_config.ProgressColumn(min_value=0,max_value=1,format="%.4f")})
    a,b = st.columns([1.5,1])
    with a, st.container(border=True):
        st.subheader("Category details")
        cid = st.selectbox("Select an action", range(119), format_func=lambda i: f"{r['class_scores'][i]['name']} · ID {i}", key="result_category")
        row = r["class_scores"][cid]
        metrics([("Probability",f"{row['score']:.4f}"),("Decision","Selected" if row["selected"] else "Not selected")])
        st.caption(f"Global ID {cid} · {'Main evaluation class' if row['main_evaluation'] else 'Extension class'} · Threshold {r['threshold']:.2f}")
    with b, st.container(border=True):
        st.subheader("Export prediction")
        st.write("Download the complete prediction with class IDs, scores, threshold and inference time.")
        st.download_button("Download CSV", export_csv(envelope), "animalact_prediction.csv", "text/csv", width="stretch", on_click="ignore")
        st.download_button("Download JSON", export_json(envelope), "animalact_prediction.json", "application/json", width="stretch", on_click="ignore")
