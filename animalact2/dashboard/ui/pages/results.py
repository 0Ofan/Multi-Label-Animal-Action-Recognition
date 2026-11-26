from animalact2.dashboard.ui.i18n import t, display_format, display_data, option_formatter
import streamlit as st

from animalact2.dashboard.ui.components.ui import header, metrics, tags, probability_chart, table, empty_results, action_timeline
from animalact2.dashboard.ui.services.presentation import score_rows, export_csv, export_json
from animalact2.dashboard.ui.services.search import llm_settings, search_actions
from animalact2.dashboard.ui.components.grounding import render_result as render_grounding_result


@st.cache_data(show_spinner="Searching action classes…", ttl=3600)
def semantic_lookup(query, vocabulary, settings):
    return search_actions(query, vocabulary, settings)


def _settings():
    try:
        secrets = st.secrets
    except Exception:
        secrets = None
    return llm_settings(secrets)


def render():
    header("PREDICTION EXPLORER", "Recognition Results", "Explore every action score from your latest video. No additional inference is run on this page.")
    envelope = st.session_state.get("recognition")
    if not envelope:
        if st.session_state.get("grounding_result") and st.session_state.get("upload"):
            st.video(st.session_state["upload"]["content"])
        if not render_grounding_result():
            empty_results()
        return
    r = envelope["result"]
    st.markdown(t(f"**{envelope['filename']}** · {r['model_name']} · {r['device'].upper()}"))
    metrics([("Predicted actions", str(len(r["predicted_labels"]))), ("Highest score", f"{max(x['score'] for x in r['class_scores']):.2%}"),
             ("Inference time", f"{r['inference_time']:.2f} s"), ("Fixed threshold", f"{r['threshold']:.2f}")])
    if st.session_state.get("upload"): st.video(st.session_state["upload"]["content"])
    render_grounding_result()
    if r.get("evaluation_unit") == "whole_video":
        st.caption(t("Whole-video ResNet-18 uses 16 uniform samples and trains all 119 classes. Short videos may reuse frames."))
    else:
        st.caption(t(f"Input window [start,end): {r['video']['window_seconds']} seconds · {r['video']['real_length']} unique real frames · feature shape {r['feature_shape']}. These are input bounds, not predicted action times."))
        st.warning(t("Defensive Rearing (ID23): raw score is shown for traceability, but it has no Train positives and is excluded from emitted labels."))
    action_timeline(envelope)
    st.write(t(""))
    with st.container(border=True):
        st.subheader(t("Predicted actions"))
        selected = [x for x in sorted(r["class_scores"], key=lambda x: -x["score"]) if x["selected"]]
        if selected:
            tags(r["predicted_labels"])
            table([{"Action": x["name"], "Probability": x["score"], "Threshold": r["threshold"], "Prediction": "Selected"} for x in selected],
                  column_config={"Probability": st.column_config.ProgressColumn(min_value=0,max_value=1,format="%.4f")})
        else: st.info(t("No class reached the validation-selected threshold. A low score does not establish that an action is absent."))
        st.caption(t("Scores are independent sigmoid outputs, not calibrated certainty. No ground truth is provided for this upload, so accuracy, precision and recall are not shown."))
    with st.container(border=True):
        st.subheader(t("Probability distribution"))
        st.caption(t("This filter searches action names only. Use the Bonus section on Video for query-based time localization."))
        query = st.text_input(t("Filter action names"), placeholder="e.g. 攀爬或跳跃的动作 / aggressive behaviour / Walking", max_chars=200, key="result_search",
                              help="Semantic search re-ranks the view only; prediction scores, threshold and emitted labels are unchanged.")
        a,b,c,d = st.columns([1.15,1.3,1.25,1])
        mode = a.selectbox(t("Search mode"), ["Semantic (LLM)","Keyword"], key="result_search_mode", format_func=option_formatter())
        scope = b.selectbox(t("Categories"), ["All 119 classes","Predicted only","Main 31 classes","Extension classes"], key="result_scope", format_func=option_formatter())
        order = c.selectbox(t("Sort"), ["Search relevance","Highest probability","Lowest probability","Action name","Class ID"], index=0, key="result_sort", format_func=option_formatter())
        limit = d.selectbox(t("Display"), ["Top 10","Top 20","All 119"], index=0, key="result_limit", format_func=option_formatter())
        semantic = bool(query) and mode.startswith("Semantic")
        match, query_ids, relevance = None, None, None
        if semantic:
            vocabulary = tuple((row["class_id"], row["name"]) for row in r["class_scores"])
            match = semantic_lookup(query, vocabulary, _settings())
            query_ids = match["ids"]
            relevance = {cid: rank for rank, cid in enumerate(query_ids)}
            if match["provider"] == "llm":
                st.caption(t(f"Semantic search · {match['model']} · {len(query_ids)} matched classes. Only the query and action names are sent to the configured API; the uploaded video is never sent."))
            elif match["error"]:
                st.warning(t(f"Semantic search is unavailable ({match['error']}). Offline alias matching was used instead."))
            else:
                st.caption(t("LLM not configured · offline alias matching is used. Add [llm] api_key under Streamlit Secrets to enable semantic search."))
            if query_ids:
                with st.expander(t("Matched classes and reasons")):
                    table([{"ID":cid,"Action":r["class_scores"][cid]["name"],"Why matched":match["reasons"].get(cid,"")} for cid in query_ids])
            else:
                st.info(t("No action class matched this query. Try another description or switch to Keyword mode."))
        rows = score_rows(r, "" if semantic else query, scope, order, limit, query_ids=query_ids, relevance=relevance)
        st.caption(t(f"Showing {len(rows)} of 119 class scores. Filters affect the view only; exports always contain all 119 classes."))
        probability_chart(rows, r["threshold"], "results_probability_chart")
        def table_row(x):
            row={"ID":x["class_id"],"Action":x["name"]}
            if semantic: row["Why matched"]=match["reasons"].get(x["class_id"],"")
            row.update({"Probability":x["score"],"Above threshold":x["above_threshold"],"Eligible":x["eligible_for_prediction"],"Predicted":x["selected"],"Main class":x["main_evaluation"]})
            return row
        table([table_row(x) for x in rows],
              column_config={"Probability":st.column_config.ProgressColumn(min_value=0,max_value=1,format="%.4f")})
    a,b = st.columns([1.5,1])
    with a, st.container(border=True):
        st.subheader(t("Category details"))
        cid = st.selectbox(t("Select an action"), range(119), format_func=display_format(lambda i: f"{r['class_scores'][i]['name']} · ID {i}"), key="result_category")
        row = r["class_scores"][cid]
        metrics([("Probability",f"{row['score']:.4f}"),("Decision","Selected" if row["selected"] else "Not selected")])
        st.caption(t(f"Global ID {cid} · {'Main evaluation class' if row['main_evaluation'] else 'Extension class'} · Threshold {r['threshold']:.2f}"))
    with b, st.container(border=True):
        st.subheader(t("Export prediction"))
        st.write(t("Download the complete prediction with class IDs, scores, threshold and inference time."))
        st.download_button(t("Download CSV"), export_csv(envelope), "animalact_prediction.csv", "text/csv", width="stretch", on_click="ignore", key='ui_results_90')
        st.download_button(t("Download JSON"), export_json(envelope), "animalact_prediction.json", "application/json", width="stretch", on_click="ignore", key='ui_results_91')
