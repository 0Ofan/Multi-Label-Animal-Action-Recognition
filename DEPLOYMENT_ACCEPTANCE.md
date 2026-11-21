# AnimalAct 2.0 portable deployment acceptance

Initial website-only publication scope, before the research supplement: current 2.0 only. Original workspace, sealed experiments and source data were not modified. No 1.0 source/model/cache, raw AnimalAct videos, annotations, per-Clip targets or secrets are included in this public repository.

Local portable validation (2026-11-21): six tests passed, including real frozen VideoMAE CPU inference using both original heads, 3-frame and single-frame alignment, 119 outputs, padding masks, real evaluation tables and Streamlit AppTest on all three pages. Test input is a synthetic control video for interface validation, not an animal accuracy claim. No training or Test inference occurred.

Portable head tensors are exported unchanged from original Phase2.4 best checkpoints reused by Phase2.5 (Seed42): Attention epoch21, threshold0.12; BiGRU Mean epoch41, threshold0.07. Original source checkpoint SHA, exported head SHA and encoder-file SHA are recorded in deployment/model_manifest.json.

Cloud adaptation changes deployment/source discovery only: single-process Streamlit, CPU Linux dependencies, project-relative model paths, streamed verified Release download. Complete-frame sampling, normalized zero alignment, position interpolation, frozen encoder FP32 operations and original heads are retained.

Actual original-project browser acceptance had separately passed with real Train/Valid animal videos, but those files are deliberately not published. Re-running the historical raw-data tests requires course data. The portable release tests are tests/test_portable.py.

Linux CPU CI and GitHub Release/download verification results are recorded below after publication. Streamlit account authorization and actual hosted-app startup remain separate from local/CI validation; a public website is only considered deployed after Streamlit reports a successful deployment.

## Published verification

- Public repository initially published with331 remote Git blob SHAs verified; no original AnimalAct folder or 1.0 directories.
- Release: https://github.com/0Ofan/Multi-Label-Animal-Action-Recognition/releases/tag/v2.0-dashboard . Both binary asset SHA256 digests match the recorded archive manifests.
- Linux CPU CI passed: https://github.com/0Ofan/Multi-Label-Animal-Action-Recognition/actions/runs/44855786598 . CPU dependency installation, public model download/restore, real pretrained inference and three-page tests all passed.
- Portable single-process CPU browser acceptance also passed locally: actual Valid video upload,119 scores, fixed threshold0.12, labels Keeping still / Sensing, approximately2.93 seconds inference, all five Evaluation tabs. Raw video/screenshots remain private.
- Portable CPU versus original CUDA maximum Logits error: Attention1.82e-5, BiGRU Mean1.53e-5; head tensors unchanged.
- The owner reported starting the Streamlit deployment. Community Cloud correctly exposes CPU only; it does not use the local RTX5060. No paid GPU services have been created.

Hosted app operation is separate from Linux CI; the actual public Streamlit URL has not yet been independently checked by the publishing script.

## Complete research supplement

The owner subsequently authorized all current-version research dependencies. A separate v2.0-research-reproduction Release now contains the sealed Clip Manifests/derived labels, complete FP32 caches, checkpoints, predictions and preparation/experiment records, plus pretrained references and project documents. Raw AnimalAct videos/JSON/demo and actual1.0 files remain excluded. See research_tools/README.md and deployment/research_archive.json.

A fresh isolated restoration verified43,903 file hashes. All six head/backbone combinations passed real-batch gradient/update checks. All12 Bests reproduced complete979-Clip Valid mAP/Macro-F1/BCE118/BCE119 exactly, including every119-class confusion count. Source5,010 Train/Valid video/annotation content hashes matched. No new formal experiments or Test inference were run. Research files do not change the web model/threshold and are never automatically downloaded by the website.
