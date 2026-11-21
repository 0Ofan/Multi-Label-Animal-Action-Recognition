# AnimalAct 2.0 portable deployment acceptance

Scope: current 2.0 only. Original workspace, sealed experiments and source data were not modified. No 1.0 source/model/cache, raw AnimalAct videos, annotations, per-Clip targets or secrets are included in this public repository.

Local portable validation (2026-11-21): six tests passed, including real frozen VideoMAE CPU inference using both original heads, 3-frame and single-frame alignment, 119 outputs, padding masks, real evaluation tables and Streamlit AppTest on all three pages. Test input is a synthetic control video for interface validation, not an animal accuracy claim. No training or Test inference occurred.

Portable head tensors are exported unchanged from original Phase2.4 best checkpoints reused by Phase2.5 (Seed42): Attention epoch21, threshold0.12; BiGRU Mean epoch41, threshold0.07. Original source checkpoint SHA, exported head SHA and encoder-file SHA are recorded in deployment/model_manifest.json.

Cloud adaptation changes deployment/source discovery only: single-process Streamlit, CPU Linux dependencies, project-relative model paths, streamed verified Release download. Complete-frame sampling, normalized zero alignment, position interpolation, frozen encoder FP32 operations and original heads are retained.

Actual original-project browser acceptance had separately passed with real Train/Valid animal videos, but those files are deliberately not published. Re-running the historical raw-data tests requires course data. The portable release tests are tests/test_portable.py.

Linux CPU CI and GitHub Release/download verification results are recorded below after publication. Streamlit account authorization and actual hosted-app startup remain separate from local/CI validation; a public website is only considered deployed after Streamlit reports a successful deployment.
