"""Release integrity and real inference on synthetic video only; no dataset access."""
import hashlib
import io
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch, Mock

from deployment import final_artifacts as assets


class FinalArtifactTests(unittest.TestCase):
    def fixture(self, root):
        payload = b"original unchanged checkpoint bytes"
        digest = hashlib.sha256(payload).hexdigest()
        blob = "sha256/" + digest
        raw = io.BytesIO()
        with zipfile.ZipFile(raw, "w") as archive:
            archive.writestr(blob, payload)
        raw = raw.getvalue()
        info = dict(bytes=len(payload), SHA256=digest, blob=blob)
        names = ["proposal/results/model.pt", "ensemble_integration/weights/model.pt"]
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps(dict(files={name: info for name in names},
            archive=dict(url="https://example.invalid/models.zip", bytes=len(raw), SHA256=hashlib.sha256(raw).hexdigest()))))
        return payload, raw, manifest, names

    def test_restore_deduplicated_payload_and_resume_without_download(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            payload, raw, manifest, names = self.fixture(root)
            with patch.object(assets, "ROOT", root), patch.object(assets, "MANIFEST", manifest), patch.object(assets, "urlopen", return_value=io.BytesIO(raw)) as download:
                assets.ensure_final_runtime()
                self.assertEqual(download.call_count, 1)
                for name in names:
                    self.assertEqual((root / name).read_bytes(), payload)
                assets.ensure_final_runtime()
                self.assertEqual(download.call_count, 1)
                self.assertTrue(assets.declared(root / names[0]))

    def test_differing_existing_checkpoint_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, _, manifest, names = self.fixture(root)
            target = root / names[0]
            target.parent.mkdir(parents=True)
            target.write_bytes(b"another experiment")
            with patch.object(assets, "ROOT", root), patch.object(assets, "MANIFEST", manifest), patch.object(assets, "urlopen") as download:
                with self.assertRaisesRegex(ValueError, "not overwritten"):
                    assets.ensure_final_runtime()
                download.assert_not_called()
                self.assertEqual(target.read_bytes(), b"another experiment")

    def test_corrupted_archive_commits_no_checkpoints(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, _, manifest, names = self.fixture(root)
            with patch.object(assets, "ROOT", root), patch.object(assets, "MANIFEST", manifest), patch.object(assets, "urlopen", return_value=io.BytesIO(b"corruption")):
                with self.assertRaisesRegex(ValueError, "checksum"):
                    assets.ensure_final_runtime()
            self.assertTrue(all(not (root / name).exists() for name in names))

    def test_unsafe_destination_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, _, manifest, _ = self.fixture(root)
            value = json.loads(manifest.read_text())
            row = next(iter(value["files"].values()))
            value["files"] = {"../outside.pt": row}
            manifest.write_text(json.dumps(value))
            with patch.object(assets, "ROOT", root), patch.object(assets, "MANIFEST", manifest), patch.object(assets, "urlopen") as download:
                with self.assertRaisesRegex(ValueError, "Unsafe"):
                    assets.ensure_final_runtime()
                download.assert_not_called()

    def test_detector_failure_preserves_classification(self):
        from animalact2.dashboard.ui.services.resources import Runtime
        runtime = Runtime()
        runtime._resnet["cpu"] = Mock()
        result = dict(inference_time=.1, predicted_labels=["Walking"])
        with patch("animalact2.dashboard.ui.services.resources.detection_checkpoint", return_value=Path("detector.pt")), patch("proposal.action_detection.predict_actions", side_effect=RuntimeError("decoder failed")):
            output = runtime._with_actions(Path("video.mp4"), result, "cpu", 0., None)
        self.assertEqual(output["predicted_labels"], ["Walking"])
        self.assertEqual(output["temporal_detection"]["status"], "unavailable")
        self.assertIn("decoder failed", output["temporal_detection"]["error"])

    @unittest.skipUnless(os.environ.get("RUN_FINAL_SMOKE") == "1", "explicit real final-model smoke")
    def test_real_final_models_on_synthetic_upload(self):
        import cv2
        import numpy as np
        from animalact2.dashboard.schema import validate_prediction
        from animalact2.dashboard.ui.services.resources import Runtime
        from animalact2.dashboard.ui.services.state import accept_upload, recognize
        from streamlit.testing.v1 import AppTest
        runtime = Runtime()
        with tempfile.TemporaryDirectory() as temp:
            video = Path(temp) / "synthetic_final_smoke.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 24, (96, 64))
            self.assertTrue(writer.isOpened())
            for i in range(24):
                frame = np.full((64, 96, 3), 30 + i * 5, dtype=np.uint8)
                frame[12:28, i:i + 12] = (40, 180, 90)
                writer.write(frame)
            writer.release()
            state = dict(model_key="resnet_ensemble")
            accept_upload(state, video.name, video.read_bytes())
            result = recognize(state, runtime, "cpu")["result"]
            validate_prediction(result)
            self.assertEqual(result["feature_shape"], [1, 16, 512])
            self.assertEqual(result["fusion_weights"], {"mean": 0., "bigru_mean": .2, "bigru_attention": .8})
            self.assertEqual(result["temporal_detection"]["status"], "ready")
            self.assertEqual(len(result["class_scores"]), 119)
            self.assertIn("cpu", runtime.loaded_devices)
            for model in ("resnet_mean", "resnet_bigru_mean", "resnet_bigru_attention"):
                other = validate_prediction(runtime.predict(video, device="cpu", model=model))
                self.assertEqual(other["feature_shape"], [1, 16, 512])
                self.assertEqual(other["temporal_detection"]["status"], "ready")
            for language in ("en", "zh"):
                app = AppTest.from_file(str(assets.ROOT / "streamlit_app.py"), default_timeout=120)
                app.query_params.update(page="results", lang=language)
                app.session_state["recognition"] = state["recognition"]
                app.session_state["upload"] = state["upload"]
                app.run()
                self.assertFalse(app.exception)
                self.assertFalse(app.error)
                self.assertGreaterEqual(len(app.get("download_button")), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
