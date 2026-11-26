"""Query wiring, session separation, exports and real original-ranker equivalence."""
import copy
import csv
import io
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.parse import urlencode
from urllib.request import build_opener, ProxyHandler
from urllib.error import HTTPError

from animalact2.dashboard import grounding
from animalact2.dashboard.ui.services import state
from animalact2.dashboard.ui.services.presentation import export_grounding_csv, export_grounding_json


def result(query="A goat is walking."):
    query, animal, action = grounding.prepare_query(query)
    info = grounding.model_info()
    return dict(task="template_temporal_grounding", query=query, animal_id=animal, action_id=action,
        action=info["classes"][action], unknown_animal=animal == 0, duration=10., inference_time=.1,
        checkpoint_SHA256=info["checkpoint_SHA256"], classification_SHA256=info["classification_SHA256"],
        time_resolution_seconds=.625,
        intervals=[dict(rank=1,start=1.,end=3.,match_score=.8),dict(rank=2,start=4.,end=7.,match_score=.6)])


def session():
    return dict(upload=dict(filename="fixture.avi", content=b"mock upload", fingerprint="video-one",
                metadata=dict(duration=10.,frames=240,fps=24.,width=96,height=64,bytes=11)),
                grounding_query="A goat is walking.", recognition=dict(existing_classification=True))


class BonusGroundingTests(unittest.TestCase):
    def test_template_validation_and_original_unknown_policy(self):
        query, animal, action = grounding.prepare_query(" A Goat is Walking. ")
        self.assertEqual(query, "A Goat is Walking.")
        self.assertGreater(animal, 0)
        self.assertEqual(action, 114)
        self.assertEqual(grounding.prepare_query("A completely unknown animal is walking.")[1], 0)
        for query in ("", "walking", "山羊正在走路", "A goat is riding a bicycle."):
            with self.subTest(query=query), self.assertRaises(ValueError):
                grounding.prepare_query(query)

    def test_query_and_video_cache_identity_and_classification_separation(self):
        data = session()
        runtime = Mock()
        runtime.ground.side_effect = lambda path, query, device: result(query)
        original = copy.deepcopy(data["recognition"])
        first = state.ground(data, runtime)
        self.assertIs(state.ground(data, runtime), first)
        self.assertEqual(runtime.ground.call_count, 1)
        data["grounding_query"] = "A goat is eating."
        second = state.ground(data, runtime)
        self.assertEqual(runtime.ground.call_count, 2)
        self.assertNotEqual(first["identity"], second["identity"])
        self.assertEqual(data["recognition"], original)
        with patch.object(state, "probe", return_value=dict(duration=10.)):
            state.accept_upload(data, "different.avi", b"different video bytes")
        self.assertNotIn("grounding_result", data)
        self.assertNotIn("recognition", data)
        self.assertFalse(data["grounding_running"])

    def test_failure_and_invalid_query_cannot_replace_classification(self):
        data = session()
        data["grounding_result"] = dict(identity=("stale",))
        runtime = Mock()
        runtime.ground.side_effect = RuntimeError("decoder failed")
        with self.assertRaisesRegex(RuntimeError, "decoder failed"):
            state.ground(data, runtime)
        self.assertEqual(data["recognition"], dict(existing_classification=True))
        self.assertNotIn("grounding_result", data)
        self.assertFalse(data["grounding_running"])
        data["grounding_query"] = "invalid query"
        runtime.reset_mock()
        with self.assertRaises(ValueError):
            state.ground(data, runtime)
        runtime.ground.assert_not_called()

    def test_query_changed_during_inference_is_not_committed(self):
        data = session()
        def run(*args, **kwargs):
            data["grounding_query"] = "A goat is eating."
            return result()
        with self.assertRaisesRegex(RuntimeError, "changed"):
            state.ground(data, Mock(ground=run))
        self.assertNotIn("grounding_result", data)

    def test_grounding_bounds_and_exports_are_independent(self):
        value = result()
        grounding.validate_result(value)
        for key, bad in (("start", -1.), ("end", 11.), ("match_score", float("nan"))):
            corrupted = copy.deepcopy(value)
            corrupted["intervals"][0][key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                grounding.validate_result(corrupted)
        envelope = dict(filename="fixture.avi", fingerprint="hash", result=value)
        rows = list(csv.DictReader(io.StringIO(export_grounding_csv(envelope).decode("utf-8-sig"))))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["candidate_meaning"], "top_prediction")
        self.assertEqual(rows[1]["candidate_meaning"], "alternative_candidate")
        saved = json.loads(export_grounding_json(envelope))
        self.assertEqual(saved["query"], value["query"])
        self.assertNotIn("class_scores", saved)

    def test_bilingual_controls_and_query_only_results(self):
        from streamlit.testing.v1 import AppTest
        for language in ("en", "zh"):
            with self.subTest(language=language):
                app = AppTest.from_file(str(grounding.ROOT / "streamlit_app.py"), default_timeout=120)
                app.query_params.update(page="video", lang=language)
                app.run()
                self.assertFalse(app.exception)
                self.assertFalse(app.error)
                self.assertTrue(app.button(key="run_grounding").disabled)
                app.button(key="grounding_use_template").click().run()
                self.assertEqual(app.text_input(key="grounding_query_input").value, "A goat is walking.")
                data = session()
                data.pop("recognition")
                data["grounding_result"] = dict(filename="fixture.avi", fingerprint=data["upload"]["fingerprint"],
                    identity=state.grounding_identity(data), result=result())
                for key, value in data.items():
                    app.session_state[key] = value
                app.radio(key="nav").set_value("Results").run()
                self.assertFalse(app.exception)
                self.assertFalse(app.error)
                self.assertEqual(len(app.get("download_button")), 2)
                self.assertTrue(any("Start (seconds)" in f.value.columns or "起点（秒）" in f.value.columns for f in app.dataframe))
                app.run()
                self.assertEqual(app.session_state["grounding_query"], "A goat is walking.")
                self.assertIn("grounding_result", app.session_state)
                app.radio(key="nav").set_value("Video").run()
                self.assertFalse(app.error)
                app.text_input(key="grounding_query_input").set_value("A goat is eating.").run()
                self.assertNotIn("grounding_result", app.session_state)

    def test_http_metadata_and_invalid_query(self):
        from animalact2.dashboard.api import Server
        server = Server(("127.0.0.1", 0))
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        base = f"http://127.0.0.1:{server.server_port}"
        opener = build_opener(ProxyHandler({}))
        try:
            with opener.open(base + "/api/grounding", timeout=20) as reply:
                self.assertEqual(json.load(reply)["best_epoch"], 3)
            with self.assertRaises(HTTPError) as error:
                opener.open(base + "/api/grounding?query=invalid", data=b"video", timeout=20)
            self.assertEqual(error.exception.code, 400)
            self.assertFalse(server.grounding_predictors)
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)

    @unittest.skipUnless(os.environ.get("RUN_FINAL_SMOKE") == "1", "explicit original-model query smoke")
    def test_real_ranker_matches_original_and_http_route(self):
        import cv2
        import numpy as np
        from proposal.predict import ProposalPredictor
        from animalact2.dashboard.api import Server
        from animalact2.dashboard.ui.services import resources
        predictor = ProposalPredictor("cpu")
        with tempfile.TemporaryDirectory() as temp:
            video = Path(temp) / "synthetic_bonus.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 24, (96, 64))
            self.assertTrue(writer.isOpened())
            for i in range(24):
                frame = np.full((64,96,3),30 + i * 5,dtype=np.uint8)
                frame[12:28,i:i+12]=(40,180,90)
                writer.write(frame)
            writer.release()
            query = "A goat is walking."
            wrapped = grounding.predict(predictor, video, query)
            original = predictor.predict(video, kind="bigru_attention", query=query)["grounding"]
            for row, raw in zip(wrapped["intervals"], original["intervals"]):
                self.assertEqual([row["start"], row["end"]], raw["interval"])
                self.assertEqual(row["match_score"], raw["score"])
            other = grounding.predict(predictor, video, "A goat is eating.")
            self.assertNotEqual(wrapped["action_id"], other["action_id"])
            grounding.validate_result(other)
            server = Server(("127.0.0.1", 0))
            server.grounding_predictors["cpu"] = predictor
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                with patch.object(resources, "DIRECT", False), patch.object(resources, "BASE", f"http://127.0.0.1:{server.server_port}"):
                    actual = resources.Runtime().ground(video, query)
                self.assertEqual(actual["intervals"], wrapped["intervals"])
                self.assertEqual(actual["checkpoint_SHA256"], grounding.model_info()["checkpoint_SHA256"])
            finally:
                server.shutdown()
                server.server_close()
                worker.join(timeout=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
