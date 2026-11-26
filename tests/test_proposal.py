import unittest
import numpy as np
import torch

from proposal.experiment import average_precision, summarize
from proposal.grounding import candidate_windows, temporal_iou, evaluate_queries, select_intervals
from proposal.models import build_model, classify
from proposal.predict import parse_query


class ProposalTests(unittest.TestCase):
    def test_shared_schema_keeps_model_class_policies(self):
        from animalact2.dashboard.registry import classes
        from animalact2.dashboard.schema import validate_prediction
        for unit, unsupported in (("whole_video", []), ("supplied_clip", [23])):
            rows = [dict(row, score=.9, above_threshold=True,
                         selected=row["class_id"] not in unsupported) for row in classes()]
            result = dict(class_scores=rows, logits=[1.] * 119, threshold=.5,
                          evaluation_unit=unit, unsupported_class_ids=unsupported,
                          predicted_labels=[row["name"] for row in rows if row["selected"]])
            validate_prediction(result)
            result["class_scores"][23]["selected"] = not rows[23]["selected"]
            with self.assertRaises(ValueError):
                validate_prediction(result)

    def test_automatic_prediction_ignores_legacy_query(self):
        from animalact2.dashboard.ui.services.state import recognition_identity
        state = dict(model_key="resnet_bigru_attention", grounding_query="A bee is flying.")
        before = recognition_identity(state)
        state["grounding_query"] = "A bee is walking."
        self.assertEqual(before, recognition_identity(state))
        state["model_key"] = "attention"
        before = recognition_identity(state)
        state["grounding_query"] = ""
        self.assertEqual(before, recognition_identity(state))

    def test_shared_runtime_routes_whole_video_and_rejects_partial_input(self):
        from unittest.mock import Mock, patch
        from pathlib import Path
        from animalact2.dashboard.ui.services.resources import Runtime
        runtime = Runtime()
        predictor = Mock()
        predictor.predict.return_value = {}
        runtime._resnet["cpu"] = predictor
        path = Path("video.mp4")
        with patch("animalact2.dashboard.ui.services.resources.detection_checkpoint", return_value=None):
            runtime.predict(path, device="cpu", model="resnet_mean", query="A bee is flying.")
        predictor.predict.assert_called_once_with(path, "mean", "A bee is flying.")
        with self.assertRaises(ValueError):
            runtime.predict(path, device="cpu", model="resnet_mean", start=1.)

    def test_dense_time_supervision_and_outside_evidence(self):
        from proposal.dense_grounding import activity_target, window_scores
        query = {"duration": 4, "references": [[0, 1], [.5, 2], [3, 4]]}
        np.testing.assert_allclose(activity_target(query, 4), [1, 1, 0, 1])
        score = window_scores(torch.tensor([[-5., 5., 5., -5.]]),
                              torch.tensor([[0., 1.], [.25, .75]]))
        self.assertGreater(score[0, 1], score[0, 0])

    def test_metrics_and_concurrent_subset(self):
        y = np.array([[1, 1], [1, 0], [0, 1], [0, 0]], dtype=bool)
        p = np.array([[.9, .2], [.7, .8], [.1, .9], [.3, .2]])
        c = np.zeros_like(y)
        c[0] = True
        result = summarize(y, p, c, [0, 1], ["a", "b"], .5)
        self.assertEqual(result["per_class"][1]["confusion_2x2_tn_fp_fn_tp"], [[1, 1], [1, 1]])
        self.assertEqual(result["concurrent_subset"]["recall"], .5)
        self.assertEqual(result["concurrent_subset"]["precision"], 1)
        self.assertEqual(average_precision(y[:, 0], np.ones(4)), .5)

    def test_existing_heads_have_gradients(self):
        for kind in ("mean", "bigru_mean", "bigru_attention"):
            model = build_model(kind)
            output = classify(model, torch.randn(2, 4, 512))
            self.assertEqual(output.shape, (2, 119))
            torch.nn.functional.binary_cross_entropy_with_logits(output, torch.zeros_like(output)).backward()
            self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_candidate_pool_and_multiple_references(self):
        features = torch.tensor([[[1.], [3.], [5.], [7.]]])
        pooled, windows = candidate_windows(features)
        index = next(i for i, (a, b) in enumerate(windows.tolist()) if a == .25 and b == .75)
        self.assertEqual(pooled[0, index, 0], 4)
        refs = [[0, .25], [.75, 1]]
        np.testing.assert_allclose(temporal_iou(np.array([[.75, 1], [.25, .75]]), refs), [1, 0])
        q = [{"duration": 1, "references": refs, "animal_id": 1}]
        scores = np.zeros((1, len(windows)))
        scores[0, -1] = 1
        result = evaluate_queries(q, scores, windows.numpy())
        self.assertEqual(result["all"]["miou"], 1)
        self.assertEqual(result["all"]["r1_iou_07"], 1)
        spans = select_intervals(scores[0], windows.numpy(), max_intervals=2)
        self.assertEqual(spans[0]["interval"], [.75, 1.])

    def test_query_retains_animal_and_action(self):
        self.assertEqual(parse_query("A honey bee is flying.", ["Honey Bee"], ["Flying"]), (1, 0))
        self.assertEqual(parse_query("A new species is flying.", ["Honey Bee"], ["Flying"]), (0, 0))
        with self.assertRaises(ValueError):
            parse_query("Where does it fly?", ["Honey Bee"], ["Flying"])


if __name__ == "__main__":
    unittest.main()
