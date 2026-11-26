import csv
import io
import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch

from proposal.action_detection import (
    ActionDetector, action_targets, decode_actions, detection_loss, evaluate_detections, predict_actions,
)


def prediction():
    from animalact2.dashboard.registry import classes
    rows = [dict(row, score=.1, above_threshold=False, eligible_for_prediction=True, selected=False)
            for row in classes()]
    return dict(class_scores=rows, logits=[-2.] * 119, threshold=.5, predicted_labels=[],
                unsupported_class_ids=[], evaluation_unit='whole_video', inference_time=.1,
                model_name='Test ResNet', device='cpu', video_id='fixture', model_id='resnet_mean',
                temporal_detection=dict(status='ready', duration=2., experimental=True,
                    segments=[dict(class_id=0, action=rows[0]['name'], start=.25, end=.75, score=.9)]))


class ActionDetectionTests(unittest.TestCase):
    def test_all_classes_and_masked_padding(self):
        model = ActionDetector().eval()
        x = torch.randn(2, 8, 512)
        lengths = torch.tensor([3, 8])
        mask = torch.arange(8)[None] < lengths[:, None]
        logits, presence = model(x, lengths, mask)
        self.assertEqual(logits.shape, (2, 8, 119, 3))
        self.assertEqual(presence.shape, (2, 119))
        x[~mask] = 1e5
        after, p_after = model(x, lengths, mask)
        torch.testing.assert_close(logits[mask], after[mask])
        torch.testing.assert_close(presence, p_after)
        target = torch.zeros_like(logits)
        target[0, 1, 0] = 1
        loss = detection_loss(after, p_after, target, mask)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_independent_boundaries_find_interior_and_repeated_actions(self):
        logits = torch.full((8, 119, 3), -20.)
        presence = torch.full((119,), -20.)
        presence[:2] = 20.
        logits[:, 0, 0] = 20.  # Even an all-video activity map must respect interior boundaries.
        logits[[1, 6], 0, 1] = 20.
        logits[[2, 7], 0, 2] = 20.
        logits[2:4, 1, 0] = 20.
        logits[2, 1, 1] = 20.
        logits[3, 1, 2] = 20.
        result = decode_actions(logits, presence, np.arange(9) * .25, [str(i) for i in range(119)])
        self.assertEqual([(r['start'], r['end']) for r in result if r['class_id'] == 0], [(.25, .75), (1.5, 2.)])
        self.assertEqual([(r['start'], r['end']) for r in result if r['class_id'] == 1], [(.5, 1.)])
        self.assertEqual(decode_actions(logits, torch.full((119,), -20.), np.arange(9) * .25, [str(i) for i in range(119)]), [])

    def test_short_action_is_positive_for_presence(self):
        logits = torch.zeros(1, 2, 119, 3, requires_grad=True)
        presence = torch.zeros(1, 119, requires_grad=True)
        target = torch.zeros_like(logits)
        target[0, 0, 0, 0] = .1  # A short event occupies only part of one time bin.
        detection_loss(logits, presence, target, torch.ones(1, 2, dtype=torch.bool)).backward()
        self.assertLess(presence.grad[0, 0], 0.)

    def test_temporal_background_is_not_drowned_by_absent_classes(self):
        logits = torch.zeros(1, 4, 119, 3, requires_grad=True)
        presence = torch.zeros(1, 119, requires_grad=True)
        target = torch.zeros_like(logits)
        target[0, :3, 0, 0] = 1
        target[0, 0, 0, 1] = 1
        target[0, 2, 0, 2] = 1
        detection_loss(logits, presence, target, torch.ones(1, 4, dtype=torch.bool)).backward()
        temporal_background = logits.grad[0, 3, 0, 0]
        absent_background = logits.grad[0, :, 1:, 0].sum()
        torch.testing.assert_close(temporal_background, absent_background)
        # Activity's three foreground cells must not outweigh either single boundary.
        activity = logits.grad[0, :3, 0, 0].sum()
        torch.testing.assert_close(activity, logits.grad[0, 0, 0, 1])
        torch.testing.assert_close(activity, logits.grad[0, 2, 0, 2])

    def test_loss_ignores_padded_labels_and_handles_empty_groups(self):
        logits = torch.zeros(1, 3, 119, 3, requires_grad=True)
        presence = torch.zeros(1, 119, requires_grad=True)
        target = torch.zeros_like(logits)
        mask = torch.tensor([[True, True, False]])
        original = detection_loss(logits, presence, target, mask)
        target[0, 2] = 1
        torch.testing.assert_close(original, detection_loss(logits, presence, target, mask))
        for fill in (0., 1.):
            value = detection_loss(logits, presence, torch.full_like(target, fill), mask)
            self.assertTrue(torch.isfinite(value))
            grad = torch.autograd.grad(value, logits, retain_graph=True)[0]
            self.assertTrue(bool((grad[0, :2] > 0).all()) if fill == 0 else bool((grad[0, :2] < 0).all()))
            self.assertEqual(grad[0, 2].abs().sum(), 0)

    def test_train_only_loader_never_opens_validation_or_test_annotations(self):
        from proposal.action_detection import load_records
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            annotation = json.dumps(dict(annotations=[dict(actions=[dict(action='0', start=0., end=1.)])])).encode()
            (root / 'train.json').write_bytes(annotation)
            (root / 'action_classes.json').write_text(json.dumps(dict(classes=[dict(class_id=i, name=str(i)) for i in range(119)])))
            rows = [dict(split=split, annotation_path='train.json' if split == 'train' else 'must_not_read.json',
                         annotation_sha256=hashlib.sha256(annotation).hexdigest(), duration=1.)
                    for split in ('train', 'valid', 'test')]
            (root / 'manifest.json').write_text(json.dumps(dict(samples=rows)))
            records, _ = load_records(root, splits=('train',))
            self.assertEqual(list(records), ['train'])
            self.assertEqual(len(records['train']), 1)

    def test_targets_merge_same_action_and_keep_concurrent_actions(self):
        events = [dict(class_id=0, interval=[.25, .75]), dict(class_id=0, interval=[.5, 1.]),
                  dict(class_id=1, interval=[.5, 1.])]
        target = action_targets(events, np.arange(9) * .25)
        self.assertEqual(target[:, 0, 1].sum(), 1)
        self.assertTrue(bool((target[2:4, :2, 0] == 1).all()))
        self.assertEqual(target[:, 2:].sum(), 0)

    def test_event_metrics_penalize_duplicates_wrong_classes_and_misses(self):
        segments = [dict(class_id=0, start=0., end=1., score=.9),
                    dict(class_id=0, start=0., end=1., score=.8),
                    dict(class_id=1, start=2., end=3., score=.7)]
        result = evaluate_detections([dict(segments=segments, references=[dict(class_id=0, interval=[0, 1]), dict(class_id=0, interval=[2, 3])])])
        self.assertEqual(result['boundary_hits'], 1)
        self.assertEqual(result['boundary_recall'], .5)
        self.assertAlmostEqual(result['boundary_precision'], 1 / 3)
        self.assertEqual(result['miou'], 1.)

    def test_schema_and_exports_include_time_intervals(self):
        from animalact2.dashboard.schema import validate_prediction
        from animalact2.dashboard.ui.services.presentation import export_intervals_csv, export_json
        result = validate_prediction(prediction())
        envelope = dict(filename='fixture.mp4', fingerprint='hash', result=result)
        rows = list(csv.DictReader(io.StringIO(export_intervals_csv(envelope).decode('utf-8-sig'))))
        self.assertEqual(rows[0]['start_seconds'], '0.25')
        self.assertIn(b'temporal_detection', export_json(envelope))
        result['temporal_detection']['segments'][0]['end'] = 3.
        with self.assertRaises(ValueError):
            validate_prediction(result)

    def test_runtime_adds_detection_without_query_or_classification_filter(self):
        from animalact2.dashboard.ui.services.resources import Runtime
        runtime = Runtime()
        predictor = Mock()
        predictor.predict.return_value = dict(inference_time=.1)
        runtime._resnet['cpu'] = predictor
        detection = prediction()['temporal_detection']
        with patch('animalact2.dashboard.ui.services.resources.detection_checkpoint', return_value=Path('detector.pt')), patch('proposal.action_detection.predict_actions', return_value=detection) as detect:
            result = runtime.predict(Path('video.mp4'), device='cpu', model='resnet_mean')
        detect.assert_called_once_with(predictor, Path('video.mp4'), Path('detector.pt'))
        self.assertEqual(len(result['temporal_detection']['segments']), 1)
        with patch('proposal.action_detection.torch.load', return_value=dict(schema='automatic-actions-v1', smoke=True)):
            with self.assertRaisesRegex(ValueError, 'smoke'):
                predict_actions(predictor, 'video.mp4', 'detector.pt')

    def test_bilingual_results_render_time_table_without_query(self):
        from streamlit.testing.v1 import AppTest
        root = Path(__file__).resolve().parents[1]
        for language, start_label in [('zh', '起点（秒）'), ('en', 'Start (seconds)')]:
            app = AppTest.from_file(str(root / 'streamlit_app.py'), default_timeout=120)
            app.query_params.update(page='results', lang=language)
            app.session_state['recognition'] = dict(filename='fixture.mp4', fingerprint='hash', result=prediction())
            app.run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertTrue(any(start_label in frame.value.columns for frame in app.dataframe))
            self.assertFalse(app.json)


if __name__ == '__main__':
    unittest.main()
