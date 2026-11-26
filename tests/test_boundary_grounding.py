import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import torch

from proposal.boundary_grounding import (
    BoundaryModel, boundary_loss, decode, dense_features, evaluate_record, summary, targets,
)


class BoundaryTests(unittest.TestCase):
    def test_targets_merge_overlaps_and_preserve_two_events(self):
        target = targets([[.25, .75], [.5, 1.], [1.5, 2.]], np.arange(0, 2.25, .25))
        np.testing.assert_allclose(target[:, 0], [0, 1, 1, 1, 0, 0, 1, 1])
        self.assertEqual(target[:, 1].sum(), 2)
        self.assertEqual(target[:, 2].sum(), 2)
        self.assertEqual(targets([], [0, .25, .5]).sum(), 0)
        with self.assertRaises(ValueError):
            targets([[0, 1]], [0, .5, .5])

    def test_decoder_returns_events_or_rejects_query(self):
        logits = torch.full((8, 3), -10.)
        logits[[1, 2, 6, 7], 0] = 10.
        logits[[1, 6], 1] = 10.
        logits[[2, 7], 2] = 10.
        edges = np.arange(9) * .25
        intervals = decode(logits, 10., edges)
        self.assertEqual([r['interval'] for r in intervals], [[.25, .75], [1.5, 2.]])
        self.assertEqual(decode(logits, -10., edges), [])

    def test_metrics_count_misses_and_match_same_occurrence(self):
        query = dict(references=[[0, 1], [2, 3]])
        wrong = evaluate_record(query, [dict(interval=[0, 3])])
        self.assertFalse(wrong['boundary_hit'])
        exact = evaluate_record(query, [dict(interval=[2, 3])])
        result = summary([exact, evaluate_record(query, []), evaluate_record(dict(references=[]), [])])
        self.assertEqual(result['boundary_hit_at_0_5s'], .5)
        self.assertEqual(result['miss_rate'], .5)
        self.assertEqual(result['negative_false_positive_rate'], 0)

    def test_loss_has_gradients_for_activity_boundaries_and_presence(self):
        model = BoundaryModel(2)
        output, presence = model(torch.randn(1, 12, 512), torch.tensor([1]), torch.tensor([3]))
        target = targets([[.5, 1.]], np.arange(13) * .25)[None]
        loss = boundary_loss(output, presence, target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))
        self.assertTrue(torch.isfinite(boundary_loss(output.detach(), presence.detach(), torch.zeros_like(target))))

    def test_sampling_retains_real_timestamps_without_repeating_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'control.avi'
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (32, 24))
            for i in range(20):
                writer.write(np.full((24, 32, 3), i * 10, dtype=np.uint8))
            writer.release()
            predictor = Mock(device='cpu')
            predictor.preprocess.side_effect = lambda frame: torch.zeros(3, 8, 8)
            predictor.encoder.side_effect = lambda x: torch.zeros(len(x), 512)
            value = dense_features(predictor, path)
            np.testing.assert_allclose(value['timestamps'], [0, .3, .5, .8, 1., 1.3, 1.5, 1.8])
            self.assertEqual(value['features'].shape, (8, 512))
            self.assertEqual(value['edges'][0], 0)
            self.assertEqual(value['edges'][-1], 2.)

    def test_inference_rejects_smoke_checkpoint_and_keeps_timestamp_geometry(self):
        from unittest.mock import patch
        from proposal.predict import ProposalPredictor
        predictor = ProposalPredictor.__new__(ProposalPredictor)
        predictor.device = 'cpu'
        checkpoint = dict(schema='boundary-v1', smoke=True)
        with patch('proposal.predict.torch.load', return_value=checkpoint):
            with self.assertRaisesRegex(ValueError, 'smoke'):
                predictor.predict_boundaries('video.mp4', 'A bee is flying.', 'model.pt')
        model = BoundaryModel(1)
        checkpoint.update(smoke=False, animals=['Bee'], classes=['Flying'], rate=4.,
                          threshold=.5, state_dict=model.state_dict())
        value = dict(features=torch.zeros(3, 512), timestamps=[0., .25, .5], edges=[0., .125, .375, .75])
        with patch('proposal.predict.torch.load', return_value=checkpoint), patch('proposal.boundary_grounding.dense_features', return_value=value):
            result = predictor.predict_boundaries('video.mp4', 'A bee is flying.', 'model.pt')
            self.assertEqual(result['sampled_timestamps'], value['timestamps'])
            self.assertEqual(result['max_time_bin_seconds'], .375)
            with self.assertRaisesRegex(ValueError, 'vocabulary'):
                predictor.predict_boundaries('video.mp4', 'A dog is flying.', 'model.pt')


if __name__ == '__main__':
    unittest.main()
