from __future__ import annotations
import unittest
import numpy as np
import torch
from animalact2.phase1_5_1 import Model1, Model2, Model3, ResNet18Backbone, ReleaseClipDataset
from animalact2.phase1_5_1.collate import collate_clips
from animalact2.phase1_5_1.transforms import FullFrameTransform, MEAN, STD
from animalact2.phase1_5_1.common import masked_mean
from animalact2.phase1_5_1.video_decoder import decode_exact_frames
from animalact2.phase1_5_1.evaluation import average_precision, evaluate_multilabel, select_validation_threshold


def sample(length, width=398):
    return dict(frames=torch.rand(length, 3, 224, width), target=torch.zeros(119),
                frame_indices=torch.arange(length), effective_length=length, valid_frame_mask=torch.ones(length, dtype=torch.bool),
                clip_id=f"synthetic_{length}", parent_video_id="synthetic", split="train", metadata={}, label_sources=[])


class DataContractTests(unittest.TestCase):
    def test_short_batch_is_not_extended_to_16(self):
        b = collate_clips([sample(1), sample(3), sample(10)])
        self.assertEqual(tuple(b["frames"].shape), (3, 10, 3, 224, 398))
        self.assertEqual(b["lengths"].tolist(), [1, 3, 10])
        self.assertTrue(torch.equal(b["padding_mask"], ~b["valid_frame_mask"]))
        self.assertTrue(torch.all(b["frames"][b["padding_mask"]] == 0))
        self.assertTrue(torch.all(b["frame_indices"][b["padding_mask"]] == -1))

    def test_collate_rejects_bad_lengths_and_shapes(self):
        with self.assertRaises(ValueError):
            collate_clips([])
        with self.assertRaises(ValueError):
            collate_clips([sample(0)])
        with self.assertRaises(ValueError):
            collate_clips([sample(1), sample(3, 300)])
        bad = sample(3)
        bad["valid_frame_mask"][1] = False
        with self.assertRaises(ValueError):
            collate_clips([bad])

    def test_full_frame_resize_color_and_edges(self):
        rgb = np.zeros((360, 640, 3), dtype=np.uint8)
        rgb[:, :50, 0] = 255
        rgb[:, -50:, 2] = 255
        x = FullFrameTransform()(rgb)
        self.assertEqual(tuple(x.shape), (3, 224, 398))
        restored = x * torch.tensor(STD)[:, None, None] + torch.tensor(MEAN)[:, None, None]
        torch.testing.assert_close(restored[:, 100, 0], torch.tensor([1., 0., 0.]), atol=1e-6, rtol=0)
        torch.testing.assert_close(restored[:, 100, -1], torch.tensor([0., 0., 1.]), atol=1e-6, rtol=0)
        portrait = FullFrameTransform()(np.zeros((640, 360, 3), dtype=np.uint8))
        self.assertEqual(tuple(portrait.shape), (3, 398, 224))

    def test_decoder_rejects_duplicates_out_of_order_and_empty(self):
        for idx in ([], [1, 1], [2, 1], [-1], [True]):
            with self.assertRaises(ValueError):
                decode_exact_frames("not_opened.mp4", idx)

    def test_test_split_is_blocked_before_any_file_read(self):
        with self.assertRaises(ValueError):
            ReleaseClipDataset("test", release_root="does_not_exist")


class ModelContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        cls.models = []
        for klass in (Model1, Model2, Model3):
            torch.manual_seed(42)
            cls.models.append(klass())

    def data(self):
        generator = torch.Generator().manual_seed(7)
        features = torch.randn(3, 5, 512, generator=generator)
        lengths = torch.tensor([1, 5, 3])  # unsorted, tests enforce_sorted=False
        valid = torch.arange(5)[None, :] < lengths[:, None]
        return features, lengths, valid

    def test_all_backbones_have_identical_official_state(self):
        first = self.models[0].backbone.encoder.state_dict()
        for model in self.models[1:]:
            other = model.backbone.encoder.state_dict()
            self.assertEqual(list(first), list(other))
            self.assertTrue(all(torch.equal(first[k], other[k]) for k in first))
        a = self.models[1].temporal.state_dict()
        b = self.models[2].temporal.state_dict()
        self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))

    def test_frozen_batchnorm_stays_eval_when_model_train(self):
        for model in self.models:
            model.train()
            self.assertFalse(any(p.requires_grad for p in model.backbone.parameters()))
            self.assertFalse(model.backbone.encoder.training)
            self.assertTrue(all(not m.training for m in model.backbone.modules() if isinstance(m, torch.nn.BatchNorm2d)))

    def test_single_frame_and_padding_invariance_all_heads(self):
        features, lengths, valid = self.data()
        poison = features.clone()
        poison[~valid] = float("nan")
        extended = torch.cat([poison, torch.full((3, 7, 512), float("nan"))], dim=1)
        extended_mask = torch.arange(12)[None, :] < lengths[:, None]
        for model in self.models:
            with self.subTest(model=type(model).__name__), torch.no_grad():
                model.eval()
                actual = model.forward_features(poison, lengths, valid)
                self.assertEqual(tuple(actual.shape), (3, 119))
                self.assertTrue(torch.isfinite(actual).all())
                torch.testing.assert_close(actual, model.forward_features(extended, lengths, extended_mask), atol=2e-6, rtol=1e-5)
                for i, n in enumerate(lengths.tolist()):
                    single = model.forward_features(features[i:i+1, :n], torch.tensor([n]), torch.ones(1, n, dtype=torch.bool))
                    torch.testing.assert_close(actual[i:i+1], single, atol=2e-6, rtol=1e-5)

    def test_attention_uses_time_softmax_and_zero_invalid_mass(self):
        features, lengths, valid = self.data()
        out = self.models[2].forward_features(features, lengths, valid, return_aux=True)
        weights = out["temporal_attention"]
        self.assertEqual(tuple(weights.shape), (3, 119, 5))
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(3, 119), atol=1e-6, rtol=0)
        self.assertTrue(torch.all(weights.masked_select(~valid[:, None, :].expand_as(weights)) == 0))
        self.assertTrue(torch.all(weights[0, :, 0] == 1))

    def test_real_step_gradients_exist_and_padding_has_zero_gradient(self):
        features, lengths, valid = self.data()
        for model in self.models:
            with self.subTest(model=type(model).__name__):
                model.train()
                model.zero_grad(set_to_none=True)
                leaf = features.clone().requires_grad_()
                logits = model.forward_features(leaf, lengths, valid)
                target = torch.zeros_like(logits)
                target[:, [1, 62]] = 1
                torch.nn.functional.binary_cross_entropy_with_logits(logits, target).backward()
                self.assertTrue(torch.all(leaf.grad[~valid] == 0))
                self.assertTrue(torch.any(leaf.grad[valid] != 0))
                for name, p in model.named_parameters():
                    if p.requires_grad:
                        self.assertIsNotNone(p.grad, name)
                        self.assertTrue(torch.isfinite(p.grad).all(), name)
                        self.assertGreater(float(p.grad.norm()), 0, name)
                self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))

    def test_invalid_zero_lengths_and_holey_masks_rejected(self):
        features, lengths, valid = self.data()
        for model in self.models:
            with self.assertRaises(ValueError):
                model.forward_features(features, torch.tensor([0, 5, 3]), valid)
            bad = valid.clone()
            bad[1, 2] = False
            with self.assertRaises(ValueError):
                model.forward_features(features, lengths, bad)

    def test_masked_mean_ignores_nan_storage(self):
        x = torch.tensor([[[1., 3.], [float("nan"), float("nan")]]])
        torch.testing.assert_close(masked_mean(x, torch.tensor([[True, False]])), torch.tensor([[1., 3.]]))


class EvaluationTests(unittest.TestCase):
    def test_ap_ties_and_unsupported_classes(self):
        self.assertAlmostEqual(average_precision([1, 0], [.5, .5]), .5)
        self.assertAlmostEqual(average_precision([1, 0, 1], [.9, .8, .7]), 5 / 6)
        self.assertIsNone(average_precision([0, 0], [.1, .9]))

    def test_fixed_31_counts_bce_sigmoid_and_threshold_ties(self):
        z = torch.full((2, 119), -10.)
        y = torch.zeros(2, 119)
        y[0] = 1
        z[0] = 10
        out = evaluate_multilabel(z, y)
        self.assertEqual(out["main_mAP"], 1.)
        self.assertEqual(out["main_macro_f1"], 1.)
        self.assertEqual(len(out["main_class_ids"]), 31)
        self.assertEqual(out["per_class"][62]["positive_clips"], 1)
        self.assertEqual(out["per_class"][62]["TP"], 1)
        self.assertEqual(out["per_class"][62]["TN"], 1)
        self.assertEqual(select_validation_threshold(z, y, split="valid")["threshold"], .05)
        self.assertAlmostEqual(out["BCE119"], float(torch.nn.functional.binary_cross_entropy_with_logits(z.double(), y.double())))

    def test_no_threshold_tuning_on_train_or_test(self):
        for split in ("train", "test"):
            with self.assertRaises(ValueError):
                select_validation_threshold(torch.zeros(1, 119), torch.zeros(1, 119), split=split)

    def test_partial_batch_cannot_claim_complete_map(self):
        z, y = torch.zeros(1, 119), torch.zeros(1, 119)
        with self.assertRaises(ValueError):
            evaluate_multilabel(z, y)
        self.assertIsNone(evaluate_multilabel(z, y, require_all_main_classes=False)["main_mAP"])


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main(verbosity=2)
