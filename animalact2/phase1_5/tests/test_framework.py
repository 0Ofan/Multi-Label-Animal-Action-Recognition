from __future__ import annotations
import copy
import io
import json
import math
import random
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from animalact2.phase1_5.config import default_config, config_hash, cohort_hash, validate_config
from animalact2.phase1_5.data import EpochShuffleSampler, training_class_support
from animalact2.phase1_5.models import build_model
from animalact2.phase1_5.losses import MultilabelLoss, UnsupportedClassPolicy
from animalact2.phase1_5.schedule import LearningRatePlan, EarlyStopping
from animalact2.phase1_5.engine import train_one_epoch, make_optimizer, Trainer
from animalact2.phase1_5.checkpoint import make_checkpoint, save_checkpoint, load_checkpoint, head_state
from animalact2.phase1_5.evaluation import action_predictions
from animalact2.phase1_5.safety import OUT


def supports():
    counts = torch.ones(119, dtype=torch.int64)
    counts[23] = 0
    return counts


class LinearFixture(nn.Module):
    """Only a test fixture for gradient algebra; not an additional research model."""
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(512,119)

    def forward(self, frames, lengths, valid):
        pooled = torch.where(valid[:, :, None], frames, 0).sum(1) / lengths[:, None]
        return self.classifier(pooled)


def fixture_batch(features, labels, split="train"):
    b,t = features.shape[:2]
    return dict(frames=features, lengths=torch.full((b,),t,dtype=torch.int64), valid_frame_mask=torch.ones(b,t,dtype=torch.bool),
                target=labels, clip_ids=[f"fixture_{i}" for i in range(b)], parent_video_ids=[f"p{i}" for i in range(b)], splits=[split]*b)


class LossAndSupportTests(unittest.TestCase):
    def test_bce_masks_unsupported_class_and_has_zero_gradient_there(self):
        logits = torch.randn(4,119,requires_grad=True)
        target = torch.zeros_like(logits)
        loss = MultilabelLoss(default_config()["loss"], supports()>0)
        actual = loss(logits,target)
        expected = nn.functional.binary_cross_entropy_with_logits(logits[:,supports()>0],target[:,supports()>0])
        torch.testing.assert_close(actual,expected)
        actual.backward()
        self.assertTrue(torch.all(logits.grad[:,23] == 0))

    def test_asl_zero_gammas_zero_clip_matches_bce_and_extremes_are_finite(self):
        c = default_config()["loss"]
        c.update(name="asl",gamma_neg=0.,gamma_pos=0.,clip=0.)
        z,y = torch.randn(3,119), torch.randint(0,2,(3,119)).float()
        criterion = MultilabelLoss(c,supports()>0)
        torch.testing.assert_close(criterion(z,y),nn.functional.binary_cross_entropy_with_logits(z[:,supports()>0],y[:,supports()>0]),atol=1e-6,rtol=1e-6)
        c.update(gamma_neg=4.,clip=.05)
        z = torch.tensor([[10000.,-10000.]*59+[0.]],requires_grad=True)
        loss = MultilabelLoss(c,supports()>0)(z,torch.zeros_like(z))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_asl_easy_negative_downweight_and_unsupported_no_emission(self):
        z = torch.full((2,119),-5.)
        y = torch.zeros_like(z)
        bce = MultilabelLoss(default_config()["loss"],supports()>0)(z,y)
        c = default_config()["loss"]
        c["name"] = "asl"
        self.assertLess(float(MultilabelLoss(c,supports()>0)(z,y)),float(bce))
        model = LinearFixture()
        policy = UnsupportedClassPolicy(model,supports())
        self.assertFalse(action_predictions(torch.full((2,119),20.),.5,policy)[:,23].any())
        self.assertEqual(policy.metadata()["training_loss_classes"],118)

    def test_class_rows_are_restored_against_adamw_decay(self):
        model = LinearFixture()
        policy = UnsupportedClassPolicy(model,supports())
        before = {k:v.detach()[23].clone() for k,v in model.named_parameters()}
        optimizer = torch.optim.AdamW(model.parameters(),lr=.1,weight_decay=.1)
        for p in model.parameters():
            p.grad = torch.zeros_like(p)
        optimizer.step()
        policy.restore_rows(model)
        for k,v in model.named_parameters():
            self.assertTrue(torch.equal(before[k],v[23]))


class LoaderAndProtocolTests(unittest.TestCase):
    def test_same_epoch_shuffle_order_all_models_and_resume(self):
        a,b = EpochShuffleSampler(list(range(19)),42),EpochShuffleSampler(list(range(19)),42)
        a.set_epoch(7)
        b.set_epoch(7)
        self.assertEqual(list(a),list(b))
        self.assertEqual(sorted(a),list(range(19)))
        b.set_epoch(8)
        self.assertNotEqual(list(a),list(b))

    def test_configs_only_differ_by_model_and_invalid_model_rejected(self):
        configs = [default_config(name) for name in ("model1","model2","model3")]
        self.assertEqual(len({cohort_hash(c) for c in configs}),1)
        c = copy.deepcopy(configs[0]);c["model"]="extra_backbone"
        with self.assertRaises(ValueError):validate_config(c)
        c = copy.deepcopy(configs[0]);c["backbone"]["frozen"]=False
        with self.assertRaises(ValueError):validate_config(c)

    def test_class_eligibility_cannot_use_valid(self):
        class Valid:
            split="valid"
        with self.assertRaises(ValueError):training_class_support(Valid())

    def test_warmup_cosine_endpoints_and_schedule_restore(self):
        c = default_config();m=LinearFixture();o=make_optimizer(m,c);s=LearningRatePlan(o,c)
        self.assertEqual(s.apply_epoch(1),.0005)
        self.assertEqual(s.apply_epoch(2),.001)
        self.assertAlmostEqual(s.apply_epoch(100),.0001)
        restored=LearningRatePlan(make_optimizer(m,c),c);restored.load_state_dict(s.state_dict())
        self.assertEqual(restored.last_epoch,100)
        self.assertEqual(restored.factor(99),s.factor(99))

    def test_early_stopping_best_checkpoint_and_resume_counts(self):
        e=EarlyStopping(2)
        self.assertEqual(e.update(.4,1),(True,False))
        self.assertEqual(e.update(.5,2),(True,False))
        self.assertEqual(e.update(.5,3),(False,False))
        restored=EarlyStopping(2);restored.load_state_dict(e.state_dict())
        self.assertEqual(restored.update(.49,4),(False,True))
        self.assertEqual(restored.best_epoch,2)

    def test_default_entries_are_preflight_only(self):
        from animalact2.phase1_5.train import main
        for name in ("model1","model2","model3"):
            with patch("sys.argv",["entry"]), patch("animalact2.phase1_5.train.Trainer",side_effect=AssertionError("Should not instantiate training")),redirect_stdout(io.StringIO()) as stream:
                main(name)
            data=json.loads(stream.getvalue())
            self.assertFalse(data["training_started"])
            self.assertEqual(data["model"],name)


class TrainingAlgebraTests(unittest.TestCase):
    def test_unequal_microbatches_and_partial_accumulation_match_full_batch(self):
        torch.manual_seed(19)
        a=LinearFixture();b=copy.deepcopy(a);c=default_config()
        pa,pb=UnsupportedClassPolicy(a,supports()),UnsupportedClassPolicy(b,supports())
        oa,ob=make_optimizer(a,c),make_optimizer(b,c)
        criterion=MultilabelLoss(c["loss"],supports()>0)
        x,y=torch.randn(5,3,512),torch.randint(0,2,(5,119)).float()
        micro=[fixture_batch(x[:2],y[:2]),fixture_batch(x[2:4],y[2:4]),fixture_batch(x[4:],y[4:])]
        result=train_one_epoch(a,micro,oa,criterion,pa,"cpu",accumulation_steps=4,grad_clip_norm=100.)
        reference=train_one_epoch(b,[fixture_batch(x,y)],ob,criterion,pb,"cpu",accumulation_steps=1,grad_clip_norm=100.)
        self.assertEqual(result["optimizer_updates"],1)
        self.assertEqual(result["examples"],5)
        self.assertAlmostEqual(result["objective"],reference["objective"],places=6)
        for va,vb in zip(a.parameters(),b.parameters()):torch.testing.assert_close(va,vb,atol=2e-6,rtol=2e-5)

    def test_training_rejects_valid_gradient_source(self):
        m=LinearFixture();c=default_config();p=UnsupportedClassPolicy(m,supports());o=make_optimizer(m,c)
        batch=fixture_batch(torch.zeros(1,1,512),torch.zeros(1,119),split="valid")
        with self.assertRaises(ValueError):
            train_one_epoch(m,[batch],o,MultilabelLoss(c["loss"],supports()>0),p,"cpu",accumulation_steps=1,grad_clip_norm=5.)


class CheckpointTests(unittest.TestCase):
    def test_roundtrip_optimizer_rng_resume_and_diagnostic_guard(self):
        c=default_config("model3");c["device"]="cpu"
        m=build_model(c);p=UnsupportedClassPolicy(m,supports());o=make_optimizer(m,c);s=LearningRatePlan(o,c);e=EarlyStopping(15)
        s.apply_epoch(1);e.update(.4,1)
        features=torch.linspace(-1,1,2*3*512).reshape(2,3,512)
        lengths=torch.tensor([1,3]);valid=torch.arange(3)[None,:]<lengths[:,None]
        target=torch.zeros(2,119)
        criterion=MultilabelLoss(c["loss"],p.active_mask)
        def step(model,opt,policy):
            opt.zero_grad(set_to_none=True)
            criterion(model.forward_features(features,lengths,valid),target).backward()
            opt.step();policy.restore_rows(model)
        step(m,o,p)
        expected_logits=m.forward_features(features,lengths,valid).detach()
        path=OUT/"diagnostics/unit_checkpoint/model3_diagnostic.pt"
        save_checkpoint(path,make_checkpoint(m,o,s,e,p,c,epoch=1,history=[],diagnostic=True))
        expected_rng=(random.random(),float(np.random.rand()),torch.rand(3))
        step(m,o,p);expected_state=head_state(m)
        restored=build_model(c);rp=UnsupportedClassPolicy(restored,supports());ro=make_optimizer(restored,c);rs=LearningRatePlan(ro,c);re=EarlyStopping(15)
        load_checkpoint(path,restored,c,optimizer=ro,schedule=rs,early=re,policy=rp,restore_random_state=True,expected_diagnostic=True)
        actual_rng=(random.random(),float(np.random.rand()),torch.rand(3))
        self.assertEqual(expected_rng[:2],actual_rng[:2]);self.assertTrue(torch.equal(expected_rng[2],actual_rng[2]))
        torch.testing.assert_close(restored.forward_features(features,lengths,valid),expected_logits,rtol=0,atol=0)
        step(restored,ro,rp)
        for name,value in head_state(restored).items():torch.testing.assert_close(value,expected_state[name],rtol=0,atol=0)
        self.assertEqual(re.best_epoch,1);self.assertEqual(rs.last_epoch,1)
        with self.assertRaises(ValueError):load_checkpoint(path,restored,c,expected_diagnostic=False)
        changed=copy.deepcopy(c);changed["seed"]=123
        with self.assertRaises(ValueError):load_checkpoint(path,restored,changed)


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main(verbosity=2)
