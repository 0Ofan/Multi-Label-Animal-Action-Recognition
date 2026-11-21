"""Checks of the resource-informed plan and the formal execution gate; no training."""
import unittest
import torch
from animalact2.phase1_5.config import load_config,cohort_hash
from animalact2.phase1_5.models import build_model
from animalact2.phase1_5.schedule import LearningRatePlan
from animalact2.phase1_5.engine import Trainer
from animalact2.phase1_5.safety import OUT


class FinalPlanTests(unittest.TestCase):
    def test_prepared_models_share_50epoch_10patience_protocol_and_asl_only_changes_loss(self):
        configs=[load_config(OUT/f"configs/{name}_seed42_bce.json") for name in ("model1","model2","model3")]
        self.assertEqual(len({cohort_hash(c) for c in configs}),1)
        for c in configs:
            self.assertEqual(c["training"]["max_epochs"],50)
            self.assertEqual(c["training"]["patience"],10)
        model=torch.nn.Linear(1,1);optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        plan=LearningRatePlan(optimizer,configs[0])
        self.assertAlmostEqual(plan.factor(1),.5)
        self.assertAlmostEqual(plan.factor(2),1.)
        self.assertAlmostEqual(plan.factor(50),.1)
        asl=load_config(OUT/"configs/asl_reference_not_first_round.json")
        asl["loss"]["name"]="bce"
        self.assertEqual(asl,configs[2])

    def test_formal_run_is_not_started_without_explicit_execution(self):
        trainer=Trainer.__new__(Trainer)
        trainer.diagnostic=False
        with self.assertRaises(RuntimeError):trainer.run(execute_training=False)
        trainer.diagnostic=True
        with self.assertRaises(RuntimeError):trainer.run(execute_training=True)


if __name__ == "__main__":unittest.main(verbosity=2)
