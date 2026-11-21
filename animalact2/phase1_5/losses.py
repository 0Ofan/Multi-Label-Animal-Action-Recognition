"""BCE/ASL with explicit Train-derived class eligibility, unchanged 119 outputs."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


class MultilabelLoss(nn.Module):
    def __init__(self, config, active_mask):
        super().__init__()
        active = torch.as_tensor(active_mask, dtype=torch.bool)
        if active.shape != (119,) or not active.any():
            raise ValueError("Need a nonempty 119-dimensional Train-supported class mask")
        self.register_buffer("active_mask", active)
        self.name = config["name"]
        self.gamma_neg, self.gamma_pos, self.clip = config["gamma_neg"], config["gamma_pos"], config["clip"]

    def per_sample(self, logits, target):
        if logits.ndim != 2 or logits.shape != target.shape or logits.shape[1] != 119:
            raise ValueError("Expected 119 independent logits and binary labels")
        z, y = logits.float(), target.float()
        if self.name == "bce":
            terms = F.binary_cross_entropy_with_logits(z, y, reduction="none")
        elif self.name == "asl":
            positive = z.sigmoid()
            negative = (1 - positive + self.clip).clamp(max=1)
            logp = positive.clamp(min=1e-8).log()
            logn = negative.clamp(min=1e-8).log()
            probability_true = positive * y + negative * (1 - y)
            gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            # Full gradient of the stated ASL objective; focal weights are not detached.
            weight = (1 - probability_true).clamp(min=0).pow(gamma)
            terms = -(y * logp + (1 - y) * logn) * weight
        else:
            raise ValueError("Only BCE/ASL are implemented")
        return terms[:, self.active_mask].mean(dim=1)

    def forward(self, logits, target):
        return self.per_sample(logits, target).mean()


class UnsupportedClassPolicy:
    """Freeze class-exclusive rows, including against AdamW decay, without new layers."""
    def __init__(self, model, train_positive_counts):
        counts = torch.as_tensor(train_positive_counts, dtype=torch.int64)
        if counts.shape != (119,):
            raise ValueError("Expected Train-only class supports")
        self.train_positive_counts = counts.cpu().clone()
        self.active_mask = counts > 0
        self.unsupported = (~self.active_mask).nonzero(as_tuple=False).flatten()
        row_names = ("classifier.weight", "classifier.bias") if hasattr(model, "classifier") else ("attention.weight", "class_weight", "class_bias")
        params = dict(model.named_parameters())
        self.anchors = {name: params[name].detach().cpu().index_select(0, self.unsupported).clone() for name in row_names}

    @torch.no_grad()
    def restore_rows(self, model):
        params = dict(model.named_parameters())
        for name, initial in self.anchors.items():
            p = params[name]
            p.index_copy_(0, self.unsupported.to(p.device), initial.to(p.device))

    def state_dict(self):
        return dict(train_positive_counts=self.train_positive_counts, unsupported_ids=self.unsupported, anchors=self.anchors)

    def load_state_dict(self, state):
        if not torch.equal(state["train_positive_counts"], self.train_positive_counts):
            raise ValueError("Train class eligibility changed")
        self.anchors = {k: v.cpu().clone() for k,v in state["anchors"].items()}

    def metadata(self):
        return dict(output_dimension=119, unsupported_from_Train_ids=self.unsupported.tolist(),
                    inference_eligible_class_ids=self.active_mask.nonzero(as_tuple=False).flatten().tolist(),
                    training_loss_classes=int(self.active_mask.sum()),
                    policy="No loss gradient or AdamW row decay; no normal action emission for unsupported classes; raw logits retained")
