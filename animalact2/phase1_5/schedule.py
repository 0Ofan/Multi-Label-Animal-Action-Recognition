"""Serializable epoch schedule and validation-only early stopping."""
import math


class LearningRatePlan:
    def __init__(self, optimizer, config):
        self.optimizer = optimizer
        self.spec = dict(config["scheduler"])
        self.max_epochs = config["training"]["max_epochs"]
        self.base_lr = config["optimizer"]["lr"]
        self.last_epoch = 0

    def factor(self, epoch):
        if not 1 <= epoch <= self.max_epochs:
            raise ValueError("Epoch outside fixed budget")
        if self.spec["name"] == "fixed":
            return 1.
        warmup, floor = self.spec["warmup_epochs"], self.spec["min_lr_ratio"]
        if warmup and epoch <= warmup:
            return epoch / warmup
        progress = (epoch-warmup) / (self.max_epochs-warmup)
        return floor + (1-floor) * .5 * (1+math.cos(math.pi*progress))

    def apply_epoch(self, epoch):
        factor = self.factor(epoch)
        for group in self.optimizer.param_groups:
            group["lr"] = self.base_lr * factor
        self.last_epoch = epoch
        return self.base_lr * factor

    def state_dict(self):
        return dict(spec=self.spec, max_epochs=self.max_epochs, base_lr=self.base_lr, last_epoch=self.last_epoch)

    def load_state_dict(self, state):
        if any(state[k] != getattr(self, k) for k in ("spec", "max_epochs", "base_lr")):
            raise ValueError("Learning rate schedule changed on resume")
        self.last_epoch = state["last_epoch"]
        if self.last_epoch:
            self.apply_epoch(self.last_epoch)


class EarlyStopping:
    def __init__(self, patience, min_delta=0.):
        self.patience, self.min_delta = patience, min_delta
        self.best_score, self.reference_score = None, None
        self.best_epoch, self.last_epoch, self.bad_epochs = 0, 0, 0

    def update(self, score, epoch):
        if score is None or not math.isfinite(score) or epoch <= self.last_epoch:
            raise ValueError("Need finite Valid mAP and increasing epoch")
        improved = self.best_score is None or score > self.best_score
        if improved:
            self.best_score, self.best_epoch = score, epoch
        if self.reference_score is None or score > self.reference_score + self.min_delta:
            self.reference_score, self.bad_epochs = score, 0
        else:
            self.bad_epochs += 1
        self.last_epoch = epoch
        return improved, self.bad_epochs >= self.patience

    def state_dict(self):
        return dict(self.__dict__)

    def load_state_dict(self, state):
        if state["patience"] != self.patience or state["min_delta"] != self.min_delta:
            raise ValueError("Early stopping settings changed")
        self.__dict__.update(state)
