"""Reuse Phase1.5.1 inputs; add deterministic epoch shuffle and shared loaders."""
from __future__ import annotations
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from animalact2.phase1_5_1.dataset import ReleaseClipDataset, RELEASE_ID
from animalact2.phase1_5_1.collate import collate_clips, move_batch_to_device


class EpochShuffleSampler(Sampler):
    def __init__(self, dataset, seed):
        self.dataset, self.seed, self.epoch = dataset, int(seed), 1

    def set_epoch(self, epoch):
        if epoch < 1:
            raise ValueError("Epoch numbering starts at 1")
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch * 1000003)
        return iter(torch.randperm(len(self.dataset), generator=generator).tolist())

    def __len__(self):
        return len(self.dataset)


def seed_worker(worker_id):
    from .safety import enable_guard
    enable_guard()
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


def make_loader(dataset, config, *, training):
    if dataset.split != ("train" if training else "valid"):
        raise ValueError("Training loader must use Train, validation loader must use Valid")
    data = config["data"]
    sampler = EpochShuffleSampler(dataset, config["seed"]) if training else None
    kwargs = dict(dataset=dataset, batch_size=data["batch_size"] if training else data["valid_batch_size"],
                  sampler=sampler, shuffle=False, drop_last=False, collate_fn=collate_clips,
                  num_workers=data["num_workers"], pin_memory=data["pin_memory"],
                  worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(config["seed"] + (0 if training else 17)))
    if data["num_workers"] > 0:
        kwargs.update(prefetch_factor=data["prefetch_factor"], persistent_workers=True)
    return DataLoader(**kwargs)


def training_class_support(dataset):
    if dataset.split != "train":
        raise ValueError("Class eligibility must be computed from Train only")
    target = torch.tensor([r["target_119"] for r in dataset.records], dtype=torch.int64)
    return target.sum(dim=0)


__all__ = ["ReleaseClipDataset", "RELEASE_ID", "collate_clips", "move_batch_to_device", "make_loader", "EpochShuffleSampler", "training_class_support"]
