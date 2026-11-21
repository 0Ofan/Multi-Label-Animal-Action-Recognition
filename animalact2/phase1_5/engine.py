"""One training/evaluation implementation for all three unchanged Proposal models."""
from __future__ import annotations
import csv
from itertools import islice
import json
from pathlib import Path
import time
import torch
from .data import ReleaseClipDataset, make_loader, training_class_support, move_batch_to_device
from .models import build_model
from .config import validate_config, config_hash, cohort_hash, development_fingerprints
from .losses import MultilabelLoss, UnsupportedClassPolicy
from .schedule import LearningRatePlan, EarlyStopping
from .evaluation import evaluate_outputs, select_validation_threshold, save_evaluation
from .checkpoint import make_checkpoint, save_checkpoint, load_checkpoint, owned
from .safety import OUT, write_json, sha


def make_optimizer(model, config):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config["optimizer"]["lr"], weight_decay=config["optimizer"]["weight_decay"])


def collect_batch(batch, output, logits, labels, ids, parents):
    if not torch.isfinite(output).all():
        raise FloatingPointError("Nonfinite logits")
    logits.append(output.detach().cpu())
    labels.append(batch["target"].detach().cpu())
    ids.extend(batch["clip_ids"])
    parents.extend(batch["parent_video_ids"])


def train_one_epoch(model, loader, optimizer, criterion, policy, device, *, accumulation_steps, grad_clip_norm, max_batches=None):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    all_z, all_y, ids, parents = [], [], [], []
    objective_sum, seen, window_samples, updates = 0., 0, 0, 0
    limit = len(loader) if max_batches is None else min(max_batches, len(loader))
    if limit < 1:
        raise ValueError("Empty training epoch")
    for j, batch in enumerate(islice(loader, limit)):
        if any(s != "train" for s in batch["splits"]):
            raise ValueError("Only Train may contribute gradients")
        batch = move_batch_to_device(batch, device)
        output = model(batch["frames"], batch["lengths"], batch["valid_frame_mask"])
        values = criterion.per_sample(output, batch["target"])
        if not torch.isfinite(values).all():
            raise FloatingPointError("Nonfinite training objective")
        values.sum().backward()
        b = len(output)
        seen += b
        window_samples += b
        objective_sum += float(values.detach().sum())
        collect_batch(batch, output, all_z, all_y, ids, parents)
        if (j+1) % accumulation_steps == 0 or j+1 == limit:
            # Average by actual samples, including the final short microbatch/window.
            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    p.grad.div_(window_samples)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip_norm, error_if_nonfinite=True)
            optimizer.step()
            policy.restore_rows(model)
            optimizer.zero_grad(set_to_none=True)
            window_samples = 0
            updates += 1
    return dict(logits=torch.cat(all_z), target=torch.cat(all_y), clip_ids=ids, parent_video_ids=parents,
                objective=objective_sum/seen, examples=seen, optimizer_updates=updates, metrics_parameter_state="online_preupdate_mixed_within_epoch")


@torch.no_grad()
def evaluate_loader(model, loader, criterion, device, *, max_batches=None):
    model.eval()
    all_z, all_y, ids, parents = [], [], [], []
    objective_sum, seen = 0., 0
    limit = len(loader) if max_batches is None else min(max_batches, len(loader))
    if limit < 1:
        raise ValueError("Empty validation loader")
    for j, batch in enumerate(islice(loader, limit)):
        if any(s != "valid" for s in batch["splits"]):
            raise ValueError("Model selection may only read Valid")
        batch = move_batch_to_device(batch, device)
        output = model(batch["frames"], batch["lengths"], batch["valid_frame_mask"])
        values = criterion.per_sample(output, batch["target"])
        objective_sum += float(values.sum())
        seen += len(output)
        collect_batch(batch, output, all_z, all_y, ids, parents)
    return dict(logits=torch.cat(all_z), target=torch.cat(all_y), clip_ids=ids, parent_video_ids=parents, objective=objective_sum/seen, examples=seen)


def summarize(output, policy, *, threshold=.5, full=True):
    result = evaluate_outputs(output["logits"], output["target"], output["parent_video_ids"], policy,
                              threshold=threshold, require_all_main_classes=full)
    result["training_objective"] = output["objective"]
    return result


def write_history(directory, history):
    write_json(directory / "history.json", history)
    (directory / "history.jsonl").write_text("".join(json.dumps(row, allow_nan=False)+"\n" for row in history), encoding="utf-8")
    with (directory / "history.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


class Trainer:
    def __init__(self, config, output_dir, *, train_dataset=None, valid_dataset=None, diagnostic=False, train_positive_counts=None):
        self.config = validate_config(config)
        self.output_dir = owned(output_dir)
        self.diagnostic = diagnostic
        self.train_dataset = train_dataset if train_dataset is not None else ReleaseClipDataset("train")
        self.valid_dataset = valid_dataset if valid_dataset is not None else ReleaseClipDataset("valid")
        if self.train_dataset.split != "train" or self.valid_dataset.split != "valid":
            raise ValueError("Fixed Train/Valid only")
        if not diagnostic and (len(self.train_dataset) != 4667 or len(self.valid_dataset) != 979):
            raise ValueError("Formal training must use all released Train/Valid clips")
        self.device = torch.device(config["device"])
        self.model = build_model(config).to(self.device)
        if train_positive_counts is not None and not diagnostic:
            raise ValueError("Formal eligibility must be derived directly from the full Train dataset")
        counts = train_positive_counts if train_positive_counts is not None else training_class_support(self.train_dataset)
        self.policy = UnsupportedClassPolicy(self.model, counts)
        if not diagnostic and self.policy.unsupported.tolist() != [23]:
            raise ValueError("Unexpected Train eligibility; inspect data before training")
        self.criterion = MultilabelLoss(config["loss"], self.policy.active_mask).to(self.device)
        self.optimizer = make_optimizer(self.model, config)
        self.schedule = LearningRatePlan(self.optimizer, config)
        self.early = EarlyStopping(config["training"]["patience"], config["training"]["min_delta"])
        self.train_loader = make_loader(self.train_dataset, config, training=True)
        self.valid_loader = make_loader(self.valid_dataset, config, training=False)

    def checkpoint(self, epoch, history):
        return make_checkpoint(self.model, self.optimizer, self.schedule, self.early, self.policy, self.config, epoch=epoch, history=history, diagnostic=self.diagnostic)

    def run(self, *, resume=None, execute_training=False):
        if self.diagnostic or not execute_training:
            raise RuntimeError("Formal run requires an explicit execution flag; diagnostics use bounded epoch functions")
        audit = json.loads((OUT / "data_audit/summary.json").read_text(encoding="utf-8"))
        if audit["status"] != "passed" or audit["clips_checked"] != 5646:
            raise RuntimeError("Full Train/Valid data audit must pass first")
        if resume is None:
            if self.output_dir.exists():
                raise FileExistsError("Never overwrite an existing experiment")
            self.output_dir.mkdir(parents=True)
            history, start_epoch = [], 1
        else:
            if owned(resume).parent != self.output_dir:
                raise ValueError("Resume checkpoint belongs to another experiment")
            state = load_checkpoint(resume, self.model, self.config, optimizer=self.optimizer, schedule=self.schedule,
                                    early=self.early, policy=self.policy, restore_random_state=True, expected_diagnostic=False)
            history, start_epoch = state["history"], state["epoch"]+1
            if history:
                write_history(self.output_dir, history)
            # Repair interruption after last.pt was committed but before best.pt.
            if self.early.best_epoch == state["epoch"]:
                save_checkpoint(self.output_dir / "best.pt", state)
        write_json(self.output_dir / "config.json", self.config)
        write_json(self.output_dir / "provenance.json", dict(config_SHA256=config_hash(self.config), cohort_SHA256=cohort_hash(self.config),
                   data_fingerprints=development_fingerprints(), class_policy=self.policy.metadata(), Test_access=False, model_definition="unchanged phase1_5_1"))
        for epoch in range(start_epoch, self.config["training"]["max_epochs"]+1):
            if self.early.bad_epochs >= self.early.patience:
                break
            begin = time.perf_counter()
            self.train_loader.sampler.set_epoch(epoch)
            lr = self.schedule.apply_epoch(epoch)
            train = train_one_epoch(self.model, self.train_loader, self.optimizer, self.criterion, self.policy, self.device,
                                    accumulation_steps=self.config["data"]["accumulation_steps"], grad_clip_norm=self.config["optimizer"]["grad_clip_norm"])
            valid = evaluate_loader(self.model, self.valid_loader, self.criterion, self.device)
            tm, vm = summarize(train, self.policy), summarize(valid, self.policy)
            improved, stop = self.early.update(vm["main_mAP"], epoch)
            record = dict(epoch=epoch, lr=lr, train_online_objective=train["objective"], valid_objective=valid["objective"],
                          train_online_mAP=tm["main_mAP"], valid_mAP=vm["main_mAP"], train_online_Macro_F1_at_05=tm["main_macro_f1"], valid_Macro_F1_at_05=vm["main_macro_f1"],
                          train_online_BCE119=tm["BCE119"], valid_BCE119=vm["BCE119"], optimizer_updates=train["optimizer_updates"],
                          elapsed_seconds=time.perf_counter()-begin, bad_epochs=self.early.bad_epochs, best_epoch=self.early.best_epoch)
            history.append(record)
            state = self.checkpoint(epoch, history)
            save_checkpoint(self.output_dir / "last.pt", state)
            if improved:
                save_checkpoint(self.output_dir / "best.pt", state)
            write_history(self.output_dir, history)
            print(json.dumps(record, allow_nan=False), flush=True)
            if stop:
                break
        load_checkpoint(self.output_dir / "best.pt", self.model, self.config, policy=self.policy, expected_diagnostic=False)
        final_valid = evaluate_loader(self.model, self.valid_loader, self.criterion, self.device)
        threshold = select_validation_threshold(final_valid["logits"], final_valid["target"], split="valid")
        metrics = summarize(final_valid, self.policy, threshold=threshold["threshold"])
        save_evaluation(self.output_dir / "best_valid", metrics)
        write_json(self.output_dir / "threshold_selection.json", threshold)
        torch.save(dict(logits=final_valid["logits"], target=final_valid["target"], clip_ids=final_valid["clip_ids"], parent_video_ids=final_valid["parent_video_ids"],
                        threshold=threshold["threshold"], class_policy=self.policy.metadata(), data_fingerprints=development_fingerprints(), checkpoint_SHA256=sha(self.output_dir/"best.pt")), self.output_dir/"valid_predictions.pt")
        write_json(self.output_dir / "results.json", dict(status="completed", config=self.config, best_epoch=self.early.best_epoch,
                   best_valid=metrics, threshold_selection=threshold, completed_epochs=len(history), Test_evaluated=False,
                   checkpoint="best.pt", history="history.csv", training_seconds=sum(r["elapsed_seconds"] for r in history)))
        return metrics
