"""Allowlisted Train/Valid experiment reader. No Test, legacy 1.0 or inference."""
import csv
from .registry import ROOT,read_json

def csv_rows(path):
    with path.open(encoding="utf-8-sig",newline="") as f: return list(csv.DictReader(f))

def run_path(model,seed):
    if model not in ("model1","model2","model3") or seed not in (42,123,2026): raise ValueError("Invalid experiment")
    return ROOT / (f"animalact2/phase2_4/runs/{model}_bce_seed42" if seed==42 else f"animalact2/phase2_5/runs/{model}_seed{seed}")

def evaluation(kind="summary",model="model3",seed=42):
    base=ROOT/"animalact2/phase2_5"
    mapping={"summary":"summary_mean_std.csv","runs":"all_seed_results.csv","backbones":"ResNet_comparison.csv",
             "per_class":"per_class_AP.csv","stability":"per_class_AP_stability.csv"}
    if kind in mapping:
        path=base/mapping[kind]; data=csv_rows(path)
    elif kind in ("history","confusion"):
        path=run_path(model,seed)/("history.csv" if kind=="history" else "best_valid/confusion_main31.csv")
        data=csv_rows(path)
    else: raise ValueError("Unknown evaluation dataset")
    return dict(split="Train/Valid" if kind in ("runs","per_class","stability","summary","history") else "Valid",
        source=str(path.relative_to(ROOT)),rows=data,Test_available=False,
        notes="ResNet18 comparison has Seed42 only; VideoMAE mean/std has N=3, ddof=1. Train history is online, fixed-checkpoint Train metrics are in runs.")
