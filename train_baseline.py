"""
train_baseline.py — bare-minimum ResNet-18 baseline.

- ImageNet pretrained weights, fc replaced for 5 classes
- CrossEntropyLoss with inverse-frequency class weights (same as augmented)
- SGD lr=0.001, momentum=0.9
- No LR scheduler, no early stopping, no augmentation
- Trains for fixed epochs, saves final checkpoint

Usage:
    python train_baseline.py [--epochs 50] [--runs 3]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from config import CFG
from dataset import load_metadata, make_patient_splits, make_clf_loaders, class_weights
from utils import compute_metrics


def build_resnet18(n_classes: int = 5) -> nn.Module:
    """ImageNet-pretrained ResNet-18 with fc swapped for n_classes output."""
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def train_one_run(splits, epochs: int, run_id: int, ckpt_path: Path,
                  synth_dirs=None, aug_ratio=0.0, model_name="baseline"):
    device = CFG.device
    loaders = make_clf_loaders(splits, synth_dirs, aug_ratio)

    model = build_resnet18().to(device)
    cw = class_weights(splits["train"]).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    best_val_f1 = -1.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0

        for imgs, labels in loaders["train"]:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            opt.step()
            running_loss += loss.item()

        val_metrics = compute_metrics(model, loaders["val"], device)
        print(f"    [{model_name}] run={run_id} ep={epoch+1}/{epochs}  "
              f"loss={running_loss/len(loaders['train']):.4f}  "
              f"val_acc={val_metrics['accuracy']:.4f}  "
              f"KL3={val_metrics.get('recall_3',0):.4f}  "
              f"KL4={val_metrics.get('recall_4',0):.4f}  "
              f"({time.time()-t0:.0f}s)")

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    torch.save(model.state_dict(), ckpt_path)
    return compute_metrics(model, loaders["test"], device)


def aggregate_runs(run_results):
    keys = [k for k in run_results[0] if k != "cm"]
    return {k: (float(np.mean([r[k] for r in run_results])),
                float(np.std([r[k]  for r in run_results])))
            for k in keys}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=CFG.n_epochs_clf)
    parser.add_argument("--runs",   type=int, default=CFG.n_clf_runs)
    parser.add_argument("--joint",  type=str, default=None,
                        help="Joint group to train on: dip, pip, mcp, or None for pooled")
    args = parser.parse_args()

    CFG.makedirs()
    meta = load_metadata()

    # Filter to joint group if specified
    joint = args.joint
    if joint:
        from dataset import filter_joint
        meta = filter_joint(meta, joint)
        print(f"  Training baseline on joint group: {joint}")

    splits = make_patient_splits(meta)

    label = f"baseline_{joint}" if joint else "baseline"

    print("\n" + "="*60)
    print(f"Baseline ResNet-18 — {joint if joint else 'pooled'} — no augmentation")
    print("="*60)

    run_results = []
    for run in range(args.runs):
        print(f"\n  Run {run+1}/{args.runs}")
        ckpt = Path(CFG.ckpt_dir) / f"baseline_run{run}.pth"
        metrics = train_one_run(splits, args.epochs, run+1, ckpt, model_name=label)
        run_results.append(metrics)
        print(f"  Test → acc={metrics['accuracy']:.4f}  "
              f"macroF1={metrics['macro_f1']:.4f}  "
              f"KL3={metrics.get('recall_3',0):.4f}  "
              f"KL4={metrics.get('recall_4',0):.4f}")

    agg = aggregate_runs(run_results)
    print("\n  mean ± std across runs:")
    for k, (m, s) in agg.items():
        print(f"    {k:20s}: {m:.4f} ± {s:.4f}")

    out = Path(CFG.results_dir) / f"{label}_results.json"
    with open(out, "w") as f:
        json.dump({"runs": run_results,
                   "aggregate": {k: {"mean": m, "std": s}
                                  for k, (m, s) in agg.items()}}, f, indent=2)
    print(f"\n  Saved to {out}")


# Exported for train_augmented.py
build_classifier = build_resnet18

if __name__ == "__main__":
    main()