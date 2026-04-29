"""
utils.py — shared evaluation helpers.
"""
from __future__ import annotations
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, confusion_matrix, classification_report
)


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_metrics(model, loader, device, n_classes: int = 5) -> dict:
    model.eval()
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        preds = model(imgs).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None,
                                    labels=list(range(n_classes)), zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    result = {
        "accuracy":  float(acc),
        "macro_f1":  float(macro_f1),
        "cm":        cm.tolist(),
    }
    for i in range(n_classes):
        result[f"recall_{i}"] = float(per_class_recall[i])

    return result


def print_metrics(metrics: dict, title: str = ""):
    if title:
        print(f"\n  {title}")
    print(f"    Accuracy : {metrics['accuracy']:.4f}")
    print(f"    Macro F1 : {metrics['macro_f1']:.4f}")
    for g in range(5):
        k = f"recall_{g}"
        if k in metrics:
            marker = " ◄" if g in (3, 4) else ""
            print(f"    KL{g} Recall: {metrics[k]:.4f}{marker}")


# ──────────────────────────────────────────────────────────────────────────────
# Early stopping
# ──────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = -float("inf")
        self.counter = 0

    def __call__(self, score: float) -> bool:
        if score > self.best + self.min_delta:
            self.best = score
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


# ──────────────────────────────────────────────────────────────────────────────
# FID (requires clean-fid)
# ──────────────────────────────────────────────────────────────────────────────

def compute_fid(real_dir: str, fake_dir: str, device: str = "cuda") -> float:
    """Compute FID between real and generated images using clean-fid."""
    try:
        from cleanfid import fid
        score = fid.compute_fid(real_dir, fake_dir, device=device,
                                 num_workers=4, verbose=False)
        return float(score)
    except ImportError:
        print("  [WARN] clean-fid not installed. FID skipped.")
        return float("nan")
    except Exception as e:
        print(f"  [WARN] FID computation failed: {e}")
        return float("nan")


# ──────────────────────────────────────────────────────────────────────────────
# Label faithfulness (frozen reference classifier)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def label_faithfulness(
    synth_dir: str,
    ref_clf_ckpt: str,
    target_grade: int,
    device: torch.device,
    n_classes: int = 5,
) -> float:
    """Fraction of synthetic images that a frozen reference classifier assigns to target_grade."""
    import os
    from pathlib import Path
    from PIL import Image
    from torchvision import models, transforms
    import torch.nn as nn

    ckpt_path = Path(ref_clf_ckpt)
    if not ckpt_path.exists():
        print(f"  [WARN] Reference classifier not found at {ref_clf_ckpt}. Skipping faithfulness.")
        return float("nan")

    # Build and load reference classifier
    clf = models.resnet18(weights=None)
    clf.fc = nn.Linear(clf.fc.in_features, n_classes)
    clf.load_state_dict(torch.load(ckpt_path, map_location=device))
    clf = clf.to(device).eval()

    tf = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    synth_path = Path(synth_dir)
    img_paths = list(synth_path.glob("*.png")) + list(synth_path.glob("*.jpg"))
    if not img_paths:
        return float("nan")

    correct = total = 0
    batch_imgs = []
    for path in img_paths:
        img = Image.open(path).convert("L")
        batch_imgs.append(tf(img))
        if len(batch_imgs) == 64:
            batch = torch.stack(batch_imgs).to(device)
            preds = clf(batch).argmax(dim=1)
            correct += (preds == target_grade).sum().item()
            total += len(batch_imgs)
            batch_imgs = []

    if batch_imgs:
        batch = torch.stack(batch_imgs).to(device)
        preds = clf(batch).argmax(dim=1)
        correct += (preds == target_grade).sum().item()
        total += len(batch_imgs)

    return correct / total if total > 0 else float("nan")


# ──────────────────────────────────────────────────────────────────────────────
# Seed
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
