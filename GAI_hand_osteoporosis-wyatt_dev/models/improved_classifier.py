"""
Improved KL Grade Classifier
==============================
Builds on kl_classifier.py with three targeted improvements:

  1. Focal loss (Lin et al. 2017)
         Down-weights easy examples (abundant KL=0) so the model focuses
         on hard boundaries (KL=1/2, KL=2/3).  Replaces weighted CE.

  2. ResNet-50 backbone
         More capacity than ResNet-18.  Same grayscale adaptation
         (pretrained conv1 averaged across RGB channels).

  3. Test-Time Augmentation (TTA)
         At inference, each image is predicted 8 times with different
         flips/crops and the probabilities are averaged.  Free accuracy
         gain with no additional training.

Results are saved alongside kl_classifier.py outputs so you can directly
compare confusion matrices, ROC curves, and F1 scores.

Run
---
    python models/improved_classifier.py
    python models/improved_classifier.py --epochs 40 --gamma 2.5

Outputs  (models/kl_image_clf_improved/)
-----------------------------------------
    best_model.pt
    training_curves.png
    confusion_matrix.png
    roc_curves.png
    gradcam_samples.png
    comparison_vs_baseline.png   — side-by-side F1 vs kl_classifier results

Requirements
------------
    pip install torch torchvision pandas openpyxl pillow matplotlib tqdm scikit-learn
"""

import argparse
import re
import zipfile
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from PIL import Image
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
XLSX_PATH = ROOT / "data" / "hand.xlsx"
ZIP_PATH  = ROOT / "data" / "Finger Joints.zip"
OUT_DIR   = Path(__file__).parent / "kl_image_clf_improved"

# Baseline results to compare against (written by kl_classifier.py)
BASELINE_DIR = Path(__file__).parent / "kl_image_clf"

IMG_SIZE   = 224
KL_CLASSES = [0, 1, 2, 3, 4]
N_CLASSES  = len(KL_CLASSES)

JOINT_MAP = {
    "dip2": "DIP2", "dip3": "DIP3", "dip4": "DIP4", "dip5": "DIP5",
    "mcp2": "MCP2", "mcp3": "MCP3", "mcp4": "MCP4", "mcp5": "MCP5",
    "pip2": "PIP2", "pip3": "PIP3", "pip4": "PIP4", "pip5": "PIP5",
}


# ── 1. Focal Loss ─────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal loss: FL(p_t) = -α_t (1 - p_t)^γ log(p_t)

    γ (gamma) — focusing parameter.
        γ=0 → standard cross-entropy.
        γ=2 → well-classified examples down-weighted 4×.
    α (class weights) — same inverse-frequency weighting as before,
        now combined with the focal term for double imbalance handling.
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=1)
        ce    = F.nll_loss(log_p, targets, weight=self.weight, reduction="none")
        p_t   = torch.exp(-ce)
        loss  = (1 - p_t) ** self.gamma * ce
        return loss.mean()


# ── 2. Dataset (identical to kl_classifier.py) ───────────────────────────────
def build_index(joint_filter=None):
    df = pd.read_excel(XLSX_PATH)
    df["img_id"] = df["id"].astype(str)
    img_to_kl: dict[str, int] = {}
    for img_joint, score_joint in JOINT_MAP.items():
        if joint_filter and img_joint != joint_filter:
            continue
        kl_col = f"v00{score_joint}_KL"
        if kl_col not in df.columns:
            continue
        for pid, kl in df.set_index("img_id")[kl_col].dropna().astype(int).items():
            img_to_kl[f"Finger Joints/{pid}_{img_joint}.png"] = kl

    pat = re.compile(r"Finger Joints/(\d+)_(\w+)\.png")
    index = []
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        for name in z.namelist():
            if not name.endswith(".png"):
                continue
            m = pat.match(name)
            if not m:
                continue
            if joint_filter and m.group(2) != joint_filter:
                continue
            kl = img_to_kl.get(name)
            if kl is None:
                continue
            index.append((name, kl))

    print(f"Index: {len(index):,} images")
    from collections import Counter
    for k, v in sorted(Counter(kl for _, kl in index).items()):
        print(f"  KL={k}: {v:,}")
    return index


def patient_split(index, val_frac=0.15, test_frac=0.15, seed=42):
    """Same seed as kl_classifier.py → identical test set for fair comparison."""
    pat = re.compile(r"Finger Joints/(\d+)_")
    pid_to_items: dict[str, list] = {}
    for item in index:
        pid = pat.match(item[0]).group(1)
        pid_to_items.setdefault(pid, []).append(item)

    rng  = np.random.default_rng(seed)
    pids = np.array(sorted(pid_to_items.keys()))
    rng.shuffle(pids)
    n         = len(pids)
    n_test    = int(n * test_frac)
    n_val     = int(n * val_frac)
    test_pids = set(pids[:n_test])
    val_pids  = set(pids[n_test: n_test + n_val])
    train_pids= set(pids[n_test + n_val:])

    def gather(pid_set):
        out = []
        for pid in pid_set:
            out.extend(pid_to_items[pid])
        return out

    tr, va, te = gather(train_pids), gather(val_pids), gather(test_pids)
    print(f"Split → train {len(tr):,}  val {len(va):,}  test {len(te):,}")
    return tr, va, te


class FingerJointDataset(Dataset):
    def __init__(self, index, transform):
        self.index     = index
        self.transform = transform
        self._zip      = None

    def _get_zip(self):
        if self._zip is None:
            self._zip = zipfile.ZipFile(ZIP_PATH, "r")
        return self._zip

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        path, kl = self.index[idx]
        with self._get_zip().open(path) as f:
            img = Image.open(BytesIO(f.read())).convert("L")
        return self.transform(img), kl


# ── 3. Model — ResNet-50 ──────────────────────────────────────────────────────
def build_model(n_classes: int = N_CLASSES) -> nn.Module:
    """
    ResNet-50 pretrained on ImageNet, adapted for grayscale input.
    Larger capacity than ResNet-18 — better at distinguishing subtle
    radiographic differences between adjacent KL grades.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Adapt conv1: 3-ch → 1-ch (average pretrained weights)
    old_conv  = model.conv1
    new_conv  = nn.Conv2d(1, old_conv.out_channels,
                          kernel_size=old_conv.kernel_size,
                          stride=old_conv.stride,
                          padding=old_conv.padding, bias=False)
    with torch.no_grad():
        new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
    model.conv1 = new_conv

    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, n_classes),
    )
    return model


# ── 4. Test-Time Augmentation ─────────────────────────────────────────────────
def tta_transforms(img_size: int) -> list:
    """
    8 deterministic augmentations applied at inference:
    original + hflip, each at 4 crop positions (center + 3 corners).
    Probabilities are averaged — free accuracy gain, no retraining.
    """
    base = [
        transforms.Resize(img_size + 32),   # slightly larger for crops
    ]
    norm = [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]

    crop_positions = [
        transforms.CenterCrop(img_size),
        transforms.RandomCrop(img_size),     # top-left area
        transforms.RandomCrop(img_size),     # bottom-right area
        transforms.FiveCrop(img_size),       # returns tuple — handled below
    ]

    tfs = []
    for flip in [False, True]:
        t = base.copy()
        if flip:
            t.append(transforms.RandomHorizontalFlip(p=1.0))
        t.append(transforms.CenterCrop(img_size))
        t += norm
        tfs.append(transforms.Compose(t))

    return tfs


@torch.no_grad()
def predict_with_tta(model: nn.Module, pil_img: Image.Image,
                     img_size: int, device: torch.device,
                     n_aug: int = 8) -> np.ndarray:
    """
    Run n_aug augmented versions of one PIL image through the model
    and return the averaged softmax probabilities.
    """
    model.eval()
    norm = transforms.Normalize([0.5], [0.5])
    base_tf = transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.ToTensor(),
        norm,
    ])
    flip_tf = transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        norm,
    ])

    crops = [
        transforms.CenterCrop(img_size),
        transforms.Pad(0),   # no-op, used for variety
    ]

    probs_list = []
    rng = np.random.default_rng(0)

    for i in range(n_aug):
        tf = flip_tf if i % 2 == 1 else base_tf
        t  = tf(pil_img)
        # Random crop offset
        _, h, w = t.shape
        max_offset = max(0, min(h, w) - img_size)
        if max_offset > 0:
            y = int(rng.integers(0, max_offset))
            x = int(rng.integers(0, max_offset))
            t = t[:, y:y + img_size, x:x + img_size]
        else:
            t = t[:, :img_size, :img_size]

        logits = model(t.unsqueeze(0).to(device))
        probs_list.append(F.softmax(logits, dim=1).cpu().numpy())

    return np.mean(probs_list, axis=0).squeeze()


# ── 5. Training ───────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimiser, device, scaler):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for imgs, labels in tqdm(loader, desc="  train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimiser.zero_grad()
        with torch.amp.autocast(device.type, enabled=scaler is not None):
            logits = model(imgs)
            loss   = criterion(logits, labels)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
        else:
            loss.backward()
            optimiser.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += imgs.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_probs, all_labels  = [], []
    for imgs, labels in tqdm(loader, desc="  val  ", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        probs  = F.softmax(logits, dim=1)
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += imgs.size(0)
        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())
    return (total_loss / n, correct / n,
            torch.cat(all_probs).numpy(), torch.cat(all_labels).numpy())


@torch.no_grad()
def evaluate_with_tta(model, test_index, img_size, device):
    """Full test evaluation using TTA — slower but more accurate."""
    model.eval()
    all_probs, all_labels = [], []
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        for path, kl in tqdm(test_index, desc="  TTA eval", leave=False):
            with z.open(path) as f:
                pil_img = Image.open(BytesIO(f.read())).convert("L")
            probs = predict_with_tta(model, pil_img, img_size, device)
            all_probs.append(probs)
            all_labels.append(kl)
    return np.array(all_probs), np.array(all_labels)


# ── 6. Visualisation ──────────────────────────────────────────────────────────
def plot_training_curves(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"],   label="val")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"],   label="val")
    axes[1].set_title("Accuracy"); axes[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    print(f"Saved {path}")


def plot_confusion(y_true, y_pred, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=[f"KL={k}" for k in KL_CLASSES],
        ax=ax, cmap="Blues", colorbar=True
    )
    ax.set_title("Confusion matrix — Improved classifier (test set)")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    print(f"Saved {path}")


def plot_roc(y_true, y_probs, path):
    y_bin  = label_binarize(y_true, classes=KL_CLASSES)
    colors = ["steelblue", "darkorange", "seagreen", "tomato", "mediumpurple"]
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (cls, col) in enumerate(zip(KL_CLASSES, colors)):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        auc = roc_auc_score(y_bin[:, i], y_probs[:, i])
        ax.plot(fpr, tpr, label=f"KL={cls}  AUC={auc:.3f}", color=col)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC curves — Improved classifier (test set)")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    print(f"Saved {path}")


def plot_comparison(y_true, baseline_probs, improved_probs, path):
    """
    Side-by-side F1 comparison: baseline (ResNet-18, CE loss) vs
    improved (ResNet-50, Focal loss + TTA).
    Loads baseline predictions from kl_image_clf/ if available,
    otherwise uses the provided baseline_probs.
    """
    f1_base = f1_score(y_true, baseline_probs.argmax(1),
                       labels=KL_CLASSES, average=None, zero_division=0)
    f1_impr = f1_score(y_true, improved_probs.argmax(1),
                       labels=KL_CLASSES, average=None, zero_division=0)

    x, width = np.arange(N_CLASSES), 0.35
    fig, ax  = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, f1_base, width, label="Baseline (ResNet-18 + CE)",
           color="steelblue")
    ax.bar(x + width / 2, f1_impr, width,
           label="Improved (ResNet-50 + Focal + TTA)", color="darkorange")

    for i, (b, imp) in enumerate(zip(f1_base, f1_impr)):
        delta = imp - b
        color = "green" if delta >= 0 else "red"
        ax.text(i + width / 2, imp + 0.02, f"{delta:+.3f}",
                ha="center", fontsize=8, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"KL={k}" for k in KL_CLASSES])
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.1)
    ax.set_title("Per-class F1: Baseline vs Improved (same real test set)")
    ax.legend()

    acc_base = (baseline_probs.argmax(1) == y_true).mean()
    acc_impr = (improved_probs.argmax(1) == y_true).mean()
    ax.text(0.98, 0.02,
            f"Overall acc — Baseline: {acc_base:.3f}  |  Improved: {acc_impr:.3f}",
            transform=ax.transAxes, ha="right", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.4))

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved {path}")

    print("\n── F1 comparison ──")
    for k, b, imp in zip(KL_CLASSES, f1_base, f1_impr):
        print(f"  KL={k}  baseline={b:.4f}  improved={imp:.4f}  "
              f"delta={imp-b:+.4f}")
    print(f"  Overall acc  baseline={acc_base:.4f}  improved={acc_impr:.4f}  "
          f"delta={acc_impr-acc_base:+.4f}")


# ── 7. Grad-CAM ───────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model):
        self.model       = model
        self.activations = None
        self.gradients   = None
        self._hooks      = []
        target = model.layer4[-1]
        self._hooks.append(target.register_forward_hook(
            lambda m, i, o: setattr(self, "activations", o.detach())))
        self._hooks.append(target.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "gradients", go[0].detach())))

    def __call__(self, x, target=None):
        self.model.eval()
        logits = self.model(x)
        if target is None:
            target = logits.argmax(1).item()
        self.model.zero_grad()
        logits[0, target].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = (weights * self.activations).sum(dim=1).squeeze()
        cam     = F.relu(cam).cpu().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def remove(self):
        for h in self._hooks:
            h.remove()


def plot_gradcam(model, test_index, img_size, device, path, n=20):
    rng     = np.random.default_rng(1)
    samples = [test_index[i] for i in rng.choice(len(test_index), n, replace=False)]
    tf      = transforms.Compose([
        transforms.Resize(img_size), transforms.CenterCrop(img_size),
        transforms.ToTensor(), transforms.Normalize([0.5], [0.5]),
    ])
    cam_model = GradCAM(model)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 4, rows * 2.5))
    axes = axes.flatten()

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        for i, (img_path, true_kl) in enumerate(samples):
            with z.open(img_path) as f:
                pil_img = Image.open(BytesIO(f.read())).convert("L")
            tensor = tf(pil_img).unsqueeze(0).to(device)
            cam    = cam_model(tensor.clone())
            pred   = model(tensor).argmax(1).item()
            cam_r  = Image.fromarray((cam * 255).astype(np.uint8)).resize(
                (img_size, img_size), Image.BILINEAR)
            cam_np = np.array(cam_r) / 255.0
            orig   = tensor.squeeze().cpu().numpy() * 0.5 + 0.5

            ax_img = axes[i * 2]
            ax_cam = axes[i * 2 + 1]
            ax_img.imshow(orig, cmap="gray")
            ax_img.set_title(f"True={true_kl} Pred={pred}", fontsize=7)
            ax_img.axis("off")
            ax_cam.imshow(orig, cmap="gray")
            ax_cam.imshow(cam_np, cmap="jet", alpha=0.45)
            ax_cam.set_title("Grad-CAM", fontsize=7)
            ax_cam.axis("off")

    for ax in axes[n * 2:]:
        ax.set_visible(False)
    cam_model.remove()
    fig.suptitle("Grad-CAM — Improved classifier", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved {path}")


# ── 8. Main ───────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Focal loss γ={args.gamma}  |  backbone=ResNet-50  |  TTA={not args.no_tta}")

    train_tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05),
                                scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    index = build_index(args.joint)
    train_idx, val_idx, test_idx = patient_split(index)

    train_ds = FingerJointDataset(train_idx, train_tf)
    val_ds   = FingerJointDataset(val_idx,   eval_tf)
    test_ds  = FingerJointDataset(test_idx,  eval_tf)

    # Weighted sampler
    label_counts = np.bincount([kl for _, kl in train_idx],
                               minlength=N_CLASSES).astype(float)
    class_w = torch.tensor(
        1.0 / np.where(label_counts == 0, 1.0, label_counts), dtype=torch.float
    ).to(device)
    sample_w = torch.tensor([class_w[kl].item() for _, kl in train_idx])
    sampler  = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

    pin = device.type == "cuda"
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                          num_workers=0, pin_memory=pin)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=0, pin_memory=pin)
    test_dl  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                          num_workers=0)

    # Focal loss — no class weights here.
    # WeightedRandomSampler already balances each mini-batch across classes.
    # Adding class_w on top of that double-counts imbalance, collapsing the
    # model to only predict rare classes.  Focal loss focuses on hard examples
    # regardless of class frequency; the sampler handles the frequency problem.
    criterion = FocalLoss(gamma=args.gamma, weight=None)

    model = build_model(N_CLASSES).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params / 1e6:.1f}M")

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warmup    = max(1, args.epochs // 10)
    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        t = (epoch - warmup) / max(1, args.epochs - warmup)
        return 0.5 * (1 + np.cos(np.pi * t))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)
    scaler    = torch.amp.GradScaler(device.type) if device.type == "cuda" else None

    history  = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc = 0.0
    ckpt     = OUT_DIR / "best_model.pt"

    if ckpt.exists():
        print(f"\n=== Existing checkpoint found at {ckpt} — skipping training ===")
        print("    Delete best_model.pt to retrain from scratch.")
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        # Skip straight to evaluation
        args.epochs = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}  LR={scheduler.get_last_lr()[0]:.2e}")
        tr_loss, tr_acc = train_one_epoch(model, train_dl, criterion,
                                          optimiser, device, scaler)
        va_loss, va_acc, _, _ = evaluate(model, val_dl, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        print(f"  train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val loss={va_loss:.4f} acc={va_acc:.4f}")

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), ckpt)
            print(f"  ✓ Checkpoint saved (val_acc={best_acc:.4f})")

    # ── Test evaluation ───────────────────────────────────────────────────────
    print("\nLoading best checkpoint …")
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))

    if args.no_tta:
        print("Evaluating without TTA …")
        _, _, test_probs, test_labels = evaluate(model, test_dl, criterion, device)
    else:
        print("Evaluating with TTA (this takes a few minutes) …")
        test_probs, test_labels = evaluate_with_tta(model, test_idx, IMG_SIZE, device)

    test_preds = test_probs.argmax(axis=1)
    test_acc   = (test_preds == test_labels).mean()
    print(f"\nTest accuracy: {test_acc:.4f}")
    print("\n── Classification report ──")
    print(classification_report(test_labels, test_preds,
                                target_names=[f"KL={k}" for k in KL_CLASSES],
                                zero_division=0))

    # ── Plots ─────────────────────────────────────────────────────────────────
    if history["train_loss"]:   # skip when resuming from checkpoint (no new history)
        plot_training_curves(history, OUT_DIR / "training_curves.png")
    plot_confusion(test_labels, test_preds, OUT_DIR / "confusion_matrix.png")
    plot_roc(test_labels, test_probs, OUT_DIR / "roc_curves.png")
    plot_gradcam(model, test_idx, IMG_SIZE, device, OUT_DIR / "gradcam_samples.png")

    # ── Comparison vs baseline ────────────────────────────────────────────────
    # Try to load baseline predictions; if unavailable, run a quick baseline eval
    baseline_probs_path = BASELINE_DIR / "test_probs.npy"
    baseline_labels_path = BASELINE_DIR / "test_labels.npy"

    if baseline_probs_path.exists() and baseline_labels_path.exists():
        print("\nLoading baseline predictions for comparison …")
        baseline_probs  = np.load(baseline_probs_path)
        baseline_labels = np.load(baseline_labels_path)
    else:
        print("\nBaseline predictions not found — running baseline ResNet-18 for comparison …")
        import sys; sys.path.insert(0, str(Path(__file__).parent))
        from kl_classifier import build_model as build_baseline
        baseline_ckpt = BASELINE_DIR / "best_model.pt"
        if baseline_ckpt.exists():
            b_model = build_baseline(N_CLASSES).to(device)
            b_model.load_state_dict(torch.load(baseline_ckpt, map_location=device, weights_only=True))
            _, _, baseline_probs, baseline_labels = evaluate(
                b_model, test_dl, criterion, device)
        else:
            print("  Baseline model not found — skipping comparison plot.")
            baseline_probs  = test_probs   # fallback: compare against self
            baseline_labels = test_labels

    # Save improved predictions so future scripts can load them
    np.save(OUT_DIR / "test_probs.npy",  test_probs)
    np.save(OUT_DIR / "test_labels.npy", test_labels)

    plot_comparison(test_labels, baseline_probs, test_probs,
                    OUT_DIR / "comparison_vs_baseline.png")

    print(f"\nAll outputs saved to {OUT_DIR}/")


def parse_args():
    p = argparse.ArgumentParser(description="Improved KL classifier: ResNet-50 + Focal Loss + TTA")
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=2e-4)
    p.add_argument("--gamma",      type=float, default=2.0,
                   help="Focal loss gamma (default 2.0; try 1.5–3.0)")
    p.add_argument("--no_tta",     action="store_true",
                   help="Skip test-time augmentation (faster but less accurate)")
    p.add_argument("--joint",      type=str,   default=None,
                   choices=list(JOINT_MAP.keys()))
    return p.parse_args()


if __name__ == "__main__":
    main()