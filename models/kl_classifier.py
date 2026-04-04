"""
Image-based KL Grade Classifier (CNN)
======================================
Fine-tunes a pretrained ResNet-18 on finger-joint X-ray images to classify
KL grade (0–4) directly from pixels — no manual scoring required.

Dataset linkage
---------------
Images are read from  data/Finger Joints.zip.
The patient ID in each filename (e.g. 9000099_dip2.png) is matched to the
`id` column in hand.xlsx.  The v00 KL score for the matching joint is the label.
All 12 joint types are included by default (~41 K images total).
Use --joint to restrict to one joint type.

Key design choices
------------------
* Patient-level train/val/test split  — all images for a patient stay in the
  same split, preventing data leakage across patients.
* WeightedRandomSampler + weighted cross-entropy  — handles severe class
  imbalance (KL=0: 31 510 images vs KL=4: 233 images).
* ResNet-18 with grayscale input  — pretrained conv1 weights are averaged
  across the 3 RGB channels to initialise the single grayscale channel,
  preserving ImageNet features.
* Cosine annealing LR with linear warmup.
* Grad-CAM on the final ResNet layer  — shows which image regions drive
  each prediction.

Run
---
    python models/kl_image_classifier.py
    python models/kl_image_classifier.py --joint dip2 --epochs 30 --img_size 128

Outputs  (models/kl_image_clf/)
--------------------------------
    best_model.pt           — weights of the best validation-accuracy checkpoint
    training_curves.png
    confusion_matrix.png
    roc_curves.png
    gradcam_samples.png     — Grad-CAM overlays for 20 random test images

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
from PIL import Image
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
XLSX_PATH = ROOT / "data" / "hand.xlsx"
ZIP_PATH  = ROOT / "data" / "Finger Joints.zip"
OUT_DIR   = Path(__file__).parent / "kl_image_clf"

# ── Constants ────────────────────────────────────────────────────────────────
JOINT_MAP = {
    "dip2": "DIP2", "dip3": "DIP3", "dip4": "DIP4", "dip5": "DIP5",
    "mcp2": "MCP2", "mcp3": "MCP3", "mcp4": "MCP4", "mcp5": "MCP5",
    "pip2": "PIP2", "pip3": "PIP3", "pip4": "PIP4", "pip5": "PIP5",
}
KL_CLASSES = [0, 1, 2, 3, 4]
N_CLASSES  = len(KL_CLASSES)


# ── 1. Dataset ────────────────────────────────────────────────────────────────
class FingerJointKLDataset(Dataset):
    """
    Reads finger-joint PNG images from the zip and assigns KL labels.

    Images are NOT pre-loaded — the zip is kept open and images are read
    on demand (memory-efficient for 41 K images).  Pass a pre-built index
    (list of (path, label)) so the zip scan only happens once.
    """

    def __init__(self, index: list[tuple[str, int]], zip_path: Path, transform):
        self.index     = index   # list of (zip_path_str, kl_int)
        self.zip_path  = zip_path
        self.transform = transform
        self._zip: zipfile.ZipFile | None = None   # opened lazily per-worker

    def _get_zip(self):
        if self._zip is None:
            self._zip = zipfile.ZipFile(self.zip_path, "r")
        return self._zip

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        path, label = self.index[idx]
        with self._get_zip().open(path) as f:
            img = Image.open(BytesIO(f.read())).convert("L")
        return self.transform(img), label

    def __del__(self):
        if self._zip is not None:
            self._zip.close()


def build_index(xlsx_path: Path, zip_path: Path,
                joint_filter: str | None) -> list[tuple[str, int]]:
    """Scan the zip once and return a list of (zip_path, kl_label) pairs."""
    df = pd.read_excel(xlsx_path)
    df["img_id"] = df["id"].astype(str)

    # patient_id → {joint: kl_grade}
    pid_kl: dict[str, dict[str, int]] = {}
    for img_joint, score_joint in JOINT_MAP.items():
        if joint_filter and img_joint != joint_filter:
            continue
        kl_col = f"v00{score_joint}_KL"
        if kl_col not in df.columns:
            continue
        for _, row in df[["img_id", kl_col]].dropna().iterrows():
            pid = row["img_id"]
            kl  = int(row[kl_col])
            pid_kl.setdefault(pid, {})[img_joint] = kl

    pat = re.compile(r"Finger Joints/(\d+)_(\w+)\.png")
    index = []
    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            if not name.endswith(".png"):
                continue
            m = pat.match(name)
            if not m:
                continue
            pid, img_joint = m.group(1), m.group(2)
            if joint_filter and img_joint != joint_filter:
                continue
            kl = pid_kl.get(pid, {}).get(img_joint)
            if kl is None:
                continue
            index.append((name, kl))

    print(f"Index built: {len(index):,} images")
    from collections import Counter
    dist = Counter(kl for _, kl in index)
    for k in sorted(dist):
        print(f"  KL={k}: {dist[k]:,}")
    return index


def patient_split(index: list[tuple[str, int]],
                  val_frac: float = 0.15, test_frac: float = 0.15,
                  seed: int = 42) -> tuple[list, list, list]:
    """
    Split at patient level so no patient appears in more than one fold.
    Returns (train_index, val_index, test_index).
    """
    pat = re.compile(r"Finger Joints/(\d+)_")
    pid_to_items: dict[str, list] = {}
    for item in index:
        pid = pat.match(item[0]).group(1)
        pid_to_items.setdefault(pid, []).append(item)

    rng  = np.random.default_rng(seed)
    pids = np.array(sorted(pid_to_items.keys()))
    rng.shuffle(pids)

    n        = len(pids)
    n_test   = int(n * test_frac)
    n_val    = int(n * val_frac)
    test_pids = set(pids[:n_test])
    val_pids  = set(pids[n_test: n_test + n_val])
    train_pids= set(pids[n_test + n_val:])

    def gather(pid_set):
        out = []
        for pid in pid_set:
            out.extend(pid_to_items[pid])
        return out

    tr, va, te = gather(train_pids), gather(val_pids), gather(test_pids)
    print(f"Split → train {len(tr):,}  val {len(va):,}  test {len(te):,}  "
          f"(patients: {len(train_pids)} / {len(val_pids)} / {len(test_pids)})")
    return tr, va, te


def make_weighted_sampler(index: list[tuple[str, int]]) -> WeightedRandomSampler:
    labels      = [kl for _, kl in index]
    class_counts= np.bincount(labels, minlength=N_CLASSES).astype(float)
    class_w     = 1.0 / np.where(class_counts == 0, 1.0, class_counts)
    sample_w    = torch.tensor([class_w[lbl] for lbl in labels], dtype=torch.float)
    return WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)


# ── 2. Model — ResNet-18 adapted for grayscale ────────────────────────────────
def build_model(n_classes: int = N_CLASSES, freeze_backbone: bool = False) -> nn.Module:
    """
    ResNet-18 pretrained on ImageNet, adapted for grayscale input.

    The original 3-channel conv1 weights are averaged across the channel
    dimension to initialise the single grayscale conv1, preserving the
    learned low-level features.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Adapt conv1: 3-ch → 1-ch
    old_conv = model.conv1
    new_conv = nn.Conv2d(1, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                         stride=old_conv.stride, padding=old_conv.padding, bias=False)
    with torch.no_grad():
        new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
    model.conv1 = new_conv

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

    # Replace classifier head
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, n_classes),
    )
    return model


# ── 3. Grad-CAM ───────────────────────────────────────────────────────────────
class GradCAM:
    """
    Grad-CAM for ResNet-18.  Hooks the last residual block (layer4).
    Usage:
        cam  = GradCAM(model)
        mask = cam(image_tensor, target_class)   # (H, W) numpy array in [0,1]
        cam.remove()
    """

    def __init__(self, model: nn.Module):
        self.model       = model
        self.activations = None
        self.gradients   = None
        self._hooks      = []

        target_layer = model.layer4[-1]
        self._hooks.append(
            target_layer.register_forward_hook(self._save_activations)
        )
        self._hooks.append(
            target_layer.register_full_backward_hook(self._save_gradients)
        )

    def _save_activations(self, _, __, output):
        self.activations = output.detach()

    def _save_gradients(self, _, __, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x: torch.Tensor, target: int | None = None) -> np.ndarray:
        self.model.eval()
        logits = self.model(x)
        if target is None:
            target = logits.argmax(dim=1).item()
        self.model.zero_grad()
        logits[0, target].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = (weights * self.activations).sum(dim=1).squeeze()
        cam     = F.relu(cam)
        cam     = cam.cpu().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def remove(self):
        for h in self._hooks:
            h.remove()


# ── 4. Training ───────────────────────────────────────────────────────────────
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
    all_probs  = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    return total_loss / n, correct / n, all_probs, all_labels


# ── 5. Visualisation helpers ─────────────────────────────────────────────────
def plot_training_curves(history: dict, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"],   label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"],   label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved {path}")


def plot_confusion(y_true, y_pred, path: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=[f"KL={k}" for k in KL_CLASSES],
        ax=ax, cmap="Blues", colorbar=True
    )
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved {path}")


def plot_roc(y_true, y_probs, path: Path):
    y_bin  = label_binarize(y_true, classes=KL_CLASSES)
    colors = ["steelblue", "darkorange", "seagreen", "tomato", "mediumpurple"]
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (cls, col) in enumerate(zip(KL_CLASSES, colors)):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        auc = roc_auc_score(y_bin[:, i], y_probs[:, i])
        ax.plot(fpr, tpr, label=f"KL={cls}  AUC={auc:.3f}", color=col)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("One-vs-rest ROC curves (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved {path}")


def plot_gradcam(model, test_index, zip_path, img_size, device, path: Path, n: int = 20):
    """Sample n test images, run Grad-CAM, and save overlay grid."""
    rng     = np.random.default_rng(0)
    samples = [test_index[i] for i in rng.choice(len(test_index), n, replace=False)]

    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    cam_model = GradCAM(model)
    cols  = 5
    rows  = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 4, rows * 2.5))
    axes = axes.flatten()

    with zipfile.ZipFile(zip_path, "r") as z:
        for i, (img_path, true_kl) in enumerate(samples):
            with z.open(img_path) as f:
                pil_img = Image.open(BytesIO(f.read())).convert("L")

            tensor = transform(pil_img).unsqueeze(0).to(device)
            cam    = cam_model(tensor.clone())
            pred   = model(tensor).argmax(1).item()

            # Resize CAM to image size
            cam_resized = Image.fromarray((cam * 255).astype(np.uint8)).resize(
                (img_size, img_size), Image.BILINEAR
            )
            cam_np = np.array(cam_resized) / 255.0

            # Original image (denorm)
            orig = tensor.squeeze().cpu().numpy()
            orig = (orig * 0.5 + 0.5)

            ax_img = axes[i * 2]
            ax_cam = axes[i * 2 + 1]

            ax_img.imshow(orig, cmap="gray")
            ax_img.set_title(f"True KL={true_kl}  Pred={pred}", fontsize=7)
            ax_img.axis("off")

            ax_cam.imshow(orig, cmap="gray")
            ax_cam.imshow(cam_np, cmap="jet", alpha=0.45)
            ax_cam.set_title("Grad-CAM", fontsize=7)
            ax_cam.axis("off")

    for ax in axes[n * 2:]:
        ax.set_visible(False)

    cam_model.remove()
    fig.suptitle("Grad-CAM: regions driving KL predictions", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved {path}")


# ── 6. Main ───────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Transforms
    train_tf = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.CenterCrop(args.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    # ── Build index and split
    index           = build_index(XLSX_PATH, ZIP_PATH, args.joint)
    train_idx, val_idx, test_idx = patient_split(index, args.val_frac, args.test_frac)

    train_ds = FingerJointKLDataset(train_idx, ZIP_PATH, train_tf)
    val_ds   = FingerJointKLDataset(val_idx,   ZIP_PATH, eval_tf)
    test_ds  = FingerJointKLDataset(test_idx,  ZIP_PATH, eval_tf)

    sampler    = make_weighted_sampler(train_idx)
    train_dl   = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                            num_workers=0, pin_memory=device.type == "cuda")
    val_dl     = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=device.type == "cuda")
    test_dl    = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                            num_workers=0)

    # ── Class weights for loss (extra push on rare classes)
    label_counts = np.bincount([kl for _, kl in train_idx], minlength=N_CLASSES).astype(float)
    class_w      = torch.tensor(
        1.0 / np.where(label_counts == 0, 1.0, label_counts), dtype=torch.float
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w)

    # ── Model
    model = build_model(N_CLASSES, freeze_backbone=args.freeze_backbone).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total  /  {train_params:,} trainable")

    optimiser = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )

    # Cosine annealing with linear warmup
    warmup_epochs  = max(1, args.epochs // 10)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        t = (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * t))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # ── Training loop
    history  = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}  LR={scheduler.get_last_lr()[0]:.2e}")
        tr_loss, tr_acc = train_one_epoch(model, train_dl, criterion, optimiser, device, scaler)
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
            torch.save(model.state_dict(), OUT_DIR / "best_model.pt")
            print(f"  ✓ New best val acc={best_acc:.4f}  — checkpoint saved")

    # ── Test evaluation
    print("\nLoading best checkpoint for test evaluation …")
    model.load_state_dict(torch.load(OUT_DIR / "best_model.pt", map_location=device))
    _, test_acc, test_probs, test_labels = evaluate(model, test_dl, criterion, device)
    test_preds = test_probs.argmax(axis=1)

    print(f"\nTest accuracy: {test_acc:.4f}")
    print("\n── Classification report ──")
    print(classification_report(test_labels, test_preds,
                                 target_names=[f"KL={k}" for k in KL_CLASSES],
                                 zero_division=0))

    # ── Plots
    plot_training_curves(history, OUT_DIR / "training_curves.png")
    plot_confusion(test_labels, test_preds, OUT_DIR / "confusion_matrix.png")
    plot_roc(test_labels, test_probs, OUT_DIR / "roc_curves.png")
    plot_gradcam(model, test_idx, ZIP_PATH, args.img_size, device,
                 OUT_DIR / "gradcam_samples.png")

    print(f"\nAll outputs saved to {OUT_DIR}/")


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="ResNet-18 KL grade classifier on X-ray images")
    p.add_argument("--epochs",           type=int,   default=25)
    p.add_argument("--batch_size",       type=int,   default=32)
    p.add_argument("--lr",               type=float, default=3e-4)
    p.add_argument("--img_size",         type=int,   default=224,
                   help="Resize images to img_size×img_size (default 224)")
    p.add_argument("--val_frac",         type=float, default=0.15)
    p.add_argument("--test_frac",        type=float, default=0.15)
    p.add_argument("--joint",            type=str,   default=None,
                   choices=list(JOINT_MAP.keys()),
                   help="Restrict to one joint type (default: all joints)")
    p.add_argument("--freeze_backbone",  action="store_true",
                   help="Freeze ResNet backbone, train only the head (faster, lower accuracy)")
    return p.parse_args()


if __name__ == "__main__":
    main()
