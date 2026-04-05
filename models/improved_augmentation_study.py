"""
Improved Augmentation Study
============================
Same experimental question as augmentation_study.py — does synthetic data
improve classification? — but now using the improved architecture:

  ResNet-50  +  Focal Loss (γ=2.0)  +  Test-Time Augmentation (TTA)

Two models are compared on the SAME real held-out test set:

  Baseline (improved)  : kl_image_clf_improved/best_model.pt
                         Already trained — loaded directly, no retraining.
  Augmented (improved) : Same ResNet-50 + Focal architecture trained on
                         real images + synthetic images for KL=3 and KL=4.

This lets us isolate the effect of synthetic data within the best model
architecture, rather than confounding it with architecture differences.

Run
---
    python models/improved_augmentation_study.py --generator cyclegan
    python models/improved_augmentation_study.py --generator diffusion

Outputs  (models/improved_augmentation_study/)
-----------------------------------------------
    augmented_model.pt
    confusion_matrix_comparison.png
    f1_comparison.png
    roc_comparison.png
    improved_augmentation_results.csv
"""

import argparse
import re
import sys
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
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
XLSX_PATH = ROOT / "data" / "hand.xlsx"
ZIP_PATH  = ROOT / "data" / "Finger Joints.zip"
OUT_DIR   = Path(__file__).parent / "improved_augmentation_study"

BASELINE_CKPT = Path(__file__).parent / "kl_image_clf_improved" / "best_model.pt"

IMG_SIZE   = 224
KL_CLASSES = [0, 1, 2, 3, 4]
N_CLASSES  = len(KL_CLASSES)

JOINT_MAP = {
    "dip2": "DIP2", "dip3": "DIP3", "dip4": "DIP4", "dip5": "DIP5",
    "mcp2": "MCP2", "mcp3": "MCP3", "mcp4": "MCP4", "mcp5": "MCP5",
    "pip2": "PIP2", "pip3": "PIP3", "pip4": "PIP4", "pip5": "PIP5",
}

# Augment only the truly rare grades
AUGMENT_GRADES        = [3, 4]
N_SYNTHETIC_PER_GRADE = 150


# ── Focal Loss (identical to improved_classifier.py) ──────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight=None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, logits, targets):
        log_p = F.log_softmax(logits, dim=1)
        ce    = F.nll_loss(log_p, targets, weight=self.weight, reduction="none")
        p_t   = torch.exp(-ce)
        return ((1 - p_t) ** self.gamma * ce).mean()


# ── Model — ResNet-50 (identical to improved_classifier.py) ───────────────────
def build_model():
    m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    old = m.conv1
    new = nn.Conv2d(1, old.out_channels, kernel_size=old.kernel_size,
                    stride=old.stride, padding=old.padding, bias=False)
    with torch.no_grad():
        new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
    m.conv1 = new
    m.fc    = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.fc.in_features, N_CLASSES))
    return m


# ── Datasets ───────────────────────────────────────────────────────────────────
def build_index(joint_filter=None):
    import pandas as pd
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

    pat   = re.compile(r"Finger Joints/(\d+)_(\w+)\.png")
    index = []
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        for name in z.namelist():
            if not name.endswith(".png"):
                continue
            m = pat.match(name)
            if not m or (joint_filter and m.group(2) != joint_filter):
                continue
            kl = img_to_kl.get(name)
            if kl is not None:
                index.append((name, kl))
    return index


def patient_split(index, val_frac=0.15, test_frac=0.15, seed=42):
    """Same seed=42 as all other scripts → identical test set."""
    pat = re.compile(r"Finger Joints/(\d+)_")
    pid_to_items: dict[str, list] = {}
    for item in index:
        pid = pat.match(item[0]).group(1)
        pid_to_items.setdefault(pid, []).append(item)

    rng   = np.random.default_rng(seed)
    pids  = np.array(sorted(pid_to_items.keys()))
    rng.shuffle(pids)
    n        = len(pids)
    n_test   = int(n * test_frac)
    n_val    = int(n * val_frac)
    test_p   = set(pids[:n_test])
    val_p    = set(pids[n_test: n_test + n_val])
    train_p  = set(pids[n_test + n_val:])

    def gather(s):
        out = []
        for pid in s:
            out.extend(pid_to_items[pid])
        return out

    return gather(train_p), gather(val_p), gather(test_p)


class RealDataset(Dataset):
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


class SyntheticDataset(Dataset):
    def __init__(self, images, transform):
        self.images    = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        tensor, kl = self.images[idx]
        pil = transforms.ToPILImage()(tensor * 0.5 + 0.5)
        return self.transform(pil), kl


# ── TTA evaluation ─────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_with_tta(model, pil_img, device, n_aug=8):
    model.eval()
    norm    = transforms.Normalize([0.5], [0.5])
    base_tf = transforms.Compose([transforms.Resize(IMG_SIZE + 32), transforms.ToTensor(), norm])
    flip_tf = transforms.Compose([transforms.Resize(IMG_SIZE + 32),
                                  transforms.RandomHorizontalFlip(p=1.0),
                                  transforms.ToTensor(), norm])
    rng        = np.random.default_rng(0)
    probs_list = []
    for i in range(n_aug):
        tf = flip_tf if i % 2 == 1 else base_tf
        t  = tf(pil_img)
        _, h, w    = t.shape
        max_offset = max(0, min(h, w) - IMG_SIZE)
        if max_offset > 0:
            y = int(rng.integers(0, max_offset))
            x = int(rng.integers(0, max_offset))
            t = t[:, y:y + IMG_SIZE, x:x + IMG_SIZE]
        else:
            t = t[:, :IMG_SIZE, :IMG_SIZE]
        logits = model(t.unsqueeze(0).to(device))
        probs_list.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.mean(probs_list, axis=0).squeeze()


@torch.no_grad()
def evaluate_with_tta(model, test_index, device):
    model.eval()
    all_probs, all_labels = [], []
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        for path, kl in tqdm(test_index, desc="  TTA eval", leave=False):
            with z.open(path) as f:
                pil = Image.open(BytesIO(f.read())).convert("L")
            all_probs.append(predict_with_tta(model, pil, device))
            all_labels.append(kl)
    return np.array(all_probs), np.array(all_labels)


@torch.no_grad()
def evaluate_fast(model, test_ds, device):
    """Quick evaluation without TTA — used during training validation."""
    loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
    model.eval()
    all_probs, all_labels = [], []
    for imgs, kls in loader:
        probs = F.softmax(model(imgs.to(device)), dim=1)
        all_probs.append(probs.cpu())
        all_labels.append(kls)
    return torch.cat(all_probs).numpy(), torch.cat(all_labels).numpy()


# ── Training ───────────────────────────────────────────────────────────────────
def train_model(train_ds, val_ds, device, epochs=20, lr=2e-4, label="",
                init_model=None):
    """
    init_model: if provided, fine-tune from these weights instead of
                initialising from scratch.  Pass the baseline model to
                ensure the only experimental variable is the training data.
    """
    def get_labels(ds):
        if hasattr(ds, "index"):
            return [kl for _, kl in ds.index]
        if hasattr(ds, "images"):
            return [kl for _, kl in ds.images]
        out = []
        for sub in ds.datasets:
            out.extend(get_labels(sub))
        return out

    all_labels = get_labels(train_ds)
    counts     = np.bincount(all_labels, minlength=N_CLASSES).astype(float)
    cls_w      = 1.0 / np.where(counts == 0, 1.0, counts)
    sample_w   = torch.tensor([cls_w[l] for l in all_labels])
    sampler    = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

    # Focal loss without class weights — sampler already balances frequencies
    criterion  = FocalLoss(gamma=2.0, weight=None)

    pin      = device.type == "cuda"
    train_dl = DataLoader(train_ds, batch_size=32, sampler=sampler,
                          num_workers=0, pin_memory=pin)
    val_dl   = DataLoader(val_ds,   batch_size=32, shuffle=False,
                          num_workers=0, pin_memory=pin)

    m = init_model if init_model is not None else build_model().to(device)
    optimiser = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    warmup    = max(1, epochs // 10)
    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        t = (ep - warmup) / max(1, epochs - warmup)
        return 0.5 * (1 + np.cos(np.pi * t))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)
    scaler    = torch.amp.GradScaler(device.type) if device.type == "cuda" else None

    best_acc, best_state = 0.0, None

    for epoch in range(1, epochs + 1):
        m.train()
        for imgs, kls in tqdm(train_dl, desc=f"  [{label}] Epoch {epoch}/{epochs}",
                               leave=False):
            imgs, kls = imgs.to(device), kls.to(device)
            optimiser.zero_grad()
            with torch.amp.autocast(device.type, enabled=scaler is not None):
                loss = criterion(m(imgs), kls)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimiser)
                scaler.update()
            else:
                loss.backward()
                optimiser.step()
        scheduler.step()

        m.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, kls in val_dl:
                preds    = m(imgs.to(device)).argmax(1)
                correct += (preds.cpu() == kls).sum().item()
                total   += kls.size(0)
        val_acc = correct / total
        if val_acc > best_acc:
            best_acc   = val_acc
            best_state = {k: v.clone() for k, v in m.state_dict().items()}
        print(f"  [{label}] Epoch {epoch:3d}  val_acc={val_acc:.4f}  best={best_acc:.4f}")

    m.load_state_dict(best_state)
    return m


# ── Synthetic image generation (same as augmentation_study.py) ────────────────
def generate_cyclegan(source_kl, target_kl, n, train_index, device):
    sys.path.insert(0, str(Path(__file__).parent))
    from kl_transition_generator import ResNetGenerator

    ckpt_dir = Path(__file__).parent / f"kl{source_kl}_to_kl{target_kl}_cyclegan"
    if not (ckpt_dir / "G_final.pt").exists():
        print(f"  No CycleGAN found for KL={source_kl}→{target_kl}, "
              f"falling back to diffusion for KL={target_kl}.")
        return generate_diffusion(target_kl, n, device)

    G = ResNetGenerator().to(device)
    G.load_state_dict(torch.load(ckpt_dir / "G_final.pt",
                                 map_location=device, weights_only=True))
    G.eval()

    source_paths = [p for p, kl in train_index if kl == source_kl]
    np.random.shuffle(source_paths)
    tf = transforms.Compose([
        transforms.Resize(64), transforms.CenterCrop(64),
        transforms.ToTensor(), transforms.Normalize([0.5], [0.5]),
    ])

    results = []
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        for path in source_paths[:n]:
            with z.open(path) as f:
                img = Image.open(BytesIO(f.read())).convert("L")
            tensor = tf(img).unsqueeze(0).to(device)
            with torch.no_grad():
                fake = G(tensor).squeeze(0).cpu()
            results.append((fake, target_kl))

    print(f"  Generated {len(results)} KL={target_kl} images via CycleGAN.")
    return results


def generate_diffusion(target_kl, n, device):
    sys.path.insert(0, str(Path(__file__).parent))
    from diffusion_generator import UNet, NoiseSchedule, ddim_sample

    ckpt = Path(__file__).parent / "diffusion_kl" / "unet_best.pt"
    if not ckpt.exists():
        print(f"  Diffusion checkpoint not found at {ckpt}.")
        return []

    schedule = NoiseSchedule(device=device)
    unet     = UNet(n_classes=N_CLASSES).to(device)
    unet.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))

    results, generated = [], 0
    while generated < n:
        bsz  = min(32, n - generated)
        cls  = torch.full((bsz,), target_kl, dtype=torch.long, device=device)
        imgs = ddim_sample(unet, schedule, cls, bsz, device, ddim_steps=50)
        for img in imgs:
            results.append((img.cpu(), target_kl))
        generated += bsz

    print(f"  Generated {len(results)} KL={target_kl} images via Diffusion.")
    return results


# ── Comparison plots ───────────────────────────────────────────────────────────
def plot_confusion_pair(y_true, y_base, y_aug, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, preds, title in zip(
        axes,
        [y_base, y_aug],
        ["Improved Baseline (real only)", "Improved Augmented (real + synthetic)"],
    ):
        ConfusionMatrixDisplay.from_predictions(
            y_true, preds, display_labels=[f"KL={k}" for k in KL_CLASSES],
            ax=ax, cmap="Blues", colorbar=False,
        )
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved {path}")


def plot_f1_comparison(y_true, y_base, y_aug, path):
    f1_base = f1_score(y_true, y_base, labels=KL_CLASSES, average=None, zero_division=0)
    f1_aug  = f1_score(y_true, y_aug,  labels=KL_CLASSES, average=None, zero_division=0)

    x, width = np.arange(N_CLASSES), 0.35
    fig, ax  = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, f1_base, width,
           label="Improved Baseline (ResNet-50 + Focal + TTA, real only)",
           color="steelblue")
    ax.bar(x + width / 2, f1_aug, width,
           label="Improved Augmented (+ synthetic KL=3/4)",
           color="darkorange")
    for i, (b, a) in enumerate(zip(f1_base, f1_aug)):
        delta = a - b
        ax.text(i + width / 2, a + 0.02, f"{delta:+.3f}",
                ha="center", fontsize=8, fontweight="bold",
                color="green" if delta >= 0 else "red")
    ax.set_xticks(x)
    ax.set_xticklabels([f"KL={k}" for k in KL_CLASSES])
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.15)
    ax.set_title("Per-class F1: Improved Baseline vs Improved Augmented (real test set)")
    ax.legend(fontsize=8)

    acc_base = (y_base == y_true).mean()
    acc_aug  = (y_aug  == y_true).mean()
    ax.text(0.98, 0.02,
            f"Overall acc — Baseline: {acc_base:.3f}  |  Augmented: {acc_aug:.3f}",
            transform=ax.transAxes, ha="right", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.4))

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved {path}")
    return f1_base, f1_aug


def plot_roc_comparison(y_true, base_probs, aug_probs, path):
    y_bin  = label_binarize(y_true, classes=KL_CLASSES)
    colors = ["steelblue", "darkorange", "seagreen", "tomato", "mediumpurple"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, probs, title in zip(axes, [base_probs, aug_probs],
                                ["Improved Baseline", "Improved Augmented"]):
        for i, (kl, col) in enumerate(zip(KL_CLASSES, colors)):
            if y_bin[:, i].sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
            auc = roc_auc_score(y_bin[:, i], probs[:, i])
            ax.plot(fpr, tpr, color=col, label=f"KL={kl} AUC={auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_title(f"ROC — {title}")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved {path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Generator: {args.generator}")

    train_tf = transforms.Compose([
        transforms.Resize(IMG_SIZE), transforms.CenterCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(), transforms.Normalize([0.5], [0.5]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(IMG_SIZE), transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(), transforms.Normalize([0.5], [0.5]),
    ])

    print("\nBuilding dataset index …")
    index = build_index(args.joint)
    train_idx, val_idx, test_idx = patient_split(index)

    from collections import Counter
    print("Train KL dist:", dict(sorted(Counter(k for _, k in train_idx).items())))
    print("Test  KL dist:", dict(sorted(Counter(k for _, k in test_idx).items())))

    train_real = RealDataset(train_idx, train_tf)
    val_ds     = RealDataset(val_idx,   eval_tf)

    # ── Baseline — load already-trained improved_classifier.py model ──────────
    if not BASELINE_CKPT.exists():
        print(f"\nBaseline checkpoint not found at {BASELINE_CKPT}.")
        print("Run this first:  python models/improved_classifier.py --epochs 30")
        return

    print(f"\n=== Loading BASELINE from {BASELINE_CKPT} ===")
    baseline_model = build_model().to(device)
    baseline_model.load_state_dict(
        torch.load(BASELINE_CKPT, map_location=device, weights_only=True))

    # Try to load pre-saved TTA predictions (saves ~10 min)
    base_probs_cache  = Path(__file__).parent / "kl_image_clf_improved" / "test_probs.npy"
    base_labels_cache = Path(__file__).parent / "kl_image_clf_improved" / "test_labels.npy"

    if base_probs_cache.exists() and base_labels_cache.exists():
        print("Loading cached baseline TTA predictions …")
        base_probs = np.load(base_probs_cache)
        y_true     = np.load(base_labels_cache)
    else:
        print("Evaluating baseline with TTA (this takes a few minutes) …")
        base_probs, y_true = evaluate_with_tta(baseline_model, test_idx, device)
        np.save(base_probs_cache,  base_probs)
        np.save(base_labels_cache, y_true)

    base_preds = base_probs.argmax(axis=1)
    print("\nBaseline classification report:")
    print(classification_report(y_true, base_preds,
                                target_names=[f"KL={k}" for k in KL_CLASSES],
                                zero_division=0))

    # ── Generate synthetic images ─────────────────────────────────────────────
    print("\n=== Generating synthetic images ===")
    synthetic = []
    for target_kl in AUGMENT_GRADES:
        source_kl = target_kl - 1
        if args.generator == "cyclegan":
            ckpt_dir = Path(__file__).parent / f"kl{source_kl}_to_kl{target_kl}_cyclegan"
            if not (ckpt_dir / "G_final.pt").exists():
                print(f"  No CycleGAN for KL={source_kl}→{target_kl}, using diffusion.")
                synthetic += generate_diffusion(target_kl, N_SYNTHETIC_PER_GRADE, device)
            else:
                synthetic += generate_cyclegan(source_kl, target_kl,
                                               N_SYNTHETIC_PER_GRADE, train_idx, device)
        else:
            synthetic += generate_diffusion(target_kl, N_SYNTHETIC_PER_GRADE, device)

    if not synthetic:
        print("No synthetic images generated — train a generator first. Exiting.")
        return

    from collections import Counter
    print(f"Synthetic images added: {dict(Counter(kl for _, kl in synthetic))}")

    synth_ds  = SyntheticDataset(synthetic, eval_tf)
    train_aug = ConcatDataset([train_real, synth_ds])

    # ── Augmented model — fine-tune from baseline weights ────────────────────
    # Starting from the baseline checkpoint ensures the only variable is the
    # training data (real vs real+synthetic). Training from scratch would
    # introduce random initialisation as a confound and require many more
    # epochs to reach the same quality level.
    aug_ckpt = OUT_DIR / "augmented_model.pt"
    if aug_ckpt.exists():
        print(f"\n=== Loading existing AUGMENTED model from {aug_ckpt} ===")
        aug_model = build_model().to(device)
        aug_model.load_state_dict(
            torch.load(aug_ckpt, map_location=device, weights_only=True))
    else:
        print("\n=== Fine-tuning AUGMENTED model from baseline weights ===")
        print("    (initialised from kl_image_clf_improved/best_model.pt)")
        aug_model = build_model().to(device)
        aug_model.load_state_dict(
            torch.load(BASELINE_CKPT, map_location=device, weights_only=True))
        # Lower LR for fine-tuning — model is already well-trained
        aug_model = train_model(train_aug, val_ds, device,
                                epochs=args.epochs, lr=5e-5, label="Augmented",
                                init_model=aug_model)
        torch.save(aug_model.state_dict(), aug_ckpt)
        print(f"Augmented model saved → {aug_ckpt}")

    print("\nEvaluating augmented model with TTA …")
    aug_probs, _ = evaluate_with_tta(aug_model, test_idx, device)
    aug_preds    = aug_probs.argmax(axis=1)

    print("\nAugmented classification report:")
    print(classification_report(y_true, aug_preds,
                                target_names=[f"KL={k}" for k in KL_CLASSES],
                                zero_division=0))

    # ── Comparison plots ──────────────────────────────────────────────────────
    print("\n=== Saving comparison plots ===")
    plot_confusion_pair(y_true, base_preds, aug_preds,
                        OUT_DIR / "confusion_matrix_comparison.png")
    f1_base, f1_aug = plot_f1_comparison(y_true, base_preds, aug_preds,
                                          OUT_DIR / "f1_comparison.png")
    plot_roc_comparison(y_true, base_probs, aug_probs,
                        OUT_DIR / "roc_comparison.png")

    # ── Summary CSV ───────────────────────────────────────────────────────────
    acc_base = (base_preds == y_true).mean()
    acc_aug  = (aug_preds  == y_true).mean()
    results  = pd.DataFrame({
        "KL_grade":           [f"KL={k}" for k in KL_CLASSES] + ["Overall accuracy"],
        "F1_Improved_Base":   list(f1_base.round(4)) + [round(acc_base, 4)],
        "F1_Improved_Aug":    list(f1_aug.round(4))  + [round(acc_aug,  4)],
        "Delta":              list((f1_aug - f1_base).round(4)) + [round(acc_aug - acc_base, 4)],
    })
    print("\n── Summary ──")
    print(results.to_string(index=False))

    csv_path = OUT_DIR / "improved_augmentation_results.csv"
    results.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    print(f"All outputs in {OUT_DIR}/")


def parse_args():
    p = argparse.ArgumentParser(
        description="Augmentation study using ResNet-50 + Focal Loss + TTA")
    p.add_argument("--generator", choices=["cyclegan", "diffusion"], default="cyclegan")
    p.add_argument("--epochs",    type=int, default=10,
                   help="Fine-tuning epochs for the augmented model (default 10)")
    p.add_argument("--joint",     type=str, default=None,
                   choices=list(JOINT_MAP.keys()))
    return p.parse_args()


if __name__ == "__main__":
    main()