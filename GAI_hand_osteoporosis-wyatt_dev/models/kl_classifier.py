"""
kl_classifier.py
================
ResNet18-based KL grade classifier for finger-joint OA X-rays.

Designed to evaluate CycleGAN output quality by training on:
  1. Real images only       → baseline
  2. Real + fake images     → measures if GAN adds value

One classifier is trained per joint type so anatomical differences
between joints don't confuse the model.

Outputs (saved per run under models/kl_classifier/joint_label_datetime/)
--------------------------------------------------------------------------
    best_model.pt           — best checkpoint by val accuracy
    training_curves.png     — loss and accuracy over epochs
    confusion_matrix.png    — test set confusion matrix
    roc_curves.png          — per-class ROC curves
    results.json            — accuracy, macro F1, AUC, per-class F1
    test_probs.npy          — raw probability outputs for later analysis
    test_labels.npy         — ground truth labels
    f1_comparison.png       — F1 comparison vs real-only (synthetic runs only)

Usage
-----
>>> from models.kl_classifier import KLGradeClassifier
>>> from data_prep.preprocessing import CleanImageDataset, _SplitDataset
>>>
>>> # Real only
>>> clf = KLGradeClassifier(joint="pip", out_dir="outputs/pip_real")
>>> train_ds, val_ds, test_ds = clean_ds.split()
>>> clf.train(train_ds, val_ds)
>>> clf.evaluate(test_ds)
>>>
>>> # Real + fake
>>> clf_aug = KLGradeClassifier(joint="pip", out_dir="outputs/pip_real_fake")
>>> clf_aug.train(train_ds_with_fake, val_ds)
>>> clf_aug.evaluate(test_ds)           # always evaluate on real only
"""

from __future__ import annotations

import sys
import json
import time
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from PIL import Image
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_prep"))
from transforms import CLAHE, NLMFilter
from preprocessing import _SplitDataset

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT   = Path(__file__).resolve().parents[1]
_DEFAULT_OUT    = _PROJECT_ROOT / "models" / "kl_classifier"
_WEIGHTS_PATH   = _PROJECT_ROOT / "pretrained" / "ResNet18" / "resnet18_imagenet.pth"

# ImageNet grayscale normalization constants
# Mean/std averaged across RGB channels for single-channel input
_IMAGENET_MEAN  = 0.485
_IMAGENET_STD   = 0.229


def compute_mean_std(split_ds: _SplitDataset, pipeline: transforms.Compose) -> tuple[float, float]:
    """
    Compute mean and std from a split's images using the provided pipeline
    (without normalization) so stats match what the model actually sees.

    Parameters
    ----------
    split_ds : training _SplitDataset
    pipeline : transform pipeline WITHOUT ToTensor normalization step —
               should end with ToTensor only
    """
    pixel_sum   = 0.0
    pixel_sq    = 0.0
    pixel_count = 0

    print("Computing dataset mean and std...")
    with zipfile.ZipFile(split_ds.zip_path, "r") as zf:
        for _, row in split_ds.data.iterrows():
            with zf.open(split_ds.file_map[row["filename"]]) as fh:
                img = Image.open(fh).convert("L")
                img.load()
            t = pipeline(img)
            pixel_sum   += t.sum().item()
            pixel_sq    += (t ** 2).sum().item()
            pixel_count += t.numel()

    mean = pixel_sum / pixel_count
    std  = ((pixel_sq / pixel_count) - mean ** 2) ** 0.5
    print(f"  mean={mean:.4f}  std={std:.4f}")
    return mean, std


# ---------------------------------------------------------------------------
# Model — standard ResNet18 adapted for grayscale
# ---------------------------------------------------------------------------


def _build_resnet18(n_classes: int) -> nn.Module:
    """
    Standard ResNet18 pretrained on ImageNet, adapted for single-channel
    grayscale input by averaging the 3-channel conv1 weights.

    Weights are loaded in this order:
      1. Local file at models/resnet18_imagenet.pth  (fastest, works offline)
      2. Download from PyTorch hub                   (requires internet, cached after first run)
      3. Random initialisation                       (fallback if both fail)
    """
    model = models.resnet18(weights=None)

    if _WEIGHTS_PATH.exists():
        print(f"Loading pretrained weights from {_WEIGHTS_PATH}")
        state = torch.load(_WEIGHTS_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    else:
        try:
            print("Downloading pretrained ResNet18 weights from PyTorch hub...")
            pretrained = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            model.load_state_dict(pretrained.state_dict())
            # Cache locally for future offline use
            _WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), _WEIGHTS_PATH)
            print(f"Weights cached at pretrained/ResNet18/resnet18_imagenet.pth")
        except Exception as e:
            print(f"Could not load pretrained weights ({e}) — initialising from scratch.")

    # Adapt conv1: average RGB weights into single grayscale channel
    # Shape: (64, 3, 7, 7) → (64, 1, 7, 7)
    rgb_weights = model.conv1.weight.data
    gray_weight = rgb_weights.mean(dim=1, keepdim=True)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.conv1.weight.data = gray_weight

    # Replace final FC layer for our number of classes
    model.fc = nn.Linear(model.fc.in_features, n_classes)

    return model


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class KLGradeClassifier:
    """
    ResNet18 KL grade classifier for a single joint type.

    Parameters
    ----------
    joint     : joint name e.g. 'pip', 'dip', 'mcp'
    n_classes : number of KL grades to classify (default 5: KL0-KL4)
    out_dir   : directory to save model and results
    device    : 'cuda', 'cpu', or None (auto-detect)
    """

    def __init__(
        self,
        joint:     str | None = None,
        n_classes: int = 5,
        out_dir:   Path | str | None = None,
        device:    str | None = None,
        use_fake:  bool = False,
    ) -> None:
        self.joint     = joint    # may be None until train() is called
        self.n_classes = n_classes
        self.use_fake  = use_fake

        timestamp      = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        label          = "real_fake" if use_fake else "real_only"
        folder_joint   = joint or "all"
        self.out_dir   = Path(out_dir or _DEFAULT_OUT / f"{folder_joint}_{label}_{timestamp}")
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"KLGradeClassifier — joint={self.joint or 'inferred from data'}, device={self.device}")

        self.model    = _build_resnet18(n_classes).to(self.device)
        self._eval_tf: "transforms.Compose | None" = None
        self._history: dict[str, list] = {
            "train_loss": [], "val_loss": [],
            "train_acc":  [], "val_acc":  [],
        }

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        train_ds:        "_SplitDataset",
        val_ds:          "_SplitDataset",
        epochs:          int   = 30,
        batch_size:      int   = 32,
        lr:              float = 2e-4,
        optimizer_cls:   type  = torch.optim.AdamW,
        optimizer_kwargs: dict | None = None,
        sampler:         bool  = True,
        patience:        int   = 5,
        min_delta:       float = 1e-3,
        train_transform: "transforms.Compose | None" = None,
        eval_transform:  "transforms.Compose | None" = None,
    ) -> None:
        """
        Train the classifier on the given splits.

        Parameters
        ----------
        train_ds         : training _SplitDataset from CleanImageDataset.split()
        val_ds           : validation _SplitDataset
        epochs           : maximum number of training epochs
        batch_size       : samples per batch
        lr               : initial learning rate
        optimizer_cls    : any torch.optim optimizer class (default AdamW)
                           e.g. torch.optim.SGD, torch.optim.Adam
        optimizer_kwargs : extra kwargs passed to optimizer e.g. {'momentum': 0.9}
        sampler          : use WeightedRandomSampler to balance KL grade frequency
        patience         : stop if val accuracy doesn't improve for this many epochs
        min_delta        : minimum improvement in val accuracy to count as progress
        train_transform  : custom pipeline for training split
                           defaults to standard ResNet18 ImageNet pipeline
        eval_transform   : custom pipeline for val and test splits
                           defaults to standard ResNet18 ImageNet pipeline
        """
        # Read joint type directly from the data
        joints_in_data = train_ds.data["joint"].unique().tolist()
        if self.joint is None:
            self.joint = joints_in_data[0] if len(joints_in_data) == 1 else "mixed"
        print(f"Joint type: {self.joint.upper()} — grades: {sorted(train_ds.data['label'].unique().tolist())}")

        ckpt = self.out_dir / "best_model.pt"
        if ckpt.exists():
            print(f"Checkpoint found at {ckpt} — loading and skipping training.")
            print("Delete best_model.pt to retrain.")
            self.load(ckpt)
            return

        # Default pipelines follow standard ResNet18 ImageNet convention
        train_tf = train_transform or transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[_IMAGENET_MEAN], std=[_IMAGENET_STD]),
        ])
        eval_tf = eval_transform or transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[_IMAGENET_MEAN], std=[_IMAGENET_STD]),
        ])

        train_ds.transform = train_tf
        val_ds.transform   = eval_tf
        self._eval_tf      = eval_tf   # store for use in predict()

        train_loader = self._make_loader(train_ds, batch_size, shuffle=True,  use_sampler=sampler)
        val_loader   = self._make_loader(val_ds,   batch_size, shuffle=False, use_sampler=False)

        criterion = nn.CrossEntropyLoss(
            weight=self._class_weights(train_ds).to(self.device)
        )

        opt_kwargs = {"lr": lr, "weight_decay": 1e-4}
        opt_kwargs.update(optimizer_kwargs or {})
        optimizer  = optimizer_cls(self.model.parameters(), **opt_kwargs)
        scheduler  = self._build_scheduler(optimizer, epochs)
        scaler     = torch.amp.GradScaler() if self.device.type == "cuda" else None

        # Build human-readable config to save alongside the model
        self._config = {
            "joint":            self.joint,
            "use_fake":         self.use_fake,
            "n_classes":        self.n_classes,
            "epochs_max":       epochs,
            "batch_size":       batch_size,
            "optimizer":        optimizer_cls.__name__,
            "optimizer_kwargs": opt_kwargs,
            "patience":         patience,
            "min_delta":        min_delta,
            "mean":             _IMAGENET_MEAN,
            "std":              _IMAGENET_STD,
            "train_transform":  str(train_tf),
            "eval_transform":   str(eval_tf),
        }
        with open(self.out_dir / "config.json", "w") as f:
            json.dump(self._config, f, indent=2)
        print(f"Config saved → {self.out_dir / 'config.json'}")

        best_acc      = 0.0
        epochs_no_imp = 0
        epoch_times: list[float] = []

        print(f"\nTraining {self.joint.upper()} classifier for up to {epochs} epochs "
              f"| optimizer={optimizer_cls.__name__} lr={lr} "
              f"| patience={patience} min_delta={min_delta}")

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            tr_loss, tr_acc = self._train_epoch(
                train_loader, criterion, optimizer, scaler
            )
            va_loss, va_acc = self._eval_epoch(val_loader, criterion)
            scheduler.step()

            epoch_time = time.time() - t0
            epoch_times.append(epoch_time)
            avg_time   = sum(epoch_times) / len(epoch_times)
            remaining  = avg_time * (epochs - epoch)

            self._history["train_loss"].append(tr_loss)
            self._history["val_loss"].append(va_loss)
            self._history["train_acc"].append(tr_acc)
            self._history["val_acc"].append(va_acc)

            improved = va_acc > best_acc + min_delta
            print(f"  Epoch {epoch:02d}/{epochs} "
                  f"[{epoch_time:.0f}s | est. remaining {remaining/60:.1f}m] | "
                  f"train loss={tr_loss:.4f} acc={tr_acc:.3f} | "
                  f"val loss={va_loss:.4f} acc={va_acc:.3f}"
                  + (" ✓" if improved else f" (no improvement {epochs_no_imp + 1}/{patience})"))

            if improved:
                best_acc      = va_acc
                epochs_no_imp = 0
                torch.save({
                    "model_state": self.model.state_dict(),
                    "mean":        _IMAGENET_MEAN,
                    "std":         _IMAGENET_STD,
                    "joint":       self.joint,
                    "config":      self._config,
                }, ckpt)
                print(f"    ✓ Saved (val_acc={best_acc:.4f})")
            else:
                epochs_no_imp += 1
                if epochs_no_imp >= patience:
                    print(f"\n  Early stopping — no improvement for {patience} epochs.")
                    break

        total = sum(epoch_times)
        print(f"\nBest val acc: {best_acc:.4f}  "
              f"(stopped at epoch {epoch}/{epochs} | "
              f"total time {total/60:.1f}m)")
        print(f"\nSaved to {self.out_dir}/")
        print(f"  best_model.pt")

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        test_ds:        "_SplitDataset",
        batch_size:     int = 32,
        eval_transform: "transforms.Compose | None" = None,
    ) -> dict:
        """
        Evaluate on test set and save all results.

        Parameters
        ----------
        test_ds        : test _SplitDataset — should always be real images only
        batch_size     : samples per batch
        eval_transform : custom eval pipeline — defaults to the same as train()

        Returns
        -------
        dict with keys: accuracy, macro_f1, macro_auc, f1_per_class, report
        """
        eval_tf = eval_transform or self._eval_tf or transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[_IMAGENET_MEAN], std=[_IMAGENET_STD]),
        ])
        test_ds.transform = eval_tf
        self._eval_tf     = eval_tf
        test_loader = self._make_loader(test_ds, batch_size, shuffle=False, use_sampler=False)

        all_probs  = []
        all_labels = []
        self.model.eval()

        with torch.no_grad():
            for imgs, labels, _ in tqdm(test_loader, desc="Evaluating"):
                imgs = imgs.to(self.device)
                logits = self.model(imgs)
                probs  = F.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)
                all_labels.extend(labels.numpy())

        probs  = np.concatenate(all_probs)
        labels = np.array(all_labels)
        preds  = probs.argmax(axis=1)

        acc       = (preds == labels).mean()
        macro_f1  = f1_score(labels, preds, average="macro",    zero_division=0)
        per_class_f1 = f1_score(labels, preds, average=None,    zero_division=0)

        # AUC — one-vs-rest
        try:
            lb  = label_binarize(labels, classes=list(range(self.n_classes)))
            auc = roc_auc_score(lb, probs, multi_class="ovr", average="macro")
        except ValueError:
            auc = float("nan")

        # Store on instance for easy access
        self.results = {
            "accuracy":      float(acc),
            "macro_f1":      float(macro_f1),
            "macro_auc":     float(auc),
            "f1_per_class":  {f"KL{i}": float(f) for i, f in enumerate(per_class_f1)},
        }

        print(f"\nTest accuracy : {acc:.4f}")
        print(f"Macro F1      : {macro_f1:.4f}")
        print(f"Macro AUC     : {auc:.4f}")
        print("\nPer-class F1:")
        for grade, score in self.results["f1_per_class"].items():
            print(f"  {grade}: {score:.4f}")
        print("\n── Classification Report ──")
        report = classification_report(
            labels, preds,
            target_names=[f"KL{i}" for i in range(self.n_classes)],
            zero_division=0,
        )
        print(report)
        self.results["report"] = report

        # Save raw outputs
        np.save(self.out_dir / "test_probs.npy",  probs)
        np.save(self.out_dir / "test_labels.npy", labels)

        # Save results to JSON for easy loading later
        with open(self.out_dir / "results.json", "w") as f:
            json.dump({k: v for k, v in self.results.items() if k != "report"}, f, indent=2)

        # Save plots
        self._plot_training_curves()
        self._plot_confusion(labels, preds)
        self._plot_roc(labels, probs)

        print(f"\nResults saved to {self.out_dir}/")
        return self.results

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(
        self,
        image: "np.ndarray | Image.Image",
    ) -> dict:
        """
        Predict KL grade and confidence for a single image.
        Uses the eval transform set during train() or evaluate().

        Parameters
        ----------
        image : uint8 grayscale ndarray or PIL Image

        Returns
        -------
        dict with keys:
            predicted_grade : int   — predicted KL grade
            confidence      : float — probability of predicted grade
            probabilities   : dict  — probability for each KL grade
        """
        if self._eval_tf is None:
            raise RuntimeError(
                "No eval transform set. Call train() or evaluate() before predict()."
            )

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        tensor = self._eval_tf(image).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            probs = F.softmax(self.model(tensor), dim=1).squeeze().cpu().numpy()

        grade = int(probs.argmax())
        return {
            "predicted_grade": grade,
            "confidence":      float(probs[grade]),
            "probabilities":   {f"KL{i}": float(p) for i, p in enumerate(probs)},
        }

    # ── Save / Load ───────────────────────────────────────────────────────────

    def load(self, path: Path | str | None = None) -> None:
        """Load model weights and config from checkpoint."""
        path = Path(path or self.out_dir / "best_model.pt")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self._config = ckpt.get("config", {})
        print(f"Loaded checkpoint from {path}")
        if self._config:
            print(f"  optimizer : {self._config.get('optimizer')}")
            print(f"  pipeline  : {self._config.get('train_transform', 'unknown')}")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _make_loader(
        self,
        split_ds:   _SplitDataset,
        batch_size: int,
        shuffle:    bool,
        use_sampler: bool,
    ) -> DataLoader:
        if use_sampler:
            weights = self._sample_weights(split_ds)
            s       = WeightedRandomSampler(weights, len(weights), replacement=True)
            return DataLoader(split_ds, batch_size=batch_size, sampler=s, num_workers=0)
        return DataLoader(split_ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    def _class_weights(self, split_ds) -> torch.Tensor:
        """Inverse frequency class weights for CrossEntropyLoss."""
        import numpy as np
        counts = np.bincount(
            split_ds.data["label"].astype(int).values,
            minlength=self.n_classes,
        ).astype(float)
        counts = np.where(counts == 0, 1.0, counts)
        return torch.tensor(1.0 / counts, dtype=torch.float)

    def _sample_weights(self, split_ds) -> torch.Tensor:
        """Per-sample weights for WeightedRandomSampler."""
        class_w    = self._class_weights(split_ds)
        labels     = split_ds.data["label"].astype(int).values
        return torch.tensor([class_w[l].item() for l in labels], dtype=torch.float)

    def _build_scheduler(self, optimizer, epochs: int):
        """Linear warmup then cosine annealing decay."""
        warmup = max(1, epochs // 10)
        def lr_lambda(epoch):
            if epoch < warmup:
                return (epoch + 1) / warmup
            t = (epoch - warmup) / max(1, epochs - warmup)
            return 0.5 * (1 + np.cos(np.pi * t))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _train_epoch(self, loader, criterion, optimizer, scaler) -> tuple[float, float]:
        self.model.train()
        total_loss = correct = total = 0

        for imgs, labels, _ in tqdm(loader, desc="  Train", leave=False):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            optimizer.zero_grad()

            if scaler:
                with torch.amp.autocast(self.device.type):
                    logits = self.model(imgs)
                    loss   = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                with torch.no_grad():
                    preds = logits.argmax(1)
            else:
                logits = self.model(imgs)
                loss   = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                preds  = logits.argmax(1)

            total_loss += loss.item() * imgs.size(0)
            correct    += (preds == labels).sum().item()
            total      += imgs.size(0)

        return total_loss / total, correct / total

    def _eval_epoch(self, loader, criterion) -> tuple[float, float]:
        self.model.eval()
        total_loss = correct = total = 0

        with torch.no_grad():
            for imgs, labels, _ in tqdm(loader, desc="  Val  ", leave=False):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                logits       = self.model(imgs)
                loss         = criterion(logits, labels)
                total_loss  += loss.item() * imgs.size(0)
                correct     += (logits.argmax(1) == labels).sum().item()
                total       += imgs.size(0)

        return total_loss / total, correct / total

    # ── Plots ─────────────────────────────────────────────────────────────────

    def _plot_training_curves(self) -> None:
        if not self._history["train_loss"]:
            return
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        epochs = range(1, len(self._history["train_loss"]) + 1)

        ax1.plot(epochs, self._history["train_loss"], label="Train")
        ax1.plot(epochs, self._history["val_loss"],   label="Val")
        ax1.set_title("Loss")
        ax1.set_xlabel("Epoch")
        ax1.legend()

        ax2.plot(epochs, self._history["train_acc"], label="Train")
        ax2.plot(epochs, self._history["val_acc"],   label="Val")
        ax2.set_title("Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.legend()

        fig.suptitle(f"{self.joint.upper()} — Training Curves")
        fig.savefig(self.out_dir / "training_curves.png", dpi=120)
        plt.close(fig)
        print(f"Saved training_curves.png")

    def _plot_confusion(self, labels, preds) -> None:
        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
        ConfusionMatrixDisplay.from_predictions(
            labels, preds,
            display_labels=[f"KL{i}" for i in range(self.n_classes)],
            ax=ax, colorbar=False,
        )
        ax.set_title(f"{self.joint.upper()} — Confusion Matrix")
        fig.savefig(self.out_dir / "confusion_matrix.png", dpi=120)
        plt.close(fig)
        print(f"Saved confusion_matrix.png")

    def _plot_roc(self, labels, probs) -> None:
        lb  = label_binarize(labels, classes=list(range(self.n_classes)))
        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)

        for i in range(self.n_classes):
            if lb[:, i].sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(lb[:, i], probs[:, i])
            try:
                auc = roc_auc_score(lb[:, i], probs[:, i])
            except ValueError:
                auc = float("nan")
            ax.plot(fpr, tpr, label=f"KL{i} (AUC={auc:.2f})")

        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{self.joint.upper()} — ROC Curves")
        ax.legend(fontsize=8)
        fig.savefig(self.out_dir / "roc_curves.png", dpi=120)
        plt.close(fig)
        print(f"Saved roc_curves.png")

    def __repr__(self) -> str:
        return (f"KLGradeClassifier(joint={self.joint!r}, "
                f"n_classes={self.n_classes}, "
                f"use_fake={self.use_fake}, "
                f"device={self.device})")


# ---------------------------------------------------------------------------
# Comparison utility
# ---------------------------------------------------------------------------

def compare_f1(
    real_clf:      "KLGradeClassifier | Path | str",
    synthetic_clf: "KLGradeClassifier | Path | str",
    save_path:     "Path | str | None" = None,
) -> None:
    """
    Plot a side-by-side bar chart comparing per-class F1 scores between a
    model trained on real data only and one trained on real + synthetic data.
    Annotates each synthetic bar with the percentage improvement over real.

    Parameters
    ----------
    real_clf      : KLGradeClassifier instance (after evaluate()) OR path to
                    its output folder containing results.json
    synthetic_clf : KLGradeClassifier instance (after evaluate()) OR path to
                    its output folder containing results.json
    save_path     : save figure here instead of displaying it — if None and
                    both inputs are classifier instances, saves to the
                    synthetic model's output folder automatically

    Example
    -------
    >>> compare_f1(clf_real, clf_fake)
    >>> compare_f1("models/kl_classifier/pip_real_only_2025-04-08_14-30",
    ...            "models/kl_classifier/pip_real_fake_2025-04-08_16-05")
    """
    import json

    def _load_f1(source) -> tuple[dict, str, Path | None]:
        """Returns (f1_per_class dict, label, out_dir or None)."""
        if isinstance(source, KLGradeClassifier):
            if not hasattr(source, "results"):
                raise ValueError(
                    "Classifier has no results — call evaluate() first."
                )
            label = f"{source.joint.upper()} {'Real+Fake' if source.use_fake else 'Real Only'}"
            return source.results["f1_per_class"], label, source.out_dir
        else:
            folder = Path(source)
            path   = folder / "results.json"
            if not path.exists():
                raise FileNotFoundError(f"results.json not found in {folder}")
            with open(path) as f:
                data = json.load(f)
            label = folder.name
            return data["f1_per_class"], label, folder

    real_f1,      real_label,      _          = _load_f1(real_clf)
    synthetic_f1, synthetic_label, synth_dir  = _load_f1(synthetic_clf)

    grades    = sorted(real_f1.keys())           # ['KL0', 'KL1', ...]
    real_vals = [real_f1[g]      for g in grades]
    syn_vals  = [synthetic_f1[g] for g in grades]

    x     = np.arange(len(grades))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    ax.bar(x - width / 2, real_vals, width,
           label=real_label, color="#4C72B0", alpha=0.85)
    ax.bar(x + width / 2, syn_vals,  width,
           label=synthetic_label, color="#DD8452", alpha=0.85)

    # Annotate each synthetic bar with % improvement
    for i, (rv, sv) in enumerate(zip(real_vals, syn_vals)):
        if rv > 0:
            pct_change = (sv - rv) / rv * 100
            colour     = "#2ca02c" if pct_change >= 0 else "#d62728"
            sign       = "+" if pct_change >= 0 else ""
            ax.text(x[i] + width / 2, sv + 0.01,
                    f"{sign}{pct_change:.1f}%",
                    ha="center", va="bottom", fontsize=8,
                    color=colour, fontweight="bold")
        else:
            ax.text(x[i] + width / 2, sv + 0.01,
                    f"{sv:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(grades)
    ax.set_xlabel("KL Grade")
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, min(1.15, max(max(real_vals), max(syn_vals)) + 0.15))
    ax.set_title("F1 Score Comparison: Real Only vs Real + Synthetic")
    ax.legend()
    ax.axhline(y=0, color="black", linewidth=0.5)

    # Auto-save to synthetic model's output folder if no path given
    if save_path is None and synth_dir is not None:
        save_path = synth_dir / "f1_comparison.png"

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"F1 comparison saved → {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from data_prep.raw_data import RawImageDataset
    from data_prep.preprocessing import CleanImageDataset

    raw = RawImageDataset()

    # Use 5% stratified subsample for fast smoke test
    ds  = CleanImageDataset(raw, joint="pip", small=True, pct=0.05)
    print(f"Smoke test dataset: {len(ds)} images")
    print(ds.summary())

    train_ds, val_ds, test_ds = ds.split()
    print(f"Split → train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # No out_dir — uses default _PROJECT_ROOT/models/kl_classifier/pip_real_only_datetime
    clf = KLGradeClassifier(joint="pip", use_fake=False)
    print(f"Output folder: {clf.out_dir}")
    clf.train(train_ds, val_ds, epochs=2, batch_size=16)
    results = clf.evaluate(test_ds)

    # Test predict on a single image
    import zipfile
    from data_prep.raw_data import _read_image
    row = test_ds.data.iloc[0]
    with zipfile.ZipFile(raw.zip_path, "r") as zf:
        img = _read_image(zf, raw._file_map[row["filename"]])

    pred = clf.predict(img)
    print(f"\nSingle image prediction:")
    print(f"  Predicted grade : KL{pred['predicted_grade']}")
    print(f"  Confidence      : {pred['confidence']:.2%}")
    print(f"  All probs       : {pred['probabilities']}")