"""
dataset.py — data loading with patient-level train/val/test splits.

Expected CSV columns (configurable in config.py):
    patient_id | image_path | kl_grade  [| joint_type  (optional)]

image_path may be absolute or relative to config.data_root.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import CFG


# ──────────────────────────────────────────────────────────────────────────────
# Split
# ──────────────────────────────────────────────────────────────────────────────

def make_patient_splits(
    meta: pd.DataFrame,
    seed: int = CFG.random_seed,
    train_r: float = CFG.train_ratio,
    val_r: float = CFG.val_ratio,
    low_data_frac: float = 1.0,
) -> Dict[str, pd.DataFrame]:
    """Return {'train': df, 'val': df, 'test': df} split at patient level.
    
    low_data_frac: fraction of real KL3/KL4 training images to keep (default 1.0 = all).
    e.g. low_data_frac=0.2 keeps only 20% of real KL3/KL4 training images.
    """
    rng = np.random.default_rng(seed)
    patients = meta[CFG.patient_col].unique()
    rng.shuffle(patients)

    n = len(patients)
    n_train = int(n * train_r)
    n_val = int(n * val_r)

    train_pats = set(patients[:n_train])
    val_pats = set(patients[n_train : n_train + n_val])

    splits = {}
    train_df = meta[meta[CFG.patient_col].isin(train_pats)].reset_index(drop=True)

    # Low-data regime: subsample KL3/KL4 in training set
    if low_data_frac < 1.0:
        minority = train_df[train_df[CFG.grade_col].isin([3, 4])]
        majority = train_df[~train_df[CFG.grade_col].isin([3, 4])]
        n_keep = max(1, int(len(minority) * low_data_frac))
        minority_sub = minority.sample(n=n_keep, random_state=seed)
        train_df = pd.concat([majority, minority_sub]).reset_index(drop=True)
        print(f"  [LOW-DATA] Keeping {low_data_frac*100:.0f}% of KL3/KL4 training images "
              f"({n_keep}/{len(minority)})")

    splits["train"] = train_df
    splits["val"] = meta[meta[CFG.patient_col].isin(val_pats)].reset_index(drop=True)
    splits["test"] = meta[~meta[CFG.patient_col].isin(train_pats | val_pats)].reset_index(drop=True)

    for k, df in splits.items():
        print(f"  {k}: {len(df[CFG.patient_col].unique()):4d} patients, "
              f"{len(df):6d} images  | "
              + "  ".join(f"KL{g}={v}" for g, v in
                          sorted(df[CFG.grade_col].value_counts().items())))
    return splits


JOINT_GROUPS = {
    "dip": ["dip2", "dip3", "dip4", "dip5"],
    "pip": ["pip2", "pip3", "pip4", "pip5"],
    "mcp": ["mcp2", "mcp3", "mcp4", "mcp5"],
}


def filter_joint(meta: pd.DataFrame, joint: Optional[str]) -> pd.DataFrame:
    """Filter metadata to a specific joint or joint group.
    
    joint can be:
      - None / 'pooled'  → all joints
      - 'dip','pip','mcp' → joint family group
      - 'dip2','pip3' etc → single joint
    """
    if joint is None or joint == "pooled":
        return meta
    if CFG.joint_col not in meta.columns:
        raise ValueError(f"Column '{CFG.joint_col}' not found. "
                         "Run setup_data.py first or set train_per_joint=False in config.py.")

    # Check if it's a group
    group_joints = JOINT_GROUPS.get(joint.lower())
    if group_joints:
        df = meta[meta[CFG.joint_col].str.lower().isin(group_joints)]
    else:
        df = meta[meta[CFG.joint_col].str.lower() == joint.lower()]

    if df.empty:
        raise ValueError(f"No images found for joint/group '{joint}'. "
                         f"Available: {sorted(meta[CFG.joint_col].unique())}")
    return df.reset_index(drop=True)


def load_metadata(csv_path: Optional[str] = None) -> pd.DataFrame:
    """Load and validate metadata CSV."""
    path = csv_path or CFG.metadata_csv
    meta = pd.read_csv(path)
    for col in [CFG.patient_col, CFG.image_col, CFG.grade_col]:
        if col not in meta.columns:
            raise ValueError(f"Column '{col}' not found in {path}. "
                             f"Available: {list(meta.columns)}")
    # Resolve relative paths
    data_root = Path(CFG.data_root)
    def resolve(p):
        p = Path(p)
        return str(p) if p.is_absolute() else str(data_root / p)
    meta[CFG.image_col] = meta[CFG.image_col].apply(resolve)
    meta[CFG.grade_col] = meta[CFG.grade_col].astype(int)
    return meta


# ──────────────────────────────────────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────────────────────────────────────

def gen_transform(size: int = CFG.img_size, augment: bool = False) -> transforms.Compose:
    """Transform for generative models — returns tensors in [-1, 1]."""
    ops = [transforms.Grayscale(1), transforms.Resize((size, size))]
    if augment:
        ops += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(5)]
    ops += [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    return transforms.Compose(ops)


def clf_transform(size: int = CFG.clf_img_size, augment: bool = False) -> transforms.Compose:
    """Transform for ResNet-18 classifier — returns tensors in [0, 1] then normalised."""
    ops = [transforms.Grayscale(3), transforms.Resize((size, size))]
    if augment:
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(ops)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset classes
# ──────────────────────────────────────────────────────────────────────────────

class OADataset(Dataset):
    """General OA radiograph dataset."""

    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row[CFG.image_col]).convert("L")
        label = int(row[CFG.grade_col])
        if self.transform:
            img = self.transform(img)
        return img, label


class GradeFilteredDataset(OADataset):
    """Dataset filtered to specific KL grades (for GAN training on target grades)."""

    def __init__(self, df: pd.DataFrame, grades: List[int], transform=None):
        sub = df[df[CFG.grade_col].isin(grades)]
        super().__init__(sub, transform)


class UnpairedGradeDataset(Dataset):
    """Returns one image from domain A and one from domain B independently
    (for CycleGAN training). Lengths differ → use the shorter one to set epoch length."""

    def __init__(self, df_A: pd.DataFrame, df_B: pd.DataFrame, transform=None):
        self.ds_A = OADataset(df_A, transform)
        self.ds_B = OADataset(df_B, transform)

    def __len__(self):
        return min(len(self.ds_A), len(self.ds_B))

    def __getitem__(self, idx):
        idx_b = idx % len(self.ds_B)
        img_a, label_a = self.ds_A[idx]
        img_b, label_b = self.ds_B[idx_b]
        return img_a, img_b, label_a, label_b


class AugmentedDataset(Dataset):
    """Combines real training data with synthetic images at a given ratio.

    Args:
        real_df:      DataFrame of real training images.
        synth_dirs:   {grade: path_to_dir_of_synthetic_pngs}
        aug_ratio:    Fraction of real-grade-count to add synthetically.
        transform:    Classifier transform.
    """

    def __init__(
        self,
        real_df: pd.DataFrame,
        synth_dirs: Dict[int, str],
        aug_ratio: float,
        transform=None,
        use_best: bool = False,  # if True, select top-n by pixel quality score
    ):
        self.transform = transform
        entries: List[Tuple[str, int]] = []

        # All real images
        for _, row in real_df.iterrows():
            entries.append((row[CFG.image_col], int(row[CFG.grade_col])))

        # Synthetic images at requested ratio
        for grade, synth_dir_or_list in synth_dirs.items():
            real_count = len(real_df[real_df[CFG.grade_col] == grade])
            n_add = int(real_count * aug_ratio)

            # Collect all paths from one or multiple dirs
            dirs = synth_dir_or_list if isinstance(synth_dir_or_list, list) else [synth_dir_or_list]
            paths = []
            for d in dirs:
                paths += sorted(Path(d).glob("*.png"))
                paths += sorted(Path(d).glob("*.jpg"))

            if not paths:
                print(f"  [WARN] No synthetic images found for KL{grade}")
                continue

            if use_best and len(paths) > n_add:
                # Score by pixel quality and select top-n
                import numpy as np
                scores = []
                for p in paths:
                    try:
                        from PIL import Image as PILImage
                        arr = np.array(PILImage.open(p).convert("L"), dtype=np.float32)
                        mean, std = arr.mean(), arr.std()
                        if mean < 5 or std < 4 or arr.max() - arr.min() < 10:
                            scores.append(-999.0)
                        else:
                            scores.append(float(std - abs(mean - 90) * 0.05))
                    except Exception:
                        scores.append(-999.0)
                ranked = sorted(range(len(paths)), key=lambda i: scores[i], reverse=True)
                chosen = [str(paths[i]) for i in ranked[:n_add]]
            else:
                chosen = [str(paths[i % len(paths)]) for i in range(n_add)]

            for p in chosen:
                entries.append((p, grade))

        self.entries = entries
        print(f"  AugmentedDataset: {len(real_df)} real + "
              f"{len(entries) - len(real_df)} synthetic = {len(entries)} total")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        path, label = self.entries[idx]
        img = Image.open(path).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, label


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_clf_loaders(
    splits: Dict[str, pd.DataFrame],
    synth_dirs: Optional[Dict[int, str]] = None,
    aug_ratio: float = 0.0,
    use_best: bool = False,
) -> Dict[str, DataLoader]:
    """Build train/val/test DataLoaders for the downstream classifier."""
    if synth_dirs and aug_ratio > 0:
        train_ds = AugmentedDataset(
            splits["train"], synth_dirs, aug_ratio,
            transform=clf_transform(augment=True), use_best=use_best
        )
    else:
        train_ds = OADataset(splits["train"], transform=clf_transform(augment=True))

    val_ds = OADataset(splits["val"], transform=clf_transform(augment=False))
    test_ds = OADataset(splits["test"], transform=clf_transform(augment=False))

    kw = dict(num_workers=CFG.num_workers, pin_memory=True)
    return {
        "train": DataLoader(train_ds, batch_size=CFG.batch_size_clf, shuffle=True, **kw),
        "val": DataLoader(val_ds, batch_size=CFG.batch_size_clf, shuffle=False, **kw),
        "test": DataLoader(test_ds, batch_size=CFG.batch_size_clf, shuffle=False, **kw),
    }


def make_gen_loader(
    df: pd.DataFrame,
    grades: Optional[List[int]] = None,
    shuffle: bool = True,
) -> DataLoader:
    """DataLoader for generative model training (single domain or all grades)."""
    if grades is not None:
        ds = GradeFilteredDataset(df, grades, transform=gen_transform(augment=True))
    else:
        ds = OADataset(df, transform=gen_transform(augment=True))
    return DataLoader(
        ds,
        batch_size=CFG.batch_size_gen,
        shuffle=shuffle,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )


def class_weights(df: pd.DataFrame) -> torch.Tensor:
    """Inverse-frequency class weights for CrossEntropyLoss."""
    counts = df[CFG.grade_col].value_counts().sort_index()
    grades = sorted(counts.index)
    weights = torch.zeros(5)
    for g in grades:
        weights[g] = 1.0 / counts[g]
    weights = weights / weights.sum() * len(grades)
    return weights