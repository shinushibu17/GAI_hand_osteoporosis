"""
preprocessing.py
================
Provides:
  - Custom torchvision-compatible transforms (CLAHE, NLMFilter)
  - CleanImageDataset — EDA, visualization, transform comparison
  - _SplitDataset — lazy PyTorch Dataset ready for DataLoader

CleanImageDataset is purely for exploration. It holds only metadata and
reads images on demand. Use split() to get _SplitDataset objects, then
wrap them in DataLoaders in your classifier or generative model class.

Typical usage
-------------
>>> from data_prep.preprocessing import CleanImageDataset, _SplitDataset
>>> from data_prep.preprocessing import CLAHE, NLMFilter
>>>
>>> ds = CleanImageDataset(raw)
>>> ds.display_image(0)
>>>
>>> # Compare transforms on a single image
>>> from torchvision import transforms
>>> pipelines = {
... "NLM + CLAHE": transforms.Compose([NLMFilter(), CLAHE(), transforms.ToTensor()]),
... "CLAHE only": transforms.Compose([CLAHE(), transforms.ToTensor()]),
... }
>>> ds.compare_images(0, pipelines)
>>>
>>> # Get splits ready for DataLoader
>>> train_ds, val_ds, test_ds = ds.split(transform=your_pipeline)
>>> train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
"""

from __future__ import annotations

import math
import sys
import zipfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from raw_data import RawImageDataset, _read_image
from transforms import CLAHE, NLMFilter, BilateralFilter, MedianFilter, InvertGrayscale


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_ZIP = _PROJECT_ROOT / "data" / "Finger Joints.zip"

# ---------------------------------------------------------------------------
# _SplitDataset — lazy PyTorch Dataset, ready to wrap in a DataLoader
# ---------------------------------------------------------------------------

class _SplitDataset(Dataset):
    """
    Lazy PyTorch Dataset for a single split (train / val / test).
    Reads images from the zip on demand in __getitem__.

    transform is set by the classifier/generative model class before
    wrapping in a DataLoader:
        train_ds.transform = your_pipeline
        train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

    Parameters
    ----------
    df : DataFrame slice with columns filename | label | joint
    zip_path : path to the PNG zip archive
    file_map : dict mapping bare filename → full zip entry path
    transform : torchvision Compose pipeline — set externally before use
    """

    def __init__(
        self,
        df: pd.DataFrame,
        zip_path: Path,
        file_map: dict[str, str],
        transform: transforms.Compose | None = None,
    ) -> None:
        self.data = df.reset_index(drop=True)
        self.zip_path = zip_path
        self.file_map = file_map
        self.transform = transform
        self.labels = torch.tensor(self.data["label"].astype(int).values, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        if self.transform is None:
            raise RuntimeError(
                "No transform set. Assign a pipeline before using the DataLoader:\n"
                " dataset.transform = transforms.Compose([...])"
            )
        row = self.data.iloc[idx]
        with zipfile.ZipFile(self.zip_path, "r") as zf:
            img = Image.fromarray(_read_image(zf, self.file_map[row["filename"]]))
        return self.transform(img), int(row["label"]), str(row["joint"])

    def __repr__(self) -> str:
        return f"_SplitDataset(n={len(self)}, transform={self.transform})"


# ---------------------------------------------------------------------------
# CleanImageDataset — EDA and visualization only
# ---------------------------------------------------------------------------

class CleanImageDataset:
    """
    EDA and visualization layer on top of RawImageDataset.

    Does NOT apply transforms or own DataLoader logic. Use split() to
    produce _SplitDataset objects, then wrap them in DataLoaders in your
    model-specific class with whatever transform pipeline you choose.

    Parameters
    ----------
    raw : RawImageDataset instance
    small : subsample the dataset stratified by KL grade
    pct : fraction to keep when small=True (default 0.1)
    joint : joint type(s) to include — str, list[str], or None for all
    seed : random seed
    """

    def __init__(
        self,
        raw: RawImageDataset,
        small: bool = False,
        pct: float = 0.1,
        joint: str | list[str] | None = None,
        seed: int = 42,
    ) -> None:
        self.seed = seed
        self.zip_path = raw.zip_path
        self._file_map = raw._file_map
        self.data = raw.data.copy()

        if joint is not None:
            joints = [joint] if isinstance(joint, str) else list(joint)
            invalid = set(joints) - set(self.data["joint"].unique())
            if invalid:
                raise ValueError(
                    f"Unknown joint type(s): {invalid}. "
                    f"Available: {sorted(self.data['joint'].unique())}"
                )
            self.data = self.data[self.data["joint"].isin(joints)].reset_index(drop=True)

        if small:
            if not 0 < pct <= 1:
                raise ValueError(f"pct must be in (0, 1], got {pct}.")
            self.data = (
                self.data
                .groupby("label", group_keys=False)
                .apply(lambda g: g.sample(max(1, math.floor(len(g) * pct)),
                                          random_state=seed))
                .reset_index(drop=True)
            )

        self.classes = sorted(self.data["label"].unique().tolist())

    def __len__(self) -> int:
        return len(self.data)

    # ── Split → _SplitDataset objects ────────────────────────────────────────

    def split(
        self,
        train_pct: float = 0.70,
        val_pct: float = 0.15,
        test_pct: float = 0.15,
    ) -> tuple[_SplitDataset, _SplitDataset, _SplitDataset]:
        """
        Stratified split by KL grade → three _SplitDataset objects.

        Transforms and DataLoaders are set in your classifier/generative
        model class via dataset.transform = your_pipeline.

        Parameters
        ----------
        train_pct : fraction for training (default 0.70)
        val_pct : fraction for validation (default 0.15)
        test_pct : fraction for test (default 0.15)

        Returns
        -------
        (train_ds, val_ds, test_ds)
        """
        if not math.isclose(train_pct + val_pct + test_pct, 1.0, abs_tol=1e-6):
            raise ValueError("train_pct + val_pct + test_pct must sum to 1.0")

        train_rows, val_rows, test_rows = [], [], []

        for _, group in self.data.groupby("label"):
            g = group.sample(frac=1, random_state=self.seed)
            n = len(g)
            n_train = math.floor(n * train_pct)
            n_val = math.floor(n * val_pct)
            n_test = n - n_train - n_val

            train_rows.append(g.iloc[:n_train])
            val_rows.append( g.iloc[n_train : n_train + n_val])
            test_rows.append( g.iloc[n_train + n_val : n_train + n_val + n_test])

        def _make(rows):
            return _SplitDataset(
                df = pd.concat(rows),
                zip_path = self.zip_path,
                file_map = self._file_map,
            )

        return _make(train_rows), _make(val_rows), _make(test_rows)

    # ── Display ───────────────────────────────────────────────────────────────

    def _resolve_row(self, key: int | str) -> pd.Series:
        if isinstance(key, str):
            matches = self.data[self.data["filename"] == key]
            if matches.empty:
                raise KeyError(f"Filename '{key}' not found.")
            return matches.iloc[0]
        elif isinstance(key, int):
            if key < 0 or key >= len(self.data):
                raise IndexError(f"Index {key} out of range (0–{len(self.data) - 1}).")
            return self.data.iloc[key]
        raise TypeError(f"key must be int or str, got {type(key).__name__}.")

    def _read(self, filename: str) -> np.ndarray:
        with zipfile.ZipFile(self.zip_path, "r") as zf:
            return _read_image(zf, self._file_map[filename])

    def display_image(
        self,
        key: int | str,
        window_title: str | None = None,
    ) -> tuple[np.ndarray, int, str]:
        """Display raw image inline and return (image, label, joint)."""
        row = self._resolve_row(key)
        img = self._read(row["filename"])
        label = int(row["label"])
        joint = str(row["joint"])
        title = window_title or f"{row['filename']} | {joint.upper()} | KL {label}"

        fig, ax = plt.subplots(figsize=(4, 5), constrained_layout=True)
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        plt.show()

        return img, label, joint

    def compare_images(
        self,
        key: int | str,
        pipelines: dict[str, transforms.Compose],
    ) -> None:
        """
        Apply multiple transform pipelines to a single image and display
        them side by side for visual comparison.

        Parameters
        ----------
        key : integer index OR filename string
        pipelines : dict mapping panel title → torchvision Compose pipeline
                    The original resized image is always shown as the first panel.

        Example
        -------
        >>> from torchvision import transforms
        >>> pipelines = {
        ... "NLM + CLAHE": transforms.Compose([NLMFilter(), CLAHE(), transforms.ToTensor()]),
        ... "CLAHE only": transforms.Compose([CLAHE(), transforms.ToTensor()]),
        ... "Median + CLAHE":transforms.Compose([transforms.GaussianBlur(3), CLAHE(), transforms.ToTensor()]),
        ... }
        >>> ds.compare_images(0, pipelines)
        """
        row = self._resolve_row(key)
        raw_img = self._read(row["filename"])
        resized = np.array(Image.fromarray(raw_img).resize((224, 224), Image.BILINEAR))

        # Build panels — original first, then each pipeline
        panels: list[tuple[np.ndarray, str]] = [("Original (resized)", resized)]

        for title, pipeline in pipelines.items():
            pil = Image.fromarray(resized)
            out = pipeline(pil)
            # Handle tensor output — convert back to displayable numpy
            if isinstance(out, torch.Tensor):
                arr = out.squeeze().numpy()
                # Rescale to [0, 255] for display regardless of normalisation
                arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
            else:
                arr = np.array(out)
            panels.append((title, arr))

        n = len(panels)
        cols = min(n, 3)
        rows = math.ceil(n / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows),
                                 constrained_layout=True)
        axes = np.array(axes).flatten()

        fig.suptitle(f"{row['filename']} | KL {int(row['label'])}", fontsize=10)

        for ax, (title, img) in zip(axes, panels):
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.set_title(title, fontsize=9)
            ax.axis("off")

        for ax in axes[n:]:
            ax.set_visible(False)

        plt.show()

    # ── Introspection ─────────────────────────────────────────────────────────

    def summary(self) -> pd.DataFrame:
        """KL grade distribution as percentages."""
        counts = self.data["label"].value_counts().sort_index()
        pct = (counts / counts.sum() * 100).round(1)
        return pd.DataFrame({"count": counts, "pct": pct}).rename_axis("KL Grade")

    def __repr__(self) -> str:
        return (
            f"CleanImageDataset(n={len(self)}, "
            f"classes={self.classes}, "
            f"joints={sorted(self.data['joint'].unique().tolist())})"
        )


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path
    from torchvision import transforms
    from raw_data import RawImageDataset

    raw = RawImageDataset()
    ds = CleanImageDataset(raw)

    print(ds)
    print(ds.summary())

    # ── compare_images with all pipelines we evaluated ─────────────────────
    pipelines = {
        "CLAHE only\n(clip=1.0, tile=16)": transforms.Compose([
            transforms.Resize((224, 224)),
            CLAHE(clip_limit=1.0, tile_grid=(16, 16)),
            transforms.ToTensor(),
        ]),
        "Median + CLAHE\n(clip=2.0)": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.1)),
            CLAHE(clip_limit=2.0, tile_grid=(8, 8)),
            transforms.ToTensor(),
        ]),
        "NLM + CLAHE\n(h=5, clip=1.0)": transforms.Compose([
            transforms.Resize((224, 224)),
            NLMFilter(h=5, template_window=5, search_window=11),
            CLAHE(clip_limit=1.0, tile_grid=(16, 16)),
            transforms.ToTensor(),
        ]),
        "Bilateral + CLAHE\n(clip=1.0)": transforms.Compose([
            transforms.Resize((224, 224)),
            BilateralFilter(),
            CLAHE(clip_limit=1.0, tile_grid=(16, 16)),
            transforms.ToTensor(),
        ]),
        "CLAHE + Inverse": transforms.Compose([
            transforms.Resize((224, 224)),
            CLAHE(clip_limit=1.0, tile_grid=(16, 16)),
            InvertGrayscale(),
            transforms.ToTensor(),
        ]),
    }

    print("\nRunning compare_images on index 0...")
    ds.compare_images(0, pipelines)

    print("\nTesting split...")
    train_ds, val_ds, test_ds = ds.split()
    print(f" train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    print(f" train + val + test = {len(train_ds) + len(val_ds) + len(test_ds)} (total={len(ds)})")
    print("Done.")
