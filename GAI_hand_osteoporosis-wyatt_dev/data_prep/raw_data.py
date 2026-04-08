
"""
raw_data.py
===========
Lazy-loading dataset for finger-joint OA X-ray images.

Images are NOT loaded into memory at construction time. Only metadata
(filename, label, joint) is stored in ds.data. Images are read from the
zip on demand in __getitem__ and display_image(), making this safe for
41k+ image datasets.

    ds.data — DataFrame with columns:
                  filename | label (int) | joint (str)

Typical usage
-------------
>>> from data_prep.raw_data import RawImageDataset
>>> ds = RawImageDataset()
>>> ds.data.head()
>>> img, label, joint = ds[0]
"""

from __future__ import annotations

import math
import os
import warnings
import zipfile
from io import BytesIO
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# ---------------------------------------------------------------------------
# Default paths (GAI_hand_osteoporosis/data/)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_ZIP = _PROJECT_ROOT / "data" / "Finger Joints.zip"
_DEFAULT_META = _PROJECT_ROOT / "data" / "hand.xlsx"

_EXCLUDE_KEYWORDS = {"Hand", "Flag", "Review", "STT", "CMC1", "MCP1"}
_EXCLUDE_EXACT = {"v00IP1_KL"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_files(*paths: Path | str) -> None:
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n " + "\n ".join(missing))


def _strip_joint_digits(col: str) -> str:
    """'pip2' -> 'pip', 'dip3' -> 'dip'"""
    return "".join(c for c in col if not c.isdigit())


def _read_image(zf: zipfile.ZipFile, entry: str) -> np.ndarray:
    """Read a single PNG from an open ZipFile into a uint8 grayscale array."""
    with zf.open(entry) as fh:
        return np.array(Image.open(BytesIO(fh.read())).convert("L"))


# ---------------------------------------------------------------------------
# Step 1 — index zip filenames only (no pixel data)
# ---------------------------------------------------------------------------

def _index_zip(zip_path: Path) -> dict[str, str]:
    """
    Returns
    -------
    file_map : dict bare_filename -> full zip entry path
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        return {
            os.path.basename(f): f
            for f in zf.namelist()
            if f.endswith(".png")
        }


# ---------------------------------------------------------------------------
# Step 2 — parse metadata → df_labels (filename, label, joint)
# ---------------------------------------------------------------------------

def _build_labels(meta_path: Path) -> pd.DataFrame:
    raw = pd.read_excel(meta_path)

    def _keep(col: str) -> bool:
        if col == "id":
            return True
        if "v00" not in col or "KL" not in col:
            return False
        if col in _EXCLUDE_EXACT:
            return False
        if any(kw in col for kw in _EXCLUDE_KEYWORDS):
            return False
        return True

    meta = (
        raw.loc[:, [c for c in raw.columns if _keep(c)]]
        .drop_duplicates()
        .reset_index(drop=True)
        .dropna(subset=["id"])
    )

    meta.columns = (
        meta.columns
        .str.removeprefix("v00")
        .str.lower()
        .str.removesuffix("_kl")
    )

    joint_cols = [c for c in meta.columns if c != "id"]
    records: list[dict] = []

    for _, row in meta.iterrows():
        patient_id = int(row["id"])
        for col in joint_cols:
            label = row[col]
            if pd.isna(label):
                continue
            records.append({
                "filename": f"{patient_id}_{col}.png",
                "label": int(label),
                "joint": _strip_joint_digits(col),
            })

    return pd.DataFrame(records, columns=["filename", "label", "joint"])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RawImageDataset:
    """
    Lazy-loading dataset for finger-joint OA X-ray images.

    Only metadata is stored in memory. Images are read from the zip on
    demand, making this suitable for large datasets (40k+ images).

    ds.data — DataFrame: filename | label | joint

    Parameters
    ----------
    zip_path : path to PNG zip archive (default: data/Finger Joints.zip)
    meta_path : path to Excel metadata (default: data/hand.xlsx)
    """

    def __init__(
        self,
        zip_path: Path | str | None = None,
        meta_path: Path | str | None = None,
    ) -> None:
        self.zip_path = Path(zip_path or _DEFAULT_ZIP)
        self.meta_path = Path(meta_path or _DEFAULT_META)

        _check_files(self.zip_path, self.meta_path)

        print("Indexing zip...")
        self._file_map: dict[str, str] = _index_zip(self.zip_path)

        print("Parsing labels from metadata...")
        df_labels = _build_labels(self.meta_path)

        # Keep only filenames that exist in both zip and metadata
        in_zip = set(self._file_map.keys())
        in_labels = set(df_labels["filename"])

        unlabelled = in_zip - in_labels
        missing = in_labels - in_zip

        if unlabelled:
            warnings.warn(
                f"{len(unlabelled)} image(s) in zip have no label — excluded.",
                UserWarning, stacklevel=2,
            )
        if missing:
            warnings.warn(
                f"{len(missing)} label entries have no matching image in zip — excluded.",
                UserWarning, stacklevel=2,
            )

        self.unlabelled_files: list[str] = sorted(unlabelled)
        self.missing_files: list[str] = sorted(missing)

        # Final data — only rows where both image and label exist
        self.data: pd.DataFrame = (
            df_labels[df_labels["filename"].isin(in_zip)]
            .reset_index(drop=True)
        )

        print(f"Done — {len(self.data)} labelled images indexed.")

        if self.unlabelled_files:
            print(f"\nIn zip, no label ({len(self.unlabelled_files)}):")
            for f in self.unlabelled_files:
                print(f" {f}")

        if self.missing_files:
            print(f"\nIn labels, no image ({len(self.missing_files)}):")
            for f in self.missing_files:
                print(f" {f}")

    # ── Image reading ────────────────────────────────────────────────────────

    def read_image(self, filename: str) -> np.ndarray:
        """Read a single image by filename from the zip."""
        if filename not in self._file_map:
            raise KeyError(f"'{filename}' not found in zip.")
        with zipfile.ZipFile(self.zip_path, "r") as zf:
            return _read_image(zf, self._file_map[filename])

    # ── Sequence protocol ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int, str]:
        row = self.data.iloc[idx]
        img = self.read_image(row["filename"])
        return img, int(row["label"]), str(row["joint"])

    # ── Display ──────────────────────────────────────────────────────────────

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

    def display_image(
        self,
        key: int | str,
        window_title: str | None = None,
    ) -> tuple[np.ndarray, int, str]:
        """Display a single image inline and return (image, label, joint)."""
        row = self._resolve_row(key)
        img = self.read_image(row["filename"])
        label = int(row["label"])
        joint = str(row["joint"])
        title = window_title or f"{row['filename']} | {joint.upper()} | KL {label}"

        fig, ax = plt.subplots(figsize=(4, 5), constrained_layout=True)
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        plt.show()

        return img, label, joint

    # ── Introspection ────────────────────────────────────────────────────────

    def summary(self, save_path: Path | str | None = None) -> pd.DataFrame:
        """KL grade % distribution per joint with bar charts."""
        df = self.data.copy()

        counts = (
            df.groupby(["joint", "label"])
            .size()
            .unstack(fill_value=0)
            .rename_axis("joint")
            .rename_axis("KL Grade", axis=1)
        )
        pct = counts.div(counts.sum(axis=1), axis=0).mul(100).round(1)
        result = pct

        joints = sorted(df["joint"].unique())
        all_grades = sorted(df["label"].unique())
        n_plots = len(joints) + 1
        cols = 2
        rows = math.ceil(n_plots / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows),
                                 constrained_layout=True)
        axes = axes.flatten()
        cmap = plt.colormaps["tab10"].resampled(len(all_grades))
        colors = [cmap(i) for i in range(len(all_grades))]

        def _bar(ax, grade_pcts, title):
            bars = ax.bar(
                [str(g) for g in all_grades],
                [grade_pcts.get(g, 0) for g in all_grades],
                color=colors, edgecolor="white", linewidth=0.6,
            )
            ax.set_title(title, fontsize=11, pad=8)
            ax.set_xlabel("KL Grade")
            ax.set_ylabel("% of joint images")
            ax.set_ylim(0, 100)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 1.5,
                            f"{h:.1f}%", ha="center", va="bottom", fontsize=8)

        for i, joint in enumerate(joints):
            sub = df[df["joint"] == joint]
            gpct = sub["label"].value_counts(normalize=True).mul(100).to_dict()
            _bar(axes[i], gpct, f"{joint.upper()} (n={len(sub)})")

        overall_pct = df["label"].value_counts(normalize=True).mul(100).to_dict()
        _bar(axes[len(joints)], overall_pct, f"All Joints Combined (n={len(df)})")

        for j in range(len(joints) + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("KL Grade Distribution by Joint", fontsize=14)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Figure saved → {save_path}")
        else:
            plt.show()

        return result

    def compute_histograms(
        self,
        joints: set[str] | None = None,
        save_path: Path | str | None = None,
    ) -> None:
        """Per-joint grayscale intensity histograms (reads images lazily)."""
        df = self.data.copy()
        if joints is not None:
            df = df[df["joint"].isin(joints)]

        target_joints = sorted(df["joint"].unique())
        hists: dict[str, np.ndarray] = {j: np.zeros(256) for j in target_joints}
        total = np.zeros(256)

        with zipfile.ZipFile(self.zip_path, "r") as zf:
            for _, row in df.iterrows():
                img = _read_image(zf, self._file_map[row["filename"]])
                hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
                hists[row["joint"]] += hist
                total += hist

        hists["total"] = total

        n = len(hists)
        cols = 2
        rows = math.ceil(n / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows),
                                 sharex=True, constrained_layout=True)
        axes = axes.flatten()

        for i, (joint_name, hist) in enumerate(hists.items()):
            axes[i].plot(hist, linewidth=1.2)
            axes[i].fill_between(range(256), hist, alpha=0.25)
            axes[i].set_title(f"{joint_name.upper()} — Intensity Distribution")
            axes[i].set_xlim([0, 255])
            axes[i].set_xlabel("Pixel Intensity")
            axes[i].set_ylabel("Frequency")

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Grayscale Intensity Distributions by Joint", fontsize=14)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()

    def plot_image_size(self, save_path: Path | str | None = None) -> pd.DataFrame:
        """
        Scatter plot of image dimensions coloured by joint type.
        Reads all images once to extract shape — may be slow on 40k+ images.
        """
        records = []
        with zipfile.ZipFile(self.zip_path, "r") as zf:
            for _, row in self.data.iterrows():
                img = _read_image(zf, self._file_map[row["filename"]])
                records.append({
                    "filename": row["filename"],
                    "joint": row["joint"],
                    "width": img.shape[1],
                    "height": img.shape[0],
                })

        sizes = pd.DataFrame(records)
        joints = sorted(sizes["joint"].dropna().unique())
        cmap = plt.colormaps["tab10"].resampled(len(joints))
        j_color = {j: cmap(i) for i, j in enumerate(joints)}

        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

        for joint in joints:
            sub = sizes[sizes["joint"] == joint]
            ax.scatter(sub["width"], sub["height"],
                       c=[j_color[joint]], alpha=0.65, s=22,
                       label=f"{joint.upper()} (n={len(sub)})", zorder=2)

        ax.set_xlabel("Width (px)")
        ax.set_ylabel("Height (px)")
        ax.set_title("Image Dimensions by Joint Type")
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()

        return sizes

    def __repr__(self) -> str:
        return (f"RawImageDataset(n={len(self)}, "
                f"joints={sorted(self.data['joint'].unique().tolist())})")


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ds = RawImageDataset()

    print("\nLabel distribution:")
    print(ds.summary().to_string())

    img, label, joint = ds[0]
    print(f"\nSample 0 → joint={joint!r}, KL={label}, shape={img.shape}, dtype={img.dtype}")

    ds.compute_histograms()
    ds.plot_image_size()
    ds.display_image(0)