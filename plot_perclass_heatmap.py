"""
plot_perclass_heatmap.py — visualize per-class accuracy across all augmentation conditions.

Loads the JSON produced by eval_checkpoints.py and generates:
  1. Per-class accuracy heatmap (absolute values)
  2. Per-class accuracy delta vs baseline (so you can see what each condition trades off)

Usage:
    python3 plot_perclass_heatmap.py --results results/augmented_results_dip_reeval.json \
                                     --baseline results/baseline_results.json \
                                     --out figures/

If per-class recalls (recall_0..recall_4) are missing from the JSON,
the script falls back to extracting them from the confusion matrix diagonal.
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

CLASSES = ["KL0", "KL1", "KL2", "KL3", "KL4"]
N_CLASSES = len(CLASSES)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_per_class_recall(agg: dict, runs: list) -> np.ndarray:
    """
    Return per-class recall as a (5,) array.
    Priority:
      1. recall_0..recall_4 stored directly in aggregate means
      2. confusion matrix diagonal from individual runs (averaged)
    """
    keys = [f"recall_{i}" for i in range(N_CLASSES)]

    # Option 1: explicit per-class recalls in aggregate
    if all(k in agg for k in keys):
        return np.array([agg[k]["mean"] for k in keys])

    # Option 2: derive from confusion matrix stored in each run
    cms = []
    for run in runs:
        if "cm" in run:
            cm = np.array(run["cm"])  # shape (5, 5), row = true, col = pred
            with np.errstate(divide="ignore", invalid="ignore"):
                row_sums = cm.sum(axis=1, keepdims=True)
                normed = np.where(row_sums > 0, cm / row_sums, 0.0)
            cms.append(np.diag(normed))
    if cms:
        return np.mean(cms, axis=0)

    # Option 3: fall back to recall_3 / recall_4 only, fill rest with NaN
    out = np.full(N_CLASSES, np.nan)
    for i in range(N_CLASSES):
        k = f"recall_{i}"
        if k in agg:
            out[i] = agg[k]["mean"]
    return out


def load_results(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def build_matrix(results: dict, models: list, ratios: list):
    """
    Returns:
        matrix  : (n_conditions, 5) per-class accuracy
        labels  : list of condition label strings
    """
    matrix, labels = [], []
    for model in models:
        if model not in results:
            continue
        for ratio in ratios:
            key = str(ratio)
            if key not in results[model]:
                continue
            entry = results[model][key]
            recall = extract_per_class_recall(entry["aggregate"], entry["runs"])
            matrix.append(recall)
            labels.append(f"{model}\nr={ratio}")
    return np.array(matrix), labels


def get_baseline_recall(baseline_results: dict) -> np.ndarray:
    """
    Baseline JSON can be either:
      { "aggregate": {...}, "runs": [...] }   (single entry)
      or the same model/ratio nested structure with key "baseline"
    """
    if "aggregate" in baseline_results:
        return extract_per_class_recall(
            baseline_results["aggregate"], baseline_results.get("runs", [])
        )
    # Try nested: baseline_results["baseline"]["aggregate"]
    if "baseline" in baseline_results:
        entry = baseline_results["baseline"]
        if isinstance(entry, dict) and "aggregate" in entry:
            return extract_per_class_recall(entry["aggregate"], entry.get("runs", []))
    raise ValueError(
        "Could not parse baseline JSON. Expected top-level 'aggregate' key "
        "or nested 'baseline' > 'aggregate'."
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_absolute(matrix: np.ndarray, labels: list, baseline: np.ndarray,
                  out_dir: Path):
    """Heatmap of absolute per-class accuracy, baseline shown as first row."""
    full_matrix = np.vstack([baseline[np.newaxis, :], matrix])
    full_labels = ["baseline"] + labels

    fig, ax = plt.subplots(figsize=(max(8, len(CLASSES) * 1.4),
                                    max(5, len(full_labels) * 0.45 + 1.5)))

    cmap = plt.cm.Blues
    im = ax.imshow(full_matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(N_CLASSES))
    ax.set_xticklabels(CLASSES, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(full_labels)))
    ax.set_yticklabels(full_labels, fontsize=8)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    # Annotate cells
    for r in range(full_matrix.shape[0]):
        for c in range(N_CLASSES):
            val = full_matrix[r, c]
            if np.isnan(val):
                txt, color = "—", "grey"
            else:
                txt = f"{val:.2f}"
                color = "white" if val > 0.65 else "black"
            ax.text(c, r, txt, ha="center", va="center",
                    fontsize=7.5, color=color)

    # Highlight baseline row
    for c in range(N_CLASSES):
        ax.add_patch(plt.Rectangle((c - 0.5, -0.5), 1, 1,
                                   fill=False, edgecolor="orange",
                                   linewidth=1.5))

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Per-class Accuracy")
    ax.set_title("Per-Class Accuracy — All Augmentation Conditions",
                 fontsize=13, fontweight="bold", pad=16)
    fig.tight_layout()
    out = out_dir / "perclass_accuracy_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_delta(matrix: np.ndarray, labels: list, baseline: np.ndarray,
               out_dir: Path):
    """Heatmap of per-class accuracy delta vs baseline (diverging colormap)."""
    delta = matrix - baseline[np.newaxis, :]  # (n_conditions, 5)

    abs_max = np.nanmax(np.abs(delta))
    abs_max = max(abs_max, 0.05)  # floor so colormap isn't flat

    fig, ax = plt.subplots(figsize=(max(8, N_CLASSES * 1.4),
                                    max(5, len(labels) * 0.45 + 1.5)))

    cmap = plt.cm.RdYlGn
    im = ax.imshow(delta, cmap=cmap, aspect="auto",
                   vmin=-abs_max, vmax=abs_max)

    ax.set_xticks(range(N_CLASSES))
    ax.set_xticklabels(CLASSES, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    for r in range(delta.shape[0]):
        for c in range(N_CLASSES):
            val = delta[r, c]
            if np.isnan(val):
                txt, color = "—", "grey"
            else:
                sign = "+" if val >= 0 else ""
                txt = f"{sign}{val:.2f}"
                color = "black"
            ax.text(c, r, txt, ha="center", va="center",
                    fontsize=7.5, color=color)

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02,
                 label="Δ Accuracy vs Baseline")
    ax.set_title("Per-Class Accuracy Delta vs Baseline\n"
                 "(green = better than baseline, red = worse)",
                 fontsize=13, fontweight="bold", pad=16)
    fig.tight_layout()
    out = out_dir / "perclass_accuracy_delta_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_tradeoff_scatter(matrix: np.ndarray, labels: list,
                          baseline: np.ndarray, out_dir: Path,
                          focus_classes=(3, 4), other_classes=(0, 1, 2)):
    """
    Scatter: x = mean accuracy gain on KL3+KL4,
             y = mean accuracy change on KL0+KL1+KL2.
    Each point = one augmentation condition.
    Makes the tradeoff structure immediately visible.
    """
    delta = matrix - baseline[np.newaxis, :]

    x = delta[:, list(focus_classes)].mean(axis=1)   # KL3+KL4 gain
    y = delta[:, list(other_classes)].mean(axis=1)   # KL0+KL1+KL2 change

    fig, ax = plt.subplots(figsize=(8, 6))

    # Color by model family
    model_colors = {
        "cyclegan": "#2ca02c",
        "wgan_gp":  "#1f77b4",
        "cvae":     "#ff7f0e",
        "cyclegan_vgg": "#9467bd",
    }
    for i, label in enumerate(labels):
        model = label.split("\n")[0]
        color = model_colors.get(model, "grey")
        ax.scatter(x[i], y[i], color=color, s=80, zorder=3)
        ax.annotate(label.replace("\n", " "), (x[i], y[i]),
                    textcoords="offset points", xytext=(5, 3),
                    fontsize=6.5, color=color)

    # Reference lines
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    # Quadrant labels
    ax.text(0.97, 0.97, "KL3/4 ↑ others ↑\n(ideal)", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color="green", alpha=0.7)
    ax.text(0.97, 0.03, "KL3/4 ↑ others ↓\n(tradeoff)", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="red", alpha=0.7)
    ax.text(0.03, 0.03, "KL3/4 ↓ others ↓\n(avoid)", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=8, color="darkred", alpha=0.7)
    ax.text(0.03, 0.97, "KL3/4 ↓ others ↑\n(baseline better)", transform=ax.transAxes,
            ha="left", va="top", fontsize=8, color="grey", alpha=0.7)

    # Legend
    for model, color in model_colors.items():
        ax.scatter([], [], color=color, label=model, s=60)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1, 0.5))

    ax.set_xlabel("Mean Δ Accuracy on KL3 + KL4", fontsize=10)
    ax.set_ylabel("Mean Δ Accuracy on KL0 + KL1 + KL2", fontsize=10)
    ax.set_title("Augmentation Tradeoff: Target Classes vs Rest",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = out_dir / "perclass_tradeoff_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True,
                        help="Path to augmented_results_*_reeval.json")
    parser.add_argument("--baseline", type=Path, required=True,
                        help="Path to baseline results JSON")
    parser.add_argument("--models", nargs="+",
                        default=["cyclegan", "wgan_gp", "cvae", "cyclegan_vgg"])
    parser.add_argument("--ratios", nargs="+", type=float,
                        default=[0.3, 0.5, 1.0, 5.0, 10.0])
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    results = load_results(args.results)
    baseline_raw = load_results(args.baseline)
    baseline = get_baseline_recall(baseline_raw)

    print(f"Baseline per-class accuracy: "
          + " | ".join(f"{c}={v:.3f}" for c, v in zip(CLASSES, baseline)))

    matrix, labels = build_matrix(results, args.models, args.ratios)

    if len(matrix) == 0:
        print("No conditions found — check --results path and model/ratio args.")
        return

    plot_absolute(matrix, labels, baseline, args.out)
    plot_delta(matrix, labels, baseline, args.out)
    plot_tradeoff_scatter(matrix, labels, baseline, args.out)

    print("\nDone. Outputs:")
    print(f"  {args.out}/perclass_accuracy_heatmap.png")
    print(f"  {args.out}/perclass_accuracy_delta_heatmap.png")
    print(f"  {args.out}/perclass_tradeoff_scatter.png")


if __name__ == "__main__":
    main()
