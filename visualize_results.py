"""
visualize_results.py — generates training curves, confusion matrices, and summary plots.

Usage:
    python3 visualize_results.py
    python3 visualize_results.py --joint dip --models cyclegan wgan_gp cvae

Outputs (saved to outputs/figures/):
    confusion_matrix_baseline.png
    confusion_matrix_{model}_{ratio}.png
    recall_comparison.png
    f1_comparison.png
    training_curves_{group}_{model}.png  (from logs)
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from config import CFG

FIG_DIR = Path(CFG.output_dir) / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "baseline":  "#64748B",
    "cyclegan":  "#0D9488",
    "wgan_gp":   "#0D1B3E",
    "cvae":      "#F59E0B",
    "ddpm":      "#14B8A6",
}
KL_NAMES = ["KL0", "KL1", "KL2", "KL3", "KL4"]


# ── Confusion Matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(cm, title, out_path, normalize=True):
    cm = np.array(cm)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.where(row_sums > 0, cm / row_sums, 0)
    else:
        cm_norm = cm

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(KL_NAMES, fontsize=10)
    ax.set_yticklabels(KL_NAMES, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    thresh = 0.5
    for i in range(5):
        for j in range(5):
            val = cm_norm[i, j]
            raw = int(cm[i, j]) if not normalize else f"{val:.2f}"
            ax.text(j, i, raw, ha="center", va="center", fontsize=9,
                    color="white" if val > thresh else "black")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def parse_gen_log(log_path: str, model: str):
    """Parse generator/discriminator loss from generative model log."""
    epochs, g_loss, d_loss = [], [], []
    if model == "cyclegan":
        pattern = re.compile(r"Epoch \[(\d+)/\d+\].*?G=([\d.]+).*?D=([\d.]+)")
    elif model == "wgan_gp":
        pattern = re.compile(r"Epoch \[(\d+)/\d+\].*?C=([-\d.]+).*?G=([-\d.]+)")
    elif model == "cvae":
        pattern = re.compile(r"Epoch \[(\d+)/\d+\].*?Loss=([\d.nan]+).*?Recon=([\d.nan]+)")
    else:
        return [], [], []
    try:
        with open(log_path) as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    epochs.append(int(m.group(1)))
                    try:
                        g_loss.append(float(m.group(2)))
                        d_loss.append(float(m.group(3)))
                    except ValueError:
                        g_loss.append(float("nan"))
                        d_loss.append(float("nan"))
    except Exception:
        pass
    return epochs, g_loss, d_loss


def plot_all_confusion_matrices(baseline_path, augmented_path):
    # Baseline mean CM
    if Path(baseline_path).exists():
        data = json.load(open(baseline_path))
        cms = [np.array(r["cm"]) for r in data.get("runs", []) if "cm" in r]
        if cms:
            avg_cm = np.mean(cms, axis=0)
            plot_confusion_matrix(avg_cm, "Baseline — Mean (3 runs)",
                                  FIG_DIR / "cm_baseline_mean.png")

    # Augmented — mean CM per condition
    if Path(augmented_path).exists():
        data = json.load(open(augmented_path))
        for model, ratios in data.items():
            for ratio, cond in ratios.items():
                cms = [np.array(r["cm"]) for r in cond.get("runs", []) if "cm" in r]
                if cms:
                    avg_cm = np.mean(cms, axis=0)
                    plot_confusion_matrix(
                        avg_cm,
                        f"{model} | ratio={ratio} | Mean",
                        FIG_DIR / f"cm_{model}_{ratio}_mean.png"
                    )


# ── Recall / F1 Comparison Bar Charts ────────────────────────────────────────

def parse_gen_log(log_path: str, model: str):
    """Parse generator/discriminator loss from generative model log."""
    epochs, g_loss, d_loss = [], [], []
    if model == "cyclegan":
        pattern = re.compile(r"Epoch \[(\d+)/\d+\].*?G=([\d.]+).*?D=([\d.]+)")
    elif model == "wgan_gp":
        pattern = re.compile(r"Epoch \[(\d+)/\d+\].*?C=([-\d.]+).*?G=([-\d.]+)")
    elif model == "cvae":
        pattern = re.compile(r"Epoch \[(\d+)/\d+\].*?Loss=([\d.nan]+).*?Recon=([\d.nan]+)")
    else:
        return [], [], []
    try:
        with open(log_path) as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    epochs.append(int(m.group(1)))
                    try:
                        g_loss.append(float(m.group(2)))
                        d_loss.append(float(m.group(3)))
                    except ValueError:
                        g_loss.append(float("nan"))
                        d_loss.append(float("nan"))
    except Exception:
        pass
    return epochs, g_loss, d_loss


def plot_all_confusion_matrices(baseline_path, augmented_path):
    # Baseline — average across runs
    if Path(baseline_path).exists():
        data = json.load(open(baseline_path))
        cms = [np.array(r["cm"]) for r in data.get("runs", []) if "cm" in r]
        if cms:
            avg_cm = np.mean(cms, axis=0)
            plot_confusion_matrix(avg_cm, "Baseline — Mean (3 runs)",
                                  FIG_DIR / "cm_baseline_mean.png")

    # Augmented — average across runs per condition
    if Path(augmented_path).exists():
        data = json.load(open(augmented_path))
        for model, ratios in data.items():
            for ratio, cond in ratios.items():
                cms = [np.array(r["cm"]) for r in cond.get("runs", []) if "cm" in r]
                if cms:
                    avg_cm = np.mean(cms, axis=0)
                    plot_confusion_matrix(
                        avg_cm,
                        f"{model} | ratio={ratio} | Mean ({len(cms)} runs)",
                        FIG_DIR / f"cm_{model}_{ratio}_mean.png"
                    )


def plot_recall_comparison(baseline_path, augmented_path, metric="recall_3", ylabel="KL3 Recall"):
    fig, ax = plt.subplots(figsize=(12, 5))

    labels = ["Baseline"]
    means = []
    stds = []
    colors = [COLORS["baseline"]]

    if Path(baseline_path).exists():
        data = json.load(open(baseline_path))
        agg = data.get("aggregate", {})
        m = agg.get(metric, {})
        means.append(m.get("mean", 0))
        stds.append(m.get("std", 0))

    if Path(augmented_path).exists():
        data = json.load(open(augmented_path))
        for model, ratios in data.items():
            for ratio, cond in ratios.items():
                agg = cond.get("aggregate", {})
                m = agg.get(metric, {})
                labels.append(f"{model}\n{ratio}")
                means.append(m.get("mean", 0))
                stds.append(m.get("std", 0))
                colors.append(COLORS.get(model, "#888"))

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.85,
                  edgecolor="white", linewidth=0.5)

    # Baseline reference line
    if means:
        ax.axhline(means[0], color=COLORS["baseline"], linestyle="--",
                   linewidth=1.5, alpha=0.7, label=f"Baseline ({means[0]:.3f})")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f"{ylabel} — All Augmentation Conditions", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)

    # Value labels on bars
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=7.5)

    plt.tight_layout()
    fname = FIG_DIR / f"comparison_{metric}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ── Training Curves from Logs ─────────────────────────────────────────────────

def parse_training_log(log_path: str, model: str):
    """Parse val_acc, KL3, KL4 from classifier log — returns per-condition averaged curves."""
    import re
    # Group by condition (model_augX.X) and run
    conditions = {}  # condition -> run_id -> {epoch -> metrics}
    pattern = re.compile(
        r"\[([\w._]+)\] run=(\d+) ep=(\d+)/\d+.*?val_acc=([\d.]+).*?KL3=([\d.]+).*?KL4=([\d.]+)"
    )
    try:
        with open(log_path) as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    cond = m.group(1)
                    run = int(m.group(2))
                    epoch = int(m.group(3))
                    if cond not in conditions:
                        conditions[cond] = {}
                    if run not in conditions[cond]:
                        conditions[cond][run] = {"epochs": [], "val_acc": [], "kl3": [], "kl4": []}
                    conditions[cond][run]["epochs"].append(epoch)
                    conditions[cond][run]["val_acc"].append(float(m.group(4)))
                    conditions[cond][run]["kl3"].append(float(m.group(5)))
                    conditions[cond][run]["kl4"].append(float(m.group(6)))
    except Exception:
        pass
    return conditions


def plot_training_curves(log_dir="logs"):
    log_dir = Path(log_dir)
    if not log_dir.exists():
        return

    # Generative model training curves
    for log_file in log_dir.glob("group_*.out"):
        group = log_file.stem.replace("group_", "")
        for model in ["cyclegan", "wgan_gp", "cvae"]:
            epochs, g_loss, d_loss = parse_gen_log(str(log_file), model)
            if not epochs:
                continue
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(epochs, g_loss, label="G loss" if model != "cvae" else "Total loss",
                    color=COLORS.get(model, "#888"), linewidth=1.5)
            if any(not np.isnan(x) for x in d_loss):
                ax.plot(epochs, d_loss, label="D loss" if model != "cvae" else "Recon loss",
                        color="#EF4444", linewidth=1.5, linestyle="--")
            ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
            ax.set_title(f"{model.upper()} Training — {group.upper()} group", fontweight="bold")
            ax.legend(fontsize=9); ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
            plt.tight_layout()
            fname = FIG_DIR / f"training_curve_{group}_{model}.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {fname}")

    # Classifier validation curves — one line per condition, averaged across runs
    for log_file in list(log_dir.glob("clf_*.out")) + list(log_dir.glob("baseline*.out")):
        name = log_file.stem
        conditions = parse_training_log(str(log_file), name)
        if not conditions:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        cmap = plt.cm.tab10
        colors = [cmap(i) for i in range(len(conditions))]

        for (cond, runs), color in zip(conditions.items(), colors):
            # Average across runs for each epoch
            max_ep = max(max(r["epochs"]) for r in runs.values())
            epochs = list(range(1, max_ep + 1))

            def avg_metric(key):
                run_arrays = []
                for run_data in runs.values():
                    arr = np.full(max_ep, np.nan)
                    for ep, val in zip(run_data["epochs"], run_data[key]):
                        if ep - 1 < max_ep:
                            arr[ep - 1] = val
                    run_arrays.append(arr)
                return np.nanmean(np.array(run_arrays), axis=0)

            label = cond.replace("_aug", " r=")
            axes[0].plot(epochs, avg_metric("val_acc"), color=color, linewidth=1.5, label=label)
            axes[1].plot(epochs, avg_metric("kl3"), color=color, linewidth=1.5, linestyle="-", label=f"{label} KL3")
            axes[1].plot(epochs, avg_metric("kl4"), color=color, linewidth=1.0, linestyle="--")

        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
        axes[0].set_title("Validation Accuracy (avg across runs)", fontweight="bold")
        axes[0].set_ylim(0, 1); axes[0].yaxis.grid(True, alpha=0.3)
        axes[0].legend(fontsize=7, ncol=2)

        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Recall")
        axes[1].set_title("KL3 Recall per condition (solid) | KL4 (dashed)", fontweight="bold")
        axes[1].set_ylim(0, 1); axes[1].yaxis.grid(True, alpha=0.3)
        axes[1].legend(fontsize=7, ncol=2)

        fig.suptitle(f"Training Curves — {name}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        fname = FIG_DIR / f"val_curve_{name}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")


# ── Summary Dashboard ─────────────────────────────────────────────────────────

def plot_summary_dashboard(baseline_path, augmented_path):
    if not Path(baseline_path).exists():
        print("  No baseline results found")
        return

    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    metrics = [
        ("accuracy",  "Accuracy",   gs[0, 0]),
        ("macro_f1",  "Macro F1",   gs[0, 1]),
        ("recall_3",  "KL3 Recall", gs[0, 2]),
        ("recall_4",  "KL4 Recall", gs[1, 0]),
        ("recall_1",  "KL1 Recall", gs[1, 1]),
        ("recall_2",  "KL2 Recall", gs[1, 2]),
    ]

    baseline_data = json.load(open(baseline_path))
    aug_data = json.load(open(augmented_path)) if Path(augmented_path).exists() else {}

    for metric, title, pos in metrics:
        ax = fig.add_subplot(pos)

        labels, means, colors_list = [], [], []

        # Baseline
        agg = baseline_data.get("aggregate", {})
        m = agg.get(metric, {})
        labels.append("Base"); means.append(m.get("mean", 0))
        colors_list.append(COLORS["baseline"])

        # Augmented — best ratio per model
        for model, ratios in aug_data.items():
            best_mean = -1; best_label = ""
            for ratio, cond in ratios.items():
                v = cond.get("aggregate", {}).get(metric, {}).get("mean", 0)
                if v > best_mean:
                    best_mean = v; best_label = f"{model[:3]}\n{ratio}"
            if best_mean >= 0:
                labels.append(best_label); means.append(best_mean)
                colors_list.append(COLORS.get(model, "#888"))

        x = np.arange(len(labels))
        ax.bar(x, means, color=colors_list, alpha=0.85, edgecolor="white")
        ax.axhline(means[0], color=COLORS["baseline"], linestyle="--", linewidth=1, alpha=0.6)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)

    # Legend
    patches = [mpatches.Patch(color=c, label=m) for m, c in COLORS.items()]
    fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Performance Summary — Best Augmentation per Model vs Baseline",
                 fontsize=13, fontweight="bold")
    fname = FIG_DIR / "summary_dashboard.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline",   default="outputs/results/baseline_results.json")
    parser.add_argument("--augmented",  default="outputs/results/augmented_results_pooled.json")
    parser.add_argument("--logs",       default="logs")
    parser.add_argument("--no-curves",  action="store_true")
    args = parser.parse_args()

    print(f"\nGenerating figures → {FIG_DIR}")

    print("\n── Confusion matrices ───────────────────────────────────────")
    plot_all_confusion_matrices(args.baseline, args.augmented)

    print("\n── Recall / F1 comparison charts ────────────────────────────")
    for metric, label in [
        ("recall_3", "KL3 Recall"),
        ("recall_4", "KL4 Recall"),
        ("macro_f1", "Macro F1"),
        ("accuracy", "Accuracy"),
    ]:
        plot_recall_comparison(args.baseline, args.augmented, metric, label)

    print("\n── Summary dashboard ────────────────────────────────────────")
    plot_summary_dashboard(args.baseline, args.augmented)

    if not args.no_curves:
        print("\n── Training curves ──────────────────────────────────────────")
        plot_training_curves(args.logs)

    print(f"\nAll figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()