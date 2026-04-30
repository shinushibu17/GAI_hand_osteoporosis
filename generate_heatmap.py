"""Generate per-class recall delta heatmap for DIP CycleGAN-VGG augmentation."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

# DIP baseline per-class recall (from baseline_dip_results.json aggregate)
baseline = {
    "KL0": 0.8634,
    "KL1": 0.3976,
    "KL2": 0.6966,
    "KL3": 0.5256,
    "KL4": 0.5833,
}

# DIP CycleGAN-VGG per-class recall at each ratio
# (from augmented_results_dip.json, cyclegan_vgg aggregate means)
augmented = {
    "0.3×": {"KL0": 0.8436, "KL1": 0.5055, "KL2": 0.6241, "KL3": 0.4615, "KL4": 0.5500},
    "0.5×": {"KL0": 0.8254, "KL1": 0.5115, "KL2": 0.6603, "KL3": 0.5256, "KL4": 0.3500},
    "1.0×": {"KL0": 0.8256, "KL1": 0.5103, "KL2": 0.6350, "KL3": 0.4744, "KL4": 0.5167},
    "5.0×": {"KL0": 0.8156, "KL1": 0.4958, "KL2": 0.6476, "KL3": 0.4103, "KL4": 0.4667},
    "10.0×": {"KL0": 0.8246, "KL1": 0.4715, "KL2": 0.6150, "KL3": 0.4744, "KL4": 0.4167},
}

classes = ["KL0", "KL1", "KL2", "KL3", "KL4"]
ratios = list(augmented.keys())

# Build delta matrix: shape (5 classes, 5 ratios)
delta = np.array([
    [augmented[r][c] - baseline[c] for r in ratios]
    for c in classes
])

# ---- Plot ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4.5))

# Diverging colormap, symmetric around 0
vmax = 0.25
norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
cmap = plt.cm.RdBu

im = ax.imshow(delta, cmap=cmap, norm=norm, aspect="auto")

# Axes
ax.set_xticks(range(len(ratios)))
ax.set_xticklabels(ratios, fontsize=12)
ax.set_yticks(range(len(classes)))
ax.set_yticklabels(classes, fontsize=12)
ax.set_xlabel("Augmentation Ratio (CycleGAN-VGG, DIP group)", fontsize=12)
ax.set_ylabel("KL Grade", fontsize=12)
ax.set_title("Per-Class Recall Δ vs. Baseline (DIP, CycleGAN-VGG)",
             fontsize=13, fontweight="bold")

# Annotate every cell
for i in range(len(classes)):
    for j in range(len(ratios)):
        val = delta[i, j]
        sign = "+" if val >= 0 else ""
        color = "white" if abs(val) > 0.12 else "black"
        ax.text(j, i, f"{sign}{val:.3f}", ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=color)

# Colorbar
cbar = fig.colorbar(im, ax=ax, pad=0.02)
cbar.set_label("Recall Δ (augmented − baseline)", fontsize=11)
cbar.ax.tick_params(labelsize=10)

# Draw grid lines between cells
ax.set_xticks(np.arange(-0.5, len(ratios), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(classes), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=2)
ax.tick_params(which="minor", bottom=False, left=False)

# Highlight minority-class rows
for i in [3, 4]:  # KL3, KL4
    for spine in ["top", "bottom", "left", "right"]:
        rect = plt.Rectangle((-0.5, i - 0.5), len(ratios), 1,
                              fill=False, edgecolor="black", linewidth=2.0,
                              zorder=10)
        ax.add_patch(rect)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__),
                        "outputs", "figures", "recall_delta_heatmap.png")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"Saved: {out_path}")
