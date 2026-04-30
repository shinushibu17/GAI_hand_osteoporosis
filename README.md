# Generative Data Augmentation for Hand Osteoarthritis KL Grade Classification

**CS790 Computer Vision · Boston University · April 2026**  
**Author:** Ittoop Shinu Shibu

---

## Overview

This project investigates whether generative data augmentation can improve detection of severe hand osteoarthritis (OA) grades (KL3, KL4) from finger joint radiographs. We compare three generative architectures — CycleGAN with VGG perceptual loss, WGAN-GP, and Conditional VAE — and evaluate their downstream impact on a ResNet-18 classifier across five augmentation ratios. The primary experimental focus is on **distal interphalangeal (DIP) joints**.

**Key finding:** No generative model reliably improves KL4 recall at any tested augmentation ratio. KL4 drops below the class-weighted baseline for all three models across all five ratios. CVAE achieves the highest single KL3 result (0.641 at 1.0×) but exhibits posterior collapse — its outputs are effectively noise and the result is not reproducible. These results reveal a critical gap between image realism and clinical faithfulness.

---

## Dataset

- **Source:** OAI (Osteoarthritis Initiative) Hand Radiograph Dataset
- **Images:** 41,051 finger joint X-rays
- **Patients:** 3,556
- **Joints:** DIP2–5, PIP2–5, MCP2–5 (12 joint types)
- **KL Grade Distribution:** KL0=76.7%, KL1=11.1%, KL2=10.9%, KL3=0.8%, KL4=0.6%
- **Imbalance ratio:** 135:1 (KL0 vs KL3)
- **Primary focus:** DIP (distal interphalangeal) joints

---

## Generative Models

| Model | Approach | Notes |
|---|---|---|
| CycleGAN+VGG | KL1→KL3, KL2→KL4 unpaired translation | Sharpest outputs; VGG perceptual loss preserves bone texture |
| WGAN-GP | Conditional generation from noise + gradient penalty | Anatomically plausible; partial mode-collapse artifacts |
| CVAE | Latent space sampling conditioned on KL grade | Posterior collapse confirmed; outputs are blurry/uninformative |

All models trained for up to **500 epochs with early stopping** (patience on validation FID). Hyperparameters selected via FID-based search (30 epochs, 100 samples).

---

## Pipeline

```
setup_data.py          # Extract zip + build metadata.csv from hand.xlsx
tune_models.py         # Hyperparameter search (configs x 30 epochs, FID-based)
train_cyclegan.py      # CycleGAN+VGG training, saves best checkpoint
train_wgan_gp.py       # WGAN-GP training
train_cvae.py          # CVAE training
generate_samples.py    # Generate 1000 synthetic KL3/KL4 images per model
train_baseline.py      # ResNet-18 baseline (DIP focus)
train_augmented.py     # ResNet-18 with augmentation (ratios: 0.3, 0.5, 1.0, 5.0, 10.0)
evaluate.py            # Downstream classification evaluation
generate_heatmap.py    # Per-class recall delta heatmap
visualize_results.py   # Confusion matrices, recall charts, training curves
submit_jobs.sh         # Orchestrates full pipeline across GPUs
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate metadata (requires hand.xlsx with KL grades)
python3 setup_data.py --zip "Finger Joints.zip" --data_root ./data --xlsx hand.xlsx

# 3. Run full pipeline (auto-detects GPUs)
bash submit_jobs.sh --zip "Finger Joints.zip"

# Optional flags
--skip-baseline    # Skip baseline training
--skip-clf         # Skip classifier training (generation only)
--fast             # Quick test run (5 epochs, 20 samples)
```

---

## Results

### DIP Baseline (ResNet-18, class-weighted CE loss, 3 runs)

| Accuracy | Macro F1 | KL3 Recall | KL4 Recall |
|---|---|---|---|
| 0.761 ± 0.010 | 0.587 ± 0.014 | 0.526 ± 0.096 | 0.583 ± 0.047 |

### Augmented Classifier — Best KL3 per Model (DIP)

| Model | Ratio | KL3 Recall | KL4 Recall | Note |
|---|---|---|---|---|
| CycleGAN+VGG | 1.0× | 0.551 ± 0.036 | 0.433 ± 0.118 | Best reliable KL3 gain |
| WGAN-GP | 10.0× | 0.603 ± 0.127 | 0.367 ± 0.085 | High variance |
| CVAE | 1.0× | **0.641 ± 0.065** | 0.467 ± 0.047 | Posterior collapse — unreliable |

**KL4 recall never exceeds baseline (0.583) for any model at any ratio.**

### Generation Quality

FID and label faithfulness were not computed in this experimental run. Qualitative assessment from final-epoch checkpoint images:

| Model | Visual Quality |
|---|---|
| CycleGAN+VGG | Sharp; clear bone trabeculae and joint-space narrowing |
| WGAN-GP | Sharp; recurring texture artifacts (partial mode collapse) |
| CVAE | Severely blurry; posterior collapse — one sample fully white |

---

## Repository Structure

```
config.py                          # Hyperparameters and paths
dataset.py                         # Dataset classes, patient-level splits
models/networks.py                 # Generator, PatchGAN, CVAE, WGAN architectures
train_*.py                         # Training scripts (cyclegan, wgan_gp, cvae)
generate_samples.py                # Synthetic image generation
evaluate.py                        # Classification evaluation
generate_heatmap.py                # Per-class recall delta heatmap
visualize_results.py               # Figures and confusion matrices
submit_jobs.sh                     # Full pipeline orchestration
outputs/checkpoints/dip/           # Final-epoch checkpoint images per model
outputs/results/                   # JSON result files (augmented_results_dip_reeval.json)
outputs/figures/                   # Heatmap and other figures
paper.tex                          # CS790 paper (LaTeX)
poster.tex                         # Conference-style poster (LaTeX)
presentation.tex                   # Beamer slide deck (LaTeX)
```

---

## Citation

```
Shibu, I. S. (2026). Generative Data Augmentation for Hand Osteoarthritis
KL Grade Classification. CS790 Computer Vision, Boston University.
```

Related work:
- Cao et al. (2025). CycleGAN and EfficientNetB7 for hand OA classification. ACR Convergence 2025, Abstract 2562.
- Zhu et al. (2017). Unpaired image-to-image translation using cycle-consistent adversarial networks. ICCV.
- Gulrajani et al. (2017). Improved training of Wasserstein GANs. NeurIPS.
