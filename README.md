# Generative Data Augmentation for Hand Osteoarthritis KL Grade Classification

**CS790 Computer Vision · Boston University · April 2026**  
**Author:** Ittoop Shinu Shibu

---

## Overview

This project investigates whether generative data augmentation can improve detection of severe hand osteoarthritis (KL grades 3 and 4) using finger joint radiographs. We compare four generative models — CycleGAN, WGAN-GP, CVAE, and DDPM — for synthesising minority-class images and evaluate their downstream impact on a ResNet-18 classifier.

**Key finding:** Pooled augmentation across all joints does not reliably beat a class-weighted baseline. Targeted per-joint-group training with hyperparameter tuning shows directional KL4 recall improvement.

---

## Dataset

- **Source:** OAI (Osteoarthritis Initiative) Hand Radiograph Dataset
- **Images:** 41,051 finger joint X-rays
- **Patients:** 3,556
- **Joints:** DIP2-5, PIP2-5, MCP2-5 (12 joint types)
- **KL Grade Distribution:** KL0=76.7%, KL1=11.1%, KL2=10.9%, KL3=0.8%, KL4=0.6%
- **Imbalance ratio:** 135:1

---

## Models

| Model | Approach | Strength |
|---|---|---|
| CycleGAN | KL1→KL3, KL2→KL4 image translation | Preserves bone structure |
| WGAN-GP | Conditional generation from noise | Grade-specific synthesis |
| CVAE | Latent space sampling conditioned on grade | Smooth interpolation |
| DDPM | Classifier-free guidance diffusion | Highest quality ceiling |

---

## Pipeline

```
setup_data.py          # Extract zip + build metadata.csv from hand.xlsx
tune_models.py         # Hyperparameter search (3 configs x 30 epochs, FID-based)
train_cyclegan.py      # CycleGAN training with best hparams, saves best.pth
train_wgan_gp.py       # WGAN-GP training
train_cvae.py          # CVAE training
train_ddpm.py          # DDPM training (optional)
generate_samples.py    # Generate 1000 synthetic KL3/KL4 images per model
train_baseline.py      # ResNet-18 baseline (pooled + per-group)
train_augmented.py     # ResNet-18 with augmentation (aug ratios: 0.3,0.5,1.0,5.0,10.0)
evaluate.py            # FID, faithfulness, comparison table
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
--ddpm             # Include DDPM (slow, 150+ epochs needed)
--fast             # Quick test run (5 epochs, 20 samples)
```

---

## Results

### Baseline (ResNet-18, class-weighted CE loss, 3 runs)

| Group | Accuracy | Macro F1 | KL3 Recall | KL4 Recall |
|---|---|---|---|---|
| Pooled | 0.793 +/- 0.007 | 0.620 +/- 0.005 | 0.532 +/- 0.030 | 0.633 +/- 0.076 |
| DIP | pending | pending | pending | pending |
| PIP | pending | pending | pending | pending |
| MCP | pending | pending | pending | pending |

### Image Quality (FID & Faithfulness)

| Model | Avg FID | KL3 Faithfulness | KL4 Faithfulness |
|---|---|---|---|
| CycleGAN | 220.8 | 0.377 | 0.297 |
| WGAN-GP | 352.8 | 0.167 | 0.069 |
| CVAE | 421.7 | 0.034 | 0.000 |
| DDPM | n/a | n/a | n/a |

---

## Repository Structure

```
config.py              # All hyperparameters and paths
dataset.py             # Dataset classes, patient-level splits, joint group filtering
models/networks.py     # ResNetGenerator, PatchGAN, CVAE, WGAN architectures
train_*.py             # Training scripts
generate_samples.py    # Synthetic image generation
evaluate.py            # FID, faithfulness, comparison table
visualize_results.py   # Figures and charts
submit_jobs.sh         # Full pipeline orchestration
outputs/figures/       # Charts and confusion matrices
outputs/results/       # JSON result files
logs/                  # Training logs
```

---

## Citation

If you use this code, please cite:

```
Ittoop, S. (2026). Generative Data Augmentation for Hand Osteoarthritis
KL Grade Classification. CS790 Computer Vision, Boston University.
```

Related work:
- Cao et al. (2025). CycleGAN augmentation for hand OA. ACR Convergence 2025.
- Prezja et al. (2022). DeepFake knee OA X-rays. Scientific Reports.