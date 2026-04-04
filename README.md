# Hand Osteoporosis — KL Grade Classification & Image Generation

A deep learning project for classifying Kellgren-Lawrence (KL) grades from finger-joint X-ray images, generating synthetic X-rays that simulate disease progression, and measuring whether synthetic data improves classifier performance.

## Project structure

```
GAI_hand_osteoporosis/
├── data/
│   ├── hand.xlsx                    # 3 590 patients × 275 radiographic score columns
│   └── Finger Joints.zip            # 41 060 cropped finger-joint PNG images
├── models/
│   ├── kl_classifier.py             # ResNet-18 CNN — predicts KL grade from X-ray images
│   ├── kl_transition_generator.py   # CycleGAN — KL1→KL2 and KL2→KL3 image translation
│   ├── diffusion_generator.py       # Conditional DDPM — state-of-the-art generation + SDEdit translation
│   └── augmentation_study.py        # Experiment: does synthetic data improve classification?
└── eda_hand_osteoporosis.ipynb      # Extensive EDA across tabular data and images
```

## Dataset

### hand.xlsx
- **3 590 patients**, **275 columns**
- Two visit timepoints: `v00` (baseline) and `v06` (6-month follow-up)
- **16 joint types** per hand: DIP2–5, IP1, PIP2–5, MCP1–5, CMC1, STT
- **7 radiographic score types** per joint:

| Suffix | Score | Range |
|--------|-------|-------|
| `_KL`  | Kellgren-Lawrence grade | 0–4 |
| `_OP`  | Osteophytes | 0–3 |
| `_JSN` | Joint Space Narrowing | 0–3 |
| `_PW`  | Periarticular Width | 0–1 |
| `_ER`  | Erosion | 0–1 |
| `_ME`  | Malalignment/Erosion | 0–1 |
| `_CY`  | Cysts | 0–1 |

### Finger Joints.zip
- **41 060 PNG images** across **3 556 unique patients**
- 12 joint types: `dip2–5`, `pip2–5`, `mcp2–5`
- Filename format: `{patient_id}_{joint}.png` (e.g. `9000099_dip2.png`)
- Patient IDs match the `id` column in `hand.xlsx`
- KL distribution: KL=0 (31 510) · KL=1 (4 539) · KL=2 (4 485) · KL=3 (308) · KL=4 (233)

## EDA notebook

Covers 19 sections including missing value analysis, score distributions, joint-level KL profiles, v00→v06 disease progression, correlation heatmaps, image size analysis, KL-stratified image galleries, and pixel intensity statistics by KL grade.

```bash
jupyter notebook eda_hand_osteoporosis.ipynb
```

## Models

### KL Image Classifier (`models/kl_classifier.py`)

Fine-tunes a pretrained **ResNet-18** (grayscale-adapted) to classify KL grade (0–4) directly from finger-joint X-ray images.

**Key design choices:**
- Patient-level train/val/test split — no data leakage across patients
- `WeightedRandomSampler` + weighted cross-entropy loss for class imbalance
- Pretrained conv1 weights averaged across RGB channels → single grayscale channel
- Cosine annealing LR with linear warmup
- Grad-CAM visualisations showing which image regions drive predictions

**Outputs** → `models/kl_image_clf/`
- `best_model.pt` — best validation checkpoint
- `training_curves.png`, `confusion_matrix.png`, `roc_curves.png`, `gradcam_samples.png`

```bash
python models/kl_classifier.py                         # all joints, 25 epochs
python models/kl_classifier.py --joint dip2            # single joint
python models/kl_classifier.py --epochs 50 --img_size 128
python models/kl_classifier.py --freeze_backbone       # head-only training
```

---

### KL Transition Generator (`models/kl_transition_generator.py`)

**CycleGAN** for unpaired image-to-image translation between KL grades, simulating disease progression.

Why CycleGAN? We have no *paired* images (the same joint photographed at two KL grades). CycleGAN learns the mapping from unpaired examples using cycle-consistency: `G(F(b)) ≈ b` and `F(G(a)) ≈ a`.

- Generator: ResNet-based with 6 residual blocks + instance normalisation
- Discriminator: 70×70 PatchGAN with spectral normalisation
- Losses: LSGAN adversarial + cycle consistency (λ=10) + identity (λ=5)
- LR decay: constant for first half of training, linear decay to 0

**Outputs** → `models/kl{S}_to_kl{T}_cyclegan/`
- `G_final.pt` — forward generator (source → target KL)
- `F_final.pt` — inverse generator (target → source KL)
- `samples/epoch_XXXX.png`, `training_losses.png`

```bash
python models/kl_transition_generator.py --source_kl 1 --target_kl 2   # KL1 → KL2
python models/kl_transition_generator.py --source_kl 2 --target_kl 3   # KL2 → KL3
python models/kl_transition_generator.py --source_kl 1 --target_kl 2 --joint dip2 --epochs 300
```

---

### Diffusion Generator (`models/diffusion_generator.py`)

**Conditional DDPM** — the state-of-the-art approach, outperforming GANs in image quality, diversity, and training stability (no mode collapse).

Supports two inference modes:
- **Generation**: sample from noise conditioned on any KL grade
- **SDEdit translation**: add noise to a real KL=1 image partway, then denoise conditioning on KL=2 — produces a translated image that preserves patient anatomy

Architecture:
- UNet with time + class embeddings (sinusoidal + learnable)
- ResBlocks with GroupNorm and SiLU activations
- Self-attention at 16×16 and 8×8 resolutions
- Channels: [64, 128, 256, 512] — ~28M parameters
- DDIM sampling (50 steps) for fast inference

**Outputs** → `models/diffusion_kl/`
- `unet_best.pt`
- `samples/kl{K}_epoch_XXXX.png`, `training_loss.png`

```bash
python models/diffusion_generator.py --mode train                            # train
python models/diffusion_generator.py --mode generate --kl_grade 2           # generate KL=2
python models/diffusion_generator.py --mode generate --kl_grade 3           # generate KL=3
python models/diffusion_generator.py --mode translate --source_kl 1 --target_kl 2  # SDEdit
python models/diffusion_generator.py --mode translate --source_kl 2 --target_kl 3
```

---

### Augmentation Study (`models/augmentation_study.py`)

Answers the key question: **does training on synthetic data improve classification of real images?**

Experimental design:
- Same patient-level train/val/test split (same seed) as the classifier
- **Baseline**: ResNet-18 trained on real images only
- **Augmented**: ResNet-18 trained on real images + synthetic images for rare KL grades (KL=2 and KL=3, 500 images each)
- Both models evaluated on the **same real test set** — synthetic images never enter the test set
- Comparison: per-class F1, confusion matrices, ROC curves, summary CSV

**Outputs** → `models/augmentation_study/`
- `confusion_matrix_comparison.png`
- `f1_comparison.png` — per-class F1 delta (green = improved)
- `roc_comparison.png`
- `augmentation_study_results.csv`
- `baseline_model.pt`, `augmented_model.pt`

```bash
# Run after training the transition generator or diffusion model
python models/augmentation_study.py --generator cyclegan    # use CycleGAN outputs
python models/augmentation_study.py --generator diffusion   # use Diffusion outputs
```

## Installation

```bash
pip install -r requirements.txt

# GPU (recommended — check your CUDA version with nvidia-smi)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Recommended run order

1. **EDA notebook** — understand the data before training
2. **KL classifier** — validates the image–label linkage, establishes baseline CNN performance
3. **KL transition generator** — train KL1→KL2 and KL2→KL3 CycleGANs
4. **Diffusion generator** — train the DDPM (better quality, slower)
5. **Augmentation study** — compare baseline vs. augmented classifier on real test images
