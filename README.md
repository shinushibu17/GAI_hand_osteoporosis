# OA Augmentation Comparison — Setup & Run Guide

## Quickstart (SCC)

```bash
# 1. Request GPU node
srun --pty -p gpu -l gpus=1 --mem=32G --time=08:00:00 bash

# 2. Load modules (adjust for your SCC setup)
module load python3/3.10.12 cuda/12.1

# 3. Create env (first time only)
conda create -n oa_aug python=3.10 -y
conda activate oa_aug
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 4. Point to your dataset — edit config.py:
#    data_root  = "/path/to/your/images"
#    metadata_csv = "/path/to/metadata.csv"

# 5. Run the full pipeline
bash run_pipeline.sh
```

## Dataset CSV format

The metadata CSV must have at minimum these columns (column names configurable in `config.py`):

| patient_id | image_path              | kl_grade |
|------------|-------------------------|----------|
| P001       | images/P001_DIP2_L.png  | 0        |
| P001       | images/P001_PIP3_L.png  | 2        |
| P002       | images/P002_MCP2_R.png  | 3        |

- `image_path` can be absolute or relative to `data_root`
- `kl_grade` must be integer 0–4

## Run options

```bash
bash run_pipeline.sh           # CycleGAN + WGAN-GP + CVAE (recommended for 8hr)
bash run_pipeline.sh --ddpm    # also trains DDPM (needs ~3 extra hours)
bash run_pipeline.sh --fast    # smoke test (5 epochs, 1 run)
bash run_pipeline.sh --resume  # resume interrupted training
```

## Running individual steps

```bash
python train_baseline.py                     # Step 1: baseline
python train_cyclegan.py --pair 1,3          # CycleGAN KL1→KL3 only
python train_cyclegan.py --pair 2,4          # CycleGAN KL2→KL4 only
python train_wgan_gp.py                      # WGAN-GP
python train_cvae.py                         # CVAE
python train_ddpm.py                         # DDPM (optional)
python generate_samples.py --n 1000          # generate synthetic images
python train_augmented.py                    # augmented classifiers
python evaluate.py                           # FID, faithfulness, table
```

## Output structure

```
outputs/
├── checkpoints/
│   ├── cyclegan_kl1_to_kl3/latest.pth
│   ├── cyclegan_kl2_to_kl4/latest.pth
│   ├── wgan_gp/latest.pth
│   ├── cvae/latest.pth
│   ├── ddpm/latest.pth          (if trained)
│   ├── baseline_run0.pth
│   └── ...
├── synthetic/
│   ├── cyclegan/kl3/*.png
│   ├── cyclegan/kl4/*.png
│   ├── wgan_gp/kl3/*.png
│   ├── wgan_gp/kl4/*.png
│   ├── cvae/kl3/*.png
│   └── cvae/kl4/*.png
└── results/
    ├── baseline_results.json        ← baseline test metrics (3 runs)
    ├── augmented_results.json       ← all augmented conditions
    ├── fid_faithfulness.json        ← FID + label faithfulness
    ├── comparison_table.csv         ← main deliverable
    └── comparison_table.txt         ← human-readable table
```

## Expected runtime (A100, 128×128 images)

| Step               | Time     |
|--------------------|----------|
| Baseline (3 runs)  | ~15 min  |
| CycleGAN ×2        | ~2.5 hr  |
| WGAN-GP            | ~1.5 hr  |
| CVAE               | ~45 min  |
| DDPM (optional)    | ~2.5 hr  |
| Generate samples   | ~10 min  |
| Augmented CLF      | ~1.5 hr  |
| Evaluation (FID)   | ~10 min  |
| **Total (no DDPM)**| **~7 hr**|

## Key config knobs (config.py)

| Parameter         | Default | Notes                                   |
|-------------------|---------|------------------------------------------|
| `img_size`        | 128     | Generation resolution (128 = fast, 256 = better quality) |
| `n_epochs_cyclegan` | 200   | Reduce to 100 if short on time          |
| `n_epochs_wgan`   | 200     | Same                                    |
| `batch_size_gen`  | 32      | Increase to 64 if VRAM allows           |
| `n_clf_runs`      | 3       | Reduce to 1 for a quick pass            |
| `aug_ratios`      | [0.3, 0.5, 1.0] | Per-grade augmentation ratios  |

## Adjusting for VRAM

- **< 16 GB**: reduce `img_size` to 64, `batch_size_gen` to 16
- **16-40 GB A100**: defaults work fine
- **V100 (16 GB)**: keep `img_size=128`, `batch_size_gen=16-32`

## Notes

- Patient-level split is fixed at first run and reused across all conditions
- All classifiers use the same split — direct comparison is valid
- Weighted sampling during generator training prioritises KL3/KL4
- CycleGAN trains separate models for KL1→KL3 and KL2→KL4 (more control per pair)
- WGAN-GP and CVAE train on all grades with weighted sampling (condition on grade label)
