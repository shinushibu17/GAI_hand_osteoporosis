# GAI Hand Osteoarthritis

Synthetic finger-joint X-ray generation and evaluation pipeline for osteoarthritis (OA) progression using CycleGAN. A ResNet18 classifier trained on real KL-graded images is used to evaluate the realism and clinical accuracy of generated images.

---

## Project Structure

```
GAI_hand_osteoporosis/
├── data/
│   ├── Finger Joints.zip        ← PNG X-ray images (41k+)
│   └── hand.xlsx                ← KL grade metadata
├── data_prep/
│   ├── __init__.py
│   ├── raw_data.py              ← lazy image loading + EDA
│   ├── preprocessing.py         ← dataset splitting + transform comparison
│   └── transforms.py            ← custom torchvision-compatible transforms
├── models/
│   └── kl_classifier/
│       └── pip_real_only_2025-04-08_14-30-22/   ← timestamped run output
│           ├── best_model.pt
│           ├── config.json
│           ├── results.json
│           ├── training_curves.png
│           ├── confusion_matrix.png
│           ├── roc_curves.png
│           ├── test_probs.npy
│           └── test_labels.npy
├── pretrained/
│   └── ResNet18/
│       └── resnet18_imagenet.pth   ← ImageNet pretrained weights (cached locally)
├── notebooks/
│   └── EDA_Display.ipynb
├── kl_classifier.py
├── requirements.txt
└── README.md
```

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/GAI_hand_osteoporosis.git
cd GAI_hand_osteoporosis
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add data files**

Place the following in `data/`:
- `Finger Joints.zip` — PNG X-ray archive
- `hand.xlsx` — KL grade metadata

**4. Download pretrained ResNet18 weights**

Run once on a machine with internet access:
```python
import torch
from torchvision import models
from pathlib import Path

path = Path("pretrained/ResNet18/resnet18_imagenet.pth")
path.parent.mkdir(parents=True, exist_ok=True)
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
torch.save(model.state_dict(), path)
```

If `pretrained/ResNet18/resnet18_imagenet.pth` exists, the classifier loads from it automatically — no internet required.

---

## Data Pipeline

### `data_prep/raw_data.py` — `RawImageDataset`

Lazy-loading dataset. Only metadata is held in memory — images are read from the zip on demand, making it safe for 40k+ images.

```python
from data_prep.raw_data import RawImageDataset

raw = RawImageDataset()
print(raw.data.head())       # filename | label | joint
raw.summary()                # KL grade % distribution with bar charts
raw.compute_histograms()     # per-joint grayscale intensity histograms
raw.plot_image_size()        # scatter plot of image dimensions by joint
raw.display_image(0)         # display single image inline
```

### `data_prep/preprocessing.py` — `CleanImageDataset`

EDA and splitting layer. Does not apply transforms — that belongs in the model class.

```python
from data_prep.preprocessing import CleanImageDataset

ds = CleanImageDataset(raw)
ds = CleanImageDataset(raw, joint="pip")              # single joint
ds = CleanImageDataset(raw, joint=["pip", "dip"])     # multiple joints
ds = CleanImageDataset(raw, small=True, pct=0.1)      # 10% stratified subsample

# Compare preprocessing pipelines visually
from torchvision import transforms
from data_prep.transforms import NLMFilter, CLAHE

pipelines = {
    "NLM + CLAHE": transforms.Compose([NLMFilter(), CLAHE(), transforms.ToTensor()]),
    "CLAHE only":  transforms.Compose([CLAHE(), transforms.ToTensor()]),
}
ds.compare_images(0, pipelines)

# Stratified split → _SplitDataset objects (transform set by model class)
train_ds, val_ds, test_ds = ds.split()
```

### `data_prep/transforms.py` — Custom Transforms

Torchvision-compatible grayscale transforms (PIL → PIL). Slot directly into `transforms.Compose`.

| Transform | Description |
|---|---|
| `CLAHE(clip_limit, tile_grid)` | Local contrast enhancement — not available in torchvision |
| `NLMFilter(h, template_window, search_window)` | Non-local means denoising — suppresses scan line artifacts |
| `BilateralFilter(d, sigma_color, sigma_space)` | Edge-preserving smoothing |
| `MedianFilter(kernel_size)` | Median blur for salt-and-pepper noise |
| `InvertGrayscale()` | Bitwise inversion |

```python
from data_prep.transforms import CLAHE, NLMFilter
from torchvision import transforms

pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    NLMFilter(h=5),
    CLAHE(clip_limit=1.0, tile_grid=(16, 16)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])
```

---

## KL Grade Classifier

### `kl_classifier.py` — `KLGradeClassifier`

ResNet18 classifier trained per joint type to evaluate CycleGAN output quality. Trained on real images only (baseline) and real + synthetic images (experimental) to measure the impact of generated data.

**Training pipeline per image:**
```
Resize(224×224) → RandomRotation(±10°) → NLMFilter → CLAHE → ToTensor → Normalize
```

**Evaluation pipeline (no augmentation):**
```
Resize(224×224) → NLMFilter → CLAHE → ToTensor → Normalize
```

Normalization mean and std are computed from the actual training data rather than ImageNet stats.

#### Train

```python
from kl_classifier import KLGradeClassifier

# Real only — baseline
clf_real = KLGradeClassifier(joint="pip", use_fake=False)
clf_real.train(train_ds, val_ds, epochs=30, batch_size=32)
clf_real.evaluate(test_ds)   # always evaluate on real images only

# Real + synthetic — experimental
clf_fake = KLGradeClassifier(joint="pip", use_fake=True)
clf_fake.train(train_ds_with_fake, val_ds, epochs=30)
clf_fake.evaluate(test_ds)   # same test set — fair comparison
```

#### Custom optimizer

```python
clf.train(
    train_ds, val_ds,
    optimizer_cls=torch.optim.SGD,
    optimizer_kwargs={"momentum": 0.9, "nesterov": True},
)
```

#### Predict — CycleGAN evaluation

```python
result = clf.predict(generated_image)
# {
#   "predicted_grade": 3,
#   "confidence": 0.81,
#   "probabilities": {"KL0": 0.02, "KL1": 0.05, "KL2": 0.08, "KL3": 0.81, "KL4": 0.04}
# }
```

#### Compare F1 — real vs synthetic

```python
from kl_classifier import compare_f1

# Using classifier instances (after evaluate())
compare_f1(clf_real, clf_fake)

# Using saved result folders
compare_f1(
    "models/kl_classifier/pip_real_only_2025-04-08_14-30-22",
    "models/kl_classifier/pip_real_fake_2025-04-08_16-05-11",
)
```

#### Outputs per run

Every training run creates a timestamped folder under `models/kl_classifier/`:

```
pip_real_only_2025-04-08_14-30-22/
├── best_model.pt          ← weights + embedded config
├── config.json            ← optimizer, transforms, hyperparameters
├── results.json           ← accuracy, macro F1, AUC, per-class F1
├── training_curves.png
├── confusion_matrix.png
├── roc_curves.png
├── test_probs.npy
├── test_labels.npy
└── f1_comparison.png      ← synthetic runs only
```

---

## Notebook

`notebooks/EDA_Display.ipynb` — exploratory data analysis. Add this at the top of the notebook:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path().resolve().parent))

from data_prep.raw_data import RawImageDataset
from data_prep.preprocessing import CleanImageDataset
from data_prep.transforms import CLAHE, NLMFilter
```

---

## Dependencies

```
torch >= 2.1
torchvision >= 0.16
opencv-python >= 4.8
scikit-learn >= 1.3
pandas >= 2.0
numpy >= 1.24
Pillow >= 10.0
matplotlib >= 3.8
tqdm >= 4.66
openpyxl >= 3.1
```

Install: `pip install -r requirements.txt`
