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
│   ├── kl_classifier.py         ← ResNet18 KL grade classifier
│   └── kl_classifier/           ← timestamped run outputs
│       └── pip_real_only_2025-04-08_14-30-22/
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
│       └── resnet18_imagenet.pth   ← ImageNet pretrained weights
├── notebooks/
│   └── EDA_Display.ipynb
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

Once saved the classifier loads it automatically — no internet required on subsequent runs. If the file is missing it falls back to downloading from PyTorch hub.

---

## Data Pipeline

### `data_prep/raw_data.py` — `RawImageDataset`

Lazy-loading dataset. Only metadata (`filename | label | joint`) is held in memory — images are read from the zip on demand, making it safe for 40k+ images.

```python
from data_prep.raw_data import RawImageDataset

raw = RawImageDataset()
print(raw.data.head())       # filename | label | joint
raw.summary()                # KL grade % distribution with bar charts
raw.compute_histograms()     # per-joint grayscale intensity histograms
raw.plot_image_size()        # scatter plot of image dimensions by joint
raw.display_image(0)         # display by index
raw.display_image("9000099_pip2.png")   # display by filename
```

### `data_prep/preprocessing.py` — `CleanImageDataset`

EDA and splitting layer. Does not apply transforms — transforms are set by the model class after splitting.

`compare_images()` accepts a dict of named pipelines to visually compare any combination of transforms on a single image.

```python
from data_prep.preprocessing import CleanImageDataset

ds = CleanImageDataset(raw)
ds = CleanImageDataset(raw, joint="pip")              # single joint
ds = CleanImageDataset(raw, joint=["pip", "dip"])     # multiple joints
ds = CleanImageDataset(raw, small=True, pct=0.1)      # 10% stratified subsample

# Compare preprocessing pipelines visually
from torchvision import transforms
from data_prep.transforms import NLMFilter, CLAHE, BilateralFilter

pipelines = {
    "NLM + CLAHE":       transforms.Compose([NLMFilter(), CLAHE(), transforms.ToTensor()]),
    "Bilateral + CLAHE": transforms.Compose([BilateralFilter(), CLAHE(), transforms.ToTensor()]),
    "CLAHE only":        transforms.Compose([CLAHE(), transforms.ToTensor()]),
}
ds.compare_images(0, pipelines)

# Stratified split → _SplitDataset objects
# Transforms are NOT applied here — the model class sets them
train_ds, val_ds, test_ds = ds.split()
```

### `data_prep/transforms.py` — Custom Transforms

Torchvision-compatible grayscale transforms (PIL → PIL). All slot directly into `transforms.Compose`. None are available in torchvision natively.

| Transform | Description |
|---|---|
| `CLAHE(clip_limit, tile_grid)` | Local contrast enhancement |
| `NLMFilter(h, template_window, search_window)` | Non-local means denoising — suppresses scan line artifacts |
| `BilateralFilter(d, sigma_color, sigma_space)` | Edge-preserving smoothing |
| `MedianFilter(kernel_size)` | Median blur |
| `InvertGrayscale()` | Bitwise inversion |

```python
from data_prep.transforms import CLAHE, NLMFilter
from torchvision import transforms

custom_pipeline = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    NLMFilter(h=5),
    CLAHE(clip_limit=1.0, tile_grid=(16, 16)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]),
])
```

---

## KL Grade Classifier

### `models/kl_classifier.py` — `KLGradeClassifier`

ResNet18 classifier trained per joint type to evaluate CycleGAN output quality. Two experiments are run per joint — real images only (baseline) and real + synthetic images — using the same held-out test set for fair comparison.

The joint type is inferred automatically from the training data — no need to specify it manually.

**Default training pipeline (standard ResNet18 / ImageNet convention):**
```
Resize(256) → CenterCrop(224) → RandomHorizontalFlip → RandomRotation(±10°) → ToTensor → Normalize(0.485, 0.229)
```

**Default eval/predict pipeline:**
```
Resize(256) → CenterCrop(224) → ToTensor → Normalize(0.485, 0.229)
```

`predict()` automatically uses the same eval pipeline set during `train()` or `evaluate()` — no manual pipeline management needed.

#### Train

```python
from models.kl_classifier import KLGradeClassifier

# Real only — baseline (joint inferred from data)
clf_real = KLGradeClassifier(use_fake=False)
clf_real.train(train_ds, val_ds, epochs=30, batch_size=32)
clf_real.evaluate(test_ds)   # always evaluate on real images only

# Real + synthetic — experimental
clf_fake = KLGradeClassifier(use_fake=True)
clf_fake.train(train_ds_with_fake, val_ds, epochs=30)
clf_fake.evaluate(test_ds)   # same test set — fair comparison
```

#### Custom transform pipeline

```python
from data_prep.transforms import NLMFilter, CLAHE
from torchvision import transforms

my_train_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    NLMFilter(h=5),
    CLAHE(clip_limit=1.0),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]),
])
my_eval_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    NLMFilter(h=5),
    CLAHE(clip_limit=1.0),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]),
])
clf.train(train_ds, val_ds, train_transform=my_train_tf, eval_transform=my_eval_tf)
```

#### Custom optimizer

```python
clf.train(
    train_ds, val_ds,
    optimizer_cls=torch.optim.SGD,
    optimizer_kwargs={"momentum": 0.9, "nesterov": True},
)
```

#### Early stopping

Stops automatically if val accuracy does not improve by `min_delta` for `patience` consecutive epochs:

```python
clf.train(train_ds, val_ds, patience=5, min_delta=1e-3)
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
from models.kl_classifier import compare_f1

# Using classifier instances (after evaluate())
compare_f1(clf_real, clf_fake)

# Using saved result folders
compare_f1(
    "models/kl_classifier/pip_real_only_2025-04-08_14-30-22",
    "models/kl_classifier/pip_real_fake_2025-04-08_16-05-11",
)
```

Saves `f1_comparison.png` to the synthetic model's output folder automatically.

#### Outputs per run

Every training run creates a timestamped folder under `models/kl_classifier/`:

```
pip_real_only_2025-04-08_14-30-22/
├── best_model.pt          ← weights + embedded config
├── config.json            ← optimizer, transform pipelines, hyperparameters
├── results.json           ← accuracy, macro F1, AUC, per-class F1 per KL grade
├── training_curves.png
├── confusion_matrix.png
├── roc_curves.png
├── test_probs.npy
├── test_labels.npy
└── f1_comparison.png      ← synthetic runs only
```

---

## Notebook

`notebooks/EDA_Display.ipynb` — exploratory data analysis. Add this at the top:

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
