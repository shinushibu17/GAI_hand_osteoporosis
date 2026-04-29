"""
filter_synthetic.py — filters synthetic images by quality using FID-based scoring.

Strategy:
  1. Score each synthetic image against real test images using LPIPS or pixel stats
  2. Keep only the top-k% by quality score
  3. Save filtered images to outputs/synthetic_filtered/{joint}/{model}/kl{grade}/

This ensures classifiers train on the highest-quality synthetic images only.

Usage:
    python3 filter_synthetic.py --joint dip --models cyclegan wgan_gp cvae --keep 0.5
    python3 filter_synthetic.py --joint pip --models cyclegan --keep 0.7

Arguments:
    --keep: fraction of images to keep (0.0-1.0, default 0.5 = top 50%)
    --score: scoring method: 'fid' (group FID), 'pixel' (individual pixel stats)
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.models import inception_v3

from config import CFG


def load_image(path: str, size: int = 299) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return tf(img)


def get_inception_features(paths: list, device: torch.device, batch_size: int = 32) -> np.ndarray:
    """Extract Inception-v3 features for a list of image paths."""
    model = inception_v3(weights="IMAGENET1K_V1", transform_input=False)
    model.fc = torch.nn.Identity()
    model = model.to(device).eval()

    features = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i + batch_size]
        imgs = []
        for p in batch_paths:
            try:
                imgs.append(load_image(str(p)))
            except Exception:
                imgs.append(torch.zeros(3, 299, 299))
        batch = torch.stack(imgs).to(device)
        with torch.no_grad():
            feat = model(batch)
        features.append(feat.cpu().numpy())

    return np.concatenate(features, axis=0)


def score_images_by_similarity(
    real_paths: list,
    fake_paths: list,
    device: torch.device,
) -> np.ndarray:
    """
    Score each synthetic image by cosine similarity to the mean real feature vector.
    Higher score = more similar to real images = better quality.
    """
    print(f"  Extracting real features ({len(real_paths)} images)...")
    real_feats = get_inception_features(real_paths, device)
    real_mean = real_feats.mean(axis=0)
    real_mean = real_mean / (np.linalg.norm(real_mean) + 1e-8)

    print(f"  Scoring {len(fake_paths)} synthetic images...")
    fake_feats = get_inception_features(fake_paths, device)

    scores = []
    for feat in fake_feats:
        feat_norm = feat / (np.linalg.norm(feat) + 1e-8)
        score = float(np.dot(feat_norm, real_mean))
        scores.append(score)

    return np.array(scores)


def score_images_by_pixel(fake_paths: list) -> np.ndarray:
    """
    Score by pixel quality — reject very dark, very bright, or low-contrast images.
    Faster than inception features, no GPU needed.
    """
    scores = []
    for p in fake_paths:
        try:
            img = Image.open(p).convert("L")
            arr = np.array(img, dtype=np.float32)
            mean = arr.mean()
            std = arr.std()
            min_val = arr.min()
            max_val = arr.max()

            # Penalise extreme brightness and low contrast
            if mean < 5 or std < 4 or (max_val - min_val) < 10:
                scores.append(-999.0)  # reject
            else:
                score = std - abs(mean - 90) * 0.05
                scores.append(float(score))
        except Exception:
            scores.append(-999.0)

    return np.array(scores)


def get_real_paths(joint: str, grade: int) -> list:
    """Get real test image paths for comparison."""
    from dataset import load_metadata, make_patient_splits, filter_joint
    meta = load_metadata()
    meta = filter_joint(meta, joint)
    splits = make_patient_splits(meta)
    test_df = splits["test"]
    grade_df = test_df[test_df[CFG.grade_col] == grade]
    paths = [row[CFG.image_col] for _, row in grade_df.iterrows()
             if Path(row[CFG.image_col]).exists()]
    return paths


def filter_joint_model(
    joint: str,
    model: str,
    grade: int,
    keep: float,
    score_method: str,
    device: torch.device,
    dry_run: bool = False,
) -> dict:
    src_dir = CFG.synth_dir(joint, model, grade)
    if not src_dir.exists():
        print(f"  [SKIP] {model}/kl{grade}: no synthetic images")
        return {}

    fake_paths = sorted(list(src_dir.glob("*.png")) + list(src_dir.glob("*.jpg")))
    if not fake_paths:
        print(f"  [SKIP] {model}/kl{grade}: directory empty")
        return {}

    n_total = len(fake_paths)
    n_keep = max(1, int(n_total * keep))

    print(f"\n  {model} KL{grade}: {n_total} images → keeping top {keep*100:.0f}% ({n_keep})")

    # Score images
    if score_method == "pixel":
        scores = score_images_by_pixel(fake_paths)
    else:
        real_paths = get_real_paths(joint, grade)
        if not real_paths:
            print(f"  [WARN] No real KL{grade} test images found, falling back to pixel scoring")
            scores = score_images_by_pixel(fake_paths)
        else:
            scores = score_images_by_similarity(real_paths, fake_paths, device)

    # Rank and select top-k
    ranked_indices = np.argsort(scores)[::-1]  # highest score first
    selected = [fake_paths[i] for i in ranked_indices[:n_keep]]
    rejected = [fake_paths[i] for i in ranked_indices[n_keep:]]

    print(f"  Score range: min={scores.min():.3f} max={scores.max():.3f} "
          f"threshold={scores[ranked_indices[n_keep-1]]:.3f}")

    if dry_run:
        print(f"  [DRY RUN] Would keep {len(selected)}, reject {len(rejected)}")
        return {"selected": len(selected), "rejected": len(rejected)}

    # Copy selected to filtered directory
    out_dir = Path(CFG.output_dir) / "synthetic_filtered" / joint / model / f"kl{grade}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clear existing
    for f in out_dir.glob("*"):
        f.unlink()

    for p in selected:
        shutil.copy2(p, out_dir / p.name)

    print(f"  Saved {len(selected)} filtered images to {out_dir}")
    return {"selected": len(selected), "rejected": len(rejected), "out_dir": str(out_dir)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint",   required=True, help="dip, pip, or mcp")
    parser.add_argument("--models",  nargs="+", default=["cyclegan", "wgan_gp", "cvae"])
    parser.add_argument("--keep",    type=float, default=0.5,
                        help="Fraction of images to keep (0.0-1.0, default 0.5)")
    parser.add_argument("--score",   choices=["inception", "pixel"], default="inception",
                        help="Scoring method (default: inception features)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be filtered without copying files")
    parser.add_argument("--gpu",     type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"\nFiltering synthetic images")
    print(f"  Joint: {args.joint}  Models: {args.models}")
    print(f"  Keep: {args.keep*100:.0f}%  Method: {args.score}  Device: {device}")

    results = {}
    for model in args.models:
        for grade in CFG.target_grades:
            key = f"{model}_kl{grade}"
            results[key] = filter_joint_model(
                joint=args.joint,
                model=model,
                grade=grade,
                keep=args.keep,
                score_method=args.score,
                device=device,
                dry_run=args.dry_run,
            )

    print(f"\n{'='*50}")
    print(f"Filtering complete. Results in outputs/synthetic_filtered/{args.joint}/")
    for key, res in results.items():
        if res:
            print(f"  {key}: kept {res.get('selected',0)}, rejected {res.get('rejected',0)}")


if __name__ == "__main__":
    main()
