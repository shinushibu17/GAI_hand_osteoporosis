"""
eval_checkpoints.py — re-evaluate all saved classifier checkpoints on test set.

Usage:
    python3 eval_checkpoints.py --joint dip
    python3 eval_checkpoints.py --joint dip --models cyclegan wgan_gp cvae cyclegan_vgg
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import resnet18

from config import CFG
from dataset import load_metadata, make_patient_splits, filter_joint, OADataset, clf_transform
from torch.utils.data import DataLoader
from utils import compute_metrics


def build_resnet18():
    model = resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, 5)
    return model


def evaluate_checkpoint(ckpt_path: Path, splits: dict, device: torch.device) -> dict:
    model = build_resnet18().to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()

    test_ds = OADataset(splits["test"], transform=clf_transform(augment=False))
    test_loader = DataLoader(test_ds, batch_size=CFG.batch_size_clf,
                             shuffle=False, num_workers=CFG.num_workers)
    return compute_metrics(model, test_loader, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint", type=str, default="dip")
    parser.add_argument("--models", nargs="+",
                        default=["cyclegan", "wgan_gp", "cvae", "cyclegan_vgg"])
    parser.add_argument("--ratios", nargs="+", type=float,
                        default=[0.3, 0.5, 1.0, 5.0, 10.0])
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Joint: {args.joint}")

    meta = load_metadata()
    meta = filter_joint(meta, args.joint)
    splits = make_patient_splits(meta)

    ckpt_base = Path(CFG.ckpt_dir)
    results = {}

    # Baseline
    baseline_ckpts = list(ckpt_base.glob("baseline*.pth")) + \
                     list(ckpt_base.glob(f"*{args.joint}*baseline*.pth"))

    # Look for augmented checkpoints
    for model in args.models:
        results[model] = {}
        for ratio in args.ratios:
            run_results = []
            for run in range(args.runs):
                # Check root checkpoints dir directly
                ckpt = ckpt_base / f"{model}_aug{ratio}_run{run}.pth"
                if not ckpt.exists():
                    print(f"  [SKIP] {model} ratio={ratio} run={run}: not found at {ckpt}")
                    continue
                print(f"  Evaluating {ckpt.name}...")
                metrics = evaluate_checkpoint(ckpt, splits, device)
                run_results.append(metrics)
                print(f"    acc={metrics['accuracy']:.4f} F1={metrics['macro_f1']:.4f} "
                      f"KL3={metrics.get('recall_3',0):.4f} KL4={metrics.get('recall_4',0):.4f}")

            if run_results:
                import numpy as np
                keys = [k for k in run_results[0] if k != "cm"]
                agg = {k: {"mean": float(np.mean([r[k] for r in run_results])),
                            "std":  float(np.std([r[k]  for r in run_results]))}
                       for k in keys}
                results[model][str(ratio)] = {"runs": run_results, "aggregate": agg}
                print(f"  {model} {ratio}: KL3={agg['recall_3']['mean']:.3f}±{agg['recall_3']['std']:.3f} "
                      f"KL4={agg['recall_4']['mean']:.3f}±{agg['recall_4']['std']:.3f} "
                      f"F1={agg['macro_f1']['mean']:.3f}±{agg['macro_f1']['std']:.3f}")

    out = Path(CFG.results_dir) / f"augmented_results_{args.joint}_reeval.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
