"""
train_augmented.py — trains ResNet-18 under each (generative model, aug_ratio) condition.

Identical training setup to train_baseline.py — same optimizer, same loss, same
hyperparameters. The ONLY difference is the training data (real + synthetic vs real only).
This ensures any performance difference is attributable to the augmentation, not the
training procedure.

Usage:
    python train_augmented.py [--epochs 50] [--runs 3] [--models cyclegan wgan_gp cvae ddpm]

Results saved to: outputs/results/augmented_results_{joint}.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from config import CFG
from dataset import load_metadata, make_patient_splits
from train_baseline import train_one_run, aggregate_runs


def get_synth_dirs(gen_model: str, joint: str) -> dict:
    """Return {grade: list_of_dirs} pooled across all joints, or single joint."""
    from dataset import AugmentedDataset

    # Joints that have synthetic images
    joints_to_check = CFG.all_joints if joint == "pooled" else [joint]

    # Collect all available dirs per grade across joints
    grade_dirs: dict = {grade: [] for grade in CFG.target_grades}
    for jt in joints_to_check:
        for grade in CFG.target_grades:
            d = CFG.synth_dir(jt, gen_model, grade)
            if d.exists() and any(d.iterdir()):
                grade_dirs[grade].append(str(d))

    # For AugmentedDataset compatibility, merge dirs into one path per grade
    # by creating a symlink-free flat mapping: return {grade: comma-joined paths}
    # We'll handle multi-dir in AugmentedDataset below
    result = {}
    for grade, dirs in grade_dirs.items():
        if dirs:
            result[grade] = dirs  # list of dirs
        else:
            print(f"  [WARN] No synthetic images for {gen_model}/kl{grade}.")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=CFG.n_epochs_clf)
    parser.add_argument("--runs", type=int, default=CFG.n_clf_runs)
    parser.add_argument("--joint", type=str, default=None)
    parser.add_argument("--models", nargs="+",
                        default=["cyclegan", "wgan_gp", "cvae", "ddpm"],
                        choices=["cyclegan", "wgan_gp", "cvae", "ddpm"])
    args = parser.parse_args()

    joint = args.joint or "pooled"
    CFG.makedirs(joint)
    meta = load_metadata()
    from dataset import filter_joint
    meta = filter_joint(meta, joint)
    splits = make_patient_splits(meta)

    all_results = {}

    for gen_model in args.models:
        synth_dirs = get_synth_dirs(gen_model, joint)
        if not synth_dirs:
            print(f"\n  [SKIP] {gen_model}: no synthetic images found.")
            continue

        all_results[gen_model] = {}

        for aug_ratio in CFG.aug_ratios:
            condition = f"{gen_model}_aug{aug_ratio}"
            print(f"\n{'='*60}")
            print(f"  {gen_model.upper()}  aug_ratio={aug_ratio}")
            print(f"{'='*60}")

            run_results = []
            for run in range(args.runs):
                print(f"\n  Run {run+1}/{args.runs}")
                ckpt = Path(CFG.ckpt_dir) / f"{gen_model}_aug{aug_ratio}_run{run}.pth"
                metrics = train_one_run(
                    splits, args.epochs, run + 1, ckpt,
                    synth_dirs=synth_dirs, aug_ratio=aug_ratio,
                    model_name=condition
                )
                run_results.append(metrics)
                print(f"  Test acc={metrics['accuracy']:.4f}  "
                      f"macroF1={metrics['macro_f1']:.4f}  "
                      f"KL3={metrics.get('recall_3',0):.4f}  "
                      f"KL4={metrics.get('recall_4',0):.4f}")

            agg = aggregate_runs(run_results)
            all_results[gen_model][str(aug_ratio)] = {
                "runs": run_results,
                "aggregate": {k: {"mean": m, "std": s} for k, (m, s) in agg.items()}
            }
            print(f"\n  [{gen_model} | ratio={aug_ratio}] mean ± std")
            for k, (m, s) in agg.items():
                if "recall" in k or "accuracy" in k or "f1" in k:
                    print(f"    {k:20s}: {m:.4f} ± {s:.4f}")

    out = Path(CFG.results_dir) / f"augmented_results_{joint}.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll augmented results saved to {out}")


if __name__ == "__main__":
    main()
