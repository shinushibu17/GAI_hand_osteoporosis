"""
evaluate.py — computes FID, label faithfulness, and generates the comparison table.

Usage:
    python evaluate.py [--ref_clf outputs/checkpoints/baseline_run0.pth]

Reads:
    outputs/results/baseline_results.json
    outputs/results/augmented_results.json
    outputs/synthetic/{model}/kl{grade}/*.png  (for FID + faithfulness)
    data/   (real images from test split for FID real set)

Outputs:
    outputs/results/comparison_table.csv
    outputs/results/comparison_table.txt  (pretty-printed)
    outputs/results/fid_faithfulness.json
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from config import CFG
from dataset import load_metadata, make_patient_splits
from utils import compute_fid, label_faithfulness


def export_real_test_images(splits, device):
    """Save real KL3 and KL4 test images to a temp dir for FID computation."""
    from torchvision.utils import save_image
    from dataset import OADataset, gen_transform
    from torch.utils.data import DataLoader

    real_dirs = {}
    for grade in CFG.target_grades:
        out_dir = Path(CFG.output_dir) / "real_test" / f"kl{grade}"
        out_dir.mkdir(parents=True, exist_ok=True)

        if any(out_dir.iterdir()):
            real_dirs[grade] = str(out_dir)
            continue

        df_g = splits["test"][splits["test"][CFG.grade_col] == grade]
        if df_g.empty:
            continue

        from dataset import GradeFilteredDataset
        tf = gen_transform(CFG.img_size, augment=False)
        ds = GradeFilteredDataset(splits["test"], [grade], transform=tf)
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

        saved = 0
        for imgs, _ in loader:
            for img in imgs:
                save_image(img * 0.5 + 0.5, out_dir / f"real_{grade}_{saved:05d}.png")
                saved += 1

        print(f"  Exported {saved} real KL{grade} test images to {out_dir}")
        real_dirs[grade] = str(out_dir)

    return real_dirs


def compute_generation_quality(splits, device_str: str, ref_clf: str):
    """FID and label faithfulness for each model × grade."""
    real_dirs = export_real_test_images(splits, device_str)
    results = {}

    gen_models = ["cyclegan", "wgan_gp", "cvae", "ddpm"]
    for model_name in gen_models:
        results[model_name] = {}
        for grade in CFG.target_grades:
            synth_dir = Path(CFG.synthetic_dir) / model_name / f"kl{grade}"
            if not synth_dir.exists() or not any(synth_dir.iterdir()):
                print(f"  [SKIP] {model_name}/kl{grade}: no images")
                continue

            fid_score = compute_fid(real_dirs.get(grade, ""), str(synth_dir), device_str)
            faith = label_faithfulness(str(synth_dir), ref_clf, grade,
                                       device=__import__("torch").device(device_str))
            results[model_name][f"kl{grade}_fid"] = fid_score
            results[model_name][f"kl{grade}_faithfulness"] = faith
            print(f"  {model_name:12s} KL{grade}  FID={fid_score:.2f}  "
                  f"faithfulness={faith:.4f}")

    return results


def build_comparison_table(baseline_path: str, augmented_path: str,
                            gen_quality: dict) -> pd.DataFrame:
    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(augmented_path) as f:
        augmented = json.load(f)

    rows = []

    # Baseline
    b_agg = baseline.get("aggregate", {})
    rows.append({
        "model": "Baseline (no aug)",
        "aug_ratio": "—",
        "KL3_recall": f"{b_agg.get('recall_3', {}).get('mean', 0):.4f} ± {b_agg.get('recall_3', {}).get('std', 0):.4f}",
        "KL4_recall": f"{b_agg.get('recall_4', {}).get('mean', 0):.4f} ± {b_agg.get('recall_4', {}).get('std', 0):.4f}",
        "accuracy":   f"{b_agg.get('accuracy', {}).get('mean', 0):.4f} ± {b_agg.get('accuracy', {}).get('std', 0):.4f}",
        "macro_f1":   f"{b_agg.get('macro_f1', {}).get('mean', 0):.4f} ± {b_agg.get('macro_f1', {}).get('std', 0):.4f}",
        "FID_KL3": gen_quality.get("baseline", {}).get("kl3_fid", "—"),
        "FID_KL4": gen_quality.get("baseline", {}).get("kl4_fid", "—"),
        "faith_KL3": "—",
        "faith_KL4": "—",
    })

    # Augmented conditions
    for gen_model, ratios in augmented.items():
        gq = gen_quality.get(gen_model, {})
        for ratio_str, cond in ratios.items():
            agg = cond.get("aggregate", {})
            rows.append({
                "model": gen_model,
                "aug_ratio": ratio_str,
                "KL3_recall": f"{agg.get('recall_3', {}).get('mean', 0):.4f} ± {agg.get('recall_3', {}).get('std', 0):.4f}",
                "KL4_recall": f"{agg.get('recall_4', {}).get('mean', 0):.4f} ± {agg.get('recall_4', {}).get('std', 0):.4f}",
                "accuracy":   f"{agg.get('accuracy', {}).get('mean', 0):.4f} ± {agg.get('accuracy', {}).get('std', 0):.4f}",
                "macro_f1":   f"{agg.get('macro_f1', {}).get('mean', 0):.4f} ± {agg.get('macro_f1', {}).get('std', 0):.4f}",
                "FID_KL3":  gq.get("kl3_fid", "—"),
                "FID_KL4":  gq.get("kl4_fid", "—"),
                "faith_KL3": gq.get("kl3_faithfulness", "—"),
                "faith_KL4": gq.get("kl4_faithfulness", "—"),
            })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_clf", type=str,
                        default=str(Path(CFG.ckpt_dir) / "baseline_run0.pth"),
                        help="Path to frozen reference classifier for faithfulness check")
    args = parser.parse_args()

    CFG.makedirs()
    device_str = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    meta = load_metadata()
    splits = make_patient_splits(meta)

    print("\n── Generation quality (FID + faithfulness) ─────────────────────")
    gen_quality = compute_generation_quality(splits, device_str, args.ref_clf)

    out_gq = Path(CFG.results_dir) / "fid_faithfulness.json"
    with open(out_gq, "w") as f:
        json.dump(gen_quality, f, indent=2)
    print(f"  Saved to {out_gq}")

    print("\n── Comparison table ─────────────────────────────────────────────")
    baseline_path = Path(CFG.results_dir) / "baseline_results.json"
    augmented_path = Path(CFG.results_dir) / "augmented_results.json"

    if not baseline_path.exists():
        print(f"  [WARN] {baseline_path} not found. Run train_baseline.py first.")
        return
    if not augmented_path.exists():
        print(f"  [WARN] {augmented_path} not found. Run train_augmented.py first.")
        df = build_comparison_table(str(baseline_path), "{}", gen_quality)
    else:
        df = build_comparison_table(str(baseline_path), str(augmented_path), gen_quality)

    csv_out = Path(CFG.results_dir) / "comparison_table.csv"
    txt_out = Path(CFG.results_dir) / "comparison_table.txt"
    df.to_csv(csv_out, index=False)

    with open(txt_out, "w") as f:
        f.write(df.to_string(index=False))

    print(df.to_string(index=False))
    print(f"\n  Saved to {csv_out} and {txt_out}")


if __name__ == "__main__":
    main()
