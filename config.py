"""
config.py — central hyperparameter file for OA augmentation comparison.
Edit paths and epochs here; everything else imports this.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import torch


@dataclass
class Config:
    # ── Data ──────────────────────────────────────────────────────────────────
    # CSV must have at minimum: patient_id, image_path (absolute or relative to
    # data_root), kl_grade (int 0-4).
    data_root: str = "."
    metadata_csv: str = "./data/metadata.csv"
    image_col: str = "image_path"
    patient_col: str = "patient_id"
    grade_col: str = "kl_grade"

    # ── Image ─────────────────────────────────────────────────────────────────
    img_size: int = 128          # used for all generative models
    clf_img_size: int = 224      # ResNet-18 input size
    channels: int = 1            # grayscale radiographs

    # ── Splits ────────────────────────────────────────────────────────────────
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42

    # ── Augmentation experiment ────────────────────────────────────────────────
    aug_ratios: List[float] = field(default_factory=lambda: [0.3, 0.5, 1.0, 5.0, 10.0])
    target_grades: List[int] = field(default_factory=lambda: [3, 4])

    # ── CycleGAN ──────────────────────────────────────────────────────────────
    # Translation pairs: (source_grade, target_grade)
    cyclegan_pairs: List[tuple] = field(default_factory=lambda: [(1, 3), (2, 4)])
    n_epochs_cyclegan: int = 200
    lambda_cycle: float = 10.0
    lambda_identity: float = 5.0
    n_resblocks: int = 6         # 6 for 128px, 9 for 256px
    lr_cyclegan: float = 2e-4
    beta1_cyclegan: float = 0.5

    # ── WGAN-GP ───────────────────────────────────────────────────────────────
    n_epochs_wgan: int = 200
    latent_dim: int = 128
    n_critic: int = 5
    lambda_gp: float = 10.0
    lr_wgan: float = 1e-4
    beta1_wgan: float = 0.0
    beta2_wgan: float = 0.9

    # ── CVAE ──────────────────────────────────────────────────────────────────
    n_epochs_cvae: int = 200
    latent_dim_vae: int = 128
    lr_cvae: float = 1e-3
    kl_weight: float = 1.0

    # ── DDPM ──────────────────────────────────────────────────────────────────
    n_epochs_ddpm: int = 100
    n_timesteps: int = 1000
    guidance_scale: float = 3.0
    p_uncond: float = 0.1        # prob of dropping conditioning during training
    lr_ddpm: float = 1e-4

    # ── Downstream classifier ─────────────────────────────────────────────────
    n_epochs_clf: int = 50
    lr_clf: float = 1e-4
    weight_decay_clf: float = 1e-4
    # H200 141GB: batch_size_gen=128, batch_size_clf=256
    # H100  80GB: batch_size_gen=64,  batch_size_clf=128
    # A100  40GB: batch_size_gen=32,  batch_size_clf=64
    # V100  16GB: batch_size_gen=16,  batch_size_clf=32
    batch_size_clf: int = 256
    n_clf_runs: int = 3          # independent runs per augmentation condition

    # ── Shared training ───────────────────────────────────────────────────────
    batch_size_gen: int = 128
    num_workers: int = 8

    # ── Joint-type settings ───────────────────────────────────────────────────
    joint_col: str = "joint_type"          # column name in metadata CSV
    # All 12 joints. If None, training is pooled across all joints.
    all_joints: List[str] = field(default_factory=lambda: [
        "dip2", "dip3", "dip4", "dip5",
        "pip2", "pip3", "pip4", "pip5",
        "mcp2", "mcp3", "mcp4", "mcp5",
    ])
    train_per_joint: bool = True           # False = pool all joints together
    joint_groups: List[str] = field(default_factory=lambda: ["dip", "pip", "mcp"])

    # ── Paths ─────────────────────────────────────────────────────────────────
    output_dir: str = "./outputs"
    synthetic_dir: str = "./outputs/synthetic"
    ckpt_dir: str = "./outputs/checkpoints"
    results_dir: str = "./outputs/results"

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def makedirs(self, joint: Optional[str] = None):
        for d in [self.output_dir, self.synthetic_dir, self.ckpt_dir, self.results_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)
        joints = [joint] if joint else (self.all_joints if self.train_per_joint else ["pooled"])
        for jt in joints:
            for grade in self.target_grades:
                for src in ["cyclegan", "wgan_gp", "cvae", "ddpm"]:
                    Path(self.synthetic_dir, jt, src, f"kl{grade}").mkdir(parents=True, exist_ok=True)

    def synth_dir(self, joint: str, model: str, grade: int) -> Path:
        jt = joint if self.train_per_joint else "pooled"
        return Path(self.synthetic_dir) / jt / model / f"kl{grade}"

    def ckpt_path(self, joint: str, model: str, suffix: str = "latest.pth") -> Path:
        jt = joint if self.train_per_joint else "pooled"
        d = Path(self.ckpt_dir) / jt / model
        d.mkdir(parents=True, exist_ok=True)
        # Prefer best.pth over latest.pth if it exists
        best = d / "best.pth"
        if suffix == "latest.pth" and best.exists():
            return best
        return d / suffix


CFG = Config()
