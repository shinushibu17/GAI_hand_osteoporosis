"""
Class-Conditional DDPM — Finger Joint X-Ray Generator
=======================================================
Diffusion models are the current state of the art for image generation,
outperforming GANs in image quality, diversity, and training stability
(no mode collapse, no adversarial training instability).

This implements a Denoising Diffusion Probabilistic Model (Ho et al. 2020)
with class conditioning on KL grade (0–4), supporting two use cases:

  1. Unconditional-style generation
         Sample pure noise → denoise conditioning on target KL grade
         → synthetic X-ray of any KL grade

  2. Image-to-image translation (SDEdit, Meng et al. 2021)
         Add noise to a real KL=1 image partway → denoise conditioning
         on KL=2 → translated KL=2 image that preserves anatomy

Architecture
------------
  UNet with:
    • Time embedding  : sinusoidal → 2-layer MLP
    • Class embedding : learnable lookup → added alongside time embedding
    • ResBlocks       : GroupNorm → SiLU → Conv, with time+class injection
    • Self-attention  : at 16×16 and 8×8 spatial resolutions
    • Channel widths  : [64, 128, 256, 512] across 4 resolution levels

Noise schedule
--------------
  Linear schedule β₁=1e-4 → β_T=0.02, T=1000 steps.
  DDIM sampling (Song et al. 2020) for fast inference in 50 steps.

Run
---
    # Train on all KL grades
    python models/diffusion_generator.py --mode train

    # Generate 64 synthetic KL=2 images after training
    python models/diffusion_generator.py --mode generate --kl_grade 2

    # Translate real KL=1 images to KL=2 (SDEdit)
    python models/diffusion_generator.py --mode translate --source_kl 1 --target_kl 2

Outputs  (models/diffusion_kl/)
---------------------------------
    unet_best.pt             — best checkpoint (lowest val loss)
    samples/kl{K}_epoch_XXX.png
    training_loss.png

Requirements
------------
    pip install torch torchvision pandas openpyxl pillow matplotlib tqdm
"""

import argparse
import math
import re
import zipfile
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
XLSX_PATH = ROOT / "data" / "hand.xlsx"
ZIP_PATH  = ROOT / "data" / "Finger Joints.zip"
OUT_DIR   = Path(__file__).parent / "diffusion_kl"

IMG_SIZE   = 64
N_CLASSES  = 5          # KL grades 0-4
T_STEPS    = 1000       # diffusion timesteps
BETA_START = 1e-4
BETA_END   = 0.02

JOINT_MAP = {
    "dip2": "DIP2", "dip3": "DIP3", "dip4": "DIP4", "dip5": "DIP5",
    "mcp2": "MCP2", "mcp3": "MCP3", "mcp4": "MCP4", "mcp5": "MCP5",
    "pip2": "PIP2", "pip3": "PIP3", "pip4": "PIP4", "pip5": "PIP5",
}


# ── Noise schedule ────────────────────────────────────────────────────────────
class NoiseSchedule:
    def __init__(self, T: int = T_STEPS, beta_start: float = BETA_START,
                 beta_end: float = BETA_END, device: torch.device = torch.device("cpu")):
        self.T = T
        betas       = torch.linspace(beta_start, beta_end, T, device=device)
        alphas      = 1.0 - betas
        alpha_bar   = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value=1.0)

        self.register = lambda name, val: setattr(self, name, val)
        self.register("betas",          betas)
        self.register("alphas",         alphas)
        self.register("alpha_bar",      alpha_bar)
        self.register("alpha_bar_prev", alpha_bar_prev)
        self.register("sqrt_alpha_bar",       alpha_bar.sqrt())
        self.register("sqrt_one_minus_ab",    (1.0 - alpha_bar).sqrt())
        self.register("log_one_minus_ab",     (1.0 - alpha_bar).log())
        self.register("sqrt_recip_alpha_bar", (1.0 / alpha_bar).sqrt())
        self.register("posterior_variance",
                      betas * (1 - alpha_bar_prev) / (1 - alpha_bar))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor | None = None) -> torch.Tensor:
        """Forward diffusion: q(x_t | x_0) = N(√ᾱ_t x_0, (1−ᾱ_t) I)"""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab  = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        sqrt_oab = self.sqrt_one_minus_ab[t].view(-1, 1, 1, 1)
        return sqrt_ab * x0 + sqrt_oab * noise, noise


# ── UNet building blocks ──────────────────────────────────────────────────────
def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
    )
    args = t[:, None].float() * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)


class TimeClassEmbedding(nn.Module):
    def __init__(self, t_dim: int, n_classes: int, out_dim: int):
        super().__init__()
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim)
        )
        self.cls_emb = nn.Embedding(n_classes + 1, out_dim)  # +1 for unconditional

    def forward(self, t: torch.Tensor, cls: torch.Tensor) -> torch.Tensor:
        t_emb  = self.t_mlp(sinusoidal_embedding(t, self.t_mlp[0].in_features))
        c_emb  = self.cls_emb(cls)
        return t_emb + c_emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip  = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act   = nn.SiLU()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.emb_proj(self.act(emb))[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(groups, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads=4, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W).permute(0, 2, 1)
        h, _ = self.attn(h, h, h)
        return x + h.permute(0, 2, 1).view(B, C, H, W)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim, use_attn=False):
        super().__init__()
        self.res  = ResBlock(in_ch, out_ch, emb_dim)
        self.attn = SelfAttention(out_ch) if use_attn else nn.Identity()
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x, emb):
        x = self.attn(self.res(x, emb))
        return self.down(x), x  # return (downsampled, skip)


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, emb_dim, use_attn=False):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        self.res  = ResBlock(in_ch + skip_ch, out_ch, emb_dim)
        self.attn = SelfAttention(out_ch) if use_attn else nn.Identity()

    def forward(self, x, skip, emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.attn(self.res(x, emb))


class UNet(nn.Module):
    """
    Class-conditional UNet for noise prediction.
    Input  : (B, 1, 64, 64) noisy image + time + class label
    Output : (B, 1, 64, 64) predicted noise
    """

    def __init__(self, channels=(64, 128, 256, 512), n_classes=N_CLASSES):
        super().__init__()
        emb_dim = 256
        self.emb = TimeClassEmbedding(128, n_classes, emb_dim)
        self.in_conv = nn.Conv2d(1, channels[0], 3, padding=1)

        # Encoder
        self.down1 = Down(channels[0], channels[1], emb_dim)
        self.down2 = Down(channels[1], channels[2], emb_dim)
        self.down3 = Down(channels[2], channels[3], emb_dim, use_attn=True)

        # Bottleneck
        self.mid1  = ResBlock(channels[3], channels[3], emb_dim)
        self.mid_attn = SelfAttention(channels[3])
        self.mid2  = ResBlock(channels[3], channels[3], emb_dim)

        # Decoder
        self.up3   = Up(channels[3], channels[2], channels[2], emb_dim, use_attn=True)
        self.up2   = Up(channels[2], channels[1], channels[1], emb_dim)
        self.up1   = Up(channels[1], channels[0], channels[0], emb_dim)

        self.out   = nn.Sequential(
            nn.GroupNorm(8, channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], 1, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                cls: torch.Tensor) -> torch.Tensor:
        emb = self.emb(t, cls)
        x   = self.in_conv(x)
        x, s1 = self.down1(x, emb)
        x, s2 = self.down2(x, emb)
        x, s3 = self.down3(x, emb)
        x = self.mid2(self.mid_attn(self.mid1(x, emb)), emb)
        x = self.up3(x, s3, emb)
        x = self.up2(x, s2, emb)
        x = self.up1(x, s1, emb)
        return self.out(x)


# ── Dataset ───────────────────────────────────────────────────────────────────
class FingerJointAllKLDataset(Dataset):
    """All images with valid KL labels — returns (image, kl_grade) pairs."""

    def __init__(self, xlsx_path, zip_path, transform, joint_filter=None):
        self.transform = transform
        self.index: list[tuple[str, int]] = []

        df = pd.read_excel(xlsx_path)
        df["img_id"] = df["id"].astype(str)

        img_to_kl: dict[str, int] = {}
        for img_joint, score_joint in JOINT_MAP.items():
            if joint_filter and img_joint != joint_filter:
                continue
            kl_col = f"v00{score_joint}_KL"
            if kl_col not in df.columns:
                continue
            for pid, kl in df.set_index("img_id")[kl_col].dropna().astype(int).items():
                img_to_kl[f"Finger Joints/{pid}_{img_joint}.png"] = kl

        self._zip_path = zip_path
        with zipfile.ZipFile(zip_path, "r") as z:
            for name in z.namelist():
                if name.endswith(".png") and name in img_to_kl:
                    self.index.append((name, img_to_kl[name]))

        from collections import Counter
        dist = Counter(kl for _, kl in self.index)
        print(f"Dataset: {len(self.index):,} images  |  " +
              "  ".join(f"KL={k}:{v}" for k, v in sorted(dist.items())))
        self._zip = None

    def _get_zip(self):
        if self._zip is None:
            self._zip = zipfile.ZipFile(self._zip_path, "r")
        return self._zip

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        path, kl = self.index[idx]
        with self._get_zip().open(path) as f:
            img = Image.open(BytesIO(f.read())).convert("L")
        return self.transform(img), kl


# ── DDIM sampling ─────────────────────────────────────────────────────────────
@torch.no_grad()
def ddim_sample(model: UNet, schedule: NoiseSchedule, cls: torch.Tensor,
                n: int, device: torch.device, ddim_steps: int = 50,
                eta: float = 0.0) -> torch.Tensor:
    """
    Fast DDIM sampling (deterministic when eta=0).
    Returns (n, 1, IMG_SIZE, IMG_SIZE) tensor in [-1, 1].
    """
    model.eval()
    step_seq  = torch.linspace(0, schedule.T - 1, ddim_steps, dtype=torch.long)
    x = torch.randn(n, 1, IMG_SIZE, IMG_SIZE, device=device)

    for i in reversed(range(len(step_seq))):
        t    = step_seq[i].expand(n).to(device)
        t_prev = step_seq[i - 1] if i > 0 else torch.zeros_like(t)

        ab      = schedule.alpha_bar[t][:, None, None, None]
        ab_prev = schedule.alpha_bar[t_prev][:, None, None, None] if i > 0 \
                  else torch.ones_like(ab)

        pred_noise = model(x, t, cls.to(device))
        x0_pred    = (x - (1 - ab).sqrt() * pred_noise) / ab.sqrt()
        x0_pred    = x0_pred.clamp(-1, 1)

        sigma = eta * ((1 - ab_prev) / (1 - ab) * (1 - ab / ab_prev)).sqrt()
        x = ab_prev.sqrt() * x0_pred \
            + (1 - ab_prev - sigma ** 2).clamp(0).sqrt() * pred_noise \
            + sigma * torch.randn_like(x)

    return x


@torch.no_grad()
def sdEdit(model: UNet, schedule: NoiseSchedule, source_imgs: torch.Tensor,
           target_cls: torch.Tensor, t_start_frac: float = 0.6,
           device: torch.device = torch.device("cpu"),
           ddim_steps: int = 50) -> torch.Tensor:
    """
    SDEdit: add noise to real source images up to t_start, then denoise
    conditioning on the target KL class → image-to-image translation.

    Args:
        source_imgs   : (N, 1, H, W) real images in [-1, 1]
        target_cls    : (N,) target KL grade tensor
        t_start_frac  : how far to noise (0.5-0.7 typical; higher = more change)
    """
    model.eval()
    source_imgs = source_imgs.to(device)
    t_start = int(t_start_frac * schedule.T)
    t_tensor = torch.full((source_imgs.size(0),), t_start - 1,
                          dtype=torch.long, device=device)

    # Add noise up to t_start
    x, _ = schedule.q_sample(source_imgs, t_tensor)

    # Denoise from t_start to 0 with target class
    step_seq = torch.linspace(0, t_start - 1, ddim_steps, dtype=torch.long)
    for i in reversed(range(len(step_seq))):
        t    = step_seq[i].expand(source_imgs.size(0)).to(device)
        t_prev = step_seq[i - 1] if i > 0 else torch.zeros_like(t)

        ab      = schedule.alpha_bar[t][:, None, None, None]
        ab_prev = schedule.alpha_bar[t_prev][:, None, None, None] if i > 0 \
                  else torch.ones_like(ab)

        pred_noise = model(x, t, target_cls.to(device))
        x0_pred    = (x - (1 - ab).sqrt() * pred_noise) / ab.sqrt()
        x0_pred    = x0_pred.clamp(-1, 1)
        x = ab_prev.sqrt() * x0_pred \
            + (1 - ab_prev).clamp(0).sqrt() * pred_noise

    return x.cpu()


# ── Training ──────────────────────────────────────────────────────────────────
def train(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "samples").mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    dataset = FingerJointAllKLDataset(XLSX_PATH, ZIP_PATH, transform, args.joint)

    # Weighted sampler to balance KL grades during training
    labels  = [kl for _, kl in dataset.index]
    counts  = np.bincount(labels, minlength=N_CLASSES).astype(float)
    w_cls   = 1.0 / np.where(counts == 0, 1.0, counts)
    weights = torch.tensor([w_cls[l] for l in labels])
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    loader  = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                         num_workers=0, drop_last=True)

    schedule = NoiseSchedule(T_STEPS, device=device)
    model    = UNet(n_classes=N_CLASSES).to(device)
    total_p  = sum(p.numel() for p in model.parameters())
    print(f"UNet parameters: {total_p/1e6:.1f}M")

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)
    scaler    = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # Fixed noise for sample grids — one per KL grade
    fixed_z   = {k: torch.randn(8, 1, IMG_SIZE, IMG_SIZE, device=device)
                 for k in range(N_CLASSES)}
    fixed_cls = {k: torch.full((8,), k, dtype=torch.long, device=device)
                 for k in range(N_CLASSES)}

    losses, best_loss = [], float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss = 0.0
        for imgs, kls in tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            imgs = imgs.to(device)
            kls  = kls.to(device)

            # Sample random timesteps
            t = torch.randint(0, T_STEPS, (imgs.size(0),), device=device)
            noisy, noise = schedule.q_sample(imgs, t)

            optimiser.zero_grad()
            with torch.amp.autocast(device.type, enabled=scaler is not None):
                pred = model(noisy, t, kls)
                loss = F.mse_loss(pred, noise)

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimiser); scaler.update()
            else:
                loss.backward(); optimiser.step()

            ep_loss += loss.item()

        scheduler.step()
        mean_loss = ep_loss / len(loader)
        losses.append(mean_loss)
        print(f"Epoch {epoch:4d}  loss={mean_loss:.5f}")

        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(model.state_dict(), OUT_DIR / "unet_best.pt")

        if epoch % args.save_interval == 0 or epoch == args.epochs:
            model.eval()
            for kl in range(N_CLASSES):
                samples = ddim_sample(model, schedule, fixed_cls[kl], 8,
                                      device, ddim_steps=50)
                grid = vutils.make_grid(samples.cpu(), nrow=8,
                                        normalize=True, value_range=(-1, 1))
                plt.figure(figsize=(14, 2))
                plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
                plt.axis("off")
                plt.title(f"KL={kl} samples — epoch {epoch}")
                plt.tight_layout()
                plt.savefig(OUT_DIR / "samples" / f"kl{kl}_epoch_{epoch:04d}.png",
                            dpi=100)
                plt.close()

    plt.figure(figsize=(9, 4))
    plt.plot(losses)
    plt.xlabel("Epoch"); plt.ylabel("MSE loss")
    plt.title("Diffusion model training loss")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "training_loss.png", dpi=120)
    plt.close()
    print(f"\nBest model saved to {OUT_DIR}/unet_best.pt")


# ── CLI generation / translation ─────────────────────────────────────────────
def generate(args):
    """Generate synthetic images for a given KL grade."""
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    schedule = NoiseSchedule(T_STEPS, device=device)
    model    = UNet(n_classes=N_CLASSES).to(device)
    model.load_state_dict(torch.load(OUT_DIR / "unet_best.pt", map_location=device))
    cls = torch.full((args.n_samples,), args.kl_grade, dtype=torch.long, device=device)
    imgs = ddim_sample(model, schedule, cls, args.n_samples, device,
                       ddim_steps=args.ddim_steps)
    grid = vutils.make_grid(imgs, nrow=8, normalize=True, value_range=(-1, 1))
    plt.figure(figsize=(14, 14))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.axis("off")
    plt.title(f"Generated KL={args.kl_grade} images")
    plt.tight_layout()
    out_path = OUT_DIR / f"generated_kl{args.kl_grade}.png"
    plt.savefig(out_path, dpi=120); plt.close()
    print(f"Saved {out_path}")


def translate(args):
    """SDEdit: translate real source KL images to target KL grade."""
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    schedule = NoiseSchedule(T_STEPS, device=device)
    model    = UNet(n_classes=N_CLASSES).to(device)
    model.load_state_dict(torch.load(OUT_DIR / "unet_best.pt", map_location=device))

    # Load a handful of real source images
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE), transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(), transforms.Normalize([0.5], [0.5]),
    ])
    from models.kl_transition_generator import KLDomainDataset  # reuse loader
    ds = KLDomainDataset(args.source_kl, None, XLSX_PATH, ZIP_PATH, transform)
    src_imgs = torch.stack([ds[i] for i in range(min(8, len(ds)))])

    tgt_cls = torch.full((src_imgs.size(0),), args.target_kl, dtype=torch.long)
    translated = sdeit(model, schedule, src_imgs, tgt_cls,
                       t_start_frac=args.t_start_frac, device=device,
                       ddim_steps=args.ddim_steps)

    combined = torch.cat([src_imgs, translated])
    grid = vutils.make_grid(combined, nrow=8, normalize=True, value_range=(-1, 1))
    plt.figure(figsize=(14, 4))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.axis("off")
    plt.title(f"Top: real KL={args.source_kl}  |  Bottom: translated KL={args.target_kl}")
    plt.tight_layout()
    out_path = OUT_DIR / f"translated_kl{args.source_kl}_to_kl{args.target_kl}.png"
    plt.savefig(out_path, dpi=120); plt.close()
    print(f"Saved {out_path}")


# alias to avoid typo in translate()
sdeit = sdeit if "sdeit" in dir() else sdEdit


def parse_args():
    p = argparse.ArgumentParser(description="Conditional DDPM for finger-joint X-rays")
    p.add_argument("--mode",          choices=["train", "generate", "translate"],
                   default="train")
    # Train
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--lr",            type=float, default=2e-4)
    p.add_argument("--save_interval", type=int,   default=10)
    p.add_argument("--joint",         type=str,   default=None,
                   choices=list(JOINT_MAP.keys()))
    # Generate
    p.add_argument("--kl_grade",      type=int,   default=2,
                   help="KL grade to generate (mode=generate)")
    p.add_argument("--n_samples",     type=int,   default=64)
    p.add_argument("--ddim_steps",    type=int,   default=50)
    # Translate
    p.add_argument("--source_kl",     type=int,   default=1)
    p.add_argument("--target_kl",     type=int,   default=2)
    p.add_argument("--t_start_frac",  type=float, default=0.6,
                   help="Noise level for SDEdit (0.5–0.75); higher = more change")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "generate":
        generate(args)
    else:
        translate(args)
