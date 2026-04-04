"""
CycleGAN — KL Grade Transition Image Generator
================================================
Learns unpaired image-to-image translation between two KL grades to simulate
disease progression in finger-joint X-rays.

    KL=1 → KL=2  (early → moderate OA)
    KL=2 → KL=3  (moderate → severe OA)

Why CycleGAN?
-------------
We have no *paired* images (the same joint photographed at two different KL
grades).  CycleGAN learns the mapping from unpaired examples using a cycle-
consistency constraint: G(F(b)) ≈ b  and  F(G(a)) ≈ a.

Architecture
------------
Generator  : ResNet-based (6 residual blocks, instance normalisation)
             grayscale 64×64 → grayscale 64×64
Discriminator : 70×70 PatchGAN with spectral normalisation

Losses
------
  Adversarial   : LSGAN (MSE — more stable than BCE)
  Cycle         : λ_cyc × (||F(G(a))-a||₁ + ||G(F(b))-b||₁)
  Identity      : λ_id  × (||G(b)-b||₁    + ||F(a)-a||₁)

Run
---
    python models/kl_transition_generator.py --source_kl 1 --target_kl 2
    python models/kl_transition_generator.py --source_kl 2 --target_kl 3 --epochs 300

Outputs  (models/kl{S}_to_kl{T}_cyclegan/)
-------------------------------------------
    G_final.pt           generator: source KL → target KL
    F_final.pt           generator: target KL → source KL (inverse)
    samples/epoch_XXXX.png
    training_losses.png

Requirements
------------
    pip install torch torchvision pandas openpyxl pillow matplotlib tqdm
"""

import argparse
import itertools
import re
import zipfile
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.utils as vutils
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
XLSX_PATH = ROOT / "data" / "hand.xlsx"
ZIP_PATH  = ROOT / "data" / "Finger Joints.zip"

IMG_SIZE  = 64
JOINT_MAP = {
    "dip2": "DIP2", "dip3": "DIP3", "dip4": "DIP4", "dip5": "DIP5",
    "mcp2": "MCP2", "mcp3": "MCP3", "mcp4": "MCP4", "mcp5": "MCP5",
    "pip2": "PIP2", "pip3": "PIP3", "pip4": "PIP4", "pip5": "PIP5",
}


# ── Dataset ───────────────────────────────────────────────────────────────────
class KLDomainDataset(Dataset):
    """Images for a single KL grade — pre-loaded into RAM."""

    def __init__(self, kl_grade: int, joint_filter: str | None,
                 xlsx_path: Path, zip_path: Path, transform):
        self.transform = transform
        self.images: list[Image.Image] = []

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

        pat = re.compile(r"Finger Joints/\d+_(\w+)\.png")
        with zipfile.ZipFile(zip_path, "r") as z:
            for name in z.namelist():
                if not name.endswith(".png"):
                    continue
                if joint_filter and not pat.match(name).group(1) == joint_filter:
                    continue
                if img_to_kl.get(name) != kl_grade:
                    continue
                with z.open(name) as f:
                    self.images.append(Image.open(BytesIO(f.read())).convert("L"))

        if not self.images:
            raise RuntimeError(f"No images found for KL={kl_grade}.")
        print(f"KL={kl_grade}: {len(self.images):,} images loaded.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.transform(self.images[idx % len(self.images)])


class UnpairedDataset(Dataset):
    """Yields (img_A, img_B) pairs sampled independently from two domains."""

    def __init__(self, ds_A: KLDomainDataset, ds_B: KLDomainDataset):
        self.ds_A = ds_A
        self.ds_B = ds_B

    def __len__(self):
        return max(len(self.ds_A), len(self.ds_B))

    def __getitem__(self, idx):
        a = self.ds_A[idx % len(self.ds_A)]
        b = self.ds_B[np.random.randint(len(self.ds_B))]
        return a, b


# ── Generator (ResNet-based) ───────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class ResNetGenerator(nn.Module):
    """
    Encoder → 6 Residual blocks → Decoder.
    Adapted for single-channel (grayscale) images.
    """

    def __init__(self, n_res: int = 6, base_ch: int = 64):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(1, base_ch, 7, bias=False),
            nn.InstanceNorm2d(base_ch),
            nn.ReLU(inplace=True),
            # Downsample
            nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_ch * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_ch * 4),
            nn.ReLU(inplace=True),
        ]
        layers += [ResBlock(base_ch * 4) for _ in range(n_res)]
        layers += [
            # Upsample
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 3, stride=2, padding=1,
                               output_padding=1, bias=False),
            nn.InstanceNorm2d(base_ch * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_ch * 2, base_ch, 3, stride=2, padding=1,
                               output_padding=1, bias=False),
            nn.InstanceNorm2d(base_ch),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(base_ch, 1, 7),
            nn.Tanh(),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ── Discriminator (PatchGAN) ──────────────────────────────────────────────────
class PatchDiscriminator(nn.Module):
    """70×70 PatchGAN with spectral normalisation."""

    def __init__(self, base_ch: int = 64):
        super().__init__()
        sn = nn.utils.spectral_norm
        self.net = nn.Sequential(
            sn(nn.Conv2d(1, base_ch, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv2d(base_ch, base_ch * 2, 4, stride=2, padding=1)),
            nn.InstanceNorm2d(base_ch * 2),
            nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv2d(base_ch * 2, base_ch * 4, 4, stride=2, padding=1)),
            nn.InstanceNorm2d(base_ch * 4),
            nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv2d(base_ch * 4, base_ch * 8, 4, padding=1)),
            nn.InstanceNorm2d(base_ch * 8),
            nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv2d(base_ch * 8, 1, 4, padding=1)),
        )

    def forward(self, x):
        return self.net(x)


# ── Image buffer (reduces model oscillation) ──────────────────────────────────
class ImageBuffer:
    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.data: list[torch.Tensor] = []

    def push_and_pop(self, images: torch.Tensor) -> torch.Tensor:
        out = []
        for img in images:
            img = img.unsqueeze(0)
            if len(self.data) < self.max_size:
                self.data.append(img)
                out.append(img)
            elif np.random.rand() > 0.5:
                idx = np.random.randint(len(self.data))
                out.append(self.data[idx].clone())
                self.data[idx] = img
            else:
                out.append(img)
        return torch.cat(out)


# ── Training ──────────────────────────────────────────────────────────────────
def train(args):
    src, tgt = args.source_kl, args.target_kl
    out_dir = Path(__file__).parent / f"kl{src}_to_kl{tgt}_cyclegan"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  KL={src} → KL={tgt}")

    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    ds_A = KLDomainDataset(src, args.joint, XLSX_PATH, ZIP_PATH, transform)
    ds_B = KLDomainDataset(tgt, args.joint, XLSX_PATH, ZIP_PATH, transform)
    pin = device.type == "cuda"
    loader = DataLoader(UnpairedDataset(ds_A, ds_B), batch_size=args.batch_size,
                        shuffle=True, num_workers=0, drop_last=True, pin_memory=pin)

    G = ResNetGenerator().to(device)   # A → B  (source → target)
    F = ResNetGenerator().to(device)   # B → A  (target → source / inverse)
    D_A = PatchDiscriminator().to(device)
    D_B = PatchDiscriminator().to(device)

    opt_G  = torch.optim.Adam(itertools.chain(G.parameters(), F.parameters()),
                               lr=args.lr, betas=(0.5, 0.999))
    opt_D  = torch.optim.Adam(itertools.chain(D_A.parameters(), D_B.parameters()),
                               lr=args.lr, betas=(0.5, 0.999))

    # LR decay: keep constant for first half, then linearly decay to 0
    def lr_lambda(epoch):
        half = args.epochs // 2
        return 1.0 if epoch < half else 1.0 - (epoch - half) / max(1, half)
    sched_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda)
    sched_D = torch.optim.lr_scheduler.LambdaLR(opt_D, lr_lambda)

    crit_gan  = nn.MSELoss()
    crit_cyc  = nn.L1Loss()
    buf_A, buf_B = ImageBuffer(), ImageBuffer()

    # Mixed precision scaler — speeds up GPU training significantly
    scaler_G = torch.amp.GradScaler(device.type) if device.type == "cuda" else None
    scaler_D = torch.amp.GradScaler(device.type) if device.type == "cuda" else None

    fixed_A = next(iter(DataLoader(ds_A, batch_size=8, shuffle=True,
                                   num_workers=0))).to(device)

    hist = {"G": [], "D": [], "cyc": [], "id": []}

    for epoch in range(1, args.epochs + 1):
        G.train(); F.train(); D_A.train(); D_B.train()
        ep_G = ep_D = ep_cyc = ep_id = 0.0

        for real_A, real_B in tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            real_A, real_B = real_A.to(device), real_B.to(device)

            # ── Generator step ──────────────────────────────────────────────
            opt_G.zero_grad()
            with torch.amp.autocast(device.type, enabled=scaler_G is not None):
                fake_B = G(real_A)
                fake_A = F(real_B)

                # Adversarial
                ones  = torch.ones_like(D_B(fake_B))
                l_gan = crit_gan(D_B(fake_B), ones) + crit_gan(D_A(fake_A), ones)

                # Cycle consistency
                l_cyc = (crit_cyc(F(fake_B), real_A) + crit_cyc(G(fake_A), real_B)) * args.lambda_cyc

                # Identity
                l_id  = (crit_cyc(G(real_B), real_B) + crit_cyc(F(real_A), real_A)) * args.lambda_id

                l_G = l_gan + l_cyc + l_id

            if scaler_G:
                scaler_G.scale(l_G).backward()
                scaler_G.step(opt_G)
                scaler_G.update()
            else:
                l_G.backward()
                opt_G.step()

            # ── Discriminator step ──────────────────────────────────────────
            opt_D.zero_grad()
            with torch.amp.autocast(device.type, enabled=scaler_D is not None):
                fake_B_buf = buf_B.push_and_pop(fake_B.detach())
                fake_A_buf = buf_A.push_and_pop(fake_A.detach())

                zeros = torch.zeros_like(D_B(fake_B_buf))
                l_D = (
                    crit_gan(D_B(real_B), torch.ones_like(D_B(real_B)))
                    + crit_gan(D_B(fake_B_buf), zeros)
                    + crit_gan(D_A(real_A), torch.ones_like(D_A(real_A)))
                    + crit_gan(D_A(fake_A_buf), zeros)
                ) * 0.5

            if scaler_D:
                scaler_D.scale(l_D).backward()
                scaler_D.step(opt_D)
                scaler_D.update()
            else:
                l_D.backward()
                opt_D.step()

            ep_G   += l_gan.item(); ep_D += l_D.item()
            ep_cyc += l_cyc.item(); ep_id += l_id.item()

        sched_G.step(); sched_D.step()
        n = len(loader)
        hist["G"].append(ep_G / n);   hist["D"].append(ep_D / n)
        hist["cyc"].append(ep_cyc / n); hist["id"].append(ep_id / n)
        print(f"Epoch {epoch:4d}  G={ep_G/n:.3f}  D={ep_D/n:.3f}  "
              f"cyc={ep_cyc/n:.3f}  id={ep_id/n:.3f}")

        if epoch % args.save_interval == 0 or epoch == args.epochs:
            G.eval()
            with torch.no_grad():
                fake  = G(fixed_A)
            grid = vutils.make_grid(
                torch.cat([fixed_A.cpu(), fake.cpu()], dim=0),
                nrow=8, normalize=True, value_range=(-1, 1)
            )
            plt.figure(figsize=(12, 4))
            plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
            plt.axis("off")
            plt.title(f"Top: KL={src} real  |  Bottom: KL={tgt} translated  (epoch {epoch})")
            plt.tight_layout()
            plt.savefig(out_dir / "samples" / f"epoch_{epoch:04d}.png", dpi=100)
            plt.close()

    # Loss curves
    plt.figure(figsize=(10, 4))
    for k, v in hist.items():
        plt.plot(v, label=k)
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(f"CycleGAN KL={src}→KL={tgt} training losses")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / "training_losses.png", dpi=120)
    plt.close()

    torch.save(G.state_dict(), out_dir / "G_final.pt")
    torch.save(F.state_dict(), out_dir / "F_final.pt")
    print(f"\nWeights saved to {out_dir}/")


# ── Inference helper ──────────────────────────────────────────────────────────
def translate_images(source_imgs: torch.Tensor, source_kl: int, target_kl: int,
                     checkpoint_dir: Path | None = None) -> torch.Tensor:
    """
    Translate a batch of real source KL images to the target KL grade.

    Args:
        source_imgs : (N, 1, H, W) tensor in [-1, 1]
        source_kl   : KL grade of the input images
        target_kl   : desired output KL grade
    Returns:
        (N, 1, H, W) translated tensor in [-1, 1]

    Example:
        kl1_imgs = ...                        # load real KL=1 images
        kl2_fakes = translate_images(kl1_imgs, source_kl=1, target_kl=2)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = ResNetGenerator().to(device)
    ckpt_dir = checkpoint_dir or (
        Path(__file__).parent / f"kl{source_kl}_to_kl{target_kl}_cyclegan"
    )
    G.load_state_dict(torch.load(ckpt_dir / "G_final.pt", map_location=device))
    G.eval()
    with torch.no_grad():
        return G(source_imgs.to(device)).cpu()


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source_kl",      type=int, required=True,
                   help="Source KL grade (e.g. 1)")
    p.add_argument("--target_kl",      type=int, required=True,
                   help="Target KL grade (e.g. 2)")
    p.add_argument("--epochs",         type=int, default=200)
    p.add_argument("--batch_size",     type=int, default=4,
                   help="Small batch (4-8) recommended for CycleGAN")
    p.add_argument("--lr",             type=float, default=2e-4)
    p.add_argument("--lambda_cyc",     type=float, default=10.0)
    p.add_argument("--lambda_id",      type=float, default=5.0)
    p.add_argument("--save_interval",  type=int, default=10)
    p.add_argument("--joint",          type=str, default=None,
                   choices=list(JOINT_MAP.keys()))
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
