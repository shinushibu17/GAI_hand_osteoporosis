"""
train_cvae.py — trains a class-conditional VAE for KL grade-specific image synthesis.

Usage:
    python train_cvae.py [--epochs 200] [--resume]

Checkpoint: outputs/checkpoints/cvae/latest.pth
"""
import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torchvision.utils import save_image

from config import CFG
from dataset import load_metadata, make_patient_splits
from models.networks import CVAEEncoder, CVAEDecoder, init_weights


def reparameterise(mu, log_var):
    std = (0.5 * log_var).exp()
    return mu + std * torch.randn_like(std)


def vae_loss(recon, target, mu, log_var, kl_weight=1.0):
    recon_loss = F.mse_loss(recon, target, reduction="mean")
    kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_weight * kl, recon_loss, kl


def train_cvae(splits, epochs: int, resume: bool = False, ckpt_dir: Path = None):
    device = CFG.device
    if ckpt_dir is None:
        ckpt_dir = Path(CFG.ckpt_dir) / "cvae"
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Conditional VAE   device={device}")
    print(f"{'='*60}")

    enc = CVAEEncoder(in_ch=CFG.channels, n_classes=5,
                      latent_dim=CFG.latent_dim_vae, img_size=CFG.img_size).to(device)
    dec = CVAEDecoder(n_classes=5, latent_dim=CFG.latent_dim_vae,
                      out_ch=CFG.channels, img_size=CFG.img_size).to(device)

    enc.apply(init_weights); dec.apply(init_weights)

    start_epoch = 0
    if resume:
        ckpt = ckpt_dir / "latest.pth"
        if ckpt.exists():
            state = torch.load(ckpt, map_location=device)
            enc.load_state_dict(state["enc"])
            dec.load_state_dict(state["dec"])
            start_epoch = state["epoch"] + 1
            print(f"  Resumed from epoch {start_epoch}")

    opt = Adam(list(enc.parameters()) + list(dec.parameters()), lr=CFG.lr_cvae)

    from torch.utils.data import DataLoader, WeightedRandomSampler
    from dataset import OADataset, gen_transform
    import numpy as np

    tf = gen_transform(CFG.img_size, augment=True)
    train_ds = OADataset(splits["train"], transform=tf)

    if len(train_ds) == 0:
        print(f"  [SKIP] No training images for this joint — skipping CVAE.")
        return None

    grade_counts = splits["train"][CFG.grade_col].value_counts()
    sample_weights = splits["train"][CFG.grade_col].map(
        lambda g: 1.0 / grade_counts[g]
    ).values
    sampler = WeightedRandomSampler(
        torch.from_numpy(sample_weights.astype(np.float32)),
        num_samples=len(train_ds), replacement=True
    )
    loader = DataLoader(train_ds, batch_size=min(CFG.batch_size_gen, len(train_ds), 64),
                        sampler=sampler, num_workers=2,
                        pin_memory=True, drop_last=True)

    print(f"  Training on {len(train_ds)} images")

    # Fixed visualisation seeds
    fixed_labels = torch.tensor([0,1,2,3,4, 3,3,4,4,3], device=device)
    fixed_z = torch.randn(10, CFG.latent_dim_vae, device=device)

    best_recon_loss = float("inf")

    for epoch in range(start_epoch, epochs):
        enc.train(); dec.train()
        t0 = time.time()
        total_loss = recon_loss_acc = kl_acc = 0.0

        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            mu, log_var = enc(imgs, labels)
            z = reparameterise(mu, log_var)
            recon = dec(z, labels)
            loss, rl, kl = vae_loss(recon, imgs, mu, log_var, CFG.kl_weight)
            if torch.isnan(loss):
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(enc.parameters()) + list(dec.parameters()), 1.0
            )
            opt.step()
            total_loss += loss.item()
            recon_loss_acc += rl.item()
            kl_acc += kl.item()

        n = len(loader)
        elapsed = time.time() - t0
        print(f"  Epoch [{epoch+1:3d}/{epochs}]  "
              f"Loss={total_loss/n:.4f}  "
              f"Recon={recon_loss_acc/n:.4f}  "
              f"KL={kl_acc/n:.4f}  ({elapsed:.0f}s)")

        if (epoch + 1) % 10 == 0 or epoch == 0:
            dec.eval()
            with torch.no_grad():
                samples = dec(fixed_z, fixed_labels) * 0.5 + 0.5
            save_image(samples, ckpt_dir / f"sample_ep{epoch+1:03d}.png", nrow=5)
            torch.save({
                "epoch": epoch,
                "enc": enc.state_dict(),
                "dec": dec.state_dict(),
            }, ckpt_dir / "latest.pth")

        # Save best checkpoint based on reconstruction loss
        epoch_recon = recon_loss_acc / max(len(loader), 1)
        if epoch_recon < best_recon_loss and not (epoch_recon == 0.0):
            best_recon_loss = epoch_recon
            torch.save({"epoch": epoch, "enc": enc.state_dict(), "dec": dec.state_dict()},
                       ckpt_dir / "best.pth")

    # Load best checkpoint
    best_ckpt = ckpt_dir / "best.pth"
    if best_ckpt.exists():
        state = torch.load(best_ckpt, map_location=device)
        enc.load_state_dict(state["enc"])
        dec.load_state_dict(state["dec"])
        print(f"  Loaded best checkpoint (recon={best_recon_loss:.4f}) from epoch {state['epoch']+1}")

    print(f"  Done. Checkpoint at: {ckpt_dir}/best.pth")
    return dec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=CFG.n_epochs_cvae)
    parser.add_argument("--joint", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    joint = args.joint or "pooled"
    CFG.makedirs(joint)

    # Load best hparams from tuning if available
    hparam_file = Path(CFG.output_dir) / "best_hparams" / f"{joint}_cvae.json"
    if hparam_file.exists():
        import json
        best = json.load(open(hparam_file))
        print(f"  Loading tuned hparams from {hparam_file}")
        CFG.lr_cvae = best.get("lr", CFG.lr_cvae)
        CFG.latent_dim_vae = best.get("latent_dim_vae", CFG.latent_dim_vae)
        CFG.kl_weight = best.get("kl_weight", CFG.kl_weight)

    meta = load_metadata()
    from dataset import filter_joint
    meta = filter_joint(meta, joint)
    splits = make_patient_splits(meta)
    ckpt_dir = CFG.ckpt_path(joint, "cvae").parent
    train_cvae(splits, args.epochs, args.resume, ckpt_dir=ckpt_dir)


if __name__ == "__main__":
    main()
