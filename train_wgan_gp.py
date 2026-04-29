"""
train_wgan_gp.py — trains a class-conditional WGAN-GP to synthesise KL 3 and KL 4 images.

Usage:
    python train_wgan_gp.py [--epochs 200] [--resume]

Checkpoint: outputs/checkpoints/wgan_gp/latest.pth
"""
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.utils import save_image

from config import CFG
from dataset import load_metadata, make_patient_splits, make_gen_loader
from models.networks import ConditionalGenerator, ConditionalCritic, compute_gradient_penalty, init_weights


def train_wgan_gp(splits, epochs: int, resume: bool = False, ckpt_dir: Path = None):
    device = CFG.device
    if ckpt_dir is None:
        ckpt_dir = Path(CFG.ckpt_dir) / "wgan_gp"
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"WGAN-GP  (conditional, KL grades 0-4)   device={device}")
    print(f"{'='*60}")

    G = ConditionalGenerator(
        latent_dim=CFG.latent_dim, n_classes=5,
        out_ch=CFG.channels, img_size=CFG.img_size
    ).to(device)
    C = ConditionalCritic(
        in_ch=CFG.channels, n_classes=5, img_size=CFG.img_size
    ).to(device)

    G.apply(init_weights); C.apply(init_weights)

    start_epoch = 0
    if resume:
        ckpt = ckpt_dir / "latest.pth"
        if ckpt.exists():
            state = torch.load(ckpt, map_location=device)
            G.load_state_dict(state["G"])
            C.load_state_dict(state["C"])
            start_epoch = state["epoch"] + 1
            print(f"  Resumed from epoch {start_epoch}")

    opt_G = Adam(G.parameters(), lr=CFG.lr_wgan,
                 betas=(CFG.beta1_wgan, CFG.beta2_wgan))
    opt_C = Adam(C.parameters(), lr=CFG.lr_wgan,
                 betas=(CFG.beta1_wgan, CFG.beta2_wgan))

    # Use weighted sampling so severe grades are seen more frequently during
    # generative training (helps GAN stability on rare classes)
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from dataset import OADataset, gen_transform
    import numpy as np

    tf = gen_transform(CFG.img_size, augment=True)
    train_ds = OADataset(splits["train"], transform=tf)

    if len(train_ds) == 0:
        print(f"  [SKIP] No training images for this joint — skipping WGAN-GP.")
        return None

    # Cap batch size to dataset size for small per-joint splits
    batch_size = min(CFG.batch_size_gen, len(train_ds), 64)

    grade_counts = splits["train"][CFG.grade_col].value_counts()
    sample_weights = splits["train"][CFG.grade_col].map(
        lambda g: 1.0 / grade_counts[g]
    ).values
    sampler = WeightedRandomSampler(
        torch.from_numpy(sample_weights.astype(np.float32)),
        num_samples=len(train_ds), replacement=True
    )
    loader = DataLoader(train_ds, batch_size=batch_size,
                        sampler=sampler, num_workers=2,
                        pin_memory=True, drop_last=True)

    n_iter = len(loader)
    print(f"  Training on {len(train_ds)} images  iters/epoch: {n_iter}")

    # Fixed noise for visualisation
    fixed_noise = torch.randn(20, CFG.latent_dim, device=device)
    fixed_labels = torch.tensor([0, 0, 0, 0,
                                  1, 1, 1, 1,
                                  2, 2, 2, 2,
                                  3, 3, 3, 3,
                                  4, 4, 4, 4], device=device)

    best_G_loss = float("inf")

    for epoch in range(start_epoch, epochs):
        G.train(); C.train()
        t0 = time.time()
        loss_C_acc = loss_G_acc = 0.0
        g_steps = 0

        for i, (real_imgs, labels) in enumerate(loader):
            real_imgs = real_imgs.to(device)
            labels = labels.to(device)
            bs = real_imgs.size(0)

            # ── Critic update (n_critic steps per generator step) ─────────────
            opt_C.zero_grad()
            noise = torch.randn(bs, CFG.latent_dim, device=device)
            fake_imgs = G(noise, labels).detach()

            loss_real = -C(real_imgs, labels).mean()
            loss_fake =  C(fake_imgs, labels).mean()
            gp         = compute_gradient_penalty(C, real_imgs, fake_imgs, labels, device)
            loss_C     = loss_real + loss_fake + CFG.lambda_gp * gp
            loss_C.backward(); opt_C.step()
            loss_C_acc += loss_C.item()

            # ── Generator update (every n_critic critic steps) ────────────────
            if (i + 1) % CFG.n_critic == 0:
                opt_G.zero_grad()
                noise = torch.randn(bs, CFG.latent_dim, device=device)
                gen_imgs = G(noise, labels)
                loss_G   = -C(gen_imgs, labels).mean()
                loss_G.backward(); opt_G.step()
                loss_G_acc += loss_G.item()
                g_steps += 1

        elapsed = time.time() - t0
        print(f"  Epoch [{epoch+1:3d}/{epochs}]  "
              f"C={loss_C_acc/n_iter:.4f}  "
              f"G={loss_G_acc/max(g_steps,1):.4f}  "
              f"({elapsed:.0f}s)")

        if (epoch + 1) % 10 == 0 or epoch == 0:
            G.eval()
            with torch.no_grad():
                samples = G(fixed_noise, fixed_labels) * 0.5 + 0.5
            save_image(samples, ckpt_dir / f"sample_ep{epoch+1:03d}.png", nrow=4)
            torch.save({
                "epoch": epoch,
                "G": G.state_dict(),
                "C": C.state_dict(),
            }, ckpt_dir / "latest.pth")

        # Save best checkpoint based on generator loss
        epoch_G_loss = loss_G_acc / max(g_steps, 1)
        if epoch_G_loss < best_G_loss:
            best_G_loss = epoch_G_loss
            torch.save({"epoch": epoch, "G": G.state_dict(), "C": C.state_dict()},
                       ckpt_dir / "best.pth")

    # Load best checkpoint
    best_ckpt = ckpt_dir / "best.pth"
    if best_ckpt.exists():
        state = torch.load(best_ckpt, map_location=device)
        G.load_state_dict(state["G"])
        print(f"  Loaded best checkpoint (G_loss={best_G_loss:.4f}) from epoch {state['epoch']+1}")

    print(f"  Done. Checkpoint at: {ckpt_dir}/best.pth")
    return G


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=CFG.n_epochs_wgan)
    parser.add_argument("--joint", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    joint = args.joint or "pooled"
    CFG.makedirs(joint)

    # Load best hparams from tuning if available
    hparam_file = Path(CFG.output_dir) / "best_hparams" / f"{joint}_wgan_gp.json"
    if hparam_file.exists():
        import json
        best = json.load(open(hparam_file))
        print(f"  Loading tuned hparams from {hparam_file}")
        CFG.lr_wgan = best.get("lr", CFG.lr_wgan)
        CFG.latent_dim = best.get("latent_dim", CFG.latent_dim)
        CFG.n_critic = best.get("n_critic", CFG.n_critic)
        CFG.lambda_gp = best.get("lambda_gp", CFG.lambda_gp)

    meta = load_metadata()
    from dataset import filter_joint
    meta = filter_joint(meta, joint)
    splits = make_patient_splits(meta)
    ckpt_dir = CFG.ckpt_path(joint, "wgan_gp").parent
    train_wgan_gp(splits, args.epochs, args.resume, ckpt_dir=ckpt_dir)


if __name__ == "__main__":
    main()
