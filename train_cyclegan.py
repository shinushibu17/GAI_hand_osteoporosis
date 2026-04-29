"""
train_cyclegan.py — trains a CycleGAN for each (source_grade → target_grade) pair.

Usage:
    python train_cyclegan.py [--epochs 200] [--pair 1,3] [--resume]

Checkpoints saved to: outputs/checkpoints/cyclegan_kl{src}_to_kl{tgt}/
"""
import argparse
import itertools
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from torchvision.utils import save_image

from config import CFG
from dataset import load_metadata, make_patient_splits, UnpairedGradeDataset
from models.networks import (
    ResNetGenerator, PatchGANDiscriminator, ImagePool, init_weights
)


def train_cyclegan(src_grade: int, tgt_grade: int, splits, epochs: int,
                   resume: bool = False, ckpt_dir: Path = None):
    device = CFG.device
    if ckpt_dir is None:
        ckpt_dir = Path(CFG.ckpt_dir) / f"cyclegan_kl{src_grade}_to_kl{tgt_grade}"
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"CycleGAN  KL{src_grade} → KL{tgt_grade}   device={device}")
    print(f"{'='*60}")

    # ── Networks ──────────────────────────────────────────────────────────────
    G_AB = ResNetGenerator(CFG.channels, CFG.channels, n_blocks=CFG.n_resblocks).to(device)
    G_BA = ResNetGenerator(CFG.channels, CFG.channels, n_blocks=CFG.n_resblocks).to(device)
    D_A  = PatchGANDiscriminator(CFG.channels).to(device)
    D_B  = PatchGANDiscriminator(CFG.channels).to(device)

    for net in [G_AB, G_BA, D_A, D_B]:
        net.apply(init_weights)

    start_epoch = 0
    if resume:
        ckpt = ckpt_dir / "latest.pth"
        if ckpt.exists():
            state = torch.load(ckpt, map_location=device)
            G_AB.load_state_dict(state["G_AB"])
            G_BA.load_state_dict(state["G_BA"])
            D_A.load_state_dict(state["D_A"])
            D_B.load_state_dict(state["D_B"])
            start_epoch = state["epoch"] + 1
            print(f"  Resumed from epoch {start_epoch}")

    # ── Losses & optimisers ───────────────────────────────────────────────────
    criterion_GAN   = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_idt   = nn.L1Loss()

    opt_G = Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()),
                 lr=CFG.lr_cyclegan, betas=(CFG.beta1_cyclegan, 0.999))
    opt_D_A = Adam(D_A.parameters(), lr=CFG.lr_cyclegan, betas=(CFG.beta1_cyclegan, 0.999))
    opt_D_B = Adam(D_B.parameters(), lr=CFG.lr_cyclegan, betas=(CFG.beta1_cyclegan, 0.999))

    # Linear LR decay in the second half of training
    half = max(1, epochs // 2)
    sched_G   = LinearLR(opt_G,   start_factor=1.0, end_factor=0.0, total_iters=half)
    sched_D_A = LinearLR(opt_D_A, start_factor=1.0, end_factor=0.0, total_iters=half)
    sched_D_B = LinearLR(opt_D_B, start_factor=1.0, end_factor=0.0, total_iters=half)

    pool_A = ImagePool()
    pool_B = ImagePool()

    # ── Data ──────────────────────────────────────────────────────────────────
    from torch.utils.data import DataLoader
    from dataset import OADataset, GradeFilteredDataset, gen_transform

    tf = gen_transform(CFG.img_size, augment=True)
    ds_A = GradeFilteredDataset(splits["train"], [src_grade], transform=tf)
    ds_B = GradeFilteredDataset(splits["train"], [tgt_grade], transform=tf)

    # Per-joint datasets can be very small — cap batch size to dataset size
    batch_size = min(CFG.batch_size_gen, len(ds_A), len(ds_B), 16)
    if batch_size < 1:
        print(f"  [SKIP] Not enough images for KL{src_grade}→KL{tgt_grade} in this joint.")
        return None
    n_iter = min(len(ds_A), len(ds_B)) // batch_size

    loader_A = DataLoader(ds_A, batch_size=batch_size, shuffle=True,
                          num_workers=2, pin_memory=True, drop_last=True)
    loader_B = DataLoader(ds_B, batch_size=batch_size, shuffle=True,
                          num_workers=2, pin_memory=True, drop_last=True)

    print(f"  Domain A (KL{src_grade}): {len(ds_A)} imgs  "
          f"Domain B (KL{tgt_grade}): {len(ds_B)} imgs  "
          f"batch={batch_size}  iters/epoch: {n_iter}")

    if n_iter == 0:
        print(f"  [SKIP] Too few images after batching — skipping this pair.")
        return None

    if len(ds_A) < 16 or len(ds_B) < 16:
        print(f"  [WARN] Very few images for this pair — training may be unstable.")

    # ── Training loop ─────────────────────────────────────────────────────────
    real_label = 1.0
    fake_label = 0.0

    best_G_loss = float("inf")
    best_state = None
    patience = 40
    no_improve = 0

    for epoch in range(start_epoch, epochs):
        G_AB.train(); G_BA.train(); D_A.train(); D_B.train()
        t0 = time.time()
        loss_G_acc = loss_D_acc = 0.0

        iter_B = iter(loader_B)
        for i, (real_A, _) in enumerate(loader_A):
            try:
                real_B, _ = next(iter_B)
            except StopIteration:
                iter_B = iter(loader_B)
                try:
                    real_B, _ = next(iter_B)
                except StopIteration:
                    break

            real_A, real_B = real_A.to(device), real_B.to(device)
            bs = real_A.size(0)

            # ── Generator step ────────────────────────────────────────────────
            opt_G.zero_grad()
            fake_B = G_AB(real_A)
            fake_A = G_BA(real_B)
            rec_A  = G_BA(fake_B)
            rec_B  = G_AB(fake_A)
            idt_A  = G_BA(real_A)
            idt_B  = G_AB(real_B)

            patch_real = torch.ones(D_B(fake_B).shape, device=device)
            loss_G_AB  = criterion_GAN(D_B(fake_B), patch_real)
            loss_G_BA  = criterion_GAN(D_A(fake_A), torch.ones(D_A(fake_A).shape, device=device))
            loss_cyc   = (criterion_cycle(rec_A, real_A) + criterion_cycle(rec_B, real_B)) * CFG.lambda_cycle
            loss_idt   = (criterion_idt(idt_A, real_A) + criterion_idt(idt_B, real_B)) * CFG.lambda_identity * 0.5
            loss_G     = loss_G_AB + loss_G_BA + loss_cyc + loss_idt
            loss_G.backward(); opt_G.step()
            loss_G_acc += loss_G.item()

            # ── Discriminator A ───────────────────────────────────────────────
            opt_D_A.zero_grad()
            pred_real = D_A(real_A)
            pred_fake = D_A(pool_A.query(fake_A.detach()))
            loss_D_A = 0.5 * (criterion_GAN(pred_real, torch.ones_like(pred_real)) +
                               criterion_GAN(pred_fake, torch.zeros_like(pred_fake)))
            loss_D_A.backward(); opt_D_A.step()

            # ── Discriminator B ───────────────────────────────────────────────
            opt_D_B.zero_grad()
            pred_real = D_B(real_B)
            pred_fake = D_B(pool_B.query(fake_B.detach()))
            loss_D_B = 0.5 * (criterion_GAN(pred_real, torch.ones_like(pred_real)) +
                               criterion_GAN(pred_fake, torch.zeros_like(pred_fake)))
            loss_D_B.backward(); opt_D_B.step()
            loss_D_acc += (loss_D_A + loss_D_B).item()

        # LR decay in second half
        if epoch >= epochs // 2:
            sched_G.step(); sched_D_A.step(); sched_D_B.step()

        elapsed = time.time() - t0
        print(f"  Epoch [{epoch+1:3d}/{epochs}]  "
              f"G={loss_G_acc/max(n_iter,1):.4f}  "
              f"D={loss_D_acc/max(n_iter,1):.4f}  "
              f"({elapsed:.0f}s)")

        # Save sample grid every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            G_AB.eval()
            with torch.no_grad():
                sample = G_AB(real_A[:4])
            grid_path = ckpt_dir / f"sample_ep{epoch+1:03d}.png"
            save_image(
                torch.cat([real_A[:4], sample], dim=0) * 0.5 + 0.5,
                grid_path, nrow=4
            )
            # Save latest checkpoint
            torch.save({
                "epoch": epoch,
                "G_AB": G_AB.state_dict(),
                "G_BA": G_BA.state_dict(),
                "D_A": D_A.state_dict(),
                "D_B": D_B.state_dict(),
            }, ckpt_dir / "latest.pth")

        # Save best checkpoint based on generator loss
        epoch_G_loss = loss_G_acc / max(n_iter, 1)
        if epoch_G_loss < best_G_loss:
            best_G_loss = epoch_G_loss
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "G_AB": G_AB.state_dict(),
                "G_BA": G_BA.state_dict(),
                "D_A": D_A.state_dict(),
                "D_B": D_B.state_dict(),
            }, ckpt_dir / "best.pth")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

    # Use best checkpoint for generation
    best_ckpt = ckpt_dir / "best.pth"
    if best_ckpt.exists():
        state = torch.load(best_ckpt, map_location=device)
        G_AB.load_state_dict(state["G_AB"])
        print(f"  Loaded best checkpoint (G_loss={best_G_loss:.4f}) from epoch {state['epoch']+1}")

    print(f"  Done. Best checkpoint at: {ckpt_dir}/best.pth")
    return G_AB


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=CFG.n_epochs_cyclegan)
    parser.add_argument("--pair", type=str, default=None,
                        help="e.g. '1,3' to train only KL1→KL3")
    parser.add_argument("--joint", type=str, default=None,
                        help="Joint type to filter (e.g. dip2). None = all joints pooled.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    joint = args.joint or ("pooled" if not CFG.train_per_joint else None)
    CFG.makedirs(joint)

    # Load best hparams from tuning if available
    hparam_file = Path(CFG.output_dir) / "best_hparams" / f"{joint}_cyclegan.json"
    if hparam_file.exists():
        import json
        best = json.load(open(hparam_file))
        print(f"  Loading tuned hparams from {hparam_file}")
        CFG.lr_cyclegan = best.get("lr", CFG.lr_cyclegan)
        CFG.lambda_cycle = best.get("lambda_cycle", CFG.lambda_cycle)
        CFG.lambda_identity = best.get("lambda_identity", CFG.lambda_identity)
        CFG.n_resblocks = best.get("n_resblocks", CFG.n_resblocks)

    meta = load_metadata()
    from dataset import filter_joint
    meta = filter_joint(meta, joint)
    splits = make_patient_splits(meta)

    pairs = CFG.cyclegan_pairs
    if args.pair:
        src, tgt = map(int, args.pair.split(","))
        pairs = [(src, tgt)]

    for src, tgt in pairs:
        ckpt_dir = CFG.ckpt_path(joint or "pooled", f"cyclegan_kl{src}_to_kl{tgt}").parent
        train_cyclegan(src, tgt, splits, args.epochs, args.resume, ckpt_dir=ckpt_dir)


if __name__ == "__main__":
    main()