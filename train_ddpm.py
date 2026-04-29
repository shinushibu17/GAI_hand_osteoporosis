"""
train_ddpm.py — trains a class-conditioned DDPM (HuggingFace Diffusers UNet2DModel)
with classifier-free guidance for KL grade synthesis.

Usage:
    python train_ddpm.py [--epochs 100] [--resume]

Checkpoint: outputs/checkpoints/ddpm/latest.pth
NOTE: DDPM is slowest — run only if CycleGAN + WGAN-GP + CVAE are done.
"""
import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.utils import save_image

from config import CFG
from dataset import load_metadata, make_patient_splits


def train_ddpm(splits, epochs: int, resume: bool = False, ckpt_dir: Path = None):
    try:
        from diffusers import UNet2DModel, DDPMScheduler
    except ImportError:
        raise ImportError("Install diffusers: pip install diffusers accelerate")

    device = CFG.device
    if ckpt_dir is None:
        ckpt_dir = Path(CFG.ckpt_dir) / "ddpm"
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"DDPM (classifier-free guidance)   device={device}")
    print(f"{'='*60}")

    n_classes = 5

    # UNet: class conditioning via class_embed_type="simple"
    model = UNet2DModel(
        sample_size=CFG.img_size,
        in_channels=CFG.channels,
        out_channels=CFG.channels,
        layers_per_block=2,
        block_out_channels=(64, 128, 128, 256),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        num_class_embeds=n_classes + 1,   # +1 for unconditional token (index n_classes)
    ).to(device)

    noise_scheduler = DDPMScheduler(num_train_timesteps=CFG.n_timesteps, beta_schedule="linear")

    start_epoch = 0
    if resume:
        ckpt = ckpt_dir / "latest.pth"
        if ckpt.exists():
            state = torch.load(ckpt, map_location=device)
            model.load_state_dict(state["model"])
            start_epoch = state["epoch"] + 1
            print(f"  Resumed from epoch {start_epoch}")

    opt = AdamW(model.parameters(), lr=CFG.lr_ddpm)
    sched = CosineAnnealingLR(opt, T_max=epochs)

    from torch.utils.data import DataLoader, WeightedRandomSampler
    from dataset import OADataset, gen_transform
    import numpy as np

    tf = gen_transform(CFG.img_size, augment=True)
    train_ds = OADataset(splits["train"], transform=tf)
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

    print(f"  Training on {len(train_ds)} images  timesteps={CFG.n_timesteps}")

    uncond_token = torch.tensor(n_classes, device=device)

    best_loss = float("inf")
    patience = 40
    no_improve = 0

    for epoch in range(start_epoch, epochs):
        model.train()
        t0 = time.time()
        loss_acc = 0.0

        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            bs = imgs.size(0)

            mask = (torch.rand(bs, device=device) < CFG.p_uncond)
            class_labels = labels.clone()
            class_labels[mask] = n_classes

            noise = torch.randn_like(imgs)
            timesteps = torch.randint(0, CFG.n_timesteps, (bs,), device=device).long()
            noisy = noise_scheduler.add_noise(imgs, noise, timesteps)

            pred = model(noisy, timesteps, class_labels=class_labels).sample
            loss = F.mse_loss(pred, noise)

            opt.zero_grad(); loss.backward(); opt.step()
            loss_acc += loss.item()

        sched.step()
        epoch_loss = loss_acc / len(loader)
        elapsed = time.time() - t0
        print(f"  Epoch [{epoch+1:3d}/{epochs}]  Loss={epoch_loss:.4f}  ({elapsed:.0f}s)")

        if (epoch + 1) % 10 == 0 or epoch == 0:
            torch.save({"epoch": epoch, "model": model.state_dict()},
                       ckpt_dir / "latest.pth")
            _quick_sample(model, noise_scheduler, ckpt_dir, epoch, device, n_classes)

        # Best checkpoint + early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            no_improve = 0
            torch.save({"epoch": epoch, "model": model.state_dict()},
                       ckpt_dir / "best.pth")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

    # Load best checkpoint
    best_ckpt = ckpt_dir / "best.pth"
    if best_ckpt.exists():
        state = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        print(f"  Loaded best checkpoint (loss={best_loss:.4f}) from epoch {state['epoch']+1}")

    print(f"  Done. Checkpoint at: {ckpt_dir}/best.pth")
    return model, noise_scheduler


@torch.no_grad()
def _quick_sample(model, scheduler, out_dir, epoch, device, n_classes,
                  target_grades=(3, 4), n_per_grade=4, steps=50):
    from diffusers import DDPMScheduler
    model.eval()
    imgs_out = []
    for grade in target_grades:
        x = torch.randn(n_per_grade, CFG.channels, CFG.img_size, CFG.img_size, device=device)
        labels = torch.full((n_per_grade,), grade, dtype=torch.long, device=device)
        uncond = torch.full((n_per_grade,), n_classes, dtype=torch.long, device=device)

        step_indices = list(range(scheduler.config.num_train_timesteps - 1, -1, -1))
        step_indices = step_indices[::scheduler.config.num_train_timesteps // steps]

        for t in step_indices:
            t_batch = torch.full((n_per_grade,), t, dtype=torch.long, device=device)
            # CFG: combine conditional and unconditional predictions
            noise_cond   = model(x, t_batch, class_labels=labels).sample
            noise_uncond = model(x, t_batch, class_labels=uncond).sample
            noise_pred   = noise_uncond + CFG.guidance_scale * (noise_cond - noise_uncond)
            x = scheduler.step(noise_pred, t, x).prev_sample
        imgs_out.append(x)

    save_image(
        torch.cat(imgs_out) * 0.5 + 0.5,
        out_dir / f"sample_ep{epoch+1:03d}.png", nrow=n_per_grade
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=CFG.n_epochs_ddpm)
    parser.add_argument("--joint", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    joint = args.joint or "pooled"
    CFG.makedirs(joint)
    meta = load_metadata()
    from dataset import filter_joint
    meta = filter_joint(meta, joint)
    splits = make_patient_splits(meta)
    ckpt_dir = CFG.ckpt_path(joint, "ddpm").parent
    train_ddpm(splits, args.epochs, args.resume, ckpt_dir=ckpt_dir)


if __name__ == "__main__":
    main()