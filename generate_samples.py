"""
generate_samples.py — generates synthetic KL 3 and KL 4 images from each trained model.

Usage:
    python generate_samples.py [--models cyclegan wgan_gp cvae ddpm] [--n 1000]

Output structure (one dir per grade per model):
    outputs/synthetic/{model}/kl{grade}/*.png
"""
import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image
from PIL import Image

from config import CFG


# ──────────────────────────────────────────────────────────────────────────────
# CycleGAN generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_cyclegan(splits, n_per_grade: int, joint: str = "pooled", perceptual: bool = False):
    from models.networks import ResNetGenerator
    from dataset import GradeFilteredDataset, gen_transform
    from torch.utils.data import DataLoader

    device = CFG.device
    tf = gen_transform(CFG.img_size, augment=False)

    for src, tgt in CFG.cyclegan_pairs:
        model_name = f"cyclegan_vgg_kl{src}_to_kl{tgt}" if perceptual else f"cyclegan_kl{src}_to_kl{tgt}"
        synth_name = "cyclegan_vgg" if perceptual else "cyclegan"
        out_dir = CFG.synth_dir(joint, synth_name, tgt)
        if out_dir.exists() and len(list(out_dir.glob("*.png"))) >= n_per_grade // 2:
            print(f"  CycleGAN{'(VGG)' if perceptual else ''} KL{src}→KL{tgt}: already generated, skipping.")
            continue
        ckpt = CFG.ckpt_path(joint, model_name)
        if not ckpt.exists():
            print(f"  [SKIP] CycleGAN KL{src}→KL{tgt}: checkpoint not found at {ckpt}")
            continue

        state = torch.load(ckpt, map_location=device)
        G = ResNetGenerator(CFG.channels, CFG.channels, n_blocks=CFG.n_resblocks).to(device)
        G.load_state_dict(state["G_AB"])
        G.eval()

        out_dir = CFG.synth_dir(joint, synth_name, tgt)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Use source grade images as input
        ds = GradeFilteredDataset(splits["train"], [src], transform=tf)
        loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=2)

        generated = 0
        with torch.no_grad():
            for imgs, _ in loader:
                imgs = imgs.to(device)
                fakes = G(imgs)
                for j, fake in enumerate(fakes):
                    if generated >= n_per_grade:
                        break
                    img_t = fake * 0.5 + 0.5
                    save_image(img_t, out_dir / f"cyc_{src}_{tgt}_{generated:05d}.png")
                    generated += 1
                if generated >= n_per_grade:
                    break

        print(f"  CycleGAN{'(VGG)' if perceptual else ''} KL{src}→KL{tgt}: {generated} images → {out_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# WGAN-GP generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_wgan_gp(n_per_grade: int, joint: str = "pooled"):
    from models.networks import ConditionalGenerator
    import json

    device = CFG.device
    ckpt = CFG.ckpt_path(joint, "wgan_gp")
    if not ckpt.exists():
        print(f"  [SKIP] WGAN-GP: checkpoint not found at {ckpt}")
        return

    # Load best hparams if available
    latent_dim = CFG.latent_dim
    hparam_file = Path(CFG.output_dir) / "best_hparams" / f"{joint}_wgan_gp.json"
    if hparam_file.exists():
        best = json.load(open(hparam_file))
        latent_dim = best.get("latent_dim", latent_dim)
        print(f"  Using tuned latent_dim={latent_dim} from {hparam_file}")

    state = torch.load(ckpt, map_location=device)
    G = ConditionalGenerator(latent_dim=latent_dim, n_classes=5,
                              out_ch=CFG.channels, img_size=CFG.img_size).to(device)
    G.load_state_dict(state["G"])
    G.eval()

    batch = 64
    for grade in CFG.target_grades:
        out_dir = CFG.synth_dir(joint, "wgan_gp", grade)
        if out_dir.exists() and len(list(out_dir.glob("*.png"))) >= n_per_grade:
            print(f"  WGAN-GP KL{grade}: already generated, skipping.")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        generated = 0
        with torch.no_grad():
            while generated < n_per_grade:
                n_batch = min(batch, n_per_grade - generated)
                noise = torch.randn(n_batch, latent_dim, device=device)
                labels = torch.full((n_batch,), grade, dtype=torch.long, device=device)
                fakes = G(noise, labels)
                for j, fake in enumerate(fakes):
                    img_t = fake * 0.5 + 0.5
                    save_image(img_t, out_dir / f"wgan_{grade}_{generated:05d}.png")
                    generated += 1
        print(f"  WGAN-GP KL{grade}: {generated} images → {out_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# CVAE generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_cvae(n_per_grade: int, joint: str = "pooled"):
    from models.networks import CVAEDecoder
    import json

    device = CFG.device
    ckpt = CFG.ckpt_path(joint, "cvae")
    if not ckpt.exists():
        print(f"  [SKIP] CVAE: checkpoint not found at {ckpt}")
        return

    # Load best hparams if available
    latent_dim_vae = CFG.latent_dim_vae
    hparam_file = Path(CFG.output_dir) / "best_hparams" / f"{joint}_cvae.json"
    if hparam_file.exists():
        best = json.load(open(hparam_file))
        latent_dim_vae = best.get("latent_dim_vae", latent_dim_vae)
        print(f"  Using tuned latent_dim_vae={latent_dim_vae} from {hparam_file}")

    state = torch.load(ckpt, map_location=device, weights_only=False)
    dec = CVAEDecoder(n_classes=5, latent_dim=latent_dim_vae,
                      out_ch=CFG.channels, img_size=CFG.img_size).to(device)
    dec.load_state_dict(state["dec"])
    dec.eval()

    batch = 64
    for grade in CFG.target_grades:
        out_dir = CFG.synth_dir(joint, "cvae", grade)
        if out_dir.exists() and len(list(out_dir.glob("*.png"))) >= n_per_grade:
            print(f"  CVAE KL{grade}: already generated, skipping.")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        generated = 0
        with torch.no_grad():
            while generated < n_per_grade:
                n_batch = min(batch, n_per_grade - generated)
                z = torch.randn(n_batch, latent_dim_vae, device=device)
                labels = torch.full((n_batch,), grade, dtype=torch.long, device=device)
                fakes = dec(z, labels)
                for j, fake in enumerate(fakes):
                    img_t = fake * 0.5 + 0.5
                    save_image(img_t, out_dir / f"cvae_{grade}_{generated:05d}.png")
                    generated += 1
        print(f"  CVAE KL{grade}: {generated} images → {out_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# DDPM generation (full denoising, slow)
# ──────────────────────────────────────────────────────────────────────────────

def generate_ddpm(n_per_grade: int, inference_steps: int = 200, joint: str = "pooled"):
    try:
        from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler
    except ImportError:
        print("  [SKIP] DDPM: diffusers not installed")
        return

    device = CFG.device
    ckpt = CFG.ckpt_path(joint, "ddpm")
    if not ckpt.exists():
        print(f"  [SKIP] DDPM: checkpoint not found at {ckpt}")
        return

    n_classes = 5
    model = UNet2DModel(
        sample_size=CFG.img_size, in_channels=CFG.channels, out_channels=CFG.channels,
        layers_per_block=2, block_out_channels=(64, 128, 128, 256),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        num_class_embeds=n_classes + 1,
    ).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    # Use DDIM for faster inference
    scheduler = DDIMScheduler(num_train_timesteps=CFG.n_timesteps)
    scheduler.set_timesteps(inference_steps)

    batch = 16
    for grade in CFG.target_grades:
        out_dir = CFG.synth_dir(joint, "ddpm", grade)
        if out_dir.exists() and len(list(out_dir.glob("*.png"))) >= n_per_grade:
            print(f"  DDPM KL{grade}: already generated, skipping.")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        generated = 0

        with torch.no_grad():
            while generated < n_per_grade:
                n_batch = min(batch, n_per_grade - generated)
                x = torch.randn(n_batch, CFG.channels, CFG.img_size, CFG.img_size, device=device)
                labels = torch.full((n_batch,), grade, dtype=torch.long, device=device)
                uncond = torch.full((n_batch,), n_classes, dtype=torch.long, device=device)

                for t in scheduler.timesteps:
                    t_batch = torch.full((n_batch,), t, dtype=torch.long, device=device)
                    noise_cond   = model(x, t_batch, class_labels=labels).sample
                    noise_uncond = model(x, t_batch, class_labels=uncond).sample
                    noise_pred   = noise_uncond + CFG.guidance_scale * (noise_cond - noise_uncond)
                    x = scheduler.step(noise_pred, t, x).prev_sample

                for j, img in enumerate(x):
                    img_t = img * 0.5 + 0.5
                    save_image(img_t, out_dir / f"ddpm_{grade}_{generated:05d}.png")
                    generated += 1

        print(f"  DDPM KL{grade}: {generated} images → {out_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=["cyclegan", "wgan_gp", "cvae"],
                        choices=["cyclegan", "wgan_gp", "cvae", "ddpm"])
    parser.add_argument("--joint", type=str, default=None,
                        help="Joint type (e.g. dip2). None = pooled.")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--ddpm_steps", type=int, default=200)
    parser.add_argument("--perceptual", action="store_true",
                        help="Generate from VGG perceptual loss CycleGAN checkpoint")
    args = parser.parse_args()

    joint = args.joint or "pooled"
    CFG.makedirs(joint)

    if "cyclegan" in args.models:
        from dataset import load_metadata, make_patient_splits, filter_joint
        meta = load_metadata()
        meta = filter_joint(meta, joint)
        splits = make_patient_splits(meta)
        print("\n── CycleGAN generation ──────────────────────────────────────")
        generate_cyclegan(splits, args.n, joint=joint, perceptual=args.perceptual)

    if "wgan_gp" in args.models:
        print("\n── WGAN-GP generation ───────────────────────────────────────")
        generate_wgan_gp(args.n, joint=joint)

    if "cvae" in args.models:
        print("\n── CVAE generation ──────────────────────────────────────────")
        generate_cvae(args.n, joint=joint)

    if "ddpm" in args.models:
        print("\n── DDPM generation ──────────────────────────────────────────")
        generate_ddpm(args.n, args.ddpm_steps, joint=joint)

    print("\nDone. Synthetic images written to:", CFG.synthetic_dir)


if __name__ == "__main__":
    main()