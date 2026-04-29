"""
tune_models.py — quick hyperparameter search for each generative model.

Strategy:
  1. Train each config for TUNE_EPOCHS (fast)
  2. Generate 100 sample images per config
  3. Compute FID vs real images
  4. Save best config to outputs/best_hparams/{group}_{model}.json
  5. Best config is automatically picked up by train_*.py via --tune flag

Usage:
    python3 tune_models.py --joint dip --models cyclegan wgan_gp cvae
    python3 tune_models.py --joint pip --models cyclegan wgan_gp cvae

Outputs:
    outputs/best_hparams/{joint}_cyclegan.json
    outputs/best_hparams/{joint}_wgan_gp.json
    outputs/best_hparams/{joint}_cvae.json
"""
import argparse
import json
import shutil
from pathlib import Path

import torch
from config import CFG
from dataset import load_metadata, make_patient_splits, filter_joint

TUNE_EPOCHS = 30       # fast trial
TUNE_SAMPLES = 100     # images for FID check
TUNE_BATCH = 16        # small batch for speed

# ── Hyperparameter grids ──────────────────────────────────────────────────────

HPARAM_GRIDS = {
    "cyclegan": [
        {"lr": 2e-4, "lambda_cycle": 10.0, "lambda_identity": 5.0, "n_resblocks": 6},
        {"lr": 1e-4, "lambda_cycle": 5.0,  "lambda_identity": 2.0, "n_resblocks": 6},
        {"lr": 2e-4, "lambda_cycle": 10.0, "lambda_identity": 0.0, "n_resblocks": 9},
    ],
    "wgan_gp": [
        {"lr": 1e-4, "latent_dim": 128, "n_critic": 5,  "lambda_gp": 10.0},
        {"lr": 2e-4, "latent_dim": 256, "n_critic": 3,  "lambda_gp": 10.0},
        {"lr": 5e-5, "latent_dim": 128, "n_critic": 5,  "lambda_gp": 5.0},
    ],
    "cvae": [
        {"lr": 1e-3, "latent_dim_vae": 128, "kl_weight": 1.0},
        {"lr": 5e-4, "latent_dim_vae": 256, "kl_weight": 0.5},
        {"lr": 1e-3, "latent_dim_vae": 64,  "kl_weight": 2.0},
    ],
}


def compute_fid_quick(real_dir: str, fake_dir: str, device: str) -> float:
    try:
        from cleanfid import fid
        score = fid.compute_fid(real_dir, fake_dir, device=device,
                                num_workers=2, verbose=False)
        return float(score)
    except Exception as e:
        print(f"  [WARN] FID failed: {e}")
        return float("inf")


def export_real_images(splits, grade: int, joint: str, n: int = 200) -> str:
    """Export n real test images for FID comparison."""
    from torchvision.utils import save_image
    from dataset import GradeFilteredDataset, gen_transform
    from torch.utils.data import DataLoader

    out_dir = Path(CFG.output_dir) / "tune_real" / joint / f"kl{grade}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(list(out_dir.glob("*.png"))) >= n:
        return str(out_dir)

    ds = GradeFilteredDataset(splits["test"], [grade],
                               transform=gen_transform(CFG.img_size, augment=False))
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)
    saved = 0
    with torch.no_grad():
        for imgs, _ in loader:
            for img in imgs:
                if saved >= n:
                    break
                save_image(img * 0.5 + 0.5, out_dir / f"real_{saved:05d}.png")
                saved += 1
            if saved >= n:
                break
    return str(out_dir)


# ── Per-model tuning functions ────────────────────────────────────────────────

def tune_cyclegan(splits, joint: str, device: torch.device) -> dict:
    from models.networks import ResNetGenerator, PatchGANDiscriminator, ImagePool, init_weights
    from dataset import GradeFilteredDataset, gen_transform
    from torch.utils.data import DataLoader
    from torchvision.utils import save_image
    import itertools

    best_fid = float("inf")
    best_config = HPARAM_GRIDS["cyclegan"][0]

    real_dir = export_real_images(splits, 3, joint)

    for ci, cfg_dict in enumerate(HPARAM_GRIDS["cyclegan"]):
        print(f"\n  CycleGAN config {ci+1}/{len(HPARAM_GRIDS['cyclegan'])}: {cfg_dict}")
        ckpt_dir = Path(CFG.output_dir) / "tune_ckpts" / joint / f"cyclegan_c{ci}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        G_AB = ResNetGenerator(CFG.channels, CFG.channels,
                               n_blocks=cfg_dict["n_resblocks"]).to(device)
        G_BA = ResNetGenerator(CFG.channels, CFG.channels,
                               n_blocks=cfg_dict["n_resblocks"]).to(device)
        D_A = PatchGANDiscriminator(CFG.channels).to(device)
        D_B = PatchGANDiscriminator(CFG.channels).to(device)
        for net in [G_AB, G_BA, D_A, D_B]:
            net.apply(init_weights)

        import torch.nn as nn
        criterion_GAN = nn.MSELoss()
        criterion_cycle = nn.L1Loss()
        criterion_idt = nn.L1Loss()

        opt_G = torch.optim.Adam(
            itertools.chain(G_AB.parameters(), G_BA.parameters()),
            lr=cfg_dict["lr"], betas=(0.5, 0.999))
        opt_D_A = torch.optim.Adam(D_A.parameters(), lr=cfg_dict["lr"], betas=(0.5, 0.999))
        opt_D_B = torch.optim.Adam(D_B.parameters(), lr=cfg_dict["lr"], betas=(0.5, 0.999))

        tf = gen_transform(CFG.img_size, augment=True)
        ds_A = GradeFilteredDataset(splits["train"], [1], transform=tf)
        ds_B = GradeFilteredDataset(splits["train"], [3], transform=tf)
        bs = min(TUNE_BATCH, len(ds_A), len(ds_B))
        if bs < 1:
            print("  [SKIP] Not enough images")
            continue

        loader_A = DataLoader(ds_A, batch_size=bs, shuffle=True, num_workers=2, drop_last=True)
        loader_B = DataLoader(ds_B, batch_size=bs, shuffle=True, num_workers=2, drop_last=True)
        pool_A = ImagePool(); pool_B = ImagePool()

        for epoch in range(TUNE_EPOCHS):
            G_AB.train(); G_BA.train(); D_A.train(); D_B.train()
            iter_B = iter(loader_B)
            for real_A, _ in loader_A:
                try:
                    real_B, _ = next(iter_B)
                except StopIteration:
                    iter_B = iter(loader_B)
                    try:
                        real_B, _ = next(iter_B)
                    except StopIteration:
                        break

                real_A, real_B = real_A.to(device), real_B.to(device)
                fake_B = G_AB(real_A); fake_A = G_BA(real_B)
                rec_A = G_BA(fake_B); rec_B = G_AB(fake_A)

                opt_G.zero_grad()
                loss_G = (criterion_GAN(D_B(fake_B), torch.ones_like(D_B(fake_B))) +
                          criterion_GAN(D_A(fake_A), torch.ones_like(D_A(fake_A))) +
                          (criterion_cycle(rec_A, real_A) + criterion_cycle(rec_B, real_B)) * cfg_dict["lambda_cycle"] +
                          (criterion_idt(G_BA(real_A), real_A) + criterion_idt(G_AB(real_B), real_B)) * cfg_dict["lambda_identity"] * 0.5)
                loss_G.backward(); opt_G.step()

                opt_D_B.zero_grad()
                loss_D_B = 0.5 * (criterion_GAN(D_B(real_B), torch.ones_like(D_B(real_B))) +
                                   criterion_GAN(D_B(pool_B.query(fake_B.detach())), torch.zeros_like(D_B(fake_B))))
                loss_D_B.backward(); opt_D_B.step()

                opt_D_A.zero_grad()
                loss_D_A = 0.5 * (criterion_GAN(D_A(real_A), torch.ones_like(D_A(real_A))) +
                                   criterion_GAN(D_A(pool_A.query(fake_A.detach())), torch.zeros_like(D_A(fake_A))))
                loss_D_A.backward(); opt_D_A.step()

        # Generate samples and compute FID
        fake_dir = Path(CFG.output_dir) / "tune_fake" / joint / f"cyclegan_c{ci}"
        fake_dir.mkdir(parents=True, exist_ok=True)
        G_AB.eval()
        generated = 0
        with torch.no_grad():
            for imgs, _ in loader_A:
                fakes = G_AB(imgs.to(device))
                for fake in fakes:
                    if generated >= TUNE_SAMPLES:
                        break
                    save_image(fake * 0.5 + 0.5, fake_dir / f"fake_{generated:05d}.png")
                    generated += 1
                if generated >= TUNE_SAMPLES:
                    break

        fid_score = compute_fid_quick(real_dir, str(fake_dir), str(device))
        print(f"  Config {ci+1} FID = {fid_score:.2f}")

        if fid_score < best_fid:
            best_fid = fid_score
            best_config = cfg_dict
            print(f"  ✓ New best config!")

    print(f"\n  Best CycleGAN config (FID={best_fid:.2f}): {best_config}")
    return {**best_config, "best_fid": best_fid}


def tune_wgan_gp(splits, joint: str, device: torch.device) -> dict:
    from models.networks import ConditionalGenerator, ConditionalCritic, compute_gradient_penalty, init_weights
    from dataset import OADataset, gen_transform
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from torchvision.utils import save_image
    import numpy as np

    best_fid = float("inf")
    best_config = HPARAM_GRIDS["wgan_gp"][0]
    real_dir = export_real_images(splits, 3, joint)

    for ci, cfg_dict in enumerate(HPARAM_GRIDS["wgan_gp"]):
        print(f"\n  WGAN-GP config {ci+1}/{len(HPARAM_GRIDS['wgan_gp'])}: {cfg_dict}")

        G = ConditionalGenerator(latent_dim=cfg_dict["latent_dim"], n_classes=5,
                                  out_ch=CFG.channels, img_size=CFG.img_size).to(device)
        C = ConditionalCritic(in_ch=CFG.channels, n_classes=5, img_size=CFG.img_size).to(device)
        G.apply(init_weights); C.apply(init_weights)

        opt_G = torch.optim.Adam(G.parameters(), lr=cfg_dict["lr"], betas=(0.0, 0.9))
        opt_C = torch.optim.Adam(C.parameters(), lr=cfg_dict["lr"], betas=(0.0, 0.9))

        tf = gen_transform(CFG.img_size, augment=True)
        train_ds = OADataset(splits["train"], transform=tf)
        bs = min(TUNE_BATCH, len(train_ds))
        grade_counts = splits["train"][CFG.grade_col].value_counts()
        sw = splits["train"][CFG.grade_col].map(lambda g: 1.0 / grade_counts[g]).values
        sampler = WeightedRandomSampler(torch.from_numpy(sw.astype(np.float32)),
                                        num_samples=len(train_ds), replacement=True)
        loader = DataLoader(train_ds, batch_size=bs, sampler=sampler,
                            num_workers=2, pin_memory=True, drop_last=True)

        for epoch in range(TUNE_EPOCHS):
            G.train(); C.train()
            for i, (real_imgs, labels) in enumerate(loader):
                real_imgs, labels = real_imgs.to(device), labels.to(device)
                noise = torch.randn(bs, cfg_dict["latent_dim"], device=device)
                fake_imgs = G(noise, labels).detach()
                opt_C.zero_grad()
                gp = compute_gradient_penalty(C, real_imgs, fake_imgs, labels, device)
                loss_C = -C(real_imgs, labels).mean() + C(fake_imgs, labels).mean() + cfg_dict["lambda_gp"] * gp
                loss_C.backward(); opt_C.step()
                if (i + 1) % cfg_dict["n_critic"] == 0:
                    opt_G.zero_grad()
                    noise2 = torch.randn(bs, cfg_dict["latent_dim"], device=device)
                    loss_G = -C(G(noise2, labels), labels).mean()
                    loss_G.backward(); opt_G.step()

        fake_dir = Path(CFG.output_dir) / "tune_fake" / joint / f"wgan_c{ci}"
        fake_dir.mkdir(parents=True, exist_ok=True)
        G.eval()
        with torch.no_grad():
            for j in range(0, TUNE_SAMPLES, 16):
                n = min(16, TUNE_SAMPLES - j)
                noise = torch.randn(n, cfg_dict["latent_dim"], device=device)
                lbls = torch.full((n,), 3, dtype=torch.long, device=device)
                fakes = G(noise, lbls)
                for k, fake in enumerate(fakes):
                    save_image(fake * 0.5 + 0.5, fake_dir / f"fake_{j+k:05d}.png")

        fid_score = compute_fid_quick(real_dir, str(fake_dir), str(device))
        print(f"  Config {ci+1} FID = {fid_score:.2f}")
        if fid_score < best_fid:
            best_fid = fid_score
            best_config = cfg_dict
            print(f"  ✓ New best config!")

    print(f"\n  Best WGAN-GP config (FID={best_fid:.2f}): {best_config}")
    return {**best_config, "best_fid": best_fid}


def tune_cvae(splits, joint: str, device: torch.device) -> dict:
    from models.networks import CVAEEncoder, CVAEDecoder, init_weights
    from dataset import OADataset, gen_transform
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from torchvision.utils import save_image
    from train_cvae import reparameterise, vae_loss
    import numpy as np

    best_fid = float("inf")
    best_config = HPARAM_GRIDS["cvae"][0]
    real_dir = export_real_images(splits, 3, joint)

    for ci, cfg_dict in enumerate(HPARAM_GRIDS["cvae"]):
        print(f"\n  CVAE config {ci+1}/{len(HPARAM_GRIDS['cvae'])}: {cfg_dict}")

        enc = CVAEEncoder(in_ch=CFG.channels, n_classes=5,
                          latent_dim=cfg_dict["latent_dim_vae"], img_size=CFG.img_size).to(device)
        dec = CVAEDecoder(n_classes=5, latent_dim=cfg_dict["latent_dim_vae"],
                          out_ch=CFG.channels, img_size=CFG.img_size).to(device)
        enc.apply(init_weights); dec.apply(init_weights)
        opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()),
                               lr=cfg_dict["lr"])

        tf = gen_transform(CFG.img_size, augment=True)
        train_ds = OADataset(splits["train"], transform=tf)
        bs = min(TUNE_BATCH, len(train_ds))
        grade_counts = splits["train"][CFG.grade_col].value_counts()
        sw = splits["train"][CFG.grade_col].map(lambda g: 1.0 / grade_counts[g]).values
        sampler = WeightedRandomSampler(torch.from_numpy(sw.astype(np.float32)),
                                        num_samples=len(train_ds), replacement=True)
        loader = DataLoader(train_ds, batch_size=bs, sampler=sampler,
                            num_workers=2, pin_memory=True, drop_last=True)

        for epoch in range(TUNE_EPOCHS):
            enc.train(); dec.train()
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                mu, log_var = enc(imgs, labels)
                z = reparameterise(mu, log_var)
                recon = dec(z, labels)
                loss, _, _ = vae_loss(recon, imgs, mu, log_var, cfg_dict["kl_weight"])
                if torch.isnan(loss):
                    continue
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()), 1.0)
                opt.step()

        fake_dir = Path(CFG.output_dir) / "tune_fake" / joint / f"cvae_c{ci}"
        fake_dir.mkdir(parents=True, exist_ok=True)
        dec.eval()
        with torch.no_grad():
            for j in range(0, TUNE_SAMPLES, 16):
                n = min(16, TUNE_SAMPLES - j)
                z = torch.randn(n, cfg_dict["latent_dim_vae"], device=device)
                lbls = torch.full((n,), 3, dtype=torch.long, device=device)
                fakes = dec(z, lbls)
                for k, fake in enumerate(fakes):
                    save_image(fake * 0.5 + 0.5, fake_dir / f"fake_{j+k:05d}.png")

        fid_score = compute_fid_quick(real_dir, str(fake_dir), str(device))
        print(f"  Config {ci+1} FID = {fid_score:.2f}")
        if fid_score < best_fid:
            best_fid = fid_score
            best_config = cfg_dict
            print(f"  ✓ New best config!")

    print(f"\n  Best CVAE config (FID={best_fid:.2f}): {best_config}")
    return {**best_config, "best_fid": best_fid}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint",   required=True, help="dip, pip, or mcp")
    parser.add_argument("--models",  nargs="+", default=["cyclegan", "wgan_gp", "cvae"],
                        choices=["cyclegan", "wgan_gp", "cvae"])
    parser.add_argument("--gpu",     type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"\nTuning on device: {device}  joint: {args.joint}")

    CFG.makedirs(args.joint)
    out_dir = Path(CFG.output_dir) / "best_hparams"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_metadata()
    from dataset import filter_joint
    meta = filter_joint(meta, args.joint)
    splits = make_patient_splits(meta)

    print(f"  train: {len(splits['train'])} images  "
          f"KL3={len(splits['train'][splits['train'][CFG.grade_col]==3])}  "
          f"KL4={len(splits['train'][splits['train'][CFG.grade_col]==4])}")

    results = {}

    if "cyclegan" in args.models:
        print(f"\n{'='*50}\nTuning CycleGAN\n{'='*50}")
        results["cyclegan"] = tune_cyclegan(splits, args.joint, device)
        with open(out_dir / f"{args.joint}_cyclegan.json", "w") as f:
            json.dump(results["cyclegan"], f, indent=2)

    if "wgan_gp" in args.models:
        print(f"\n{'='*50}\nTuning WGAN-GP\n{'='*50}")
        results["wgan_gp"] = tune_wgan_gp(splits, args.joint, device)
        with open(out_dir / f"{args.joint}_wgan_gp.json", "w") as f:
            json.dump(results["wgan_gp"], f, indent=2)

    if "cvae" in args.models:
        print(f"\n{'='*50}\nTuning CVAE\n{'='*50}")
        results["cvae"] = tune_cvae(splits, args.joint, device)
        with open(out_dir / f"{args.joint}_cvae.json", "w") as f:
            json.dump(results["cvae"], f, indent=2)

    print(f"\n{'='*50}")
    print(f"Best configs saved to: {out_dir}")
    for model, res in results.items():
        print(f"  {model}: FID={res.get('best_fid', 'n/a'):.2f}  config={res}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
