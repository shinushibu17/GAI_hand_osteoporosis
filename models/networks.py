"""
models/networks.py — shared building blocks used across all generative models.
"""
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Normalisation helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_norm_layer(norm: str):
    if norm == "batch":
        return functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm == "instance":
        return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm == "none":
        return lambda c: nn.Identity()
    else:
        raise ValueError(f"Unknown norm: {norm}")


# ──────────────────────────────────────────────────────────────────────────────
# ResNet generator (CycleGAN style)
# ──────────────────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, dim: int, norm_layer, use_dropout: bool = False):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            norm_layer(dim),
            nn.ReLU(inplace=True),
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        layers += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            norm_layer(dim),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x)


class ResNetGenerator(nn.Module):
    """CycleGAN ResNet generator. Input/output: (B, C, H, W) in [-1, 1]."""

    def __init__(self, in_ch: int = 1, out_ch: int = 1, ngf: int = 64,
                 n_blocks: int = 6, norm: str = "instance"):
        super().__init__()
        norm_layer = get_norm_layer(norm)
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, ngf, 7),
            norm_layer(ngf),
            nn.ReLU(inplace=True),
        ]
        # Downsample
        for i in range(2):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, 3, stride=2, padding=1),
                norm_layer(ngf * mult * 2),
                nn.ReLU(inplace=True),
            ]
        # ResBlocks
        mult = 4
        for _ in range(n_blocks):
            model.append(ResBlock(ngf * mult, norm_layer))
        # Upsample
        for i in range(2):
            mult = 2 ** (2 - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, 3,
                                   stride=2, padding=1, output_padding=1),
                norm_layer(ngf * mult // 2),
                nn.ReLU(inplace=True),
            ]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, out_ch, 7), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# ──────────────────────────────────────────────────────────────────────────────
# PatchGAN discriminator (CycleGAN / WGAN-GP)
# ──────────────────────────────────────────────────────────────────────────────

class PatchGANDiscriminator(nn.Module):
    """70×70 PatchGAN. Set use_sigmoid=False for WGAN-GP (no final sigmoid)."""

    def __init__(self, in_ch: int = 1, ndf: int = 64, n_layers: int = 3,
                 norm: str = "instance", use_sigmoid: bool = False):
        super().__init__()
        norm_layer = get_norm_layer(norm)
        layers = [nn.Conv2d(in_ch, ndf, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True)]
        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            layers += [
                nn.Conv2d(nf_prev, nf, 4, stride=2, padding=1),
                norm_layer(nf),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        nf_prev = nf
        nf = min(nf * 2, 512)
        layers += [
            nn.Conv2d(nf_prev, nf, 4, stride=1, padding=1),
            norm_layer(nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, 1, 4, stride=1, padding=1),
        ]
        if use_sigmoid:
            layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ──────────────────────────────────────────────────────────────────────────────
# Conditional generator/discriminator for WGAN-GP
# ──────────────────────────────────────────────────────────────────────────────

class ConditionalGenerator(nn.Module):
    """Maps (noise, grade_label) → image via transposed convolutions."""

    def __init__(self, latent_dim: int = 128, n_classes: int = 5,
                 ngf: int = 64, out_ch: int = 1, img_size: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.embed = nn.Embedding(n_classes, latent_dim)
        self.init_size = img_size // 16   # 8 for 128px
        self.l1 = nn.Linear(latent_dim * 2, ngf * 8 * self.init_size ** 2)

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(ngf * 8),
            # 8→16
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 8, ngf * 4, 3, padding=1),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 16→32
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 4, ngf * 2, 3, padding=1),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 32→64
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 2, ngf, 3, padding=1),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # 64→128
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf, out_ch, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_emb = self.embed(labels)
        x = torch.cat([noise, label_emb], dim=1)
        x = self.l1(x)
        x = x.view(x.size(0), -1, self.init_size, self.init_size)
        return self.conv_blocks(x)


class ConditionalCritic(nn.Module):
    """Critic for WGAN-GP: conditions on grade by channel-concatenating a label map."""

    def __init__(self, in_ch: int = 1, n_classes: int = 5,
                 ndf: int = 64, img_size: int = 128):
        super().__init__()
        self.img_size = img_size
        self.embed = nn.Embedding(n_classes, img_size * img_size)

        self.model = nn.Sequential(
            # in_ch + 1 label channel
            nn.Conv2d(in_ch + 1, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # compute flattened size
        dummy = torch.zeros(1, in_ch + 1, img_size, img_size)
        feat_size = self.model(dummy).numel()
        self.fc = nn.Linear(feat_size, 1)

    def forward(self, img: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_map = self.embed(labels).view(img.size(0), 1, self.img_size, self.img_size)
        x = torch.cat([img, label_map], dim=1)
        x = self.model(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ──────────────────────────────────────────────────────────────────────────────
# CVAE encoder / decoder
# ──────────────────────────────────────────────────────────────────────────────

class CVAEEncoder(nn.Module):
    def __init__(self, in_ch: int = 1, n_classes: int = 5,
                 latent_dim: int = 128, nef: int = 32, img_size: int = 128):
        super().__init__()
        self.embed = nn.Embedding(n_classes, img_size * img_size)
        self.img_size = img_size

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + 1, nef, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(nef, nef * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(nef * 2), nn.ReLU(inplace=True),
            nn.Conv2d(nef * 2, nef * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(nef * 4), nn.ReLU(inplace=True),
            nn.Conv2d(nef * 4, nef * 8, 4, stride=2, padding=1),
            nn.BatchNorm2d(nef * 8), nn.ReLU(inplace=True),
        )
        dummy = torch.zeros(1, in_ch + 1, img_size, img_size)
        feat = self.conv(dummy).numel()
        self.fc_mu = nn.Linear(feat, latent_dim)
        self.fc_lv = nn.Linear(feat, latent_dim)

    def forward(self, img, labels):
        lmap = self.embed(labels).view(img.size(0), 1, self.img_size, self.img_size)
        x = torch.cat([img, lmap], dim=1)
        x = self.conv(x).view(img.size(0), -1)
        return self.fc_mu(x), self.fc_lv(x)


class CVAEDecoder(nn.Module):
    def __init__(self, n_classes: int = 5, latent_dim: int = 128,
                 ndf: int = 32, out_ch: int = 1, img_size: int = 128):
        super().__init__()
        self.embed = nn.Embedding(n_classes, latent_dim)
        self.init_size = img_size // 16
        self.l1 = nn.Linear(latent_dim * 2, ndf * 8 * self.init_size ** 2)

        self.deconv = nn.Sequential(
            nn.BatchNorm2d(ndf * 8),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ndf * 8, ndf * 4, 3, padding=1),
            nn.BatchNorm2d(ndf * 4), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ndf * 4, ndf * 2, 3, padding=1),
            nn.BatchNorm2d(ndf * 2), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ndf * 2, ndf, 3, padding=1),
            nn.BatchNorm2d(ndf), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ndf, out_ch, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        emb = self.embed(labels)
        x = torch.cat([z, emb], dim=1)
        x = self.l1(x).view(x.size(0), -1, self.init_size, self.init_size)
        return self.deconv(x)


# ──────────────────────────────────────────────────────────────────────────────
# Image buffer for CycleGAN (to reduce model oscillation)
# ──────────────────────────────────────────────────────────────────────────────

class ImagePool:
    """Stores 50 previously generated images and randomly returns from the pool."""

    def __init__(self, pool_size: int = 50):
        self.pool_size = pool_size
        self.pool: list = []

    def query(self, images: torch.Tensor) -> torch.Tensor:
        if self.pool_size == 0:
            return images
        return_images = []
        for img in images:
            img = img.unsqueeze(0)
            if len(self.pool) < self.pool_size:
                self.pool.append(img)
                return_images.append(img)
            else:
                if torch.rand(1).item() > 0.5:
                    idx = torch.randint(0, len(self.pool), (1,)).item()
                    tmp = self.pool[idx].clone()
                    self.pool[idx] = img
                    return_images.append(tmp)
                else:
                    return_images.append(img)
        return torch.cat(return_images, dim=0)


# ──────────────────────────────────────────────────────────────────────────────
# Gradient penalty (WGAN-GP)
# ──────────────────────────────────────────────────────────────────────────────

def compute_gradient_penalty(critic, real: torch.Tensor, fake: torch.Tensor,
                              labels: torch.Tensor, device: torch.device) -> torch.Tensor:
    alpha = torch.rand(real.size(0), 1, 1, 1, device=device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp = critic(interpolated, labels)
    gradients = torch.autograd.grad(
        outputs=d_interp,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True, retain_graph=True, only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Weight init
# ──────────────────────────────────────────────────────────────────────────────

def init_weights(m, gain: float = 0.02):
    name = type(m).__name__
    if "Conv" in name:
        nn.init.normal_(m.weight, 0.0, gain)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif "BatchNorm2d" in name:
        nn.init.normal_(m.weight, 1.0, gain)
        nn.init.constant_(m.bias, 0.0)
