"""
transforms.py
=============
Custom torchvision-compatible grayscale image transforms for finger-joint
OA X-ray preprocessing. All transforms operate on PIL Images and return
PIL Images, making them compatible with torchvision.transforms.Compose.

Available transforms
--------------------
- CLAHE : Contrast Limited Adaptive Histogram Equalization
- NLMFilter : Non-Local Means denoising
- BilateralFilter : Edge-preserving bilateral filter
- MedianFilter : Median blur
- InvertGrayscale : Bitwise inversion

Usage
-----
>>> from data_prep.transforms import CLAHE, NLMFilter
>>> from torchvision import transforms
>>>
>>> pipeline = transforms.Compose([
... transforms.Resize((224, 224)),
... NLMFilter(h=5),
... CLAHE(clip_limit=1.0),
... transforms.ToTensor(),
... transforms.Normalize(mean=[0.5], std=[0.5]),
... ])
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# CLAHE
# ---------------------------------------------------------------------------

class CLAHE:
    """
    Contrast Limited Adaptive Histogram Equalization.

    Enhances local contrast of joint space, cortical bone, and osteophytes.
    Not available in torchvision — custom implementation using OpenCV.

    Parameters
    ----------
    clip_limit : contrast clip limit — lower = gentler enhancement (default 1.0)
    tile_grid : tile size for local histogram computation (default (16, 16))
    """

    def __init__(self, clip_limit: float = 1.0, tile_grid: tuple[int, int] = (16, 16)) -> None:
        self.clip_limit = clip_limit
        self.tile_grid = tile_grid

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid)
        return Image.fromarray(clahe.apply(arr))

    def __repr__(self) -> str:
        return f"CLAHE(clip_limit={self.clip_limit}, tile_grid={self.tile_grid})"


# ---------------------------------------------------------------------------
# NLMFilter
# ---------------------------------------------------------------------------

class NLMFilter:
    """
    Non-Local Means denoising.

    Suppresses scan line artifacts and noise while preserving bone structure
    by averaging similar patches across the entire image. Dissimilar patches
    (like artifacts) get near-zero weight and are smoothed out.

    Parameters
    ----------
    h : filter strength — higher = more smoothing (default 5)
    template_window : patch size for similarity comparison (default 5)
    search_window : search area for similar patches (default 11)
    """

    def __init__(self, h: int = 5, template_window: int = 5, search_window: int = 11) -> None:
        self.h = h
        self.template_window = template_window
        self.search_window = search_window

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img)
        arr = cv2.fastNlMeansDenoising(
            arr,
            h = self.h,
            templateWindowSize = self.template_window,
            searchWindowSize = self.search_window,
        )
        return Image.fromarray(arr)

    def __repr__(self) -> str:
        return (f"NLMFilter(h={self.h}, template_window={self.template_window}, "
                f"search_window={self.search_window})")


# ---------------------------------------------------------------------------
# BilateralFilter
# ---------------------------------------------------------------------------

class BilateralFilter:
    """
    Edge-preserving bilateral filter.

    Smooths flat regions while keeping sharp edges (bone boundaries) intact
    by weighting neighbours by both spatial distance and intensity similarity.

    Parameters
    ----------
    d : diameter of pixel neighbourhood (default 7)
    sigma_color : intensity range for averaging — lower = sharper edges (default 30)
    sigma_space : spatial range for averaging (default 30)
    """

    def __init__(self, d: int = 7, sigma_color: int = 30, sigma_space: int = 30) -> None:
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img)
        arr = cv2.bilateralFilter(arr, self.d, self.sigma_color, self.sigma_space)
        return Image.fromarray(arr)

    def __repr__(self) -> str:
        return (f"BilateralFilter(d={self.d}, sigma_color={self.sigma_color}, "
                f"sigma_space={self.sigma_space})")


# ---------------------------------------------------------------------------
# MedianFilter
# ---------------------------------------------------------------------------

class MedianFilter:
    """
    Median blur.

    Removes salt-and-pepper noise by replacing each pixel with the median
    of its neighbourhood. Less effective than NLM on structured artifacts
    but much faster.

    Parameters
    ----------
    kernel_size : neighbourhood size — must be odd (default 3)
    """

    def __init__(self, kernel_size: int = 3) -> None:
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")
        self.kernel_size = kernel_size

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img)
        arr = cv2.medianBlur(arr, self.kernel_size)
        return Image.fromarray(arr)

    def __repr__(self) -> str:
        return f"MedianFilter(kernel_size={self.kernel_size})"


# ---------------------------------------------------------------------------
# InvertGrayscale
# ---------------------------------------------------------------------------

class InvertGrayscale:
    """
    Bitwise inversion of a grayscale image.

    Flips bone (bright) to dark and background (dark) to bright.
    Useful for generative models where inverted contrast may aid learning.
    """

    def __call__(self, img: Image.Image) -> Image.Image:
        return Image.fromarray(cv2.bitwise_not(np.array(img)))

    def __repr__(self) -> str:
        return "InvertGrayscale()"

