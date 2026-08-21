from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.nn import functional as F


def _to_uint8_grayscale(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("L"))
    else:
        array = np.asarray(image)
        if array.ndim == 3:
            array = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    if array.dtype == np.uint8:
        return array
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError("Image contains no finite pixel values.")
    low, high = np.percentile(finite, [0.5, 99.5])
    if high <= low:
        return np.zeros(array.shape, dtype=np.uint8)
    scaled = np.clip((array.astype(np.float32) - low) / (high - low), 0, 1)
    return np.round(255 * scaled).astype(np.uint8)


class DRScorePreprocessor:
    """CLAHE, per-image z-score normalisation, and 630-pixel resizing."""

    def __init__(
        self,
        image_size: int = 630,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid: tuple[int, int] = (8, 8),
    ) -> None:
        self.image_size = int(image_size)
        self.clahe = cv2.createCLAHE(
            clipLimit=float(clahe_clip_limit),
            tileGridSize=tuple(map(int, clahe_tile_grid)),
        )

    def __call__(
        self,
        image: Image.Image | np.ndarray,
        *,
        side: str | None = None,
        reflect_right_knee: bool = True,
    ) -> torch.Tensor:
        gray = _to_uint8_grayscale(image)
        if side is not None:
            side = side.strip().lower()
            if side not in {"left", "right"}:
                raise ValueError("side must be 'left', 'right', or None.")
            if reflect_right_knee and side == "right":
                gray = np.ascontiguousarray(np.fliplr(gray))

        equalized = self.clahe.apply(gray)
        tensor = torch.from_numpy(equalized).to(torch.float32).unsqueeze(0) / 255.0
        mean = tensor.mean()
        std = tensor.std(unbiased=False)
        if not torch.isfinite(std) or std <= 0:
            raise ValueError("Image has zero or invalid intensity variance.")
        tensor = (tensor - mean) / std
        return F.interpolate(
            tensor.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).squeeze(0)


def load_image(path: str | Path) -> Image.Image:
    with Image.open(Path(path).expanduser()) as image:
        return ImageOps.exif_transpose(image).copy()


def rescale_dr_score(
    raw_score: float | np.ndarray,
    raw_min: float,
    raw_max: float,
    *,
    clip: bool = False,
) -> float | np.ndarray:
    """Apply the prespecified linear transformation to the 0-4 DR Score scale."""
    if raw_max <= raw_min:
        raise ValueError("raw_max must be greater than raw_min.")
    score = 4.0 * (np.asarray(raw_score) - raw_min) / (raw_max - raw_min)
    if clip:
        score = np.clip(score, 0.0, 4.0)
    return float(score) if score.ndim == 0 else score
