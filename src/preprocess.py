from __future__ import annotations

"""Image preprocessing utilities shared by feature/DL pipelines."""

from pathlib import Path

import cv2
import numpy as np


def read_grayscale(path: str | Path) -> np.ndarray:
    """Load image as grayscale array; raise explicit error on failure."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def resize_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize image to `(width, height)` using area interpolation."""
    width, height = size
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def normalize_to_unit(image: np.ndarray) -> np.ndarray:
    """Scale uint8 image to float32 [0, 1]."""
    return image.astype(np.float32) / 255.0


def preprocess_for_features(
    path: str | Path,
    size: tuple[int, int] = (224, 224),
    normalize: bool = False,
) -> np.ndarray:
    """Standard preprocessing used by handcrafted feature extractors."""
    image = read_grayscale(path)
    image = resize_image(image, size=size)
    if normalize:
        return normalize_to_unit(image)
    return image
