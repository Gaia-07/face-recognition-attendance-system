"""Biometric augmentation pipeline using Albumentations."""

from typing import Any

import albumentations as A


def build_train_transform(image_size: int = 128) -> Any:
    """Builds stochastic augmentation pipeline simulating unconstrained biometric acquisition.

    Includes geometric transforms, illumination variations, sensor blur, additive noise,
    compression artifacts, and partial synthetic occlusions.
    """
    return A.Compose(
        [
            A.Affine(
                scale=(0.92, 1.08),
                translate_percent=(-0.06, 0.06),
                rotate=(-15, 15),
                p=0.5,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.5,
            ),
            A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.2, 1.5), p=0.3),
            A.GaussNoise(std_range=(0.02, 0.08), p=0.3),
            A.ImageCompression(quality_range=(70, 100), p=0.3),
            A.CoarseDropout(
                num_holes_range=(1, 3),
                hole_height_range=(0.08, 0.16),
                hole_width_range=(0.08, 0.16),
                p=0.25,
            ),
        ]
    )


def build_val_transform(image_size: int = 128) -> Any:
    """Deterministic validation transform (identity/resizing without distortion)."""
    return A.Compose([])
