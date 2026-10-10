"""Biometric image quality assessment gates and diagnostic reason codes."""

from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np


class QualityRejectionCode(StrEnum):
    PASSED = "PASSED"
    REJECT_BLURRY = "REJECT_BLURRY"
    REJECT_TOO_SMALL = "REJECT_TOO_SMALL"
    REJECT_TOO_DARK = "REJECT_TOO_DARK"
    REJECT_TOO_BRIGHT = "REJECT_TOO_BRIGHT"
    REJECT_NO_FACE = "REJECT_NO_FACE"
    REJECT_MULTIPLE_FACES = "REJECT_MULTIPLE_FACES"


@dataclass
class QualityReport:
    passed: bool
    rejection_code: QualityRejectionCode
    blur_score: float
    brightness_score: float
    face_width: int
    face_height: int
    diagnostic_message: str


def compute_laplacian_blur_variance(image_bgr_or_rgb: np.ndarray) -> float:
    """Computes variance of the Laplacian as a high-frequency blur proxy."""
    if image_bgr_or_rgb.ndim == 3:
        gray = cv2.cvtColor(image_bgr_or_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_bgr_or_rgb
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def compute_luminance(image_bgr_or_rgb: np.ndarray) -> float:
    """Computes mean pixel luminance."""
    if image_bgr_or_rgb.ndim == 3:
        gray = cv2.cvtColor(image_bgr_or_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_bgr_or_rgb
    return float(np.mean(gray))


def check_face_quality(
    face_image: np.ndarray,
    blur_threshold: float = 60.0,
    min_face_size: int = 64,
    min_brightness: float = 40.0,
    max_brightness: float = 220.0,
) -> QualityReport:
    """Evaluates face image against blur, size, and illumination quality gates."""
    h, w = face_image.shape[:2]

    # 1. Face size gate
    if w < min_face_size or h < min_face_size:
        return QualityReport(
            passed=False,
            rejection_code=QualityRejectionCode.REJECT_TOO_SMALL,
            blur_score=0.0,
            brightness_score=0.0,
            face_width=w,
            face_height=h,
            diagnostic_message=f"Face bounding box ({w}x{h}) is smaller than minimum required ({min_face_size}x{min_face_size}px).",
        )

    # 2. Blur gate
    blur_var = compute_laplacian_blur_variance(face_image)
    if blur_var < blur_threshold:
        return QualityReport(
            passed=False,
            rejection_code=QualityRejectionCode.REJECT_BLURRY,
            blur_score=blur_var,
            brightness_score=0.0,
            face_width=w,
            face_height=h,
            diagnostic_message=f"Image focus is blurry (Laplacian variance {blur_var:.1f} < {blur_threshold:.1f}).",
        )

    # 3. Illumination gate
    luminance = compute_luminance(face_image)
    if luminance < min_brightness:
        return QualityReport(
            passed=False,
            rejection_code=QualityRejectionCode.REJECT_TOO_DARK,
            blur_score=blur_var,
            brightness_score=luminance,
            face_width=w,
            face_height=h,
            diagnostic_message=f"Illumination is underexposed ({luminance:.1f} < {min_brightness:.1f}).",
        )
    if luminance > max_brightness:
        return QualityReport(
            passed=False,
            rejection_code=QualityRejectionCode.REJECT_TOO_BRIGHT,
            blur_score=blur_var,
            brightness_score=luminance,
            face_width=w,
            face_height=h,
            diagnostic_message=f"Illumination is overexposed / washed out ({luminance:.1f} > {max_brightness:.1f}).",
        )

    return QualityReport(
        passed=True,
        rejection_code=QualityRejectionCode.PASSED,
        blur_score=blur_var,
        brightness_score=luminance,
        face_width=w,
        face_height=h,
        diagnostic_message="Quality criteria satisfied.",
    )
