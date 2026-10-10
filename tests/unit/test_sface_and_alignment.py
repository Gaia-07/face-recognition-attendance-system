"""Unit tests for geometric face alignment and SFace backend error handling."""

from pathlib import Path

import numpy as np
import pytest

from fras.core.alignment import align_face_5points, crop_and_normalize_face
from fras.recognition.sface import SFaceBackend


def test_align_face_5points_normal() -> None:
    img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    landmarks = np.array(
        [
            [40.0, 50.0],
            [75.0, 50.0],
            [57.0, 70.0],
            [42.0, 90.0],
            [72.0, 90.0],
        ],
        dtype=np.float32,
    )
    aligned = align_face_5points(img, landmarks, target_size=(112, 112))
    assert aligned.shape == (112, 112, 3)


def test_align_face_5points_degenerated_fallback() -> None:
    img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    # Collinear / identical points
    landmarks = np.zeros((5, 2), dtype=np.float32)
    aligned = align_face_5points(img, landmarks, target_size=(112, 112))
    assert aligned.shape == (112, 112, 3)


def test_crop_and_normalize_face() -> None:
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    # Bbox partially out of bounds
    bbox = (-10, -10, 100, 100)
    cropped = crop_and_normalize_face(img, bbox, target_size=(128, 128))
    assert cropped.shape == (128, 128, 3)


def test_sface_backend_missing_model_raises_clean_error(tmp_path: Path) -> None:
    non_existent = tmp_path / "missing_sface.onnx"
    with pytest.raises(FileNotFoundError) as exc_info:
        SFaceBackend(model_path=non_existent)
    assert "SFace model ONNX not found" in str(exc_info.value)
