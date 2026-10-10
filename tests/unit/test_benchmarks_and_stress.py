"""Unit tests for latency benchmarks, perturbation stress tests, liveness challenges, and detectors."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

from fras.core.detection import FaceDetector
from fras.evaluation.latency import benchmark_matcher_scaling, benchmark_onnx_latency
from fras.evaluation.stress import (
    apply_additive_noise,
    apply_brightness_shift,
    apply_gaussian_blur,
    apply_jpeg_compression,
    apply_occlusion,
    apply_rotation,
    run_stress_benchmarks,
)
from fras.recognition.liveness import LivenessChallenge, LivenessEngine


def test_latency_benchmarks() -> None:
    onnx_path = Path("models/avatar_cnn.onnx")
    if onnx_path.exists():
        res = benchmark_onnx_latency(onnx_path, n_warmup=2, n_iterations=5)
        assert res["iterations"] == 5
        assert res["p50_ms"] > 0.0
        assert res["throughput_fps"] > 0.0

    scaling = benchmark_matcher_scaling(emb_dim=128, gallery_sizes=[10, 50], n_queries=5)
    assert len(scaling) == 2
    assert scaling[0]["gallery_size"] == 10
    assert scaling[0]["p50_search_ms"] >= 0.0


def test_stress_perturbations_and_suite() -> None:
    dummy_img = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)

    b_img = apply_gaussian_blur(dummy_img, sigma=1.0)
    assert b_img.shape == (128, 128, 3)

    br_img = apply_brightness_shift(dummy_img, shift=20)
    assert br_img.shape == (128, 128, 3)

    n_img = apply_additive_noise(dummy_img, sigma_frac=0.05)
    assert n_img.shape == (128, 128, 3)

    occ_img = apply_occlusion(dummy_img, patch_size=16)
    assert occ_img.shape == (128, 128, 3)

    rot_img = apply_rotation(dummy_img, angle=15.0)
    assert rot_img.shape == (128, 128, 3)

    jpg_img = apply_jpeg_compression(dummy_img, quality=50)
    assert jpg_img.shape == (128, 128, 3)

    onnx_path = Path("models/avatar_cnn.onnx")
    raw_images_dir = Path("data/raw/face_attendance_dataset/images")
    if onnx_path.exists() and raw_images_dir.exists():
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        gallery = [{"person_id": "person_001", "relative_path": "person_001/person_001_01.jpg"}]
        probes = [
            {"person_id": "person_001", "relative_path": "person_001/person_001_02.jpg"},
            {"person_id": "person_002", "relative_path": "person_002/person_002_01.jpg"},
        ]
        stress_res = run_stress_benchmarks(session, gallery, probes, raw_images_dir)
        assert "gaussian_blur" in stress_res
        assert len(stress_res["gaussian_blur"]) > 0


def test_liveness_engine_challenges() -> None:
    engine = LivenessEngine(simulation_mode=False)

    # Simulated landmarks: [0: right eye, 1: left eye, 2: nose, 3: right mouth, 4: left mouth]
    # Nose centered between eyes
    center_landmarks = np.array(
        [
            [40.0, 50.0],
            [80.0, 50.0],
            [60.0, 70.0],
            [45.0, 95.0],
            [75.0, 95.0],
        ]
    )
    res_center = engine.verify_challenge(
        center_landmarks, challenge=LivenessChallenge.BLINK_OR_HOLD
    )
    assert res_center.passed is True

    # Turned right: nose closer to left eye in 2D image or d_right > d_left
    # right eye at x=30, nose at x=75, left eye at x=85 -> d_right=45, d_left=10 -> yaw_ratio=4.5
    right_landmarks = np.array(
        [
            [30.0, 50.0],
            [85.0, 50.0],
            [75.0, 70.0],
            [35.0, 95.0],
            [80.0, 95.0],
        ]
    )
    res_right = engine.verify_challenge(right_landmarks, challenge=LivenessChallenge.TURN_RIGHT)
    assert res_right.passed is True
    assert "verified" in res_right.message

    # Turned left: right eye at x=35, nose at x=45, left eye at x=90 -> d_right=10, d_left=45 -> yaw_ratio=0.22
    left_landmarks = np.array(
        [
            [35.0, 50.0],
            [90.0, 50.0],
            [45.0, 70.0],
            [40.0, 95.0],
            [85.0, 95.0],
        ]
    )
    res_left = engine.verify_challenge(left_landmarks, challenge=LivenessChallenge.TURN_LEFT)
    assert res_left.passed is True


def test_face_detector_avatar_and_missing_sface() -> None:
    det_avatar = FaceDetector(backend_name="avatar_cnn")
    dummy = np.zeros((128, 128, 3), dtype=np.uint8)
    faces = det_avatar.detect(dummy)
    assert len(faces) == 1
    assert faces[0].bbox == (0, 0, 128, 128)
    assert faces[0].score == 1.0

    with pytest.raises(FileNotFoundError):
        _ = FaceDetector(backend_name="sface", model_path="nonexistent_yunet_model.onnx")
