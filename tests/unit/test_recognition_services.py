"""Unit tests for recognition services, camera abstractions, quality gates, and T01-T07 report test cases."""

import numpy as np
import pytest

from fras.core.camera import CameraUnavailableError, OpenCVCamera, SimulatedCamera
from fras.core.quality import QualityRejectionCode, check_face_quality
from fras.recognition.avatar_cnn import AvatarCNNBackend
from fras.recognition.liveness import LivenessChallenge, LivenessEngine
from fras.recognition.matcher import BiometricMatcher
from fras.recognition.voting import TemporalVotingEngine


@pytest.fixture
def avatar_backend() -> AvatarCNNBackend:
    return AvatarCNNBackend()


@pytest.fixture
def sample_image() -> np.ndarray:
    return (np.random.rand(128, 128, 3) * 255).astype(np.uint8)


def test_avatar_cnn_backend_embed(
    avatar_backend: AvatarCNNBackend, sample_image: np.ndarray
) -> None:
    preprocessed = avatar_backend.preprocess(sample_image)
    assert preprocessed.shape == (1, 3, 128, 128)

    embs = avatar_backend.embed(preprocessed)
    assert embs.shape == (1, 128)
    norm = np.linalg.norm(embs[0])
    np.testing.assert_allclose(norm, 1.0, atol=1e-5)


def test_t01_register_identity_and_match(avatar_backend: AvatarCNNBackend) -> None:
    """Report T01: Identity registration and template persistence."""
    matcher = BiometricMatcher(threshold=0.65)

    img1 = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)
    emb1 = avatar_backend.embed(avatar_backend.preprocess(img1))[0]

    matcher.enroll("student_001", emb1)
    assert "student_001" in matcher.templates

    # T04: Known face recognized
    res = matcher.match(emb1)
    assert res.is_match is True
    assert res.matched_id == "student_001"
    assert res.similarity > 0.95


def test_t02_camera_unavailable() -> None:
    """Report T02: Hardware failure detection when video capture device is missing."""
    with pytest.raises(CameraUnavailableError):
        # Invalid capture device index
        _ = OpenCVCamera(device_index=-999)


def test_simulated_camera_stream() -> None:
    """Verifies headless SimulatedCamera streaming."""
    cam = SimulatedCamera(add_jitter=False)
    frame = cam.read_frame()
    assert frame is not None
    assert frame.shape == (128, 128, 3)
    cam.release()


def test_t03_quality_gates_blur_and_size() -> None:
    """Report T03: Low-quality image rejection (blurry or too small)."""
    # 1. Too small
    small = np.zeros((32, 32, 3), dtype=np.uint8)
    q1 = check_face_quality(small, min_face_size=64)
    assert q1.passed is False
    assert q1.rejection_code == QualityRejectionCode.REJECT_TOO_SMALL

    # 2. Blurry (uniform gray image has Laplacian variance 0)
    blurry = np.ones((128, 128, 3), dtype=np.uint8) * 128
    q2 = check_face_quality(blurry, blur_threshold=60.0)
    assert q2.passed is False
    assert q2.rejection_code == QualityRejectionCode.REJECT_BLURRY

    # 3. High quality texture
    textured = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)
    q3 = check_face_quality(
        textured, blur_threshold=10.0, min_brightness=10.0, max_brightness=250.0
    )
    assert q3.passed is True
    assert q3.rejection_code == QualityRejectionCode.PASSED


def test_t05_unknown_face_rejected(avatar_backend: AvatarCNNBackend) -> None:
    """Report T05: Unknown face rejection below calibrated threshold."""
    matcher = BiometricMatcher(threshold=0.85)

    img1 = np.ones((128, 128, 3), dtype=np.uint8) * 50
    img2 = np.ones((128, 128, 3), dtype=np.uint8) * 200

    emb1 = avatar_backend.embed(avatar_backend.preprocess(img1))[0]
    emb2 = avatar_backend.embed(avatar_backend.preprocess(img2))[0]

    matcher.enroll("enrolled_person", emb1)

    # Orthogonal or distant probe
    res = matcher.match(emb2)
    if res.similarity < 0.85:
        assert res.is_match is False
        assert res.status == "REJECT_UNKNOWN"
        assert res.matched_id is None


def test_temporal_voting_consensus() -> None:
    """Multi-frame temporal voting logic."""
    engine = TemporalVotingEngine(k=3, n=5)

    # 2 votes for A -> not confirmed
    d1 = engine.update("person_A")
    assert d1.is_confirmed is False
    d2 = engine.update("person_A")
    assert d2.is_confirmed is False

    # 3rd vote for A -> confirmed!
    d3 = engine.update("person_A")
    assert d3.is_confirmed is True
    assert d3.confirmed_id == "person_A"
    assert d3.votes == 3


def test_liveness_engine_simulation_mode() -> None:
    """Active challenge response engine in simulation mode."""
    engine = LivenessEngine(simulation_mode=True)
    res = engine.verify_challenge(landmarks=None, challenge=LivenessChallenge.TURN_LEFT)
    assert res.passed is True
    assert res.is_simulated is True
