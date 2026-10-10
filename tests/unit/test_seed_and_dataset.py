"""Unit tests for dataset loader functions and database seeding procedures."""

from pathlib import Path

import numpy as np
import pytest

from fras.db.seed import compute_student_template, seed_users
from fras.db.session import SessionLocal
from fras.recognition.avatar_cnn import AvatarCNNBackend
from fras.training.dataset import FaceImageDataset, validate_dataset

ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = ROOT / "data" / "raw" / "face_attendance_dataset"


def test_dataset_validation_and_torch_loader() -> None:
    if not RAW_DIR.exists():
        pytest.skip("Raw dataset not found")

    stats = validate_dataset(RAW_DIR)
    assert stats.total_persons == 30
    assert stats.total_images == 450
    assert stats.corrupted_images == 0

    items = [{"person_id": "person_001", "relative_path": "person_001/person_001_01.jpg"}]
    id_map = {"person_001": 0}
    ds = FaceImageDataset(base_dir=RAW_DIR, items=items, identity_to_idx=id_map)
    assert len(ds) == 1
    tensor, label, pid = ds[0]
    assert tensor.shape == (3, 128, 128)
    assert label == 0
    assert pid == "person_001"


def test_seed_operations() -> None:
    # 1. Seed users test
    seed_users()
    with SessionLocal() as db:
        from fras.db.models import User

        users = db.query(User).all()
        assert len(users) >= 3

    # 2. Template computation
    backend = AvatarCNNBackend()
    template = compute_student_template("person_001", backend)
    assert template.shape == (128,)
    norm = np.linalg.norm(template)
    assert np.isclose(norm, 1.0, atol=1e-5)


def test_generate_splits_and_seed_flow(tmp_path: Path) -> None:
    from fras.db.seed import seed_attendance, seed_students
    from fras.training.dataset import generate_splits

    if not RAW_DIR.exists():
        pytest.skip("Raw dataset not found")

    # 1. Generate splits
    closed, open_s = generate_splits(RAW_DIR, tmp_path, seed=123)
    assert "train" in closed["splits"]
    assert "train" in open_s["splits"]
    assert (tmp_path / "closed_set.json").exists()
    assert (tmp_path / "open_set.json").exists()

    # 2. Seed students and attendance on live DB (idempotent)
    student_map = seed_students()
    assert len(student_map) == 30
    seed_attendance(student_map)
