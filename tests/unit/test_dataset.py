"""Unit tests asserting dataset integrity, schema conformity, and zero split leakage."""

import json
from pathlib import Path

import pytest

from fras.training.dataset import validate_dataset


@pytest.fixture
def dataset_paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parent.parent.parent
    raw_dir = root / "data" / "raw" / "face_attendance_dataset"
    splits_dir = root / "data" / "splits"
    return raw_dir, splits_dir


def test_dataset_validation_stats(dataset_paths: tuple[Path, Path]) -> None:
    raw_dir, _ = dataset_paths
    stats = validate_dataset(raw_dir)

    assert stats.total_persons == 30
    assert stats.total_images == 450
    assert stats.image_shape == (128, 128, 3)
    assert stats.corrupted_images == 0
    assert stats.attendance_records == 1980
    assert len(stats.departments) == 8
    assert stats.status_counts["Present"] == 1587
    assert stats.status_counts["Late"] == 250
    assert stats.status_counts["Absent"] == 143


def test_closed_set_splits_no_leakage(dataset_paths: tuple[Path, Path]) -> None:
    _, splits_dir = dataset_paths
    split_file = splits_dir / "closed_set.json"
    assert split_file.exists(), "closed_set.json must exist"

    with open(split_file, encoding="utf-8") as f:
        data = json.load(f)

    train_imgs = {x["relative_path"] for x in data["splits"]["train"]}
    val_imgs = {x["relative_path"] for x in data["splits"]["val"]}
    test_imgs = {x["relative_path"] for x in data["splits"]["test"]}

    assert len(train_imgs) == 300
    assert len(val_imgs) == 60
    assert len(test_imgs) == 90

    # Image disjointness
    assert len(train_imgs & val_imgs) == 0, "Closed-set: Train and Val overlap!"
    assert len(train_imgs & test_imgs) == 0, "Closed-set: Train and Test overlap!"
    assert len(val_imgs & test_imgs) == 0, "Closed-set: Val and Test overlap!"


def test_open_set_splits_identity_disjointness(dataset_paths: tuple[Path, Path]) -> None:
    _, splits_dir = dataset_paths
    split_file = splits_dir / "open_set.json"
    assert split_file.exists(), "open_set.json must exist"

    with open(split_file, encoding="utf-8") as f:
        data = json.load(f)

    train_ids = set(data["identities"]["train"])
    val_ids = set(data["identities"]["val"])
    test_ids = set(data["identities"]["test"])

    assert len(train_ids) == 20
    assert len(val_ids) == 5
    assert len(test_ids) == 5

    # Identity disjointness (Open-set requirement)
    assert len(train_ids & val_ids) == 0, "Open-set: Train and Val identities overlap!"
    assert len(train_ids & test_ids) == 0, "Open-set: Train and Test identities overlap!"
    assert len(val_ids & test_ids) == 0, "Open-set: Val and Test identities overlap!"

    # Gallery and probe partition checks
    val_gallery = {x["relative_path"] for x in data["splits"]["val_gallery"]}
    val_probes = {x["relative_path"] for x in data["splits"]["val_probes"]}
    assert len(val_gallery & val_probes) == 0, "Val gallery and probe overlap!"

    test_gallery = {x["relative_path"] for x in data["splits"]["test_gallery"]}
    test_probes = {x["relative_path"] for x in data["splits"]["test_probes"]}
    assert len(test_gallery & test_probes) == 0, "Test gallery and probe overlap!"
