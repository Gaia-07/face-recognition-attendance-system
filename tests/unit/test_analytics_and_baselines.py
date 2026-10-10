"""Unit tests for ML absence prediction, baseline extractors, and seed operations."""

from pathlib import Path

import pytest

from fras.attendance.analytics import train_and_predict_absence_risk
from fras.db import repository
from fras.db.session import SessionLocal
from fras.training.baselines import (
    LBPHBaseline,
    PixelHOGBaseline,
    evaluate_feature_extractor_on_pairs,
)

ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = ROOT / "data" / "raw" / "face_attendance_dataset"


def test_predict_absence_risk_on_historical_records() -> None:
    with SessionLocal() as db:
        records = repository.list_attendance_records(db)

    assert len(records) >= 100
    predictions = train_and_predict_absence_risk(records)

    assert len(predictions) > 0
    p = predictions[0]
    assert "student_id" in p
    assert "risk_score" in p
    assert "risk_tier" in p
    assert p["risk_tier"] in ("High", "Medium", "Low")
    assert 0.0 <= p["risk_score"] <= 1.0


def test_pixel_hog_baseline() -> None:
    img1 = RAW_DIR / "images" / "person_001" / "person_001_01.jpg"
    img2 = RAW_DIR / "images" / "person_002" / "person_002_01.jpg"
    probe = RAW_DIR / "images" / "person_001" / "person_001_02.jpg"

    if not img1.exists() or not img2.exists():
        pytest.skip("Raw images not available")

    baseline = PixelHOGBaseline(target_size=(32, 32))
    baseline.fit([img1, img2], ["person_001", "person_002"])

    pred_label, sim = baseline.predict(probe)
    assert pred_label == "person_001"
    assert -1.0 <= sim <= 1.0


def test_lbph_baseline() -> None:
    img1 = RAW_DIR / "images" / "person_001" / "person_001_01.jpg"
    img2 = RAW_DIR / "images" / "person_002" / "person_002_01.jpg"
    probe = RAW_DIR / "images" / "person_001" / "person_001_02.jpg"

    if not img1.exists() or not img2.exists():
        pytest.skip("Raw images not available")

    baseline = LBPHBaseline(grid_x=4, grid_y=4)
    baseline.fit([img1, img2], ["person_001", "person_002"])

    pred_label, sim = baseline.predict(probe)
    assert pred_label == "person_001"
    assert -1.0 <= sim <= 1.0


def test_evaluate_feature_extractor_on_pairs() -> None:
    gallery_items = [
        {"person_id": "person_001", "relative_path": "images/person_001/person_001_01.jpg"},
        {"person_id": "person_002", "relative_path": "images/person_002/person_002_01.jpg"},
    ]
    probe_items = [
        {"person_id": "person_001", "relative_path": "images/person_001/person_001_02.jpg"},
        {"person_id": "person_002", "relative_path": "images/person_002/person_002_02.jpg"},
    ]

    baseline = PixelHOGBaseline()
    metrics = evaluate_feature_extractor_on_pairs(
        extract_fn=baseline.extract_features,
        gallery_items=gallery_items,
        probe_items=probe_items,
        base_dir=RAW_DIR,
    )
    assert "roc_auc" in metrics
    assert "eer" in metrics
