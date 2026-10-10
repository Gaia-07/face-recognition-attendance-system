"""Unit tests for biometric evaluation protocols and attendance replay."""

from pathlib import Path

import onnxruntime as ort
import pytest

from fras.evaluation.attendance_replay import run_attendance_replay
from fras.evaluation.protocols import evaluate_closed_set, evaluate_open_set_verification

ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = ROOT / "models" / "avatar_cnn.onnx"
RAW_DATA_DIR = ROOT / "data" / "raw" / "face_attendance_dataset"


@pytest.fixture(scope="module")
def onnx_session() -> ort.InferenceSession:
    if not MODEL_PATH.exists():
        pytest.skip("ONNX model models/avatar_cnn.onnx not found")
    return ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])


def test_evaluate_closed_set(onnx_session: ort.InferenceSession) -> None:
    train_items = [
        {"person_id": "person_001", "relative_path": "images/person_001/person_001_01.jpg"},
        {"person_id": "person_001", "relative_path": "images/person_001/person_001_02.jpg"},
        {"person_id": "person_002", "relative_path": "images/person_002/person_002_01.jpg"},
        {"person_id": "person_002", "relative_path": "images/person_002/person_002_02.jpg"},
    ]
    test_items = [
        {"person_id": "person_001", "relative_path": "images/person_001/person_001_03.jpg"},
        {"person_id": "person_002", "relative_path": "images/person_002/person_002_03.jpg"},
    ]

    res = evaluate_closed_set(
        onnx_session=onnx_session,
        train_items=train_items,
        test_items=test_items,
        base_dir=RAW_DATA_DIR,
    )

    assert res["num_classes"] == 2
    assert res["num_probes"] == 2
    assert "top1_accuracy" in res
    assert "macro_f1" in res


def test_evaluate_open_set_verification(onnx_session: ort.InferenceSession) -> None:
    gallery_items = [
        {"person_id": "person_001", "relative_path": "images/person_001/person_001_01.jpg"},
        {"person_id": "person_002", "relative_path": "images/person_002/person_002_01.jpg"},
    ]
    probe_items = [
        {"person_id": "person_001", "relative_path": "images/person_001/person_001_02.jpg"},
        {"person_id": "person_003", "relative_path": "images/person_003/person_003_01.jpg"},
    ]

    res = evaluate_open_set_verification(
        onnx_session=onnx_session,
        gallery_items=gallery_items,
        probe_items=probe_items,
        base_dir=RAW_DATA_DIR,
        frozen_threshold=0.65,
        n_bootstrap=50,
    )

    assert "metrics_with_95_bootstrap_ci" in res
    assert "roc_auc" in res["metrics_with_95_bootstrap_ci"]
    assert "operational_metrics_at_frozen_threshold" in res
    assert "far" in res["operational_metrics_at_frozen_threshold"]


def test_attendance_replay_pipeline(onnx_session: ort.InferenceSession) -> None:
    splits_file = ROOT / "data" / "splits" / "open_set.json"
    if not splits_file.exists():
        pytest.skip("Splits file not found")
    import json

    with open(splits_file, encoding="utf-8") as f:
        splits_data = json.load(f)

    result = run_attendance_replay(
        onnx_session=onnx_session,
        raw_dir=RAW_DATA_DIR,
        splits_data=splits_data,
        frozen_threshold=0.65,
    )
    assert result["total_replayed_events"] > 0
    assert result["identification_accuracy"] > 0.8
    assert result["duplicate_prevention_integrity"] == 1.0
