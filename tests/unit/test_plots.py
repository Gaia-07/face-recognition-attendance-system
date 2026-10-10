"""Unit tests for evaluation visualization and chart generation routines."""

from pathlib import Path

import numpy as np

from fras.evaluation.plots import (
    plot_confusion_matrix,
    plot_intra_inter_distributions,
    plot_roc_det_curves,
    plot_stress_curves,
    plot_tsne_embeddings,
)


def test_plot_roc_det_curves(tmp_path: Path) -> None:
    fpr = [0.0001, 0.001, 0.01, 0.1, 1.0]
    tpr = [0.8, 0.85, 0.95, 0.99, 1.0]
    save_file = tmp_path / "roc.png"
    plot_roc_det_curves(fpr, tpr, save_file)
    assert save_file.exists()
    assert save_file.stat().st_size > 0


def test_plot_confusion_matrix(tmp_path: Path) -> None:
    cm = [[10, 0, 1], [0, 9, 2], [1, 1, 10]]
    classes = ["person_001", "person_002", "person_003"]
    save_file = tmp_path / "cm.png"
    plot_confusion_matrix(cm, classes, save_file)
    assert save_file.exists()
    assert save_file.stat().st_size > 0


def test_plot_intra_inter_distributions(tmp_path: Path) -> None:
    genuine = [0.85, 0.89, 0.92, 0.95]
    impostor = [0.25, 0.30, 0.45, 0.50]
    save_file = tmp_path / "dist.png"
    plot_intra_inter_distributions(genuine, impostor, threshold=0.65, save_path=save_file)
    assert save_file.exists()
    assert save_file.stat().st_size > 0


def test_plot_tsne_embeddings(tmp_path: Path) -> None:
    # 30 random embeddings across 3 identities
    embs = np.random.randn(30, 128).astype(np.float32)
    labels = ["p1"] * 10 + ["p2"] * 10 + ["p3"] * 10
    save_file = tmp_path / "tsne.png"
    plot_tsne_embeddings(embs, labels, save_file)
    assert save_file.exists()
    assert save_file.stat().st_size > 0


def test_plot_stress_curves(tmp_path: Path) -> None:
    stress_data = {
        "gaussian_blur": [{"sigma": 1.0, "eer": 0.02}],
        "brightness_shift": [{"shift": 10, "eer": 0.01}],
        "additive_noise": [{"noise_std": 5, "eer": 0.03}],
        "occlusion_patch": [{"patch_size_px": 8, "eer": 0.04}],
        "rotation_angle": [{"angle_deg": 5, "eer": 0.01}],
        "jpeg_compression": [{"quality": 90, "eer": 0.01}],
    }
    save_file = tmp_path / "stress.png"
    plot_stress_curves(stress_data, save_file)
    assert save_file.exists()
    assert save_file.stat().st_size > 0
