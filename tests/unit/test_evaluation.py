"""Unit tests for biometric evaluation metrics, bootstrap CIs, and stress perturbation functions."""

import numpy as np

from fras.evaluation.latency import benchmark_matcher_scaling
from fras.evaluation.metrics import (
    bootstrap_metric_ci,
    calculate_biometric_metrics,
    compute_roc_det_curves,
)
from fras.evaluation.stress import (
    apply_additive_noise,
    apply_brightness_shift,
    apply_gaussian_blur,
    apply_jpeg_compression,
    apply_occlusion,
    apply_rotation,
)


def test_calculate_biometric_metrics_synthetic() -> None:
    # Well-separated distributions
    genuine = np.random.normal(0.85, 0.05, 100)
    impostor = np.random.normal(0.20, 0.10, 500)

    metrics = calculate_biometric_metrics(genuine, impostor, target_far=0.01)

    assert metrics["roc_auc"] > 0.95
    assert metrics["eer"] < 0.05
    assert metrics["tar_at_far_1e2"] > 0.90
    assert metrics["calibrated_threshold_1pct"] > 0.30
    assert metrics["calibrated_far"] <= 0.015


def test_bootstrap_metric_ci() -> None:
    genuine = np.random.normal(0.80, 0.05, 50)
    impostor = np.random.normal(0.30, 0.10, 200)

    point_est, ci_low, ci_high = bootstrap_metric_ci(
        genuine, impostor, metric_name="roc_auc", n_bootstrap=50, seed=42
    )

    assert ci_low <= point_est <= ci_high
    assert 0.0 <= ci_low <= 1.0


def test_compute_roc_curves() -> None:
    genuine = np.array([0.7, 0.8, 0.9])
    impostor = np.array([0.1, 0.2, 0.3])

    fpr, tpr, thresholds = compute_roc_det_curves(genuine, impostor)
    assert len(fpr) > 0
    assert len(tpr) > 0
    assert len(thresholds) > 0


def test_stress_perturbations_execute() -> None:
    img = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)

    b = apply_gaussian_blur(img, sigma=1.5)
    assert b.shape == img.shape

    br = apply_brightness_shift(img, shift=20)
    assert br.shape == img.shape

    n = apply_additive_noise(img, sigma_frac=0.05)
    assert n.shape == img.shape

    o = apply_occlusion(img, patch_size=24)
    assert o.shape == img.shape

    r = apply_rotation(img, angle=15.0)
    assert r.shape == img.shape

    j = apply_jpeg_compression(img, quality=50)
    assert j.shape == img.shape


def test_matcher_scaling_benchmark() -> None:
    res = benchmark_matcher_scaling(emb_dim=128, gallery_sizes=[10, 20], n_queries=5)
    assert len(res) == 2
    assert "p50_search_ms" in res[0]
    assert "p95_search_ms" in res[0]
