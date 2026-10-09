"""Rigorous biometric metric calculation: ROC, DET, EER, TAR@FAR, and Bootstrap CIs."""

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


@dataclass
class BiometricMetrics:
    roc_auc: float
    eer: float
    eer_threshold: float
    tar_at_far_1e2: float  # TAR at FAR = 1%
    tar_at_far_1e3: float  # TAR at FAR = 0.1%
    calibrated_threshold_1pct: float
    calibrated_far: float
    calibrated_frr: float
    calibrated_tar: float


def compute_roc_det_curves(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes False Positive Rates (FAR), True Positive Rates (TAR), and thresholds."""
    y_true = np.concatenate([np.ones_like(genuine_scores), np.zeros_like(impostor_scores)])
    scores = np.concatenate([genuine_scores, impostor_scores])
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    return fpr, tpr, thresholds


def calculate_biometric_metrics(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    target_far: float = 0.01,
) -> dict[str, float]:
    """Calculates full biometric verification metrics from similarity scores."""
    if len(genuine_scores) == 0 or len(impostor_scores) == 0:
        return {
            "roc_auc": 0.0,
            "eer": 1.0,
            "eer_threshold": 0.0,
            "tar_at_far_1e2": 0.0,
            "tar_at_far_1e3": 0.0,
            "calibrated_threshold_1pct": 0.0,
            "calibrated_far": 1.0,
            "calibrated_frr": 1.0,
            "calibrated_tar": 0.0,
        }

    y_true = np.concatenate([np.ones_like(genuine_scores), np.zeros_like(impostor_scores)])
    scores = np.concatenate([genuine_scores, impostor_scores])

    # 1. ROC-AUC
    auc = float(roc_auc_score(y_true, scores))

    # 2. Sweep thresholds to compute FAR, FRR, EER
    # We sort distinct thresholds across score distribution
    thresholds = np.linspace(-1.0, 1.0, 2001)
    far_list = np.mean(impostor_scores[:, None] >= thresholds[None, :], axis=0)
    frr_list = np.mean(genuine_scores[:, None] < thresholds[None, :], axis=0)
    tar_list = 1.0 - frr_list

    # EER: point where abs(FAR - FRR) is minimized
    eer_idx = int(np.argmin(np.abs(far_list - frr_list)))
    eer = float((far_list[eer_idx] + frr_list[eer_idx]) / 2.0)
    eer_th = float(thresholds[eer_idx])

    # TAR @ FAR = 1e-2 (1%)
    idx_1e2 = np.where(far_list <= 0.010001)[0]
    if len(idx_1e2) > 0:
        best_1e2 = idx_1e2[0]  # lowest threshold satisfying FAR <= 0.01
        tar_at_1e2 = float(tar_list[best_1e2])
    else:
        tar_at_1e2 = float(tar_list[-1])

    # TAR @ FAR = 1e-3 (0.1%)
    idx_1e3 = np.where(far_list <= 0.001001)[0]
    if len(idx_1e3) > 0:
        best_1e3 = idx_1e3[0]
        tar_at_1e3 = float(tar_list[best_1e3])
    else:
        tar_at_1e3 = float(tar_list[-1])

    # Calibrate threshold at target_far
    calib_indices = np.where(far_list <= target_far)[0]
    if len(calib_indices) > 0:
        calib_idx = calib_indices[0]
        calib_th = float(thresholds[calib_idx])
        calib_far = float(far_list[calib_idx])
        calib_frr = float(frr_list[calib_idx])
        calib_tar = float(tar_list[calib_idx])
    else:
        calib_th = float(thresholds[-1])
        calib_far = 0.0
        calib_frr = 1.0
        calib_tar = 0.0

    metrics = BiometricMetrics(
        roc_auc=auc,
        eer=eer,
        eer_threshold=eer_th,
        tar_at_far_1e2=tar_at_1e2,
        tar_at_far_1e3=tar_at_1e3,
        calibrated_threshold_1pct=calib_th,
        calibrated_far=calib_far,
        calibrated_frr=calib_frr,
        calibrated_tar=calib_tar,
    )
    return asdict(metrics)


def bootstrap_metric_ci(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    metric_name: str = "eer",
    n_bootstrap: int = 500,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Calculates metric value with 95% bootstrap confidence intervals.

    Returns:
        (mean_value, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(seed)
    n_gen = len(genuine_scores)
    n_imp = len(impostor_scores)

    base_metrics = calculate_biometric_metrics(genuine_scores, impostor_scores)
    point_est = base_metrics.get(metric_name, 0.0)

    bootstrap_values: list[float] = []
    for _ in range(n_bootstrap):
        gen_boot = genuine_scores[rng.integers(0, n_gen, size=n_gen)]
        imp_boot = impostor_scores[rng.integers(0, n_imp, size=n_imp)]
        boot_m = calculate_biometric_metrics(gen_boot, imp_boot)
        bootstrap_values.append(boot_m.get(metric_name, 0.0))

    alpha = 1.0 - confidence_level
    lower_pct = 100 * (alpha / 2.0)
    upper_pct = 100 * (1.0 - alpha / 2.0)

    ci_lower = float(np.percentile(bootstrap_values, lower_pct))
    ci_upper = float(np.percentile(bootstrap_values, upper_pct))
    return point_est, ci_lower, ci_upper
