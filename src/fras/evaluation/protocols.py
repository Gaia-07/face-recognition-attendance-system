"""Biometric evaluation protocols: Closed-set and Open-set verification & identification."""

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score

from fras.evaluation.metrics import (
    bootstrap_metric_ci,
    compute_roc_det_curves,
)


def evaluate_closed_set(
    onnx_session: Any,
    train_items: list[dict[str, str]],
    test_items: list[dict[str, str]],
    base_dir: Path,
) -> dict[str, Any]:
    """Closed-set gallery retrieval evaluation (Nearest Neighbor matching).

    Enrolls train_items as gallery, evaluates test_items as probes.
    Calculates Top-1, Top-5 accuracy, macro-F1, and confusion matrix.
    """

    def extract_emb(path: Path) -> np.ndarray:
        with Image.open(path) as img:
            arr = np.array(img.convert("RGB")).transpose(2, 0, 1).astype(np.float32) / 255.0
            arr = (arr - 0.5) / 0.5
            arr = np.expand_dims(arr, axis=0)
            raw_emb = onnx_session.run(None, {"input": arr})[0][0]
            emb: np.ndarray = np.asarray(raw_emb, dtype=np.float32)
            norm = float(np.linalg.norm(emb))
            return emb / (norm + 1e-7)

    # Enroll gallery
    g_embs = []
    g_labels = []
    for item in train_items:
        p = base_dir / item["relative_path"]
        g_embs.append(extract_emb(p))
        g_labels.append(item["person_id"])
    g_embs_mat = np.stack(g_embs)
    g_labels_arr = np.array(g_labels)

    # Probe queries
    y_true = []
    y_pred_top1 = []
    top5_correct = 0

    for item in test_items:
        p = base_dir / item["relative_path"]
        p_emb = extract_emb(p)
        sims = np.dot(g_embs_mat, p_emb)
        ranked_indices = np.argsort(-sims)

        gt = item["person_id"]
        y_true.append(gt)

        # Top-1
        top1_pred = g_labels_arr[ranked_indices[0]]
        y_pred_top1.append(top1_pred)

        # Top-5
        top5_preds = set(g_labels_arr[ranked_indices[:5]])
        if gt in top5_preds:
            top5_correct += 1

    unique_labels = sorted(set(y_true))
    top1_acc = float(np.mean(np.array(y_true) == np.array(y_pred_top1)))
    top5_acc = float(top5_correct / len(test_items))
    macro_f1 = float(f1_score(y_true, y_pred_top1, average="macro"))
    cm = confusion_matrix(y_true, y_pred_top1, labels=unique_labels).tolist()

    return {
        "num_classes": len(unique_labels),
        "num_probes": len(test_items),
        "top1_accuracy": top1_acc,
        "top5_accuracy": top5_acc,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "classes": unique_labels,
    }


def evaluate_open_set_verification(
    onnx_session: Any,
    gallery_items: list[dict[str, str]],
    probe_items: list[dict[str, str]],
    base_dir: Path,
    frozen_threshold: float = 0.65,
    n_bootstrap: int = 500,
) -> dict[str, Any]:
    """Open-set verification evaluation on unseen identities with 95% bootstrap CIs.

    Evaluates:
      - ROC, DET curves
      - ROC-AUC with 95% CI
      - EER with 95% CI
      - TAR@FAR=1% and TAR@FAR=0.1% with 95% CIs
      - Operational FAR, FRR, and TAR at frozen_threshold
      - Open-Set Identification: Detection & Identification Rate (DIR) vs FAR
    """

    def extract_emb(path: Path) -> np.ndarray:
        with Image.open(path) as img:
            arr = np.array(img.convert("RGB")).transpose(2, 0, 1).astype(np.float32) / 255.0
            arr = (arr - 0.5) / 0.5
            arr = np.expand_dims(arr, axis=0)
            raw_emb = onnx_session.run(None, {"input": arr})[0][0]
            emb: np.ndarray = np.asarray(raw_emb, dtype=np.float32)
            norm = float(np.linalg.norm(emb))
            return emb / (norm + 1e-7)

    # 1. Extract gallery embeddings and average by identity (enrolled templates)
    gallery_by_id: dict[str, list[np.ndarray]] = {}
    for item in gallery_items:
        p = base_dir / item["relative_path"]
        emb = extract_emb(p)
        gallery_by_id.setdefault(item["person_id"], []).append(emb)

    enrolled_templates: dict[str, np.ndarray] = {
        pid: np.mean(embs, axis=0) / np.linalg.norm(np.mean(embs, axis=0))
        for pid, embs in gallery_by_id.items()
    }
    template_matrix = np.stack(list(enrolled_templates.values()))
    template_ids = list(enrolled_templates.keys())

    # 2. Extract probe embeddings
    probe_embs: list[np.ndarray] = []
    probe_ids: list[str] = []
    for item in probe_items:
        p = base_dir / item["relative_path"]
        probe_embs.append(extract_emb(p))
        probe_ids.append(item["person_id"])

    # 3. Pairwise Genuine and Impostor verification scores
    genuine_scores: list[float] = []
    impostor_scores: list[float] = []

    for p_idx, p_pid in enumerate(probe_ids):
        p_emb = probe_embs[p_idx]
        for t_idx, t_pid in enumerate(template_ids):
            sim = float(np.dot(p_emb, template_matrix[t_idx]))
            if p_pid == t_pid:
                genuine_scores.append(sim)
            else:
                impostor_scores.append(sim)

    gen_arr = np.array(genuine_scores)
    imp_arr = np.array(impostor_scores)

    # Bootstrap 95% Confidence Intervals
    auc_val, auc_low, auc_high = bootstrap_metric_ci(
        gen_arr, imp_arr, "roc_auc", n_bootstrap=n_bootstrap
    )
    eer_val, eer_low, eer_high = bootstrap_metric_ci(
        gen_arr, imp_arr, "eer", n_bootstrap=n_bootstrap
    )
    tar1_val, tar1_low, tar1_high = bootstrap_metric_ci(
        gen_arr, imp_arr, "tar_at_far_1e2", n_bootstrap=n_bootstrap
    )
    tar2_val, tar2_low, tar2_high = bootstrap_metric_ci(
        gen_arr, imp_arr, "tar_at_far_1e3", n_bootstrap=n_bootstrap
    )

    # Evaluate at frozen threshold
    op_far = float(np.mean(imp_arr >= frozen_threshold))
    op_frr = float(np.mean(gen_arr < frozen_threshold))
    op_tar = float(1.0 - op_frr)

    # Open-Set Identification: DIR vs FAR at rank 1
    # For probes of enrolled identities: probe correctly identified at rank 1 AND similarity >= threshold
    correct_identifications = 0
    total_enrolled_probes = 0
    for p_idx, p_pid in enumerate(probe_ids):
        if p_pid in enrolled_templates:
            total_enrolled_probes += 1
            p_emb = probe_embs[p_idx]
            sims = np.dot(template_matrix, p_emb)
            best_idx = int(np.argmax(sims))
            if template_ids[best_idx] == p_pid and sims[best_idx] >= frozen_threshold:
                correct_identifications += 1

    rank1_dir = (
        float(correct_identifications / total_enrolled_probes) if total_enrolled_probes > 0 else 0.0
    )

    fpr_curve, tpr_curve, _ = compute_roc_det_curves(gen_arr, imp_arr)

    return {
        "protocol": "open_set_verification_and_identification",
        "num_enrolled_templates": len(template_ids),
        "num_probes": len(probe_ids),
        "num_genuine_pairs": len(genuine_scores),
        "num_impostor_pairs": len(impostor_scores),
        "frozen_threshold": frozen_threshold,
        "operational_metrics_at_frozen_threshold": {
            "far": op_far,
            "frr": op_frr,
            "tar": op_tar,
            "rank1_dir": rank1_dir,
        },
        "metrics_with_95_bootstrap_ci": {
            "roc_auc": {"value": auc_val, "ci_lower": auc_low, "ci_upper": auc_high},
            "eer": {"value": eer_val, "ci_lower": eer_low, "ci_upper": eer_high},
            "tar_at_far_1e2": {"value": tar1_val, "ci_lower": tar1_low, "ci_upper": tar1_high},
            "tar_at_far_1e3": {"value": tar2_val, "ci_lower": tar2_low, "ci_upper": tar2_high},
        },
        "scores": {
            "genuine": genuine_scores,
            "impostor": impostor_scores,
        },
        "roc_curve": {
            "fpr": fpr_curve[:: max(1, len(fpr_curve) // 100)].tolist(),
            "tpr": tpr_curve[:: max(1, len(tpr_curve) // 100)].tolist(),
        },
    }
