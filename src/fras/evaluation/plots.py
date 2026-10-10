"""Visualization suite generating high-resolution biometric evaluation charts."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

# Headless matplotlib
plt.switch_backend("Agg")


def plot_roc_det_curves(fpr: list[float], tpr: list[float], save_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="#2980b9", lw=2.5, label="ROC Curve")
    plt.plot([0, 1], [0, 1], color="#7f8c8d", lw=1.5, linestyle="--", label="Chance")
    plt.xscale("log")
    plt.xlim(1e-4, 1.0)
    plt.ylim(0.0, 1.02)
    plt.xlabel("False Acceptance Rate (FAR, log scale)", fontsize=11)
    plt.ylabel("True Acceptance Rate (TAR)", fontsize=11)
    plt.title("Open-Set Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrix(cm: list[list[int]], classes: list[str], save_path: Path) -> None:
    cm_arr = np.array(cm)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(
        "Closed-Set Confusion Matrix (30 Enrolled Identities)", fontsize=12, fontweight="bold"
    )
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=8)
    plt.yticks(tick_marks, classes, fontsize=8)
    plt.xlabel("Predicted Identity", fontsize=11)
    plt.ylabel("Ground Truth Identity", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_intra_inter_distributions(
    genuine: list[float], impostor: list[float], threshold: float, save_path: Path
) -> None:
    plt.figure(figsize=(8, 4.8))
    plt.hist(
        impostor,
        bins=40,
        density=True,
        alpha=0.6,
        color="#e74c3c",
        label="Impostor Pairs (Inter-Class)",
    )
    plt.hist(
        genuine,
        bins=25,
        density=True,
        alpha=0.7,
        color="#2ecc71",
        label="Genuine Pairs (Intra-Class)",
    )
    plt.axvline(
        threshold,
        color="#2c3e50",
        linestyle="--",
        lw=2,
        label=f"Calibrated Threshold ({threshold:.3f})",
    )
    plt.xlabel("Cosine Similarity", fontsize=11)
    plt.ylabel("Probability Density", fontsize=11)
    plt.title(
        "Biometric Score Distribution: Intra- vs Inter-Class Separation",
        fontsize=12,
        fontweight="bold",
    )
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_tsne_embeddings(embeddings: np.ndarray, labels: list[str], save_path: Path) -> None:
    tsne = TSNE(n_components=2, perplexity=10.0, random_state=42, init="pca", learning_rate="auto")
    emb_2d = tsne.fit_transform(embeddings)

    unique_labels = sorted(set(labels))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    plt.figure(figsize=(9, 7))
    for idx, lbl in enumerate(unique_labels):
        mask = [i for i, x in enumerate(labels) if x == lbl]
        plt.scatter(
            emb_2d[mask, 0],
            emb_2d[mask, 1],
            color=colors[idx % len(colors)],
            label=lbl if idx < 10 else None,
            alpha=0.85,
            edgecolors="none",
            s=40,
        )

    plt.title("t-SNE Projection of 128-D Avatar Embeddings", fontsize=12, fontweight="bold")
    plt.xlabel("Dimension 1", fontsize=11)
    plt.ylabel("Dimension 2", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", title="Identities (Sample)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_stress_curves(stress_data: dict[str, list[dict[str, Any]]], save_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # 1. Blur
    ax = axes[0, 0]
    pts = stress_data["gaussian_blur"]
    ax.plot([p["sigma"] for p in pts], [p["eer"] * 100 for p in pts], marker="o", color="#e67e22")
    ax.set_title("Gaussian Blur")
    ax.set_xlabel("Sigma")
    ax.set_ylabel("EER (%)")
    ax.grid(True, linestyle="--", alpha=0.5)

    # 2. Brightness
    ax = axes[0, 1]
    pts = stress_data["brightness_shift"]
    ax.plot([p["shift"] for p in pts], [p["eer"] * 100 for p in pts], marker="o", color="#f1c40f")
    ax.set_title("Brightness Shift")
    ax.set_xlabel("Shift (-255 to +255)")
    ax.set_ylabel("EER (%)")
    ax.grid(True, linestyle="--", alpha=0.5)

    # 3. Noise
    ax = axes[0, 2]
    pts = stress_data["additive_noise"]
    ax.plot(
        [p["noise_std"] for p in pts], [p["eer"] * 100 for p in pts], marker="o", color="#9b59b6"
    )
    ax.set_title("Additive Gaussian Noise")
    ax.set_xlabel("Noise Std")
    ax.set_ylabel("EER (%)")
    ax.grid(True, linestyle="--", alpha=0.5)

    # 4. Occlusion
    ax = axes[1, 0]
    pts = stress_data["occlusion_patch"]
    ax.plot(
        [p["patch_size_px"] for p in pts],
        [p["eer"] * 100 for p in pts],
        marker="o",
        color="#e74c3c",
    )
    ax.set_title("Synthetic Occlusion Patch")
    ax.set_xlabel("Patch Width (px)")
    ax.set_ylabel("EER (%)")
    ax.grid(True, linestyle="--", alpha=0.5)

    # 5. Rotation
    ax = axes[1, 1]
    pts = stress_data["rotation_angle"]
    ax.plot(
        [p["angle_deg"] for p in pts], [p["eer"] * 100 for p in pts], marker="o", color="#3498db"
    )
    ax.set_title("In-Plane Rotation")
    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel("EER (%)")
    ax.grid(True, linestyle="--", alpha=0.5)

    # 6. JPEG compression
    ax = axes[1, 2]
    pts = stress_data["jpeg_compression"]
    ax.plot([p["quality"] for p in pts], [p["eer"] * 100 for p in pts], marker="o", color="#1abc9c")
    ax.set_title("JPEG Compression")
    ax.set_xlabel("Quality (10-100)")
    ax.set_ylabel("EER (%)")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle(
        "Robustness Stress Response: Open-Set EER across Synthetic Perturbations",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
