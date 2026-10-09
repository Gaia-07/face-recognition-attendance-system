"""Baseline recognition algorithms and comparative feature extractors."""

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors


class PixelHOGBaseline:
    """Baseline 1: Flattened, downsampled grayscale pixels + Cosine kNN."""

    def __init__(self, target_size: tuple[int, int] = (32, 32)) -> None:
        self.target_size = target_size
        self.gallery_features: np.ndarray | None = None
        self.gallery_labels: list[str] = []
        self.knn = NearestNeighbors(n_neighbors=1, metric="cosine")

    def extract_features(self, image_path: Path) -> np.ndarray:
        with Image.open(image_path) as img:
            gray = img.convert("L").resize(self.target_size)
            arr = np.array(gray, dtype=np.float32) / 255.0
            feat = arr.flatten()
            norm = np.linalg.norm(feat)
            return feat / (norm + 1e-7)

    def fit(self, image_paths: list[Path], labels: list[str]) -> None:
        features = [self.extract_features(p) for p in image_paths]
        self.gallery_features = np.stack(features)
        self.gallery_labels = labels
        self.knn.fit(self.gallery_features)

    def predict(self, probe_path: Path) -> tuple[str, float]:
        feat = self.extract_features(probe_path).reshape(1, -1)
        distances, indices = self.knn.kneighbors(feat)
        cosine_dist = float(distances[0][0])
        sim = 1.0 - cosine_dist
        pred_label = self.gallery_labels[indices[0][0]]
        return pred_label, sim


class LBPHBaseline:
    """Baseline 2: Local Binary Pattern Histograms (pure NumPy implementation)."""

    def __init__(self, grid_x: int = 8, grid_y: int = 8) -> None:
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.gallery_features: np.ndarray | None = None
        self.gallery_labels: list[str] = []
        self.knn = NearestNeighbors(n_neighbors=1, metric="cosine")

    def _compute_lbp(self, gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape
        lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)
        center = gray[1:-1, 1:-1]

        # 8 neighbors clockwise
        neighbors = [
            gray[0:-2, 0:-2],  # top-left
            gray[0:-2, 1:-1],  # top
            gray[0:-2, 2:],  # top-right
            gray[1:-1, 2:],  # right
            gray[2:, 2:],  # bottom-right
            gray[2:, 1:-1],  # bottom
            gray[2:, 0:-2],  # bottom-left
            gray[1:-1, 0:-2],  # left
        ]

        for i, neighbor in enumerate(neighbors):
            lbp |= ((neighbor >= center).astype(np.uint8)) << i
        return lbp

    def extract_features(self, image_path: Path) -> np.ndarray:
        with Image.open(image_path) as img:
            gray = np.array(img.convert("L").resize((64, 64)), dtype=np.float32)

        lbp = self._compute_lbp(gray)
        h, w = lbp.shape
        cell_h, cell_w = h // self.grid_y, w // self.grid_x
        histograms = []

        for r in range(self.grid_y):
            for c in range(self.grid_x):
                cell = lbp[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w]
                hist, _ = np.histogram(cell, bins=32, range=(0, 256), density=True)
                histograms.append(hist)

        feat = np.concatenate(histograms).astype(np.float32)
        norm = np.linalg.norm(feat)
        return feat / (norm + 1e-7)

    def fit(self, image_paths: list[Path], labels: list[str]) -> None:
        features = [self.extract_features(p) for p in image_paths]
        self.gallery_features = np.stack(features)
        self.gallery_labels = labels
        self.knn.fit(self.gallery_features)

    def predict(self, probe_path: Path) -> tuple[str, float]:
        feat = self.extract_features(probe_path).reshape(1, -1)
        distances, indices = self.knn.kneighbors(feat)
        cosine_dist = float(distances[0][0])
        sim = 1.0 - cosine_dist
        pred_label = self.gallery_labels[indices[0][0]]
        return pred_label, sim


def evaluate_feature_extractor_on_pairs(
    extract_fn: Any,
    gallery_items: list[dict[str, str]],
    probe_items: list[dict[str, str]],
    base_dir: Path,
) -> dict[str, float]:
    """Computes genuine and impostor pair similarities, EER, and TAR@FAR=1%."""
    from fras.evaluation.metrics import calculate_biometric_metrics

    # Extract gallery embeddings
    gallery_embeddings: list[np.ndarray] = []
    gallery_pids: list[str] = []
    for item in gallery_items:
        path = base_dir / item["relative_path"]
        emb = extract_fn(path)
        gallery_embeddings.append(emb)
        gallery_pids.append(item["person_id"])

    # Extract probe embeddings
    probe_embeddings: list[np.ndarray] = []
    probe_pids: list[str] = []
    for item in probe_items:
        path = base_dir / item["relative_path"]
        emb = extract_fn(path)
        probe_embeddings.append(emb)
        probe_pids.append(item["person_id"])

    # Build genuine and impostor pairs
    genuine_scores: list[float] = []
    impostor_scores: list[float] = []

    for p_idx, p_pid in enumerate(probe_pids):
        p_emb = probe_embeddings[p_idx]
        for g_idx, g_pid in enumerate(gallery_pids):
            sim = float(np.dot(p_emb, gallery_embeddings[g_idx]))
            if p_pid == g_pid:
                genuine_scores.append(sim)
            else:
                impostor_scores.append(sim)

    return calculate_biometric_metrics(
        np.array(genuine_scores),
        np.array(impostor_scores),
    )
