"""Avatar CNN embedding backend powered by our trained ONNX model."""

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from fras.recognition.base import EmbeddingBackend


class AvatarCNNBackend(EmbeddingBackend):
    """In-house trained compact ResNet embedder for synthetic avatar benchmarks."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.name = "avatar_cnn"
        self.dim = 128

        if model_path is None:
            root = Path(__file__).resolve().parent.parent.parent.parent
            model_path = root / "models" / "avatar_cnn.onnx"

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. Train the model first using 'make train'."
            )

        self.session = ort.InferenceSession(
            str(self.model_path), providers=["CPUExecutionProvider"]
        )

    def preprocess(self, images: np.ndarray) -> np.ndarray:
        """Preprocesses RGB images (N, H, W, 3) or single (H, W, 3) to (N, 3, 128, 128) in [-1, 1]."""
        if images.ndim == 3:
            images = np.expand_dims(images, axis=0)

        processed = []
        for img in images:
            if img.shape[:2] != (128, 128):
                img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LINEAR)
            # Normalize uint8 [0, 255] -> float32 [-1, 1]
            arr = (img.astype(np.float32) / 255.0 - 0.5) / 0.5
            arr = arr.transpose(2, 0, 1)  # HWC -> CHW
            processed.append(arr)

        return np.stack(processed, axis=0)

    def embed(self, batch: np.ndarray) -> np.ndarray:
        """Extracts unit L2-normalized embeddings from batch (N, 3, 128, 128)."""
        if batch.ndim == 3:
            batch = np.expand_dims(batch, axis=0)

        raw_out = self.session.run(None, {"input": batch})[0]
        embs: np.ndarray = np.asarray(raw_out, dtype=np.float32)
        norms: np.ndarray = np.linalg.norm(embs, axis=1, keepdims=True)
        res: np.ndarray = embs / (norms + 1e-7)
        return res
