"""SFace real-face recognition backend (OpenCV Zoo Apache-2.0)."""

from pathlib import Path

import cv2
import numpy as np

from fras.recognition.base import EmbeddingBackend


class SFaceBackend(EmbeddingBackend):
    """Real-face recognition backend utilizing OpenCV Zoo's SFace model."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.name = "sface"
        self.dim = 128

        if model_path is None:
            root = Path(__file__).resolve().parent.parent.parent.parent
            model_path = root / "models" / "face_recognition_sface_2021dec.onnx"

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"SFace model ONNX not found at '{self.model_path}'. "
                "Run 'python scripts/download_models.py' to download real-face weights, "
                "or switch active_backend to 'avatar_cnn' in config/default.yaml."
            )

        try:
            self.recognizer = cv2.FaceRecognizerSF.create(
                model=str(self.model_path),
                config="",
                backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
                target_id=cv2.dnn.DNN_TARGET_CPU,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize cv2.FaceRecognizerSF: {e}") from e

    def preprocess(self, images: np.ndarray) -> np.ndarray:
        """Preprocesses RGB images for SFace (expects 112x112 aligned face crop)."""
        if images.ndim == 3:
            images = np.expand_dims(images, axis=0)

        processed = []
        for img in images:
            if img.shape[:2] != (112, 112):
                img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)
            # SFace expects BGR format internally
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            processed.append(bgr)

        return np.stack(processed, axis=0)

    def embed(self, batch: np.ndarray) -> np.ndarray:
        """Computes unit L2-normalized 128-d embeddings for input aligned face images."""
        if batch.ndim == 3:
            batch = np.expand_dims(batch, axis=0)

        embeddings = []
        for img in batch:
            raw_feature = self.recognizer.feature(img)
            emb = np.asarray(raw_feature, dtype=np.float32).flatten()
            norm = float(np.linalg.norm(emb))
            embeddings.append(emb / (norm + 1e-7))

        return np.stack(embeddings, axis=0)
