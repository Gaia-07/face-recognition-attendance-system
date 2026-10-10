"""Face detection module supporting avatar pass-through and OpenCV Zoo YuNet."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class FaceDetection:
    bbox: tuple[int, int, int, int]  # (x, y, w, h)
    score: float
    landmarks: np.ndarray | None  # (5, 2) [re, le, nt, rm, lm]


class FaceDetector:
    """Multi-backend face detector."""

    def __init__(
        self,
        backend_name: str = "avatar_cnn",
        model_path: str | Path | None = None,
        conf_threshold: float = 0.6,
    ) -> None:
        self.backend_name = backend_name
        self.conf_threshold = conf_threshold
        self.yunet_detector = None

        if backend_name == "sface":
            if model_path is None:
                root = Path(__file__).resolve().parent.parent.parent.parent
                model_path = root / "models" / "face_detection_yunet_2023mar.onnx"

            m_path = Path(model_path)
            if not m_path.exists():
                raise FileNotFoundError(
                    f"YuNet ONNX detector not found at '{m_path}'. "
                    "Run 'python scripts/download_models.py' or use avatar_cnn mode."
                )

            self.yunet_detector = cv2.FaceDetectorYN.create(
                model=str(m_path),
                config="",
                input_size=(320, 320),
                score_threshold=conf_threshold,
                nms_threshold=0.3,
                top_k=5000,
                backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
                target_id=cv2.dnn.DNN_TARGET_CPU,
            )

    def detect(self, image_rgb_or_bgr: np.ndarray) -> list[FaceDetection]:
        """Detects faces in input image."""
        h, w = image_rgb_or_bgr.shape[:2]

        if self.backend_name == "avatar_cnn" or self.yunet_detector is None:
            # Avatar mode: pass-through bounding box covering full image
            return [FaceDetection(bbox=(0, 0, w, h), score=1.0, landmarks=None)]

        # Real-face mode via YuNet
        self.yunet_detector.setInputSize((w, h))
        # YuNet expects BGR
        _, faces = self.yunet_detector.detect(image_rgb_or_bgr)
        if faces is None:
            return []

        results: list[FaceDetection] = []
        for face in faces:
            x, y, bw, bh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            score = float(face[14])
            landmarks = np.array(
                [
                    [face[4], face[5]],  # right eye
                    [face[6], face[7]],  # left eye
                    [face[8], face[9]],  # nose tip
                    [face[10], face[11]],  # right mouth corner
                    [face[12], face[13]],  # left mouth corner
                ],
                dtype=np.float32,
            )
            results.append(FaceDetection(bbox=(x, y, bw, bh), score=score, landmarks=landmarks))

        return results
