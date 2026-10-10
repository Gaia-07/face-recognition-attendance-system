"""Camera capture abstractions supporting physical webcams and headless SimulatedCamera."""

import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


class CameraUnavailableError(RuntimeError):
    """Raised when physical video acquisition hardware is missing, busy, or disconnected."""

    pass


class OpenCVCamera:
    """Physical webcam capture wrapper via OpenCV with explicit hardware error handling."""

    def __init__(self, device_index: int = 0) -> None:
        self.device_index = device_index
        self.cap = cv2.VideoCapture(device_index)
        if not self.cap.isOpened():
            raise CameraUnavailableError(
                f"Camera device {device_index} is unavailable, in-use, or disconnected."
            )

    def read_frame(self) -> np.ndarray:
        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise CameraUnavailableError(
                f"Failed to acquire video frame from camera {self.device_index}."
            )
        return frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()


class SimulatedCamera:
    """Simulated camera streaming dataset images with stochastic jitter for headless operation."""

    def __init__(
        self,
        images_dir: str | Path | None = None,
        target_size: tuple[int, int] = (128, 128),
        add_jitter: bool = True,
        frame_interval_sec: float = 0.05,
    ) -> None:
        self.target_size = target_size
        self.add_jitter = add_jitter
        self.frame_interval_sec = frame_interval_sec
        self.last_read_time = 0.0

        if images_dir is None:
            root = Path(__file__).resolve().parent.parent.parent.parent
            images_dir = root / "data" / "raw" / "face_attendance_dataset" / "images"

        self.images_dir = Path(images_dir)
        self.image_paths: list[Path] = []
        if self.images_dir.exists():
            for p_dir in sorted(self.images_dir.iterdir()):
                if p_dir.is_dir():
                    self.image_paths.extend(sorted(p_dir.glob("*.jpg")))

        if not self.image_paths:
            # Fallback to generating synthetic avatar dummy images
            self.image_paths = []

        self.current_idx = 0

    def read_frame(self) -> np.ndarray:
        """Returns the next simulated video frame with optional temporal rate limiting."""
        now = time.time()
        elapsed = now - self.last_read_time
        if elapsed < self.frame_interval_sec:
            time.sleep(self.frame_interval_sec - elapsed)
        self.last_read_time = time.time()

        if self.image_paths:
            img_path = self.image_paths[self.current_idx % len(self.image_paths)]
            self.current_idx += 1
            with Image.open(img_path) as im:
                frame = np.array(im.convert("RGB"))
        else:
            # Synthetic 128x128 color block
            frame = np.ones((128, 128, 3), dtype=np.uint8) * 180

        if self.add_jitter:
            # Subtle random illumination shift (+/- 8)
            jitter = np.random.randint(-8, 9)
            frame = np.clip(frame.astype(np.int16) + jitter, 0, 255).astype(np.uint8)

        if frame.shape[:2] != self.target_size:
            frame = cv2.resize(frame, self.target_size)

        return frame

    def release(self) -> None:
        pass
