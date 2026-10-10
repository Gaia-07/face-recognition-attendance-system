"""Face alignment and geometric normalization routines."""

import cv2
import numpy as np

# Canonical reference landmarks for 112x112 aligned face (ArcFace / SFace standard)
CANONICAL_5PTS_112X112 = np.array(
    [
        [38.2946, 51.6963],  # right eye
        [73.5318, 51.5014],  # left eye
        [56.0252, 71.7366],  # nose
        [41.5493, 92.3655],  # right mouth corner
        [70.7299, 92.2041],  # left mouth corner
    ],
    dtype=np.float32,
)


def align_face_5points(
    image: np.ndarray,
    landmarks: np.ndarray,
    target_size: tuple[int, int] = (112, 112),
) -> np.ndarray:
    """Estimates similarity transformation aligning 5 detected facial landmarks to canonical reference points."""
    src_pts = landmarks.astype(np.float32)
    dst_pts = CANONICAL_5PTS_112X112

    # Estimate partial affine (similarity) matrix
    m, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    if m is None:
        # Fallback to standard resize if affine matrix estimation degenerates
        return cv2.resize(image, target_size)

    aligned = cv2.warpAffine(
        image,
        m,
        target_size,
        borderMode=cv2.BORDER_REFLECT,
    )
    return aligned


def crop_and_normalize_face(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    target_size: tuple[int, int] = (128, 128),
) -> np.ndarray:
    """Crops face bounding box (x, y, w, h) with bounds clamping and resizes to target_size."""
    h, w = image.shape[:2]
    x, y, bw, bh = bbox

    # Clamp coordinates to valid image bounds
    x1 = max(0, min(x, w - 1))
    y1 = max(0, min(y, h - 1))
    x2 = max(x1 + 1, min(x + bw, w))
    y2 = max(y1 + 1, min(y + bh, h))

    crop = image[y1:y2, x1:x2]
    if crop.shape[:2] != target_size:
        crop = cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)
    return crop
