"""Download official OpenCV Zoo YuNet and SFace models with SHA-256 integrity checks."""

import hashlib
import urllib.request
from pathlib import Path

MODELS = {
    "face_detection_yunet_2023mar.onnx": {
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "sha256": "473db2f2eb681ef511e40c865f3f090d23fb5a2858b4dae8a719544c4fa0bc80",
    },
    "face_recognition_sface_2021dec.onnx": {
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "sha256": "8fa84742f8c5b3d5b06bd9e1925b6a7ee7b2c58ee8a3e7efeb9fb8ebf4b16259",
    },
}


def verify_sha256(filepath: Path, expected_hash: str) -> bool:
    if not filepath.exists():
        return False
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower() == expected_hash.lower()


def download_models(target_dir: Path | None = None) -> None:
    if target_dir is None:
        target_dir = Path(__file__).resolve().parent.parent / "models"
    target_dir.mkdir(parents=True, exist_ok=True)

    for filename, meta in MODELS.items():
        out_path = target_dir / filename
        if verify_sha256(out_path, meta["sha256"]):
            print(f"[OK] {filename} verified (SHA-256 match).")
            continue

        print(f"Downloading {filename} from {meta['url']}...")
        try:
            urllib.request.urlretrieve(meta["url"], str(out_path))
            if verify_sha256(out_path, meta["sha256"]):
                print(f"[SUCCESS] Downloaded and verified {filename}.")
            else:
                print(f"[WARNING] SHA-256 verification failed for {filename}.")
        except Exception as e:
            print(f"[FAILED] Could not download {filename}: {e}")
            print("Note: The avatar_cnn backend operates fully offline without these files.")


if __name__ == "__main__":
    download_models()
