"""Dataset loading, validation, and split management for FRAS."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class DatasetStats:
    total_persons: int
    total_images: int
    image_shape: tuple[int, int, int]
    corrupted_images: int
    departments: dict[str, int]
    attendance_records: int
    status_counts: dict[str, int]
    camera_counts: dict[str, int]
    mean_luminance: float
    std_luminance: float


def validate_dataset(dataset_dir: str | Path) -> DatasetStats:
    """Validates data integrity, dimensions, referential constraints, and returns statistics.

    Raises:
        FileNotFoundError: if dataset paths do not exist.
        ValueError: if data corruption or schema violations are found.
    """
    path = Path(dataset_dir)
    images_dir = path / "images"
    metadata_dir = path / "metadata"
    attendance_dir = path / "attendance"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    persons_csv = metadata_dir / "persons.csv"
    if not persons_csv.exists():
        raise FileNotFoundError(f"persons.csv not found: {persons_csv}")

    persons_df = pd.read_csv(persons_csv)
    expected_person_cols = {"person_id", "name", "department", "enrollment_date", "num_images"}
    if not expected_person_cols.issubset(persons_df.columns):
        raise ValueError(f"persons.csv missing required columns. Found: {list(persons_df.columns)}")

    attendance_csv = attendance_dir / "attendance_log.csv"
    if not attendance_csv.exists():
        raise FileNotFoundError(f"attendance_log.csv not found: {attendance_csv}")

    attendance_df = pd.read_csv(attendance_csv)

    person_dirs = sorted([p for p in images_dir.iterdir() if p.is_dir()])
    total_persons = len(person_dirs)
    if total_persons != len(persons_df):
        raise ValueError(
            f"Directory count ({total_persons}) != persons.csv count ({len(persons_df)})"
        )

    # Referential check
    csv_ids = set(persons_df["person_id"])
    dir_ids = {p.name for p in person_dirs}
    if csv_ids != dir_ids:
        raise ValueError(f"Mismatch between folder names and person_ids: {csv_ids ^ dir_ids}")

    attendance_ids = set(attendance_df["person_id"])
    if not attendance_ids.issubset(csv_ids):
        raise ValueError(f"Attendance log references unregistered IDs: {attendance_ids - csv_ids}")

    total_images = 0
    corrupted_images = 0
    luminances: list[float] = []
    canonical_shape: tuple[int, int, int] | None = None

    for p_dir in person_dirs:
        img_files = sorted(
            [f for f in p_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        )
        total_images += len(img_files)

        for img_path in img_files:
            try:
                with Image.open(img_path) as img:
                    img_rgb = img.convert("RGB")
                    arr = np.array(img_rgb)
                    shape = arr.shape
                    if canonical_shape is None:
                        canonical_shape = shape
                    elif shape != canonical_shape:
                        raise ValueError(f"Inconsistent shape {shape} in {img_path}")

                    # Compute luminance (Rec. 601)
                    lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
                    luminances.append(float(lum.mean()))
            except Exception as e:
                logger.error("Failed to read image %s: %s", img_path, e)
                corrupted_images += 1

    if canonical_shape is None:
        raise ValueError("No valid images found in dataset.")

    department_counts = {str(k): int(v) for k, v in persons_df["department"].value_counts().items()}
    status_counts = {str(k): int(v) for k, v in attendance_df["status"].value_counts().items()}
    camera_counts = {str(k): int(v) for k, v in attendance_df["camera_id"].value_counts().items()}

    lum_arr = np.array(luminances)
    mean_lum = float(lum_arr.mean()) if len(lum_arr) > 0 else 0.0
    std_lum = float(lum_arr.std()) if len(lum_arr) > 0 else 0.0

    return DatasetStats(
        total_persons=total_persons,
        total_images=total_images,
        image_shape=canonical_shape,
        corrupted_images=corrupted_images,
        departments=department_counts,
        attendance_records=len(attendance_df),
        status_counts=status_counts,
        camera_counts=camera_counts,
        mean_luminance=mean_lum,
        std_luminance=std_lum,
    )


def generate_splits(
    dataset_dir: str | Path,
    output_dir: str | Path,
    seed: int = 42,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Generates leakage-free closed-set and open-set splits and writes them to JSON.

    Closed-set: 10 train / 2 val / 3 test images per identity across all 30 identities.
    Open-set: 20 train identities / 5 val identities / 5 unseen test identities.
              For val and test identities: 3 gallery images, 12 probe images.
    """
    rng = np.random.default_rng(seed)
    images_dir = Path(dataset_dir) / "images"
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    person_dirs = sorted([p.name for p in images_dir.iterdir() if p.is_dir()])
    all_identities = np.array(person_dirs)

    # 1. Closed-set split
    closed_train_items: list[dict[str, str]] = []
    closed_val_items: list[dict[str, str]] = []
    closed_test_items: list[dict[str, str]] = []

    for pid in all_identities:
        p_dir = images_dir / pid
        images = sorted(
            [f.name for f in p_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        )
        if len(images) != 15:
            raise ValueError(f"Expected 15 images for {pid}, found {len(images)}")

        indices = np.arange(len(images))
        rng.shuffle(indices)

        train_idx = indices[:10]
        val_idx = indices[10:12]
        test_idx = indices[12:15]

        for i in train_idx:
            closed_train_items.append({"person_id": pid, "relative_path": f"{pid}/{images[i]}"})
        for i in val_idx:
            closed_val_items.append({"person_id": pid, "relative_path": f"{pid}/{images[i]}"})
        for i in test_idx:
            closed_test_items.append({"person_id": pid, "relative_path": f"{pid}/{images[i]}"})

    closed_split_data: dict[str, Any] = {
        "protocol": "closed_set",
        "seed": seed,
        "num_identities": len(all_identities),
        "split_counts": {
            "train": len(closed_train_items),
            "val": len(closed_val_items),
            "test": len(closed_test_items),
        },
        "splits": {
            "train": closed_train_items,
            "val": closed_val_items,
            "test": closed_test_items,
        },
    }

    # 2. Open-set split (Disjoint identities)
    shuffled_pids = all_identities.copy()
    rng.shuffle(shuffled_pids)

    train_pids = list(shuffled_pids[:20])
    val_pids = list(shuffled_pids[20:25])
    test_pids = list(shuffled_pids[25:30])

    open_train_items: list[dict[str, str]] = []
    for pid in train_pids:
        p_dir = images_dir / pid
        images = sorted(
            [f.name for f in p_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        )
        for img in images:
            open_train_items.append({"person_id": pid, "relative_path": f"{pid}/{img}"})

    def partition_gallery_probe(
        pids: list[str],
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        gallery: list[dict[str, str]] = []
        probes: list[dict[str, str]] = []
        for pid in pids:
            p_dir = images_dir / pid
            images = sorted(
                [f.name for f in p_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
            )
            idx = np.arange(len(images))
            rng.shuffle(idx)
            g_idx = idx[:3]
            p_idx = idx[3:]
            for i in g_idx:
                gallery.append({"person_id": pid, "relative_path": f"{pid}/{images[i]}"})
            for i in p_idx:
                probes.append({"person_id": pid, "relative_path": f"{pid}/{images[i]}"})
        return gallery, probes

    val_gallery, val_probes = partition_gallery_probe(val_pids)
    test_gallery, test_probes = partition_gallery_probe(test_pids)

    open_split_data: dict[str, Any] = {
        "protocol": "open_set",
        "seed": seed,
        "identities": {
            "train": train_pids,
            "val": val_pids,
            "test": test_pids,
        },
        "splits": {
            "train": open_train_items,
            "val_gallery": val_gallery,
            "val_probes": val_probes,
            "test_gallery": test_gallery,
            "test_probes": test_probes,
        },
    }

    with open(out_path / "closed_set.json", "w", encoding="utf-8") as f:
        json.dump(closed_split_data, f, indent=2)

    with open(out_path / "open_set.json", "w", encoding="utf-8") as f:
        json.dump(open_split_data, f, indent=2)

    return closed_split_data, open_split_data


class FaceImageDataset(Dataset[tuple[torch.Tensor, int, str]]):
    """PyTorch Dataset loading images from relative path items with identity labels."""

    def __init__(
        self,
        base_dir: str | Path,
        items: list[dict[str, str]],
        identity_to_idx: dict[str, int],
        transform: Any = None,
    ) -> None:
        self.base_dir = Path(base_dir) / "images"
        self.items = items
        self.identity_to_idx = identity_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        item = self.items[index]
        pid = item["person_id"]
        rel_path = item["relative_path"]
        full_path = self.base_dir / rel_path

        with Image.open(full_path) as img:
            arr = np.array(img.convert("RGB"))

        if self.transform is not None:
            # Albumentations expects image=arr
            if callable(self.transform) and "image" in getattr(self.transform, "inputs", ["image"]):
                augmented = self.transform(image=arr)
                arr = augmented["image"]
            elif callable(self.transform):
                arr = self.transform(arr)

        # Convert HWC uint8 -> CHW float32 normalized to [-1, 1] or [0, 1]
        tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        # Standard normalization: mean=0.5, std=0.5 -> [-1, 1]
        tensor = (tensor - 0.5) / 0.5

        label = self.identity_to_idx[pid]
        return tensor, label, pid
