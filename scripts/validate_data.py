"""Script to validate raw dataset integrity and generate leakage-free splits."""

from pathlib import Path

from fras.training.dataset import generate_splits, validate_dataset


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    raw_dir = root / "data" / "raw" / "face_attendance_dataset"
    splits_dir = root / "data" / "splits"

    print(f"Validating dataset at: {raw_dir}")
    stats = validate_dataset(raw_dir)
    print("Validation passed successfully!")
    print(f"  Total Persons: {stats.total_persons}")
    print(f"  Total Images: {stats.total_images}")
    print(f"  Image Shape: {stats.image_shape}")
    print(f"  Corrupted Images: {stats.corrupted_images}")
    print(f"  Attendance Records: {stats.attendance_records}")
    print(f"  Departments ({len(stats.departments)}): {stats.departments}")
    print(f"  Status Counts: {stats.status_counts}")
    print(f"  Camera Counts: {stats.camera_counts}")
    print(f"  Luminance: {stats.mean_luminance:.2f} +/- {stats.std_luminance:.2f}")

    print(f"\nGenerating reproducible splits in: {splits_dir}")
    closed_split, open_split = generate_splits(raw_dir, splits_dir, seed=42)

    print("Closed-Set counts:", closed_split["split_counts"])
    print("Open-Set partitions:")
    print("  Train identities:", len(open_split["identities"]["train"]))
    print("  Val identities:", len(open_split["identities"]["val"]))
    print("  Test identities (unseen):", len(open_split["identities"]["test"]))
    print(
        f"  Val gallery/probe: {len(open_split['splits']['val_gallery'])} / {len(open_split['splits']['val_probes'])}"
    )
    print(
        f"  Test gallery/probe: {len(open_split['splits']['test_gallery'])} / {len(open_split['splits']['test_probes'])}"
    )

    # Verify no leakage
    # 1. Closed set: image path disjointness
    train_paths = {x["relative_path"] for x in closed_split["splits"]["train"]}
    val_paths = {x["relative_path"] for x in closed_split["splits"]["val"]}
    test_paths = {x["relative_path"] for x in closed_split["splits"]["test"]}
    assert len(train_paths & val_paths) == 0, "Closed-set train/val leakage detected!"
    assert len(train_paths & test_paths) == 0, "Closed-set train/test leakage detected!"
    assert len(val_paths & test_paths) == 0, "Closed-set val/test leakage detected!"

    # 2. Open set: identity disjointness
    open_train_ids = set(open_split["identities"]["train"])
    open_val_ids = set(open_split["identities"]["val"])
    open_test_ids = set(open_split["identities"]["test"])
    assert len(open_train_ids & open_val_ids) == 0, "Open-set train/val identity leakage!"
    assert len(open_train_ids & open_test_ids) == 0, "Open-set train/test identity leakage!"
    assert len(open_val_ids & open_test_ids) == 0, "Open-set val/test identity leakage!"
    print("\nVerified: All splits are mathematically disjoint with 0% data leakage!")


if __name__ == "__main__":
    main()
