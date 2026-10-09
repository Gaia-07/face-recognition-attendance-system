"""Generates visual assets and exploratory data analysis charts for documentation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# Use headless matplotlib backend
plt.switch_backend("Agg")


def generate_eda_charts() -> None:
    root = Path(__file__).resolve().parent.parent
    raw_dir = root / "data" / "raw" / "face_attendance_dataset"
    assets_dir = root / "docs" / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    persons_df = pd.read_csv(raw_dir / "metadata" / "persons.csv")
    attendance_df = pd.read_csv(raw_dir / "attendance" / "attendance_log.csv")

    # Set aesthetic style
    plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

    # 1. Department Distribution
    plt.figure(figsize=(8, 4.5))
    dept_counts = persons_df["department"].value_counts()
    bars = plt.bar(dept_counts.index, dept_counts.values, color="#2b5c8f", edgecolor="#1a365d")
    plt.title("Identity Distribution Across Departments (N=30)", fontsize=13, fontweight="bold")
    plt.xlabel("Department")
    plt.ylabel("Number of Identities")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0, yval + 0.1, int(yval), ha="center", va="bottom"
        )
    plt.savefig(assets_dir / "eda_department_dist.png", dpi=150)
    plt.close()

    # 2. Attendance Status Distribution
    plt.figure(figsize=(6, 4.5))
    status_counts = attendance_df["status"].value_counts()
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    wedges, texts, autotexts = plt.pie(
        status_counts.values,
        labels=status_counts.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        explode=(0.02, 0.05, 0.08),
    )
    plt.title(
        "Historical Attendance Status Breakdown (1,980 records)", fontsize=12, fontweight="bold"
    )
    plt.savefig(assets_dir / "eda_status_dist.png", dpi=150)
    plt.close()

    # 3. Camera Utilization
    plt.figure(figsize=(7, 4.5))
    cam_counts = attendance_df["camera_id"].value_counts()
    bars = plt.bar(cam_counts.index, cam_counts.values, color="#34495e", edgecolor="#111111")
    plt.title("Sensor Captures by Hardware Camera Endpoint", fontsize=12, fontweight="bold")
    plt.xlabel("Camera Identifier")
    plt.ylabel("Capture Events")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0, yval + 10, int(yval), ha="center", va="bottom"
        )
    plt.savefig(assets_dir / "eda_camera_dist.png", dpi=150)
    plt.close()

    # 4. Image Luminance Histogram
    images_dir = raw_dir / "images"
    luminances: list[float] = []
    for p_dir in sorted(images_dir.iterdir()):
        if not p_dir.is_dir():
            continue
        for img_path in p_dir.iterdir():
            if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                with Image.open(img_path) as img:
                    arr = np.array(img.convert("RGB"))
                    lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
                    luminances.append(float(lum.mean()))

    plt.figure(figsize=(8, 4.5))
    plt.hist(luminances, bins=25, color="#8e44ad", edgecolor="#4a235a", alpha=0.85)
    plt.axvline(
        np.mean(luminances),
        color="red",
        linestyle="dashed",
        linewidth=1.5,
        label=f"Mean: {np.mean(luminances):.1f}",
    )
    plt.title(
        "Luminance Distribution Across Avatar Dataset (450 images)", fontsize=12, fontweight="bold"
    )
    plt.xlabel("Mean Pixel Luminance (Rec. 601)")
    plt.ylabel("Image Count")
    plt.legend()
    plt.grid(linestyle="--", alpha=0.5)
    plt.savefig(assets_dir / "eda_luminance_dist.png", dpi=150)
    plt.close()

    # 5. Sample Avatar Grid (6 identities, 3 variations each)
    fig, axes = plt.subplots(3, 6, figsize=(12, 6))
    sample_pids = sorted([p.name for p in images_dir.iterdir() if p.is_dir()])[:6]
    for col_idx, pid in enumerate(sample_pids):
        p_imgs = sorted((images_dir / pid).glob("*.jpg"))[:3]
        for row_idx, img_path in enumerate(p_imgs):
            ax = axes[row_idx, col_idx]
            img = Image.open(img_path)
            ax.imshow(img)
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(pid, fontsize=10, fontweight="bold")
    plt.suptitle(
        "Sample Procedurally Generated Avatar Inputs (128x128 RGB)", fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(assets_dir / "eda_sample_grid.png", dpi=150)
    plt.close()

    print(f"Generated 5 EDA charts in {assets_dir}")


if __name__ == "__main__":
    generate_eda_charts()
