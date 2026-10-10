"""Robustness stress-testing across synthetic perturbations."""

import io
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from fras.evaluation.metrics import calculate_biometric_metrics


def apply_gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    ksize = int(2 * round(3 * sigma) + 1)
    return cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)


def apply_brightness_shift(image: np.ndarray, shift: int) -> np.ndarray:
    return np.clip(image.astype(np.int32) + shift, 0, 255).astype(np.uint8)


def apply_additive_noise(image: np.ndarray, sigma_frac: float) -> np.ndarray:
    noise = np.random.normal(0, sigma_frac * 255, image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def apply_occlusion(image: np.ndarray, patch_size: int) -> np.ndarray:
    out = image.copy()
    h, w, _ = out.shape
    top = (h - patch_size) // 2
    left = (w - patch_size) // 2
    out[top : top + patch_size, left : left + patch_size, :] = 0  # Black square occlusion
    return out


def apply_rotation(image: np.ndarray, angle: float) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, m, (w, h), borderMode=cv2.BORDER_REFLECT)


def apply_jpeg_compression(image: np.ndarray, quality: int) -> np.ndarray:
    pil_img = Image.fromarray(image)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf))


def run_stress_benchmarks(
    onnx_session: Any,
    gallery_items: list[dict[str, str]],
    probe_items: list[dict[str, str]],
    base_dir: Path,
) -> dict[str, Any]:
    """Sweeps over 6 distinct environmental stress dimensions and evaluates open-set EER & AUC."""

    def extract_from_arr(arr: np.ndarray) -> np.ndarray:
        tensor = arr.transpose(2, 0, 1).astype(np.float32) / 255.0
        tensor = (tensor - 0.5) / 0.5
        tensor = np.expand_dims(tensor, axis=0)
        raw_emb = onnx_session.run(None, {"input": tensor})[0][0]
        emb: np.ndarray = np.asarray(raw_emb, dtype=np.float32)
        norm = float(np.linalg.norm(emb))
        return emb / (norm + 1e-7)

    # Clean gallery embeddings
    gallery_embs = []
    gallery_pids = []
    for item in gallery_items:
        p = base_dir / item["relative_path"]
        with Image.open(p) as img:
            arr = np.array(img.convert("RGB"))
        gallery_embs.append(extract_from_arr(arr))
        gallery_pids.append(item["person_id"])
    gallery_mat = np.stack(gallery_embs)

    # Probe arrays (unperturbed ground truth)
    probe_raw = []
    probe_pids = []
    for item in probe_items:
        p = base_dir / item["relative_path"]
        with Image.open(p) as img:
            probe_raw.append(np.array(img.convert("RGB")))
        probe_pids.append(item["person_id"])

    def evaluate_perturbation(perturb_fn: Any) -> tuple[float, float, float]:
        gen_scores = []
        imp_scores = []
        for idx, raw_img in enumerate(probe_raw):
            pert_img = perturb_fn(raw_img)
            p_emb = extract_from_arr(pert_img)
            p_pid = probe_pids[idx]
            for g_idx, g_pid in enumerate(gallery_pids):
                sim = float(np.dot(p_emb, gallery_mat[g_idx]))
                if p_pid == g_pid:
                    gen_scores.append(sim)
                else:
                    imp_scores.append(sim)
        m = calculate_biometric_metrics(np.array(gen_scores), np.array(imp_scores))
        return m["eer"], m["roc_auc"], m["tar_at_far_1e2"]

    stress_results: dict[str, list[dict[str, Any]]] = {
        "gaussian_blur": [],
        "brightness_shift": [],
        "additive_noise": [],
        "occlusion_patch": [],
        "rotation_angle": [],
        "jpeg_compression": [],
    }

    # 1. Blur
    for s in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
        eer, auc, tar = evaluate_perturbation(lambda img, s=s: apply_gaussian_blur(img, s))
        stress_results["gaussian_blur"].append(
            {"sigma": s, "eer": eer, "roc_auc": auc, "tar_at_far_1e2": tar}
        )

    # 2. Brightness
    for b in [-60, -40, -20, 0, 20, 40, 60]:
        eer, auc, tar = evaluate_perturbation(lambda img, b=b: apply_brightness_shift(img, b))
        stress_results["brightness_shift"].append(
            {"shift": b, "eer": eer, "roc_auc": auc, "tar_at_far_1e2": tar}
        )

    # 3. Noise
    for n in [0.02, 0.05, 0.08, 0.12, 0.16, 0.20]:
        eer, auc, tar = evaluate_perturbation(lambda img, n=n: apply_additive_noise(img, n))
        stress_results["additive_noise"].append(
            {"noise_std": n, "eer": eer, "roc_auc": auc, "tar_at_far_1e2": tar}
        )

    # 4. Occlusion
    for o in [8, 16, 24, 32, 40, 48]:
        eer, auc, tar = evaluate_perturbation(lambda img, o=o: apply_occlusion(img, o))
        stress_results["occlusion_patch"].append(
            {"patch_size_px": o, "eer": eer, "roc_auc": auc, "tar_at_far_1e2": tar}
        )

    # 5. Rotation
    for r in [-45, -30, -15, 0, 15, 30, 45]:
        eer, auc, tar = evaluate_perturbation(lambda img, r=r: apply_rotation(img, r))
        stress_results["rotation_angle"].append(
            {"angle_deg": r, "eer": eer, "roc_auc": auc, "tar_at_far_1e2": tar}
        )

    # 6. JPEG quality
    for q in [95, 75, 50, 30, 20, 10]:
        eer, auc, tar = evaluate_perturbation(lambda img, q=q: apply_jpeg_compression(img, q))
        stress_results["jpeg_compression"].append(
            {"quality": q, "eer": eer, "roc_auc": auc, "tar_at_far_1e2": tar}
        )

    return stress_results
