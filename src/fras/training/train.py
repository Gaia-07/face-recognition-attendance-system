"""End-to-end training, baselines, ablations, and ONNX export for AvatarResNet with ArcFace."""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from fras.evaluation.metrics import calculate_biometric_metrics
from fras.training.augment import build_train_transform
from fras.training.baselines import (
    LBPHBaseline,
    PixelHOGBaseline,
    evaluate_feature_extractor_on_pairs,
)
from fras.training.dataset import FaceImageDataset
from fras.training.export_onnx import export_to_onnx
from fras.training.losses import ArcFaceHead
from fras.training.model import AvatarResNet, create_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_embeddings_for_items(
    model: nn.Module,
    items: list[dict[str, str]],
    base_dir: Path,
    device: torch.device,
) -> tuple[np.ndarray, list[str]]:
    """Extracts unit-normalized embeddings from a list of dataset items."""
    model.eval()
    dataset = FaceImageDataset(
        base_dir=base_dir,
        items=items,
        identity_to_idx={it["person_id"]: 0 for it in items},
        transform=None,
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    embeddings_list: list[np.ndarray] = []
    pids_list: list[str] = []

    with torch.no_grad():
        for batch_imgs, _, batch_pids in loader:
            batch_imgs = batch_imgs.to(device)
            emb = model(batch_imgs)
            embeddings_list.append(emb.cpu().numpy())
            pids_list.extend(batch_pids)

    return np.concatenate(embeddings_list, axis=0), pids_list


def evaluate_model_on_split(
    model: nn.Module,
    gallery_items: list[dict[str, str]],
    probe_items: list[dict[str, str]],
    base_dir: Path,
    device: torch.device,
    target_far: float = 0.01,
) -> dict[str, float]:
    """Computes verification metrics (AUC, EER, TAR@FAR=1%) for a PyTorch model on gallery/probe."""
    g_emb, g_pids = extract_embeddings_for_items(model, gallery_items, base_dir, device)
    p_emb, p_pids = extract_embeddings_for_items(model, probe_items, base_dir, device)

    # Compute pairwise cosine similarity matrix
    sim_matrix = np.dot(p_emb, g_emb.T)  # (N_probe, N_gallery)

    genuine_scores: list[float] = []
    impostor_scores: list[float] = []

    for p_idx, p_id in enumerate(p_pids):
        for g_idx, g_id in enumerate(g_pids):
            sim = float(sim_matrix[p_idx, g_idx])
            if p_id == g_id:
                genuine_scores.append(sim)
            else:
                impostor_scores.append(sim)

    return calculate_biometric_metrics(
        np.array(genuine_scores), np.array(impostor_scores), target_far=target_far
    )


def train_epoch(
    model: nn.Module,
    head: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    head.train()
    total_loss = 0.0
    for imgs, labels, _ in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        features = model(imgs)
        logits = head(features, labels)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())

    return total_loss / max(len(loader), 1)


def run_training_experiment(
    config: dict[str, Any],
    fast_mode: bool = False,
    override_margin: float | None = None,
    override_dim: int | None = None,
    override_epochs: int | None = None,
    use_augmentation: bool = True,
) -> tuple[AvatarResNet, dict[str, Any]]:
    """Runs a complete training run with given parameters."""
    root_dir = Path(__file__).resolve().parent.parent.parent.parent
    data_dir = root_dir / config["data"]["raw_dir"]
    splits_file = root_dir / config["data"]["splits_dir"] / "open_set.json"

    with open(splits_file, encoding="utf-8") as f:
        splits = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_items = splits["splits"]["train"]
    val_gallery = splits["splits"]["val_gallery"]
    val_probes = splits["splits"]["val_probes"]

    train_pids = sorted({x["person_id"] for x in train_items})
    pid_to_idx = {pid: idx for idx, pid in enumerate(train_pids)}

    emb_dim = override_dim if override_dim is not None else int(config["model"]["embedding_dim"])
    margin = override_margin if override_margin is not None else float(config["model"]["margin"])
    scale = float(config["model"]["scale"])

    model = create_model(embedding_dim=emb_dim).to(device)
    head = ArcFaceHead(
        in_features=emb_dim, num_classes=len(train_pids), scale=scale, margin=margin
    ).to(device)

    transform = build_train_transform() if use_augmentation else None
    train_ds = FaceImageDataset(
        data_dir, train_items, identity_to_idx=pid_to_idx, transform=transform
    )
    batch_size = (
        int(config["training"]["batch_size"]) if not fast_mode else min(16, len(train_items))
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    if fast_mode:
        epochs = 1
    elif override_epochs is not None:
        epochs = override_epochs
    else:
        epochs = int(config["training"]["epochs"])
    best_eer = 1.0
    best_state: dict[str, Any] = {}

    start_time = time.time()
    for ep in range(1, epochs + 1):
        loss = train_epoch(model, head, train_loader, criterion, optimizer, device)
        val_metrics = evaluate_model_on_split(model, val_gallery, val_probes, data_dir, device)
        val_eer = val_metrics["eer"]

        if val_eer <= best_eer:
            best_eer = val_eer
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if not fast_mode and (ep % 5 == 0 or ep == epochs):
            logger.info(
                f"Epoch {ep:02d}/{epochs:02d} - Loss: {loss:.4f} - Val EER: {val_eer:.4f} (AUC: {val_metrics['roc_auc']:.4f})"
            )

    train_duration = time.time() - start_time
    if best_state:
        model.load_state_dict(best_state)

    final_val_metrics = evaluate_model_on_split(model, val_gallery, val_probes, data_dir, device)
    final_val_metrics["train_duration_sec"] = float(train_duration)
    final_val_metrics["epochs_run"] = epochs
    final_val_metrics["embedding_dim"] = emb_dim
    final_val_metrics["margin"] = margin
    final_val_metrics["augmentation"] = use_augmentation

    return model, final_val_metrics


def run_baselines_benchmark(
    config: dict[str, Any],
    proposed_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Evaluates all 4 baselines on the identical validation split."""
    root_dir = Path(__file__).resolve().parent.parent.parent.parent
    data_dir = root_dir / config["data"]["raw_dir"] / "images"
    splits_file = root_dir / config["data"]["splits_dir"] / "open_set.json"

    with open(splits_file, encoding="utf-8") as f:
        splits = json.load(f)

    val_gallery = splits["splits"]["val_gallery"]
    val_probes = splits["splits"]["val_probes"]

    # Baseline 1: Pixel / HOG
    b1 = PixelHOGBaseline()
    b1_metrics = evaluate_feature_extractor_on_pairs(
        b1.extract_features, val_gallery, val_probes, data_dir
    )

    # Baseline 2: LBPH
    b2 = LBPHBaseline()
    b2_metrics = evaluate_feature_extractor_on_pairs(
        b2.extract_features, val_gallery, val_probes, data_dir
    )

    # Baseline 3: Softmax-CE CNN (ArcFace margin = 0.0)
    logger.info("Training Baseline 3: Softmax-CE CNN (margin = 0.0)...")
    _, b3_metrics = run_training_experiment(
        config,
        fast_mode=False,
        override_margin=0.0,
        override_dim=128,
        override_epochs=15,
        use_augmentation=True,
    )

    results = {
        "baseline_1_pixel_knn": {
            "name": "Raw Pixels / HOG + Cosine kNN",
            "roc_auc": b1_metrics["roc_auc"],
            "eer": b1_metrics["eer"],
            "tar_at_far_1e2": b1_metrics["tar_at_far_1e2"],
        },
        "baseline_2_lbph": {
            "name": "Local Binary Pattern Histograms (LBPH)",
            "roc_auc": b2_metrics["roc_auc"],
            "eer": b2_metrics["eer"],
            "tar_at_far_1e2": b2_metrics["tar_at_far_1e2"],
        },
        "baseline_3_softmax_ce": {
            "name": "Softmax Cross-Entropy CNN",
            "roc_auc": b3_metrics["roc_auc"],
            "eer": b3_metrics["eer"],
            "tar_at_far_1e2": b3_metrics["tar_at_far_1e2"],
        },
        "proposed_arcface_cnn": {
            "name": "ArcFace AvatarResNet (Proposed)",
            "roc_auc": proposed_metrics["roc_auc"],
            "eer": proposed_metrics["eer"],
            "tar_at_far_1e2": proposed_metrics["tar_at_far_1e2"],
        },
    }
    return results


def run_ablation_study(config: dict[str, Any]) -> dict[str, Any]:
    """Runs ablation experiments over augmentation, margin, and embedding dimension."""
    logger.info("Executing ablation experiments...")
    ablations: dict[str, Any] = {}
    ablation_epochs = 12

    # 1. Augmentation on vs off
    _, no_aug_m = run_training_experiment(
        config, override_epochs=ablation_epochs, use_augmentation=False
    )
    ablations["no_augmentation"] = {
        "description": "ArcFace m=0.5, d=128 without data augmentation",
        "roc_auc": no_aug_m["roc_auc"],
        "eer": no_aug_m["eer"],
        "tar_at_far_1e2": no_aug_m["tar_at_far_1e2"],
    }

    # 2. Margins: 0.0, 0.3, 0.5
    for m in [0.0, 0.3, 0.5]:
        _, m_metrics = run_training_experiment(
            config, override_epochs=ablation_epochs, override_margin=m
        )
        ablations[f"margin_{m}"] = {
            "description": f"ArcFace with margin m={m}",
            "margin": m,
            "roc_auc": m_metrics["roc_auc"],
            "eer": m_metrics["eer"],
            "tar_at_far_1e2": m_metrics["tar_at_far_1e2"],
        }

    # 3. Embedding dimension: 64, 128, 256
    for d in [64, 128, 256]:
        _, d_metrics = run_training_experiment(
            config, override_epochs=ablation_epochs, override_dim=d
        )
        ablations[f"dim_{d}"] = {
            "description": f"ArcFace with embedding dimension d={d}",
            "embedding_dim": d,
            "roc_auc": d_metrics["roc_auc"],
            "eer": d_metrics["eer"],
            "tar_at_far_1e2": d_metrics["tar_at_far_1e2"],
        }

    return ablations


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FRAS Avatar ArcFace CNN")
    parser.add_argument(
        "--config", type=str, default="config/train.yaml", help="Path to train config"
    )
    parser.add_argument(
        "--fast", action="store_true", help="Fast smoke run (1 epoch) for CI testing"
    )
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent.parent.parent
    config_path = root_dir / args.config

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Set seed
    torch.manual_seed(config["experiment"]["seed"])
    np.random.default_rng(config["experiment"]["seed"])

    models_dir = root_dir / "models"
    results_dir = root_dir / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting training run for Avatar ArcFace CNN...")
    model, val_metrics = run_training_experiment(config, fast_mode=args.fast)
    logger.info(
        f"Validation Results - ROC-AUC: {val_metrics['roc_auc']:.4f}, EER: {val_metrics['eer']:.4f}, Calibrated Thresh: {val_metrics['calibrated_threshold_1pct']:.4f}"
    )

    # Threshold calibration: freeze validation threshold
    threshold_data = {
        "metric": "cosine",
        "target_far": 0.01,
        "calibrated_threshold": val_metrics["calibrated_threshold_1pct"],
        "val_eer": val_metrics["eer"],
        "val_roc_auc": val_metrics["roc_auc"],
        "val_tar_at_far_1e2": val_metrics["tar_at_far_1e2"],
        "val_far": val_metrics["calibrated_far"],
        "val_frr": val_metrics["calibrated_frr"],
        "date_calibrated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(models_dir / "threshold.json", "w", encoding="utf-8") as f:
        json.dump(threshold_data, f, indent=2)
    logger.info(f"Saved frozen decision threshold to {models_dir / 'threshold.json'}")

    # Evaluate on unseen test identities with the frozen threshold
    data_dir = root_dir / config["data"]["raw_dir"]
    splits_file = root_dir / config["data"]["splits_dir"] / "open_set.json"
    with open(splits_file, encoding="utf-8") as f:
        splits = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_metrics = evaluate_model_on_split(
        model,
        splits["splits"]["test_gallery"],
        splits["splits"]["test_probes"],
        data_dir,
        device,
        target_far=0.01,
    )
    logger.info(
        f"Unseen Test Set - ROC-AUC: {test_metrics['roc_auc']:.4f}, EER: {test_metrics['eer']:.4f}, TAR@FAR=1%: {test_metrics['tar_at_far_1e2']:.4f}"
    )

    # Export to ONNX
    onnx_path = models_dir / "avatar_cnn.onnx"
    export_info = export_to_onnx(
        model,
        output_path=onnx_path,
        input_shape=(1, 3, 128, 128),
        opset_version=int(config["export"]["opset"]),
        tolerance=float(config["export"]["tolerance"]),
    )
    logger.info(
        f"Exported ONNX model ({export_info['file_size_bytes']} bytes) with max abs diff: {export_info['max_abs_diff']:.6e}"
    )

    # Save PyTorch checkpoint
    torch.save(model.state_dict(), models_dir / "avatar_cnn.pt")

    train_summary = {
        "experiment": config["experiment"]["name"],
        "fast_mode": args.fast,
        "validation_metrics": val_metrics,
        "test_metrics_unseen": test_metrics,
        "threshold": threshold_data,
        "onnx_export": export_info,
    }

    if not args.fast:
        # Run baselines comparison
        baselines_results = run_baselines_benchmark(config, val_metrics)
        train_summary["baselines_comparison"] = baselines_results

        # Run ablations
        ablations_results = run_ablation_study(config)
        with open(results_dir / "ablation.json", "w", encoding="utf-8") as f:
            json.dump(ablations_results, f, indent=2)
        train_summary["ablations"] = ablations_results

    with open(results_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(train_summary, f, indent=2)
    logger.info(f"Saved complete train summary to {results_dir / 'train_summary.json'}")


if __name__ == "__main__":
    main()
