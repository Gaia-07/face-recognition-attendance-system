"""Unit tests for ArcFace loss head, biometric augmentations, and training dataset components."""

import numpy as np
import torch
import torch.nn as nn

from fras.training.augment import build_train_transform, build_val_transform
from fras.training.losses import ArcFaceHead


def test_arcface_loss_head_forward_and_loss() -> None:
    batch_size = 8
    in_features = 128
    num_classes = 10

    head = ArcFaceHead(in_features=in_features, num_classes=num_classes, scale=32.0, margin=0.5)

    # Random unit embeddings
    raw_emb = torch.randn(batch_size, in_features)
    embeddings = raw_emb / torch.norm(raw_emb, dim=1, keepdim=True)
    labels = torch.randint(0, num_classes, (batch_size,))

    logits = head(embeddings, labels)
    assert logits.shape == (batch_size, num_classes)

    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, labels)
    assert not torch.isnan(loss)
    assert loss.item() > 0.0


def test_arcface_loss_easy_margin() -> None:
    head = ArcFaceHead(in_features=64, num_classes=5, scale=16.0, margin=0.3, easy_margin=True)
    embeddings = torch.randn(4, 64)
    embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)
    labels = torch.tensor([0, 1, 2, 3])

    logits = head(embeddings, labels)
    assert logits.shape == (4, 5)


def test_biometric_augmentations() -> None:
    img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)

    train_aug = build_train_transform(image_size=128)
    res_train = train_aug(image=img)["image"]
    assert res_train.shape == (128, 128, 3)

    val_aug = build_val_transform(image_size=128)
    res_val = val_aug(image=img)["image"]
    assert res_val.shape == (128, 128, 3)


def test_train_epoch_and_evaluation() -> None:
    from pathlib import Path

    from fras.training.model import AvatarResNet
    from fras.training.train import extract_embeddings_for_items, train_epoch

    model = AvatarResNet(embedding_dim=128)
    head = ArcFaceHead(in_features=128, num_classes=2)
    device = torch.device("cpu")
    optimizer = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    dummy_imgs = torch.randn(4, 3, 128, 128)
    dummy_labels = torch.tensor([0, 0, 1, 1])
    dummy_pids = ["p1", "p1", "p2", "p2"]

    class MockLoader:
        def __iter__(self):
            yield dummy_imgs, dummy_labels, dummy_pids

        def __len__(self):
            return 1

    loss = train_epoch(model, head, MockLoader(), criterion, optimizer, device)
    assert loss > 0.0

    raw_dir = Path("data/raw/face_attendance_dataset")
    if raw_dir.exists():
        items = [{"person_id": "person_001", "relative_path": "person_001/person_001_01.jpg"}]
        embs, pids = extract_embeddings_for_items(model, items, raw_dir, device)
        assert embs.shape == (1, 128)
        assert pids == ["person_001"]


def test_evaluate_model_on_split() -> None:
    from pathlib import Path

    from fras.training.model import AvatarResNet
    from fras.training.train import evaluate_model_on_split

    raw_dir = Path("data/raw/face_attendance_dataset")
    if not raw_dir.exists():
        return

    model = AvatarResNet(embedding_dim=128)
    device = torch.device("cpu")
    gallery = [{"person_id": "person_001", "relative_path": "person_001/person_001_01.jpg"}]
    probes = [
        {"person_id": "person_001", "relative_path": "person_001/person_001_02.jpg"},
        {"person_id": "person_002", "relative_path": "person_002/person_002_01.jpg"},
    ]
    metrics = evaluate_model_on_split(model, gallery, probes, raw_dir, device)
    assert "eer" in metrics
    assert "roc_auc" in metrics
