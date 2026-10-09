"""Unit tests for model architecture, ArcFace loss, and ONNX parity."""

from pathlib import Path

import numpy as np
import torch

from fras.training.export_onnx import export_to_onnx
from fras.training.losses import ArcFaceHead
from fras.training.model import create_model


def test_avatar_resnet_forward_and_l2_norm() -> None:
    model = create_model(embedding_dim=128)
    model.eval()

    batch = torch.randn(4, 3, 128, 128)
    with torch.no_grad():
        embeddings = model(batch)

    assert embeddings.shape == (4, 128)
    # Check L2 normalization: norm of each vector must be 1.0 +/- 1e-5
    norms = torch.norm(embeddings, p=2, dim=1).numpy()
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_arcface_head_logits_shape() -> None:
    num_classes = 20
    head = ArcFaceHead(in_features=128, num_classes=num_classes, scale=32.0, margin=0.5)

    embeddings = torch.randn(8, 128)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    labels = torch.randint(0, num_classes, (8,))

    logits = head(embeddings, labels)
    assert logits.shape == (8, num_classes)


def test_onnx_export_and_numerical_parity(tmp_path: Path) -> None:
    model = create_model(embedding_dim=128)
    model.eval()

    onnx_file = tmp_path / "test_model.onnx"
    result = export_to_onnx(
        model=model,
        output_path=onnx_file,
        input_shape=(1, 3, 128, 128),
        opset_version=17,
        tolerance=1e-4,
    )

    assert result["parity_verified"] is True
    assert result["max_abs_diff"] < 1e-4
    assert onnx_file.exists()
    assert (tmp_path / "test_model.onnx.sha256").exists()
