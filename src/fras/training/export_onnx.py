"""ONNX model export and numerical parity verification."""

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn


def export_to_onnx(
    model: nn.Module,
    output_path: str | Path,
    input_shape: tuple[int, int, int, int] = (1, 3, 128, 128),
    opset_version: int = 17,
    tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Exports PyTorch embedder to ONNX and asserts numerical parity with ONNX Runtime.

    Raises:
        AssertionError: If max absolute difference between PyTorch and ONNX exceeds tolerance.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy_input = torch.randn(*input_shape, dtype=torch.float32)

    # Compute PyTorch ground truth
    with torch.no_grad():
        pt_output = model(dummy_input).cpu().numpy()

    # Export ONNX via classic TorchScript exporter
    torch.onnx.export(
        model,
        (dummy_input,),
        str(out_file),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "embedding": {0: "batch_size"},
        },
        dynamo=False,
    )

    # Validate ONNX model graph structure
    onnx_model = onnx.load(str(out_file))
    onnx.checker.check_model(onnx_model)

    # Run ONNX Runtime inference
    session = ort.InferenceSession(str(out_file), providers=["CPUExecutionProvider"])
    ort_inputs = {"input": dummy_input.numpy()}
    ort_output = session.run(None, ort_inputs)[0]

    # Verify parity
    max_abs_diff = float(np.max(np.abs(pt_output - ort_output)))
    if max_abs_diff >= tolerance:
        raise AssertionError(
            f"ONNX parity violation! Max abs diff: {max_abs_diff:.6e} >= tolerance: {tolerance}"
        )

    # Compute SHA-256 checksum
    hasher = hashlib.sha256()
    with open(out_file, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    checksum = hasher.hexdigest()

    checksum_file = out_file.with_suffix(".onnx.sha256")
    with open(checksum_file, "w", encoding="utf-8") as f:
        f.write(f"{checksum}  {out_file.name}\n")

    return {
        "onnx_path": str(out_file),
        "checksum_sha256": checksum,
        "max_abs_diff": max_abs_diff,
        "file_size_bytes": out_file.stat().st_size,
        "opset_version": opset_version,
        "parity_verified": True,
    }
