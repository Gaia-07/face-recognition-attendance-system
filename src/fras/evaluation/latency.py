"""Latency, throughput, and matcher vector search scaling benchmarks."""

import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort


def benchmark_onnx_latency(
    onnx_path: Path,
    n_warmup: int = 20,
    n_iterations: int = 100,
) -> dict[str, float]:
    """Measures single-image inference latency on CPU using ONNX Runtime."""
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    dummy_input = np.random.randn(1, 3, 128, 128).astype(np.float32)

    # Warmup
    for _ in range(n_warmup):
        _ = session.run(None, {"input": dummy_input})

    latencies_ms: list[float] = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        _ = session.run(None, {"input": dummy_input})
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    p50 = float(np.percentile(latencies_ms, 50))
    p95 = float(np.percentile(latencies_ms, 95))
    p99 = float(np.percentile(latencies_ms, 99))
    mean_lat = float(np.mean(latencies_ms))
    fps = float(1000.0 / mean_lat) if mean_lat > 0 else 0.0

    return {
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "mean_ms": mean_lat,
        "throughput_fps": fps,
        "iterations": n_iterations,
    }


def benchmark_matcher_scaling(
    emb_dim: int = 128,
    gallery_sizes: list[int] | None = None,
    n_queries: int = 50,
) -> list[dict[str, Any]]:
    """Measures cosine similarity top-1 search latency as gallery scales 30 to 10,000."""
    if gallery_sizes is None:
        gallery_sizes = [30, 100, 500, 1000, 5000, 10000]

    scaling_results: list[dict[str, Any]] = []

    for size in gallery_sizes:
        gallery = np.random.randn(size, emb_dim).astype(np.float32)
        gallery /= np.linalg.norm(gallery, axis=1, keepdims=True)

        query = np.random.randn(1, emb_dim).astype(np.float32)
        query /= np.linalg.norm(query)

        latencies_ms: list[float] = []
        for _ in range(n_queries):
            t0 = time.perf_counter()
            sims = np.dot(gallery, query.T).flatten()
            _ = np.argmax(sims)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        scaling_results.append(
            {
                "gallery_size": size,
                "p50_search_ms": float(np.percentile(latencies_ms, 50)),
                "p95_search_ms": float(np.percentile(latencies_ms, 95)),
                "mean_search_ms": float(np.mean(latencies_ms)),
            }
        )

    return scaling_results
