"""CLI utility to benchmark inference latency, throughput, and vector search scaling on CPU."""

import json
from pathlib import Path

from fras.evaluation.latency import benchmark_matcher_scaling, benchmark_onnx_latency


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    onnx_path = root / "models" / "avatar_cnn.onnx"
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Benchmarking ONNX Runtime inference latency on CPU: {onnx_path}")
    onnx_perf = benchmark_onnx_latency(onnx_path)
    print(f"  p50 Latency: {onnx_perf['p50_ms']:.2f} ms")
    print(f"  p95 Latency: {onnx_perf['p95_ms']:.2f} ms")
    print(f"  Throughput:  {onnx_perf['throughput_fps']:.1f} FPS")

    print("\nBenchmarking Vector Matcher Scaling (30 to 10,000 gallery templates)...")
    scaling = benchmark_matcher_scaling()
    for row in scaling:
        print(
            f"  Gallery: {row['gallery_size']:>5} | p50: {row['p50_search_ms']:.4f} ms | p95: {row['p95_search_ms']:.4f} ms"
        )

    output_data = {
        "onnx_cpu_inference": onnx_perf,
        "matcher_scaling_benchmark": scaling,
    }
    with open(results_dir / "latency.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved latency benchmarks to {results_dir / 'latency.json'}")


if __name__ == "__main__":
    main()
