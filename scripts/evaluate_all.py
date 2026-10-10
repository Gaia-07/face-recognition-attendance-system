"""Master evaluation script running closed-set, open-set, stress tests, attendance replay, and latency."""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from fras.evaluation.attendance_replay import run_attendance_replay
from fras.evaluation.latency import benchmark_matcher_scaling, benchmark_onnx_latency
from fras.evaluation.plots import (
    plot_confusion_matrix,
    plot_intra_inter_distributions,
    plot_roc_det_curves,
    plot_stress_curves,
    plot_tsne_embeddings,
)
from fras.evaluation.protocols import evaluate_closed_set, evaluate_open_set_verification
from fras.evaluation.stress import run_stress_benchmarks


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    raw_dir = root / "data" / "raw" / "face_attendance_dataset"
    splits_dir = root / "data" / "splits"
    models_dir = root / "models"
    results_dir = root / "results"
    assets_dir = root / "docs" / "assets"

    results_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = models_dir / "avatar_cnn.onnx"
    threshold_path = models_dir / "threshold.json"

    if not onnx_path.exists():
        raise FileNotFoundError(f"Trained ONNX model not found: {onnx_path}. Run make train first.")

    with open(threshold_path, encoding="utf-8") as f:
        threshold_info = json.load(f)
    frozen_threshold = float(threshold_info["calibrated_threshold"])
    print(f"Loaded frozen calibrated decision threshold: {frozen_threshold:.4f}")

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    # 1. Closed-Set Evaluation
    print("\n[1/5] Evaluating Closed-Set Protocol...")
    with open(splits_dir / "closed_set.json", encoding="utf-8") as f:
        closed_splits = json.load(f)

    closed_res = evaluate_closed_set(
        onnx_session=session,
        train_items=closed_splits["splits"]["train"],
        test_items=closed_splits["splits"]["test"],
        base_dir=raw_dir / "images",
    )
    with open(results_dir / "closed_set.json", "w", encoding="utf-8") as f:
        json.dump(closed_res, f, indent=2)

    plot_confusion_matrix(
        cm=closed_res["confusion_matrix"],
        classes=closed_res["classes"],
        save_path=assets_dir / "closed_set_confusion_matrix.png",
    )
    print(f"  Closed-Set Top-1 Acc: {closed_res['top1_accuracy'] * 100:.2f}%")
    print(f"  Closed-Set Top-5 Acc: {closed_res['top5_accuracy'] * 100:.2f}%")
    print(f"  Closed-Set Macro-F1:  {closed_res['macro_f1']:.4f}")

    # 2. Open-Set Protocol on Unseen Identities
    print("\n[2/5] Evaluating Open-Set Protocol on Unseen Identities...")
    with open(splits_dir / "open_set.json", encoding="utf-8") as f:
        open_splits = json.load(f)

    open_res = evaluate_open_set_verification(
        onnx_session=session,
        gallery_items=open_splits["splits"]["test_gallery"],
        probe_items=open_splits["splits"]["test_probes"],
        base_dir=raw_dir / "images",
        frozen_threshold=frozen_threshold,
        n_bootstrap=500,
    )
    with open(results_dir / "open_set.json", "w", encoding="utf-8") as f:
        json.dump(open_res, f, indent=2)

    plot_roc_det_curves(
        fpr=open_res["roc_curve"]["fpr"],
        tpr=open_res["roc_curve"]["tpr"],
        save_path=assets_dir / "open_set_roc.png",
    )
    plot_intra_inter_distributions(
        genuine=open_res["scores"]["genuine"],
        impostor=open_res["scores"]["impostor"],
        threshold=frozen_threshold,
        save_path=assets_dir / "score_distribution.png",
    )

    # Compute t-SNE embeddings across all 30 classes for visualization
    print("  Generating t-SNE embedding projection...")
    all_embs = []
    all_labels = []
    images_dir = raw_dir / "images"
    for p_dir in sorted(images_dir.iterdir()):
        if not p_dir.is_dir():
            continue
        for img_p in list(p_dir.glob("*.jpg"))[:10]:
            with Image.open(img_p) as im:
                arr = np.array(im.convert("RGB")).transpose(2, 0, 1).astype(np.float32) / 255.0
                arr = (arr - 0.5) / 0.5
                e = session.run(None, {"input": np.expand_dims(arr, axis=0)})[0][0]
                all_embs.append(e / np.linalg.norm(e))
                all_labels.append(p_dir.name)

    plot_tsne_embeddings(
        embeddings=np.stack(all_embs),
        labels=all_labels,
        save_path=assets_dir / "tsne_embeddings.png",
    )

    ci_m = open_res["metrics_with_95_bootstrap_ci"]
    print(
        f"  Open-Set ROC-AUC: {ci_m['roc_auc']['value']:.4f} [95% CI: {ci_m['roc_auc']['ci_lower']:.4f} - {ci_m['roc_auc']['ci_upper']:.4f}]"
    )
    print(
        f"  Open-Set EER:     {ci_m['eer']['value'] * 100:.2f}% [95% CI: {ci_m['eer']['ci_lower'] * 100:.2f}% - {ci_m['eer']['ci_upper'] * 100:.2f}%]"
    )
    print(
        f"  TAR @ FAR=1%:     {ci_m['tar_at_far_1e2']['value'] * 100:.2f}% [95% CI: {ci_m['tar_at_far_1e2']['ci_lower'] * 100:.2f}% - {ci_m['tar_at_far_1e2']['ci_upper'] * 100:.2f}%]"
    )
    print(
        f"  Operational FAR at tau* ({frozen_threshold:.3f}): {open_res['operational_metrics_at_frozen_threshold']['far'] * 100:.2f}%"
    )
    print(
        f"  Operational FRR at tau* ({frozen_threshold:.3f}): {open_res['operational_metrics_at_frozen_threshold']['frr'] * 100:.2f}%"
    )

    # 3. Robustness Stress Benchmarks
    print("\n[3/5] Running Robustness Stress Benchmarks...")
    stress_res = run_stress_benchmarks(
        onnx_session=session,
        gallery_items=open_splits["splits"]["test_gallery"],
        probe_items=open_splits["splits"]["test_probes"],
        base_dir=raw_dir / "images",
    )
    with open(results_dir / "stress_tests.json", "w", encoding="utf-8") as f:
        json.dump(stress_res, f, indent=2)

    plot_stress_curves(
        stress_data=stress_res,
        save_path=assets_dir / "stress_curves.png",
    )
    print("  Stress curves generated and saved to docs/assets/stress_curves.png")

    # 4. End-to-End Attendance Replay (66 days)
    print("\n[4/5] Running End-to-End Attendance Replay (66 Days)...")
    replay_res = run_attendance_replay(
        onnx_session=session,
        raw_dir=raw_dir,
        splits_data=open_splits,
        frozen_threshold=frozen_threshold,
        late_cutoff="09:15:00",
    )
    with open(results_dir / "attendance_replay.json", "w", encoding="utf-8") as f:
        json.dump(replay_res, f, indent=2)

    print(f"  Replayed Historical Events: {replay_res['total_replayed_events']}")
    print(f"  Identification Accuracy:    {replay_res['identification_accuracy'] * 100:.2f}%")
    print(f"  Status Agreement Rate:      {replay_res['status_agreement_rate'] * 100:.2f}%")
    print(
        f"  Stranger Rejection Rate:    {replay_res['stranger_rejection_rate'] * 100:.2f}% ({replay_res['stranger_injections']} probes)"
    )
    print(
        f"  Duplicate Block Integrity:  {replay_res['duplicate_prevention_integrity'] * 100:.2f}% ({replay_res['duplicate_blocks']}/{replay_res['duplicate_attempts']} attempts)"
    )

    # 5. Latency & Scalability Profiling
    print("\n[5/5] Profiling Inference Latency & Gallery Scaling...")
    latency_perf = benchmark_onnx_latency(onnx_path)
    scaling_perf = benchmark_matcher_scaling()
    latency_data = {
        "onnx_cpu_inference": latency_perf,
        "matcher_scaling_benchmark": scaling_perf,
    }
    with open(results_dir / "latency.json", "w", encoding="utf-8") as f:
        json.dump(latency_data, f, indent=2)
    print(
        f"  p50 Latency: {latency_perf['p50_ms']:.2f} ms | Throughput: {latency_perf['throughput_fps']:.1f} FPS"
    )

    print(
        "\nEvaluation completed successfully. All artifacts written to results/ and docs/assets/."
    )


if __name__ == "__main__":
    main()
