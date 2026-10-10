# Biometric Evaluation & Benchmark Report

## 1. Evaluation Methodology & Biometric Metrics

Biometric verification and identification systems must adhere to rigorous statistical protocols (ISO/IEC 19795-1). In accordance with the academic foundation of this project, **no accuracy figure is presented without a documented test protocol and empirical measurement.**

### Core Metrics Defined
- **False Acceptance Rate (FAR)**: Proportion of impostor verification attempts that are incorrectly accepted:
  $$\text{FAR}(\tau) = \frac{|\{s \in \mathcal{S}_{\text{imp}} : s \ge \tau\}|}{|\mathcal{S}_{\text{imp}}|}$$
- **False Rejection Rate (FRR)**: Proportion of genuine verification attempts that are incorrectly rejected:
  $$\text{FRR}(\tau) = \frac{|\{s \in \mathcal{S}_{\text{gen}} : s < \tau\}|}{|\mathcal{S}_{\text{gen}}|}$$
- **True Acceptance Rate (TAR)**: $\text{TAR}(\tau) = 1 - \text{FRR}(\tau)$.
- **Equal Error Rate (EER)**: The operational threshold $\tau_{\text{EER}}$ where $\text{FAR}(\tau_{\text{EER}}) = \text{FRR}(\tau_{\text{EER}})$.
- **Receiver Operating Characteristic (ROC)**: Plot of $\text{TAR}(\tau)$ vs $\text{FAR}(\tau)$ across all potential decision thresholds $\tau \in [-1, 1]$.
- **Detection & Identification Rate (DIR / Rank-1 Identification)**: Proportion of probe queries for which the correct enrolled identity is returned with similarity $s \ge \tau^*$.

---

## 2. Experimental Protocols

### Protocol A: Closed-Set Identification
- **Participants**: All 30 dataset identities.
- **Data Partition**: 10 train images (300 total) / 2 validation images (60 total) / 3 test images (90 total) per identity.
- **Task**: Identify which of the 30 enrolled classes a probe image belongs to (Rank-1 Top-1 Accuracy).
- **Result**: **94.44% Top-1 Accuracy** (Macro-F1: 0.9441).

### Protocol B: Open-Set Biometric Verification (Unseen Identities)
- **Participants**: Identity-disjoint splits:
  - 20 Training Identities (`person_001` through `person_020`): Used exclusively to train the ArcFace feature extractor.
  - 5 Validation Identities (`person_021` through `person_025`): Used solely for threshold calibration $\tau^*$.
  - 5 Unseen Test Identities (`person_026` through `person_030`): Kept completely unseen until final evaluation.
- **Gallery / Probe Setup**: For validation and test identities, 3 randomly sampled images per person are enrolled in the gallery ($N_{\text{gallery}} = 15$), and the remaining 12 images per person act as probes ($N_{\text{probe}} = 60$).
- **Statistical Rigor**: 95% bootstrap confidence intervals computed across $B = 1,000$ resamples.

### Protocol C: Historical Attendance Replay (66 Days)
- **Participants**: Complete dataset log of 1,980 events spanning 66 instructional days (2025-07-01 to 2025-09-30).
- **Task**: Replay every log event through the recognition and attendance pipeline:
  - Verify deduplication integrity per `(student_id, session_id)`.
  - Validate 09:15 late check-in cutoff logic against recorded `Present`, `Late`, and `Absent` statuses.
  - Introduce synthetic stranger probes to audit boundary rejection behavior.

### Protocol D: Environmental Stress & Robustness Tests
- **Task**: Subject unseen probe images to 6 synthetic degradation dimensions and evaluate performance decay curves.

---

## 3. Threshold Calibration & Operating Decision Point ($\tau^*$)

> [!IMPORTANT]
> **Threshold Hygiene Rule**: An operational threshold must **never** be calibrated on the test split.

In production biometric systems, the threshold cannot be set to the theoretical Equal Error Rate ($\text{EER}$) because an institutional attendance kiosk cannot tolerate a 1%–5% impostor entry rate. We calibrate $\tau^*$ strictly on the **validation impostor distribution** to satisfy:
$$\tau^* = \min \left\{ \tau \in [-1, 1] : \text{FAR}_{\text{val}}(\tau) \le 0.010 \right\}$$

- **Calibrated Value**: $\tau^* = 0.653$
- **Validation Verification**: $\text{FAR}_{\text{val}}(0.653) = 0.98\% \le 1.0\%$.

### Frozen Evaluation on Unseen Test Split
Freezing $\tau^* = 0.653$ and evaluating on the unseen test split yields:
- **Test Operational FAR**: **0.42%** (Satisfies target $\text{FAR} \le 1.0\%$).
- **Test Operational FRR**: **0.00%**.
- **Test Operational TAR**: **100.00%**.
- **Rank-1 Identification Rate (DIR)**: **100.00%** (60 / 60 correct probe matches).

---

## 4. Benchmark Comparisons

All baselines evaluated on identical splits:

| Architecture / Baseline | Protocol | ROC-AUC | EER | TAR @ FAR=1% |
| :--- | :--- | :---: | :---: | :---: |
| Baseline 1: Pixel / HOG k-NN | Open-Set (Val) | 0.8130 | 25.49% | 36.11% |
| Baseline 2: OpenCV LBPH | Open-Set (Val) | 0.7794 | 26.81% | 8.33% |
| Baseline 3: Softmax-CE CNN | Open-Set (Val) | 1.0000 | 0.00% | 100.00% |
| **ArcFace AvatarResNet (Ours)** | **Open-Set (Unseen Test)** | **1.0000** | **0.00%** | **100.00%** |

---

## 5. Environmental Robustness & Stress Curves

| Perturbation Dimension | Sweep Range | Baseline EER | Severe Stress EER | Failure Threshold |
| :--- | :--- | :---: | :---: | :--- |
| **Gaussian Blur** | $\sigma \in [0.5, 3.5]$ | 0.00% | 0.07% ($\sigma=3.5$) | None ($\sigma > 3.5$ safe) |
| **Illumination Shift** | $\Delta Y \in [-60, +60]$ | 0.00% | 22.15% ($+60$) | $\Delta Y > +40$ |
| **Additive Noise** | $\sigma_{\text{noise}} \in [0.02, 0.20]$ | 0.00% | 1.67% ($\sigma=0.20$) | None (Resilient) |
| **Facial Occlusion** | Patch $\in [8, 48]\text{ px}$ | 0.00% | 2.08% ($48\text{ px}$) | None ($< 50\text{ px}$ safe) |
| **In-Plane Rotation** | $\theta \in [-45^\circ, +45^\circ]$ | 0.00% | 0.00% ($\pm 45^\circ$) | None (Augmentation-robust) |
| **JPEG Compression** | Quality $q \in [95, 10]$ | 0.00% | 2.15% ($q=10$) | None ($q \ge 10$ safe) |

---

## 6. Historical Attendance Replay (66 Days)

The historical log from 2025-07-01 to 2025-09-30 (1,980 total records; 1,587 Present, 250 Late, 143 Absent) was replayed chronologically:
- **Records Processed**: 1,837 active attendance events (excluding absent entries).
- **Identification Agreement**: 80.19%.
- **09:15 Cutoff Consistency**: 80.19% exact status concordance with ground truth.
- **Duplicate Prevention Integrity**: **100.00%** (zero duplicate attendance records permitted).
- **Stranger Adversary Probe Rejection**: **18.67%** (75 stranger probes audited against $\tau^*$).

---

## 7. Latency Benchmarks & Matcher Scalability

Execution times on standard x86 CPU via ONNX Runtime:

```
Inference Stage                Latency (p50)      Latency (p95)
────────────────────────────────────────────────────────────────
Quality Gate Filtering         0.42 ms            0.85 ms
Face Alignment & Crop          0.18 ms            0.35 ms
ArcFace Embedding Extraction   3.55 ms            34.86 ms
Cosine Vector Search (N=30)    0.0038 ms          0.0056 ms
Cosine Vector Search (N=10K)   0.2357 ms          0.3589 ms
────────────────────────────────────────────────────────────────
Total Pipeline Latency         ~4.15 ms           ~36.00 ms
Effective Throughput           145.8 FPS (CPU)
```
