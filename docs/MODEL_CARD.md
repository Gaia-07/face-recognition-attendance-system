# Biometric Model Card: ArcFace AvatarResNet

## 1. Model Details

- **Model Name**: ArcFace AvatarResNet (`avatar_cnn`)
- **Version**: 1.0.0 (ONNX Export)
- **Model Type**: Deep Convolutional Neural Network with Residual Connections and ArcFace Angular Margin Head
- **Architecture**:
  - Input: $128 \times 128 \times 3$ RGB, normalized to $[-1, 1]$.
  - Backbone: 4 residual stages (channels: 32, 64, 128, 256), batch normalization, ReLU activations.
  - Head: Global Average Pooling $\to$ Linear(256, 128) $\to$ L2 Normalization.
  - Loss Objective: ArcFace (Additive Angular Margin Loss, $s=32.0, m=0.5$).
- **Embedding Dimension**: 128-dimensional unit vector ($\|\mathbf{e}\|_2 = 1.0$).
- **Deployment Format**: Open Neural Network Exchange (ONNX, opset 17).
- **Inference Runtime**: ONNX Runtime (CPUExecutionProvider).
- **Model Checkpoint**: `models/avatar_cnn.onnx` (File size: ~2.8 MB).

---

## 2. Intended Use

### In-Scope Uses
- Reproducible benchmarking and evaluation of open-set face recognition algorithms on synthetic avatars.
- Automated attendance check-in kiosk demonstration in academic, research, and technical portfolio environments.
- Educational demonstration of biometric pipeline architecture, threshold calibration, and privacy-preserving storage.

### Out-of-Scope / Prohibited Uses
- Commercial surveillance, law enforcement, or non-consensual biometric tracking.
- Physical security access control for real human subjects without switching to the `sface` backend.
- High-stakes biometric identification without human-in-the-loop oversight.

---

## 3. Training & Validation Data

- **Dataset**: `face_attendance_dataset` (CC0 1.0 Universal public domain).
- **Nature of Data**: Procedurally generated 2D cartoon avatars with simulated photographic perturbations (rotation, illumination shifts, blur, scaling).
- **Scale**: 30 identities $\times$ 15 images/identity = 450 total images ($128 \times 128$ RGB).
- **Partitioning Protocols**:
  - **Closed-Set Split**: 10 train / 2 val / 3 test images per identity across all 30 identities (300 train, 60 val, 90 test).
  - **Open-Set Disjoint Split**: 20 train identities (300 images) / 5 validation identities (15 gallery, 60 probe) / 5 unseen test identities (15 gallery, 60 probe).
  - **Leakage Prevention**: Strictly zero identity and image overlap across splits; confirmed via automated verification tests (`tests/unit/test_seed_and_dataset.py`).

---

## 4. Quantitative Evaluation

Every metric reported below is derived from experimental runs and saved in `results/*.json`.

### 4.1 Biometric Performance on Unseen Test Identities (Open-Set)

Evaluated on 5 completely unseen identities (15 enrolled gallery templates, 60 test probes; $B=1,000$ bootstrap iterations for 95% confidence intervals):

| Metric | Measured Value | 95% Bootstrap Confidence Interval |
| :--- | :---: | :---: |
| **ROC-AUC** | **1.0000** | $[1.0000, 1.0000]$ |
| **Equal Error Rate (EER)** | **0.00%** | $[0.00\%, 0.00\%]$ |
| **TAR @ FAR = 1%** | **100.00%** | $[100.00\%, 100.00\%]$ |
| **Calibrated Threshold ($\tau^*$)** | **0.653** | Calibrated on validation impostors |
| **Test Operational FAR** | **0.42%** | Satisfies target $\text{FAR} \le 1.0\%$ |
| **Test Operational FRR** | **0.00%** | Zero false rejections |
| **Rank-1 Identification (DIR)** | **100.00%** | 60 / 60 correct gallery matches |

### 4.2 Benchmark Comparison Against Baselines

| Baseline Architecture | Protocol | ROC-AUC | EER | TAR @ FAR=1% |
| :--- | :--- | :---: | :---: | :---: |
| Baseline 1: Pixel / HOG k-NN | Open-Set (Val) | 0.8130 | 25.49% | 36.11% |
| Baseline 2: OpenCV LBPH | Open-Set (Val) | 0.7794 | 26.81% | 8.33% |
| Baseline 3: Softmax-CE CNN | Open-Set (Val) | 1.0000 | 0.00% | 100.00% |
| **ArcFace AvatarResNet (Ours)** | **Open-Set (Unseen Test)** | **1.0000** | **0.00%** | **100.00%** |

### 4.3 Computational Efficiency (CPU Benchmark)

Benchmarked on single-core CPU execution via ONNX Runtime:

| Benchmark Dimension | Metric Value |
| :--- | :---: |
| Warmup Iterations | 20 runs |
| Benchmark Iterations | 100 runs |
| Median Latency ($p_{50}$) | **3.55 ms** |
| 95th Percentile Latency ($p_{95}$) | **34.86 ms** |
| Throughput | **145.8 FPS** |
| Matcher Search ($N=30$) | **0.0038 ms** |
| Matcher Search ($N=10,000$) | **0.2357 ms** |

---

## 5. Environmental Robustness & Stress Analysis

The model was subjected to systematic synthetic degradations across 6 perturbation dimensions:

1. **Gaussian Blur**: Robust up to $\sigma = 3.5$ ($\text{EER} = 0.07\%$).
2. **Illumination Shift**: Degrades under severe overexposure ($+60$ intensity shift results in $\text{EER} = 22.15\%$). Quality gates mitigate this by rejecting frames with $\mu_Y > 220$.
3. **Additive Gaussian Noise**: Highly resilient ($\text{EER} = 1.67\%$ at $\sigma_{\text{noise}} = 0.20$).
4. **2D Facial Occlusion**: Maintained zero error up to $24 \times 24$ px center occlusion; $48 \times 48$ px occlusion yielded $\text{EER} = 2.08\%$.
5. **In-Plane Rotation**: Zero degradation across $-45^\circ \to +45^\circ$ ($\text{EER} = 0.00\%$) due to spatial augmentations during training.
6. **JPEG Compression**: Extremely stable down to quality factor $q=10$ ($\text{EER} = 2.15\%$).

---

## 6. Two-Backend Architecture Rationale

> [!IMPORTANT]
> **Synthetic vs. Real Face Non-Transferability**:
> - A model trained on 2D cartoon avatars will **not** recognize real human faces, as human skin textures, facial topography, and lighting dynamics differ fundamentally from cartoon illustrations.
> - Standard human face detectors (e.g. Haar Cascades, MTCNN, YuNet) will **not** reliably detect cartoon avatars.
> - **Solution**: FRAS ships with a **two-backend design**:
>   1. `avatar_cnn`: Evaluates and runs the reproducible avatar benchmark.
>   2. `sface`: Integrates OpenCV Zoo's pretrained YuNet + SFace models for live physical webcam deployment with real people.

---

## 7. Ethical & Privacy Considerations

- **Zero Photo Storage**: The model operates in an ephemeral pipeline. Bounding box crops are embedded into 128-d vectors and immediately discarded.
- **Biometric Encryption**: Biometric templates are stored as AES-128 ciphertext tokens.
- **Cryptographic Right-to-Erasure**: Deletion immediately purges the ciphertext token and related database entities.
- **Fairness & Demographic Bias**: The synthetic cartoon dataset does not represent human racial, ethnic, or gender diversity. Do not infer human demographic parity from avatar benchmark results.
