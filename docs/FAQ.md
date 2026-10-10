# Frequently Asked Questions (FAQ)

## 🤖 Computer Vision & Machine Learning

### Q1: Why train on synthetic cartoon avatars rather than real human faces?
The academic project specification and provided dataset (`face_attendance_dataset`) consists of 30 procedurally generated cartoon avatars with simulated photographic perturbations. Using CC0 synthetic data ensures:
1. **Zero Biometric Privacy Infringement**: No real human biometric identities are distributed in public repositories.
2. **Reproducible Benchmarks**: Standardized benchmarks with fixed seeds and deterministic image crops.
3. **Honest Engineering Boundaries**: We explicitly disclose that cartoon models do not transfer to real faces, and solve this through our **two-backend design** (`avatar_cnn` for the synthetic benchmark, `sface` for real-world webcam deployment).

### Q2: Why is open-set evaluation on unseen identities necessary?
In a naive closed-set classification task on cartoon avatars, modern convolutional networks easily achieve near-100% accuracy. That proves only that distinct colors and shapes are separable. 
Evaluating under an **open-set protocol on unseen identities** tests the model's true biometric capability: extracting discriminative identity manifolds that generalize to people never seen during training.

### Q3: Why is the operating threshold ($\tau^* = 0.653$) higher than the Equal Error Rate?
The theoretical Equal Error Rate ($\text{EER}$) equalizes false acceptances ($\text{FAR}$) and false rejections ($\text{FRR}$). However, in an institutional environment, a 2% or 5% false acceptance rate means unauthorized individuals or impostors can check in for others. 
We calibrate $\tau^*$ strictly on the **validation impostor distribution** to enforce $\text{FAR} \le 1.0\%$. On unseen test identities, $\tau^* = 0.653$ achieved $\text{FAR} = 0.42\%$ and $\text{TAR} = 100.0\%$.

### Q4: Why use ArcFace instead of standard Softmax Cross-Entropy?
Standard Softmax Cross-Entropy separates classes with linear decision boundaries in Euclidean space, but does not enforce intra-class compactness or inter-class angular margin. ArcFace adds an additive angular margin $m$ directly in normalized spherical embedding space:
$$\mathcal{L}_{\text{ArcFace}} = -\frac{1}{N} \sum_{i=1}^N \log \frac{e^{s \cos(\theta_{y_i} + m)}}{e^{s \cos(\theta_{y_i} + m)} + \sum_{j \ne y_i} e^{s \cos \theta_j}}$$
This forces embeddings from the same identity into tight angular clusters, maximizing cosine similarity margins.

### Q5: What happens when a stranger/unregistered face is detected?
The cosine similarity vector $\mathbf{s} = G p^\top$ will have $s_{\max} < \tau^* = 0.653$. The matcher rejects the probe, emits an `UNKNOWN_STRANGER` classification, logs the event to the audit trail, and **never auto-enrolls unknown faces**.

---

## 🔒 Security, Privacy & Business Logic

### Q6: Does FRAS store photos of students on disk or in the database?
**No. Zero raw photographs are stored.** When a photo is submitted during enrollment or check-in, it is processed in volatile RAM to extract a 128-dimensional embedding vector, and the raw pixel tensor is immediately freed. 

### Q7: Are the biometric embeddings vulnerable if the database is leaked?
Biometric vectors are encrypted at rest with **Fernet (AES-128-CBC + HMAC-SHA256 authenticated encryption)** before database insertion. An attacker with raw access to `fras.db` cannot view or match embeddings without the encryption key stored outside the database. Furthermore, reconstructing human facial likeness from a 128-dimensional compressed embedding is mathematically non-invertible.

### Q8: How does FRAS enforce GDPR Right-to-Erasure (Art. 17)?
When an administrator issues a deletion request:
1. The student's encrypted biometric template is permanently wiped from the database.
2. Related records are cascaded or anonymized.
3. An irreversible deletion tombstone is recorded in the append-only audit trail (`logs/audit.jsonl`).

### Q9: How are duplicate check-ins prevented?
Deduplication is enforced at both the database and service levels. The database maintains a compound unique constraint on `(student_id, session_id)`. If a student attempts to scan their face multiple times during the same active lecture, the system recognizes the existing record, flags the transaction as idempotent (`already_checked_in`), and prevents duplicate attendance entries.

### Q10: How can this system run on headless servers without a webcam?
FRAS includes a `SimulatedCameraStream` that reads probe images from disk and streams them as if they were a live video capture device. This allows tests, CI pipelines, Docker containers, and cloud demos to run without physical video hardware.
