# Biometric Privacy, Ethics, and Governance Charter

> **Institutional Origin:** Govt. Women's Polytechnic, Ranchi  
> **Reference Document:** Final Year Project Report — Chapter 16: *Biometric Privacy, Legal Compliance, and Ethical Framework*  
> **System Name:** Face Recognition Attendance System (FRAS)  
> **Applicable Standards:** GDPR (General Data Protection Regulation, Articles 5, 9, 17), ISO/IEC 19794-5 (Biometric Data Interchange), ISO/IEC 24745 (Biometric Information Protection), India Digital Personal Data Protection (DPDP) Act.

---

## 1. Executive Summary & Core Ethical Creed

Biometric facial recognition involves processing sensitive personal data that uniquely identifies individuals. At Govt. Women's Polytechnic, Ranchi, the implementation of automated biometric attendance is governed by the strict principle that **technology must empower students, not subject them to unwarranted surveillance**.

FRAS adheres to an **Ethics-by-Design** and **Privacy-by-Architecture** paradigm:
1. Attendance tracking serves solely educational administration (verifying physical presence for lectures and laboratories).
2. Biometric processing must never be used for punitive, intrusive, or tracking purposes outside the designated attendance window.
3. Every student retains full sovereignty over their identity, with guaranteed manual fallback and permanent erasure upon request.

---

## 2. Seven Biometric Architectural Safeguards

```
[Camera Stream / Ingest]
         │ (In-memory frame only)
         ▼
[Detection & Feature Extractor]
         │
         ├──► Raw RGB Frame ──► [IMMEDIATE MEMORY PURGE (0 Raw Storage)]
         ▼
[128-d Float32 Embedding]
         │
         ▼
[Fernet Authenticated Encryption] ──► AES-128-CBC + HMAC-SHA256
         │
         ▼
[Encrypted SQLite Database] ───────► Column: `students.embedding_encrypted`
```

### 1. In-Memory Processing & Zero Raw Image Retention
- **Policy:** FRAS **never persists raw photographic images** of students to disk or database during routine attendance operations.
- **Mechanism:** Facial frames captured from the video pipeline are cropped, normalized, transformed into a 128-dimensional embedding vector, and immediately released by Python garbage collection.
- **Audit:** Automated repository tests (`test_security.py`) assert that unknown rejections and attendance check-ins store only numerical similarity scores and metadata, not images.

### 2. Biometric Template Protection at Rest (ISO/IEC 24745)
- **Policy:** Enrolled facial templates are encrypted using authenticated cryptography before persisting to the database.
- **Mechanism:** Fernet symmetric encryption (AES-128-CBC cipher with HMAC-SHA256 authentication). The decryption key is injected at runtime via the `BIOMETRIC_ENCRYPTION_KEY` environment variable.
- **Irreversibility:** Because raw faces are not retained and embeddings represent high-level latent representations, compromised database tables cannot be used to reconstruct a photographic likeness.

### 3. Explicit, Timestamped Informed Consent (GDPR Art. 6 & 9)
- **Policy:** Biometric enrollment is strictly opt-in with informed parental/student consent.
- **Mechanism:** The database model `Student` enforces a non-nullable `consent_at` timestamp. Students who decline biometric enrollment are assigned standard ID-card or manual roster attendance with zero academic disparity.

### 4. Right to Erasure / "Right to be Forgotten" (GDPR Art. 17)
- **Policy:** Any student or faculty member may formally request the irrevocable deletion of their biometric templates and personal profile.
- **Mechanism:** The system provides an audited endpoint and repository method `repository.delete_student(db, external_id)` that cascades deletion to the encrypted template and student record, while writing an immutable record to `audit_log` confirming compliance.

### 5. Multi-Tier Role-Based Access Control (RBAC)
To prevent unauthorized inspection of attendance records or administrative tampering, access is partitioned into three discrete roles:

| Role | Permitted Capabilities | Restrictions |
| :--- | :--- | :--- |
| **Admin** | Full system configuration, model threshold adjustment, student enrollment, GDPR erasure, security audit review. | Restricted to authorized IT / DPO officers. |
| **Faculty** | Session initiation, live attendance monitoring, manual fallback overrides for classroom disputes. | Cannot view cryptographic keys, cannot export audit logs. |
| **Auditor** | Read-only access to anonymized reports, punctuality trends, and compliance logs. | Cannot view individual student profiles or modify records. |

### 6. Guaranteed Manual Fallback Policy
- **Policy:** Under no circumstances shall a student be marked absent or suffer disciplinary consequences due to an algorithmic false rejection, camera hardware failure, or lighting perturbation.
- **Mechanism:** Faculty dashboards include a one-click **Manual Fallback Entry** tool requiring a brief explanation (e.g., *"Medical exception"*, *"Camera glare"*, *"Network latency"*). Manual entries are clearly tagged (`is_manual_fallback=True`) and permanently logged.

### 7. Dual-Destination Immutable Audit Trail
- **Policy:** All administrative modifications, role logins, manual attendance entries, and biometric erasures are recorded in an tamper-evident audit trail.
- **Mechanism:** Every action writes both to the relational database table `audit_log` and to an append-only JSON Lines file (`logs/audit.jsonl`).

---

## 3. Data Retention and Destruction Schedule

| Data Category | Storage Location | Retention Window | Purge Procedure |
| :--- | :--- | :--- | :--- |
| **Raw Camera Frames** | Volatile RAM only | ~50 ms (frame processing duration) | Overwritten immediately by frame loop. |
| **Encrypted Biometric Templates** | SQLite (`students.embedding_encrypted`) | Active enrollment period (max 3 years) | Irrevocably scrubbed upon graduation or GDPR erasure request. |
| **Attendance Check-in Records** | SQLite (`attendance`) | 1 Academic Year (365 days) | Anonymized into aggregate statistics; logs archived. |
| **Audit Logs** | SQLite (`audit_log`) & `logs/audit.jsonl` | 3 Years (statutory audit requirement) | Encrypted cold storage archive. |

---

## 4. Synthetic Data Disclosure & Fair Use Statement

The reproducible benchmark dataset included in this repository (`face_attendance_dataset/`, 30 identities, 450 images) is **100% synthetically generated cartoon avatars** licensed under Creative Commons Zero (CC0).

- **Purpose of Avatar Dataset:** Enables open-source reproducibility, CI pipeline validation, and open-set metric verification without handling real human biometric identifiers or compromising personal privacy.
- **Real-World Deployment:** For real-world webcams, FRAS switches seamlessly to the `sface` backend using YuNet landmark detection and OpenCV Zoo SFace ONNX models trained on public biometric research benchmarks.

---

## 5. Contact & Governance Officer

* **Data Protection Officer (DPO):** Dr. MD Rashid (Project Guide)  
* **Lead System Architect:** Bhoomika (Principal ML Engineer & Lead Author)  
* **Institution:** Government Women's Polytechnic, Ranchi, Jharkhand, India  
* **Compliance Inquiry Channel:** `dpo@gwpranchi.ac.in`
