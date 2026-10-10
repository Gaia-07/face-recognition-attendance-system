# Executive Portfolio Summary (For Technical Recruiters & Hiring Managers)

## 📌 Project Overview
**Face Recognition Attendance System (FRAS)** is an end-to-end, production-grade biometric attendance and identity platform. It evolves an academic computer vision report (*Govt. Women's Polytechnic Ranchi*) into an enterprise AI product meeting institutional privacy requirements (GDPR Art. 9/17, FERPA) and rigorous machine learning evaluation standards.

---

## 🎯 Engineering Competencies Demonstrated

### 1. Computer Vision & Biometric Systems Engineering
- **Open-Set Biometric Modeling**: Built an ArcFace ResNet embedder trained with additive angular margin loss ($s=32.0, m=0.5$).
- **Principled Evaluation**: Evaluated on unseen identities under an open-set protocol with 95% bootstrap confidence intervals ($B=1,000$).
- **Threshold Hygiene**: Calibrated decision threshold ($\tau^* = 0.653$) on validation impostor distributions to satisfy $\text{FAR} \le 1.0\%$. Achieved **0.42% FAR** and **100% TAR** on unseen test subjects.
- **Two-Backend Architecture**: Decoupled engine allowing instant switching between in-house ONNX models (`avatar_cnn`) and pretrained real-face models (`sface`).
- **High Throughput Inference**: Exported to ONNX Runtime achieving **145.8 FPS** single-threaded CPU throughput and **< 0.24 ms** top-1 vector search at 10,000 enrolled identities.

### 2. Biometric Privacy, Security & Compliance
- **Privacy by Design**: Zero photographic images stored in the database or filesystem. Images are embedded in memory and immediately freed.
- **Biometric Encryption at Rest**: 128-dimensional biometric vectors stored as ciphertext tokens using Fernet authenticated encryption (AES-128-CBC + HMAC-SHA256).
- **Cryptographic Right-to-Erasure**: Complete GDPR Art. 17 deletion workflows that purge biometric templates and cascade relational records.
- **Tamper-Evident Auditability**: Dual logging architecture capturing all enrollments, check-ins, overrides, and deletions across SQLite and an append-only JSONL log.

### 3. Full-Stack Systems & Production Architecture
- **FastAPI Asynchronous Microservice**: Fully typed REST API with OAuth2 JWT role-based access control (Admin, Faculty, Auditor) and automatic Swagger/OpenAPI documentation.
- **Multi-Page Streamlit Kiosk**: 6 specialized web interfaces for live check-in, student management, records search, analytics heatmaps, model card explorer, and security auditing.
- **Reporting & Business Rules**: Automated 09:15 late cutoff enforcement, idempotent duplicate check-in prevention, and dynamic report exports in CSV, Excel (`.xlsx`), and formal academic PDF.
- **Predictive ML Analytics**: Chronological Random Forest absence risk modeling forecasting student attendance probabilities.

### 4. Software Quality & Engineering Rigor
- **Test Automation**: 74 unit tests achieving **87% overall line coverage**.
- **Static Typing**: 100% type annotations strictly verified via `mypy --strict` with zero errors across 61 source files.
- **Code Standards**: Formatted and linted with Astral `ruff`.
- **Packaging & DevOps**: Standardized `Makefile`, multi-stage `Dockerfile`, `docker-compose.yml`, and GitHub Actions CI workflow.

---

## 💻 Tech Stack Snapshot

- **Languages & Frameworks**: Python 3.11, PyTorch 2.x, ONNX Runtime, OpenCV, Albumentations
- **API & Web**: FastAPI, Uvicorn, Pydantic v2, Streamlit
- **Persistence & Cryptography**: SQLAlchemy Core/ORM, SQLite, Fernet (cryptography), Passlib, PyJWT
- **Reporting & Analytics**: ReportLab, OpenPyXL, Pandas, Scikit-Learn
- **DevOps & Tooling**: Docker, Docker Compose, Ruff, Mypy, Pytest, Git (Conventional Commits)

---

## 🚀 Live Demo Verification
Launch the complete platform with a single command:
```bash
docker compose up --build
```
- Multi-Page Portal: `http://localhost:8501`
- REST API Documentation: `http://localhost:8000/docs`
