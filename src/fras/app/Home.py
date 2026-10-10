"""Face Recognition Attendance System (FRAS) - Portal Home."""

import streamlit as st

from fras.db import repository
from fras.db.models import AttendanceRecord
from fras.db.session import SessionLocal, init_db

st.set_page_config(
    page_title="FRAS - Face Recognition Attendance System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize database if needed
init_db()

st.title("🎓 Face Recognition Attendance System (FRAS)")
st.caption(
    "Govt. Women's Polytechnic, Ranchi — Final Year Project | Lead ML Engineer: Bhoomika | Guide: MD Rashid"
)

st.markdown("---")

# Query high-level stats from database
with SessionLocal() as db:
    total_students = len(repository.list_students(db))
    sessions = repository.list_sessions(db)
    total_sessions = len(sessions)
    active_session = repository.get_active_session(db)
    records_count = db.query(AttendanceRecord).count()

# Metric summary cards
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric(label="Enrolled Students", value=total_students, delta="Active Roster")
with c2:
    st.metric(label="Recorded Sessions", value=total_sessions, delta="66-Day History")
with c3:
    status_str = f"Active: {active_session.session_name}" if active_session else "No Active Session"
    st.metric(
        label="Gate Session Status", value="Open" if active_session else "Closed", delta=status_str
    )
with c4:
    st.metric(label="Biometric Security", value="Fernet Encrypted", delta="AES-128 + HMAC")

st.markdown("### 🏛️ System Architecture & Biometric Pipeline")

st.markdown(
    """
```
camera / simulated stream / uploaded image
        │
  [Quality Gate] ──► Blur variance (Laplacian) · Face bounding box · Brightness
        │
  [Face Detector] ──► Avatar: pass-through crop  |  Real-World: YuNet (OpenCV Zoo)
        │
  [Biometric Aligner] ──► 5-point landmark affine warp / resize & normalize
        │
  [Embedding Backend] ──► avatar_cnn (In-House ResNet ONNX)  |  sface (OpenCV Zoo)
        │
  [Vector Matcher] ──► Cosine similarity · Margin check · Open-Set Rejection (tau* = 0.653)
        │
  [Temporal Voter] ──► k-of-n sequential frames consensus (prevents single-frame jitter)
        │
  [Attendance Rules] ──► Deduplication (student_id, session_id) · Late cutoff (09:15)
        │
  [Database Engine] ──► SQLite + SQLAlchemy 2.0 (Encrypted templates at rest)
```
"""
)

st.markdown("### 🚀 Module Navigation")

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.info(
        "#### 🎥 1. Live Attendance\nReal-time camera feed or avatar stream simulation with visual bounding box annotations and automated deduplication."
    )
    st.success(
        "#### 📊 4. Analytics Dashboard\nPunctuality trends, department rates, student attendance heatmaps, and ML next-day absence risk forecasting."
    )

with col_b:
    st.info(
        "#### 👥 2. Student Directory\nBrowse enrolled student profiles, review consent status, enroll new students with quality validation, and GDPR erasure."
    )
    st.success(
        "#### 🧪 5. Model & Evaluation\nBiometric ROC/DET curves, bootstrap 95% CIs, baseline comparisons, stress perturbations, and threshold slider."
    )

with col_c:
    st.info(
        "#### 📋 3. Attendance Records\nAudit historical attendance logs across 66 days with one-click export to CSV, Excel (.xlsx), and academic PDF."
    )
    st.success(
        "#### 🔒 6. Audit & Security\nReview immutable security audit trails, inspect login attempts, and verify encryption-at-rest integrity."
    )

st.markdown("---")
st.markdown(
    "🔒 **Biometric Privacy Guarantee:** Facial photos are never stored on disk or in the database. "
    "Biometric templates are encrypted at rest with Fernet. Full compliance with GDPR Article 17 Right-to-Erasure."
)
