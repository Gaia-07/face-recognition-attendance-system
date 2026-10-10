"""Database seeding script: populates default users, students with embeddings, and 66-day historical attendance."""

import argparse
import glob
import logging
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sqlalchemy import select

from fras.db.models import AttendanceRecord, AuditLog, Student, User
from fras.db.models import Session as SessionModel
from fras.db.session import SessionLocal, init_db
from fras.recognition.avatar_cnn import AvatarCNNBackend
from fras.security.auth import hash_password
from fras.security.crypto import encrypt_embedding

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data"
    / "raw"
    / "face_attendance_dataset"
)


def seed_users() -> None:
    """Creates default administrative, faculty, and auditor accounts."""
    with SessionLocal() as db:
        default_accounts = [
            ("admin", "admin123", "admin"),
            ("faculty", "faculty123", "faculty"),
            ("auditor", "auditor123", "auditor"),
        ]
        for username, password, role in default_accounts:
            existing = db.scalars(select(User).where(User.username == username)).first()
            if not existing:
                u = User(
                    username=username,
                    password_hash=hash_password(password),
                    role=role,
                )
                db.add(u)
                logger.info(f"Created default user: {username} ({role})")
        db.commit()


def compute_student_template(person_id: str, backend: AvatarCNNBackend) -> np.ndarray:
    """Computes a mean enrolled biometric template from gallery images for a person."""
    img_dir = RAW_DATA_DIR / "images" / person_id
    pattern = str(img_dir / "*.jpg")
    img_paths = sorted(glob.glob(pattern))
    if not img_paths:
        raise FileNotFoundError(f"No gallery images found for identity {person_id} at {img_dir}")

    # Use first 10 images as enrollment gallery
    enroll_paths = img_paths[:10]
    valid_imgs: list[np.ndarray] = []
    for p in enroll_paths:
        im = cv2.imread(p)
        if im is not None:
            valid_imgs.append(im)
    if not valid_imgs:
        raise ValueError(f"No valid images could be read for {person_id}")

    preprocessed = backend.preprocess(np.stack(valid_imgs, axis=0))
    embs = backend.embed(preprocessed)
    mean_emb = np.mean(embs, axis=0)
    norm = float(np.linalg.norm(mean_emb))
    res: np.ndarray = (mean_emb / (norm + 1e-7)).astype(np.float32)
    return res


def seed_students() -> dict[str, int]:
    """Imports 30 students from persons.csv and extracts encrypted biometric templates.

    Returns:
        Mapping of external person_id -> database primary key student_id.
    """
    persons_csv = RAW_DATA_DIR / "metadata" / "persons.csv"
    if not persons_csv.exists():
        raise FileNotFoundError(f"Persons metadata not found at {persons_csv}")

    df_persons = pd.read_csv(persons_csv)
    backend = AvatarCNNBackend()
    id_map: dict[str, int] = {}

    with SessionLocal() as db:
        for _, row in df_persons.iterrows():
            pid = str(row["person_id"])
            existing = db.scalars(select(Student).where(Student.external_id == pid)).first()
            if existing:
                id_map[pid] = existing.id
                continue

            template_emb = compute_student_template(pid, backend)
            enc_str = encrypt_embedding(template_emb)
            enroll_dt = datetime.strptime(str(row["enrollment_date"]), "%Y-%m-%d").replace(
                tzinfo=UTC
            )

            student = Student(
                external_id=pid,
                name=str(row["name"]),
                department=str(row["department"]),
                consent_at=enroll_dt,
                embedding_encrypted=enc_str,
                created_at=enroll_dt,
            )
            db.add(student)
            db.flush()
            id_map[pid] = student.id
            logger.info(f"Enrolled student: {pid} - {row['name']} ({row['department']})")

        db.commit()

    return id_map


def seed_attendance(student_id_map: dict[str, int]) -> None:
    """Populates historical daily sessions and 1,980 attendance records from attendance_log.csv."""
    log_csv = RAW_DATA_DIR / "attendance" / "attendance_log.csv"
    if not log_csv.exists():
        raise FileNotFoundError(f"Attendance log not found at {log_csv}")

    df_log = pd.read_csv(log_csv)
    unique_dates = df_log["date"].unique()

    with SessionLocal() as db:
        for d in unique_dates:
            d_str = str(d)
            session_name = f"Daily Gate - {d_str}"
            session = db.scalars(
                select(SessionModel).where(SessionModel.session_name == session_name)
            ).first()
            if not session:
                start_dt = datetime.strptime(f"{d_str} 08:30:00", "%Y-%m-%d %H:%M:%S").replace(
                    tzinfo=UTC
                )
                end_dt = datetime.strptime(f"{d_str} 17:30:00", "%Y-%m-%d %H:%M:%S").replace(
                    tzinfo=UTC
                )
                session = SessionModel(
                    session_name=session_name,
                    started_at=start_dt,
                    ended_at=end_dt,
                    camera_id="ENTRANCE_CAM_01",
                    created_by="system_seeder",
                )
                db.add(session)
                db.flush()

            # Process day's records
            day_records = df_log[df_log["date"] == d_str]
            for _, r in day_records.iterrows():
                pid = str(r["person_id"])
                sid = student_id_map.get(pid)
                if not sid:
                    continue

                existing_rec = db.scalars(
                    select(AttendanceRecord).where(
                        AttendanceRecord.student_id == sid,
                        AttendanceRecord.session_id == session.id,
                    )
                ).first()
                if existing_rec:
                    continue

                check_in = str(r["check_in_time"]) if pd.notna(r["check_in_time"]) else None
                check_out = str(r["check_out_time"]) if pd.notna(r["check_out_time"]) else None
                conf = (
                    float(r["recognition_confidence"])
                    if pd.notna(r["recognition_confidence"])
                    else None
                )
                cam = str(r["camera_id"]) if pd.notna(r["camera_id"]) else None

                rec = AttendanceRecord(
                    student_id=sid,
                    session_id=session.id,
                    date=d_str,
                    check_in_time=check_in,
                    check_out_time=check_out,
                    status=str(r["status"]),
                    confidence=conf,
                    camera_id=cam,
                    is_manual_fallback=False,
                )
                db.add(rec)

        # Record audit entry
        audit = AuditLog(
            actor="system_seeder",
            action="SEED_DATABASE",
            resource_type="Database",
            resource_id="fras.db",
            details="Hydrated database with 30 enrolled students and 1,980 historical attendance logs.",
        )
        db.add(audit)
        db.commit()
        logger.info(f"Seeded historical attendance across {len(unique_dates)} daily sessions.")


def run_seed() -> None:
    """Initializes tables and seeds full historical data."""
    init_db()
    logger.info("Database schema initialized.")
    seed_users()
    student_map = seed_students()
    seed_attendance(student_map)
    logger.info("Database seeding completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hydrate FRAS database with default data")
    args = parser.parse_args()
    run_seed()
