"""Typed repository layer for database access and persistence operations."""

from datetime import UTC, datetime

import numpy as np
from sqlalchemy import desc, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, joinedload

from fras.attendance.rules import (
    DuplicateAttendanceError,
    validate_check_in_prerequisites,
)
from fras.db.models import (
    AttendanceRecord,
    AuditLog,
    Student,
    UnknownEvent,
)
from fras.db.models import (
    Session as SessionModel,
)
from fras.security.crypto import decrypt_embedding, encrypt_embedding

# -----------------------------------------------------------------------------
# Student Repository
# -----------------------------------------------------------------------------


def create_student(
    db: Session,
    external_id: str,
    name: str,
    department: str,
    embedding: np.ndarray,
    consent_at: datetime | None = None,
) -> Student:
    """Enrolls a student with encrypted biometric embeddings."""
    encrypted_str = encrypt_embedding(embedding)
    student = Student(
        external_id=external_id,
        name=name,
        department=department,
        consent_at=consent_at or datetime.now(UTC),
        embedding_encrypted=encrypted_str,
    )
    db.add(student)
    db.commit()
    db.refresh(student)
    return student


def get_student_by_id(db: Session, student_id: int) -> Student | None:
    """Retrieves student by primary key ID."""
    return db.get(Student, student_id)


def get_student_by_external_id(db: Session, external_id: str) -> Student | None:
    """Retrieves student by external identity ID (e.g., 'person_001')."""
    stmt = select(Student).where(Student.external_id == external_id)
    return db.scalars(stmt).first()


def list_students(db: Session, department: str | None = None) -> list[Student]:
    """Lists all enrolled students, optionally filtered by department."""
    stmt = select(Student).order_by(Student.external_id)
    if department:
        stmt = stmt.where(Student.department == department)
    return list(db.scalars(stmt).all())


def get_student_embedding(student: Student) -> np.ndarray:
    """Decrypts student's stored biometric embedding."""
    return decrypt_embedding(student.embedding_encrypted)


def get_all_embeddings(db: Session) -> tuple[list[Student], np.ndarray]:
    """Retrieves all students and decrypts their embeddings into a 2D matrix.

    Returns:
        tuple of (students_list, embeddings_matrix [N, D]).
    """
    students = list_students(db)
    if not students:
        return [], np.empty((0, 128), dtype=np.float32)

    vecs = [get_student_embedding(s) for s in students]
    matrix = np.vstack(vecs).astype(np.float32)
    return students, matrix


def delete_student(db: Session, external_id: str, actor: str = "admin") -> bool:
    """GDPR Article 17 Right-to-Erasure: permanently deletes student and biometric data."""
    student = get_student_by_external_id(db, external_id)
    if not student:
        return False

    student_id_val = student.id
    db.delete(student)
    log_audit_event(
        db,
        actor=actor,
        action="ERASURE",
        resource_type="Student",
        resource_id=external_id,
        details=f"Permanently erased student ID {student_id_val} and biometric embedding per GDPR right-to-erasure request.",
    )
    db.commit()
    return True


# -----------------------------------------------------------------------------
# Session Repository
# -----------------------------------------------------------------------------


def create_session(
    db: Session,
    session_name: str,
    camera_id: str = "ENTRANCE_CAM_01",
    created_by: str = "system",
    started_at: datetime | None = None,
) -> SessionModel:
    """Creates a new attendance recording session."""
    session = SessionModel(
        session_name=session_name,
        camera_id=camera_id,
        created_by=created_by,
        started_at=started_at or datetime.now(UTC),
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def get_session_by_id(db: Session, session_id: int) -> SessionModel | None:
    """Retrieves session by ID."""
    return db.get(SessionModel, session_id)


def get_active_session(db: Session) -> SessionModel | None:
    """Returns the most recent active (open) session, or None."""
    stmt = (
        select(SessionModel)
        .where(SessionModel.ended_at.is_(None))
        .order_by(desc(SessionModel.started_at))
    )
    return db.scalars(stmt).first()


def end_session(db: Session, session_id: int, ended_at: datetime | None = None) -> SessionModel:
    """Marks an active session as closed."""
    session = get_session_by_id(db, session_id)
    if not session:
        raise ValueError(f"Session {session_id} not found.")

    session.ended_at = ended_at or datetime.now(UTC)
    db.commit()
    db.refresh(session)
    return session


def list_sessions(db: Session, limit: int = 100) -> list[SessionModel]:
    """Lists sessions ordered by start time descending."""
    stmt = select(SessionModel).order_by(desc(SessionModel.started_at)).limit(limit)
    return list(db.scalars(stmt).all())


# -----------------------------------------------------------------------------
# Attendance Repository
# -----------------------------------------------------------------------------


def get_attendance_record(db: Session, student_id: int, session_id: int) -> AttendanceRecord | None:
    """Checks if a student already has an attendance record in a session."""
    stmt = select(AttendanceRecord).where(
        AttendanceRecord.student_id == student_id,
        AttendanceRecord.session_id == session_id,
    )
    return db.scalars(stmt).first()


def record_attendance(
    db: Session,
    student_id: int,
    session_id: int,
    date: str,
    check_in_time: str | None = None,
    status: str = "Present",
    confidence: float | None = None,
    camera_id: str | None = None,
    is_manual_fallback: bool = False,
    manual_reason: str | None = None,
) -> AttendanceRecord:
    """Records attendance verifying session status and duplicate prevention."""
    session_obj = get_session_by_id(db, session_id)
    if not session_obj:
        raise ValueError(f"Session {session_id} does not exist.")

    existing = get_attendance_record(db, student_id, session_id)
    validate_check_in_prerequisites(
        session_ended_at=session_obj.ended_at,
        already_checked_in=existing is not None,
        student_id=student_id,
        session_id=session_id,
    )

    record = AttendanceRecord(
        student_id=student_id,
        session_id=session_id,
        date=date,
        check_in_time=check_in_time,
        status=status,
        confidence=confidence,
        camera_id=camera_id,
        is_manual_fallback=is_manual_fallback,
        manual_reason=manual_reason,
    )
    db.add(record)
    try:
        db.commit()
        db.refresh(record)
    except IntegrityError as e:
        db.rollback()
        raise DuplicateAttendanceError(
            f"Database unique constraint violated for student {student_id} in session {session_id}."
        ) from e

    return record


def record_checkout(
    db: Session, student_id: int, session_id: int, check_out_time: str
) -> AttendanceRecord | None:
    """Updates checkout timestamp for an existing attendance record."""
    record = get_attendance_record(db, student_id, session_id)
    if not record:
        return None
    record.check_out_time = check_out_time
    db.commit()
    db.refresh(record)
    return record


def list_attendance_records(
    db: Session,
    date: str | None = None,
    session_id: int | None = None,
    department: str | None = None,
    student_id: int | None = None,
) -> list[AttendanceRecord]:
    """Retrieves attendance records with optional filtering."""
    stmt = (
        select(AttendanceRecord)
        .options(joinedload(AttendanceRecord.student), joinedload(AttendanceRecord.session))
        .order_by(desc(AttendanceRecord.date), desc(AttendanceRecord.check_in_time))
    )
    if date:
        stmt = stmt.where(AttendanceRecord.date == date)
    if session_id is not None:
        stmt = stmt.where(AttendanceRecord.session_id == session_id)
    if student_id is not None:
        stmt = stmt.where(AttendanceRecord.student_id == student_id)
    if department:
        stmt = stmt.join(AttendanceRecord.student).where(Student.department == department)

    return list(db.scalars(stmt).all())


# -----------------------------------------------------------------------------
# Unknown Events & Audit Logs
# -----------------------------------------------------------------------------


def record_unknown_event(
    db: Session,
    similarity_score: float,
    camera_id: str,
    reason_code: str,
    session_id: int | None = None,
) -> UnknownEvent:
    """Records an unknown face detection incident."""
    event = UnknownEvent(
        session_id=session_id,
        similarity_score=similarity_score,
        camera_id=camera_id,
        reason_code=reason_code,
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return event


def list_unknown_events(db: Session, limit: int = 100) -> list[UnknownEvent]:
    """Lists unknown face events ordered by latest first."""
    stmt = select(UnknownEvent).order_by(desc(UnknownEvent.timestamp)).limit(limit)
    return list(db.scalars(stmt).all())


def log_audit_event(
    db: Session,
    actor: str,
    action: str,
    resource_type: str,
    resource_id: str,
    details: str,
) -> AuditLog:
    """Records an immutable security audit event."""
    log = AuditLog(
        actor=actor,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        details=details,
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log


def list_audit_logs(db: Session, limit: int = 100) -> list[AuditLog]:
    """Lists audit log entries ordered by latest first."""
    stmt = select(AuditLog).order_by(desc(AuditLog.timestamp)).limit(limit)
    return list(db.scalars(stmt).all())
