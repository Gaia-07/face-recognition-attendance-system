"""Unit tests for database schema, models, repositories, and constraints."""

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from fras.attendance.rules import DuplicateAttendanceError, SessionClosedError
from fras.db import repository
from fras.db.models import Base


@pytest.fixture
def memory_db() -> Session:
    """Provides a clean in-memory SQLite database session for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    yield session
    session.close()


def test_create_and_retrieve_student(memory_db: Session) -> None:
    dummy_emb = np.random.randn(128).astype(np.float32)
    dummy_emb /= np.linalg.norm(dummy_emb)

    student = repository.create_student(
        db=memory_db,
        external_id="test_001",
        name="Test Student",
        department="Computer Science",
        embedding=dummy_emb,
    )

    assert student.id is not None
    assert student.external_id == "test_001"
    assert student.embedding_encrypted != ""

    retrieved = repository.get_student_by_external_id(memory_db, "test_001")
    assert retrieved is not None
    assert retrieved.name == "Test Student"

    decrypted = repository.get_student_embedding(retrieved)
    assert np.allclose(decrypted, dummy_emb, atol=1e-5)


def test_unique_student_session_attendance_constraint(memory_db: Session) -> None:
    dummy_emb = np.zeros(128, dtype=np.float32)
    student = repository.create_student(
        db=memory_db,
        external_id="test_002",
        name="Alice",
        department="Engineering",
        embedding=dummy_emb,
    )

    session = repository.create_session(db=memory_db, session_name="Morning Gate")

    # First check-in succeeds
    rec1 = repository.record_attendance(
        db=memory_db,
        student_id=student.id,
        session_id=session.id,
        date="2026-10-10",
        check_in_time="09:00:00",
        status="Present",
        confidence=0.95,
        camera_id="CAM_01",
    )
    assert rec1.id is not None

    # Second check-in in same session must raise DuplicateAttendanceError
    with pytest.raises(DuplicateAttendanceError):
        repository.record_attendance(
            db=memory_db,
            student_id=student.id,
            session_id=session.id,
            date="2026-10-10",
            check_in_time="09:05:00",
            status="Present",
            confidence=0.96,
            camera_id="CAM_01",
        )


def test_session_closed_rejection(memory_db: Session) -> None:
    student = repository.create_student(
        db=memory_db,
        external_id="test_003",
        name="Bob",
        department="Mathematics",
        embedding=np.zeros(128, dtype=np.float32),
    )

    session = repository.create_session(db=memory_db, session_name="Afternoon Lab")
    repository.end_session(memory_db, session.id)

    with pytest.raises(SessionClosedError):
        repository.record_attendance(
            db=memory_db,
            student_id=student.id,
            session_id=session.id,
            date="2026-10-10",
            check_in_time="14:00:00",
            status="Present",
            confidence=0.98,
            camera_id="CAM_02",
        )


def test_gdpr_erasure_and_audit_log(memory_db: Session) -> None:
    _ = repository.create_student(
        db=memory_db,
        external_id="test_004",
        name="Charlie",
        department="Physics",
        embedding=np.zeros(128, dtype=np.float32),
    )

    success = repository.delete_student(memory_db, "test_004", actor="dpo_admin")
    assert success is True

    # Student should no longer exist
    assert repository.get_student_by_external_id(memory_db, "test_004") is None

    # Audit log entry must exist
    audit_logs = repository.list_audit_logs(memory_db)
    assert len(audit_logs) >= 1
    assert any(log.action == "ERASURE" and log.resource_id == "test_004" for log in audit_logs)
